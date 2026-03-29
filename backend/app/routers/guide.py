import uuid

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import async_session_factory
from app.dependencies import get_current_user, get_db
from app.models.postgres.navigation_log import NavigationLog
from app.models.postgres.user import User
from app.schemas.guide import NavigationRequest, NavigationResponse, NavStep, ObstacleRequest, ObstacleResponse
from app.schemas.sign import SpeakRequest, SpeakResponse
from app.services.auth_service import decode_access_token
from app.services.navigation_service import get_walking_directions, is_near_waypoint
from app.services.tts_service import speak
from app.services.vision.manager import vision_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# In-memory route cache for navigation WebSocket
_route_cache: dict[str, dict] = {}


@router.post("/obstacle", response_model=ObstacleResponse)
async def detect_obstacle(
    req: ObstacleRequest,
    user: User = Depends(get_current_user),
):
    """Analyze a camera frame for obstacles. Returns warning + audio."""
    result = await vision_manager.detect_obstacle(req.frame)

    # Skip TTS for clear paths — saves ~500ms per scan
    if result.severity == "clear":
        return ObstacleResponse(
            warning=result.text,
            severity=result.severity,
            audio_url="",
            provider=result.provider,
            latency_ms=int(result.latency_ms),
        )

    # Generate TTS only for actual warnings
    audio_file = await speak(result.text, emotion="guide")

    return ObstacleResponse(
        warning=result.text,
        severity=result.severity,
        audio_url=f"/audio/{audio_file}",
        provider=result.provider,
        latency_ms=int(result.latency_ms),
    )


@router.post("/navigate", response_model=NavigationResponse)
async def get_navigation(
    req: NavigationRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get walking directions to a destination."""
    route_data = get_walking_directions(req.lat, req.lng, req.destination)

    if not route_data["steps"]:
        return NavigationResponse(
            route_id="",
            steps=[],
            total_distance="",
            total_duration="",
        )

    # Log navigation in DB
    nav_log = NavigationLog(
        user_id=user.id,
        origin_lat=req.lat,
        origin_lng=req.lng,
        destination=req.destination,
        total_steps=len(route_data["steps"]),
    )
    db.add(nav_log)
    await db.flush()

    # Cache route for WebSocket step advancement
    route_id = route_data["route_id"]
    _route_cache[route_id] = {
        "steps": route_data["steps"],
        "nav_log_id": nav_log.id,
        "user_id": str(user.id),
        "current_step": 0,
    }

    steps = [
        NavStep(
            instruction=s["instruction"],
            distance=s["distance"],
            duration=s["duration"],
            lat=s.get("lat"),
            lng=s.get("lng"),
        )
        for s in route_data["steps"]
    ]

    return NavigationResponse(
        route_id=route_id,
        steps=steps,
        total_distance=route_data["total_distance"],
        total_duration=route_data["total_duration"],
    )


@router.websocket("/ws/nav")
async def guide_nav_websocket(websocket: WebSocket):
    """WebSocket for real-time navigation step advancement.

    Client sends location updates. Server pushes next step when close to waypoint.
    """
    token = websocket.query_params.get("token")
    route_id = websocket.query_params.get("route_id")

    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return

    user_id = decode_access_token(token)
    if not user_id:
        await websocket.close(code=4001, reason="Invalid token")
        return

    if not route_id or route_id not in _route_cache:
        await websocket.close(code=4002, reason="Invalid route_id")
        return

    await websocket.accept()
    logger.info("guide_ws_connected", user_id=user_id, route_id=route_id)

    route = _route_cache[route_id]
    steps = route["steps"]

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "location_update":
                lat = data["lat"]
                lng = data["lng"]
                current_step = route["current_step"]

                if current_step >= len(steps):
                    # Already arrived
                    await websocket.send_json({
                        "type": "arrived",
                        "message": "You have arrived at your destination.",
                    })
                    continue

                step = steps[current_step]
                step_lat = step.get("lat")
                step_lng = step.get("lng")

                if step_lat and step_lng and is_near_waypoint(lat, lng, step_lat, step_lng):
                    # Advance to next step
                    route["current_step"] = current_step + 1
                    next_step_idx = current_step + 1

                    if next_step_idx >= len(steps):
                        # Arrived
                        audio_file = await speak(
                            "You have arrived at your destination.", emotion="guide"
                        )
                        await websocket.send_json({
                            "type": "arrived",
                            "message": "You have arrived at your destination.",
                            "audio_url": f"/audio/{audio_file}",
                        })

                        # Update nav log
                        try:
                            async with async_session_factory() as db:
                                from sqlalchemy import update
                                from datetime import datetime, timezone

                                await db.execute(
                                    update(NavigationLog)
                                    .where(NavigationLog.id == route["nav_log_id"])
                                    .values(
                                        completed_steps=len(steps),
                                        ended_at=datetime.now(timezone.utc),
                                    )
                                )
                                await db.commit()
                        except Exception:
                            pass
                    else:
                        next_instruction = steps[next_step_idx]["instruction"]
                        audio_file = await speak(next_instruction, emotion="guide")
                        await websocket.send_json({
                            "type": "next_step",
                            "step_index": next_step_idx,
                            "instruction": next_instruction,
                            "audio_url": f"/audio/{audio_file}",
                        })

    except WebSocketDisconnect:
        logger.info("guide_ws_disconnected", user_id=user_id, route_id=route_id)
    except Exception:
        logger.exception("guide_ws_error")
    finally:
        # Clean up route cache after a while (don't delete immediately in case of reconnect)
        pass


@router.post("/speak", response_model=SpeakResponse)
async def guide_speak(
    req: SpeakRequest,
    user: User = Depends(get_current_user),
):
    """Speak arbitrary text in guide mode voice."""
    audio_file = await speak(req.text, emotion="guide")
    return SpeakResponse(sentence=req.text, audio_url=f"/audio/{audio_file}")
