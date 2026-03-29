import asyncio
import time
import uuid

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import async_session_factory
from app.dependencies import get_current_user, get_db
from app.models.postgres.transcript import SignSession, TranscriptEntry
from app.models.postgres.user import User
from app.schemas.sign import SOSRequest, SpeakRequest, SpeakResponse
from app.services.auth_service import decode_access_token
from app.services.session_manager import (
    clear_buffer,
    delete_session,
    get_session,
    set_language,
)
from app.services.sos_service import contains_sos_trigger, trigger_sos
from app.services.tts_service import speak
from app.services.vision.manager import vision_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Frame collection settings
COLLECTION_WINDOW_S = 7.0     # Collect frames for 7 seconds
SAMPLE_FRAMES = 8             # Sample this many frames from the window (evenly spaced)
MAX_BUFFERED_FRAMES = 50      # Max frames to hold in memory (~10s at 5fps)
IDLE_TIMEOUT_S = 3.0          # Seconds without frames before auto-processing


@router.websocket("/ws")
async def sign_websocket(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return
    user_id = decode_access_token(token)
    if not user_id:
        await websocket.close(code=4001, reason="Invalid token")
        return

    await websocket.accept()
    logger.info("sign_ws_connected", user_id=user_id)

    db_session_id = None
    try:
        async with async_session_factory() as db:
            sign_session = SignSession(user_id=uuid.UUID(user_id))
            db.add(sign_session)
            await db.flush()
            db_session_id = sign_session.id
            await db.commit()
    except Exception:
        logger.exception("sign_session_create_error")

    # Per-connection state
    frame_buffer: list[str] = []       # Collected base64 frames
    collection_start: float | None = None
    last_frame_time: float = time.monotonic()
    processing = False                  # True while pipeline is running
    language = "en"

    async def _process_frames():
        """Take buffered frames → Bedrock vision → Bedrock sentence → TTS → send."""
        nonlocal frame_buffer, collection_start, processing

        if not frame_buffer or processing:
            return

        processing = True
        frames = frame_buffer.copy()
        frame_buffer.clear()
        collection_start = None

        try:
            # Sample evenly spaced frames
            sampled = _sample_frames(frames, SAMPLE_FRAMES)
            logger.info(
                "processing_frames",
                total_frames=len(frames),
                sampled=len(sampled),
                user_id=user_id,
            )

            # Step 1: Bedrock Vision — describe ASL signs from frames
            vision_result = await vision_manager.describe_sign_sequence(sampled)
            signs, emotion = _parse_vision_output(vision_result.text)
            logger.info(
                "vision_described",
                signs=signs,
                emotion=emotion,
                provider=vision_result.provider,
                latency_ms=int(vision_result.latency_ms),
            )

            # Send intermediate result to client
            await websocket.send_json({
                "type": "signs_detected",
                "signs": signs,
                "emotion": emotion,
            })

            if signs.upper() == "NONE" or not signs.strip():
                await websocket.send_json({
                    "type": "status",
                    "message": "No signs detected, keep signing...",
                })
                return

            # Check for SOS trigger
            if contains_sos_trigger(signs):
                async with async_session_factory() as db:
                    sentence, audio_file = await trigger_sos(user_id, db)
                    await db.commit()
                await websocket.send_json({
                    "type": "sos_triggered",
                    "text": sentence,
                    "audio_url": f"/audio/{audio_file}",
                })
                return

            # Step 2: Bedrock Text — generate natural sentence from signs
            sentence_result = await vision_manager.build_sentence(signs, emotion)
            sentence = sentence_result.text
            logger.info(
                "sentence_built",
                raw=signs,
                sentence=sentence,
                provider=sentence_result.provider,
                latency_ms=int(sentence_result.latency_ms),
            )

            # Translate if non-English
            if language != "en":
                try:
                    tr = await vision_manager.translate(sentence, language)
                    sentence = tr.text
                except Exception:
                    pass

            # Step 3: ElevenLabs TTS — generate speech with emotion
            audio_file = await speak(sentence, emotion=emotion)

            # Save transcript
            if db_session_id:
                asyncio.create_task(_save_transcript(
                    db_session_id, sentence, signs, emotion, language
                ))

            # Send final result to client
            await websocket.send_json({
                "type": "sentence",
                "text": sentence,
                "emotion": emotion,
                "audio_url": f"/audio/{audio_file}",
            })

        except Exception:
            logger.exception("pipeline_error", user_id=user_id)
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": "Processing failed, please try again",
                })
            except Exception:
                pass
        finally:
            processing = False

    try:
        while True:
            # Use a timeout so we can detect idle periods
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                # Check if we should auto-process due to idle
                now = time.monotonic()
                if (
                    frame_buffer
                    and not processing
                    and (now - last_frame_time) >= IDLE_TIMEOUT_S
                ):
                    await _process_frames()
                continue

            msg_type = data.get("type")

            if msg_type == "frame":
                frame_b64 = data.get("frame", "")
                if not frame_b64:
                    continue

                last_frame_time = time.monotonic()

                # Start collection window on first frame
                if collection_start is None:
                    collection_start = last_frame_time

                # Buffer the frame
                if len(frame_buffer) < MAX_BUFFERED_FRAMES:
                    frame_buffer.append(frame_b64)

                # Check if collection window is complete
                elapsed = last_frame_time - collection_start
                if elapsed >= COLLECTION_WINDOW_S and not processing:
                    await _process_frames()

            elif msg_type == "clear_buffer":
                frame_buffer.clear()
                collection_start = None
                await websocket.send_json({"type": "letter", "letter": "", "buffer": ""})

            elif msg_type == "set_language":
                language = data.get("language", "en")
                await set_language(user_id, language)

            elif msg_type == "force_process":
                # Allow client to manually trigger processing
                if frame_buffer and not processing:
                    await _process_frames()

    except WebSocketDisconnect:
        logger.info("sign_ws_disconnected", user_id=user_id)
    except Exception:
        logger.exception("sign_ws_error", user_id=user_id)
    finally:
        # Process any remaining frames before disconnect
        if frame_buffer and not processing:
            try:
                await _process_frames()
            except Exception:
                pass
        await delete_session(user_id)
        if db_session_id:
            try:
                async with async_session_factory() as db:
                    from datetime import datetime, timezone

                    from sqlalchemy import update

                    await db.execute(
                        update(SignSession)
                        .where(SignSession.id == db_session_id)
                        .values(ended_at=datetime.now(timezone.utc))
                    )
                    await db.commit()
            except Exception:
                pass


def _sample_frames(frames: list[str], n: int) -> list[str]:
    """Evenly sample n frames from the list."""
    if len(frames) <= n:
        return frames
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def _parse_vision_output(text: str) -> tuple[str, str]:
    """Parse 'SIGNS: ...\nEMOTION: ...' from vision model output."""
    signs = "NONE"
    emotion = "neutral"
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("SIGNS:"):
            signs = line.split(":", 1)[1].strip()
        elif line.upper().startswith("EMOTION:"):
            emotion = line.split(":", 1)[1].strip().lower()
            valid = {"happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"}
            if emotion not in valid:
                emotion = "neutral"
    return signs, emotion


async def _save_transcript(session_id, text, raw_signs, emotion, language):
    try:
        async with async_session_factory() as db:
            entry = TranscriptEntry(
                session_id=session_id,
                text=text,
                raw_letters=raw_signs,
                emotion=emotion,
                language=language,
            )
            db.add(entry)
            await db.commit()
    except Exception:
        logger.exception("transcript_save_error")


@router.post("/speak", response_model=SpeakResponse)
async def manual_speak(
    req: SpeakRequest,
    user: User = Depends(get_current_user),
):
    text = req.text
    if req.language != "en":
        try:
            result = await vision_manager.translate(text, req.language)
            text = result.text
        except Exception:
            pass
    audio_file = await speak(text, emotion=req.emotion)
    return SpeakResponse(sentence=text, audio_url=f"/audio/{audio_file}")


@router.post("/sos", response_model=SpeakResponse)
async def manual_sos(
    req: SOSRequest = SOSRequest(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    sentence, audio_file = await trigger_sos(
        str(user.id), db, latitude=req.latitude, longitude=req.longitude
    )
    return SpeakResponse(sentence=sentence, audio_url=f"/audio/{audio_file}")


@router.get("/transcript")
async def get_transcript(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select

    result = await db.execute(
        select(TranscriptEntry)
        .where(TranscriptEntry.session_id == uuid.UUID(session_id))
        .order_by(TranscriptEntry.created_at)
    )
    entries = result.scalars().all()
    return {
        "entries": [
            {
                "id": str(e.id),
                "text": e.text,
                "raw_letters": e.raw_letters,
                "emotion": e.emotion,
                "language": e.language,
                "created_at": e.created_at.isoformat(),
            }
            for e in entries
        ]
    }
