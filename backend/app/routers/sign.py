import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import async_session_factory
from app.dependencies import get_current_user, get_db
from app.models.postgres.transcript import SignSession, TranscriptEntry
from app.models.postgres.user import User
from app.schemas.sign import SOSRequest, SpeakRequest, SpeakResponse
from app.services.asl_classifier import classify_asl
from app.services.auth_service import decode_access_token
from app.services.emotion_detector import detect_emotion_from_frame
from app.services.session_manager import (
    append_letter,
    clear_buffer,
    delete_session,
    get_and_clear_buffer,
    get_session,
    set_emotion,
    set_language,
    should_build_sentence,
)
from app.services.sos_service import contains_sos_trigger, trigger_sos
from app.services.tts_service import speak
from app.services.vision.manager import vision_manager
from app.utils.image import decode_base64_image
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Thread pool for CPU-bound work (MediaPipe, emotion detection)
_executor = ThreadPoolExecutor(max_workers=2)

# MediaPipe Hands — initialized once, reused for all WebSocket connections
_mp_hands = mp.solutions.hands
_hands = _mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.65,
)


def _extract_landmarks_sync(frame_b64: str) -> list[list[float]] | None:
    """Extract hand landmarks (CPU-bound, run in thread pool)."""
    try:
        image = decode_base64_image(frame_b64)
        # Downscale for faster MediaPipe processing
        h, w = image.shape[:2]
        if w > 320:
            scale = 320 / w
            image = cv2.resize(image, (320, int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = _hands.process(rgb)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            return [[l.x, l.y, l.z] for l in lm.landmark]
        return None
    except Exception:
        return None


@router.websocket("/ws")
async def sign_websocket(websocket: WebSocket):
    """Real-time SIGN mode WebSocket."""
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

    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "frame":
                frame_b64 = data.get("frame", "")
                if not frame_b64:
                    continue

                session = await get_session(user_id)
                frame_count = session["frame_count"]

                # Run MediaPipe in thread pool (non-blocking)
                landmarks = await loop.run_in_executor(
                    _executor, _extract_landmarks_sync, frame_b64
                )

                if landmarks:
                    letter = classify_asl(landmarks)
                    if letter:
                        session = await append_letter(user_id, letter)
                        buffer_str = "".join(session["letter_buffer"])
                        await websocket.send_json({
                            "type": "letter",
                            "letter": letter,
                            "buffer": buffer_str,
                        })

                # Emotion detection every 8th frame (in thread pool)
                if frame_count % 8 == 0 and frame_b64:
                    emotion, confidence = await loop.run_in_executor(
                        _executor, detect_emotion_from_frame, frame_b64
                    )
                    await set_emotion(user_id, emotion, confidence)
                    await websocket.send_json({
                        "type": "emotion",
                        "emotion": emotion,
                        "confidence": confidence,
                    })

                # Increment frame count even when no hand detected
                if not landmarks:
                    session["frame_count"] = frame_count + 1
                    from app.services.session_manager import update_session
                    await update_session(user_id, session)

                # Check if we should build a sentence
                if await should_build_sentence(user_id):
                    raw_text, emotion = await get_and_clear_buffer(user_id)

                    if raw_text:
                        if contains_sos_trigger(raw_text):
                            async with async_session_factory() as db:
                                sentence, audio_file = await trigger_sos(user_id, db)
                                await db.commit()
                            await websocket.send_json({
                                "type": "sos_triggered",
                                "text": sentence,
                                "audio_url": f"/audio/{audio_file}",
                            })
                            continue

                        # Build sentence via LLM
                        try:
                            result = await vision_manager.build_sentence(raw_text, emotion)
                            sentence = result.text
                        except Exception:
                            sentence = raw_text

                        # Translate if needed
                        lang = session.get("language", "en")
                        if lang != "en":
                            try:
                                tr = await vision_manager.translate(sentence, lang)
                                sentence = tr.text
                            except Exception:
                                pass

                        # TTS
                        audio_file = await speak(sentence, emotion=emotion)

                        # Log transcript (fire-and-forget)
                        if db_session_id:
                            asyncio.create_task(_save_transcript(
                                db_session_id, sentence, raw_text, emotion, lang
                            ))

                        await websocket.send_json({
                            "type": "sentence",
                            "text": sentence,
                            "emotion": emotion,
                            "audio_url": f"/audio/{audio_file}",
                        })

            elif msg_type == "clear_buffer":
                await clear_buffer(user_id)
                await websocket.send_json({"type": "letter", "letter": "", "buffer": ""})

            elif msg_type == "set_language":
                lang = data.get("language", "en")
                await set_language(user_id, lang)

    except WebSocketDisconnect:
        logger.info("sign_ws_disconnected", user_id=user_id)
    except Exception:
        logger.exception("sign_ws_error", user_id=user_id)
    finally:
        await delete_session(user_id)
        if db_session_id:
            try:
                async with async_session_factory() as db:
                    from sqlalchemy import update
                    from datetime import datetime, timezone

                    await db.execute(
                        update(SignSession)
                        .where(SignSession.id == db_session_id)
                        .values(ended_at=datetime.now(timezone.utc))
                    )
                    await db.commit()
            except Exception:
                pass


async def _save_transcript(
    session_id, text: str, raw_letters: str, emotion: str, language: str
) -> None:
    """Fire-and-forget transcript save."""
    try:
        async with async_session_factory() as db:
            entry = TranscriptEntry(
                session_id=session_id,
                text=text,
                raw_letters=raw_letters,
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
