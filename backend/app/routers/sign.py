import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import async_session_factory
from app.dependencies import get_current_user, get_db
from app.models.postgres.transcript import SignSession, TranscriptEntry
from app.models.postgres.user import User
from app.schemas.sign import SOSRequest, SpeakRequest, SpeakResponse
from app.services.auth_service import decode_access_token
from app.services.emotion_detector import detect_emotion_from_frame
from app.services.session_manager import (
    add_sign,
    clear_buffer,
    delete_session,
    get_and_clear_buffer,
    get_session,
    set_emotion,
    set_language,
    should_build_sentence,
)
from app.services.sign_classifier import classify_sign
from app.services.sos_service import contains_sos_trigger, trigger_sos
from app.services.tts_service import speak
from app.services.vision.manager import vision_manager
from app.utils.image import decode_base64_image
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_executor = ThreadPoolExecutor(max_workers=4)

# MediaPipe — hand detection + landmark extraction only
_mp_hands = mp.solutions.hands
_hands = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# Stability: need N consecutive frames with same sign to accept it
CONFIRM_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.55

# Collection window: accumulate signs, then build sentence
COLLECTION_TIMEOUT = 10.0  # Max seconds before auto-building sentence
NO_HAND_TRIGGER = 10  # ~2s at 5fps without hand → trigger sentence build
MIN_SIGNS_FOR_SENTENCE = 2

# Emotion detection frequency (every Nth frame)
EMOTION_EVERY_N_FRAMES = 5


def _extract_landmarks_and_frame(frame_b64: str) -> tuple[list[list[float]] | None, np.ndarray | None]:
    """Extract MediaPipe hand landmarks AND return the decoded frame."""
    try:
        image = decode_base64_image(frame_b64)
        h, w = image.shape[:2]
        if w > 480:
            scale = 480 / w
            image = cv2.resize(image, (480, int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = _hands.process(rgb)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            landmarks = [[l.x, l.y, l.z] for l in lm.landmark]
            return landmarks, image
        return None, image
    except Exception:
        return None, None


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

    loop = asyncio.get_event_loop()

    # Per-connection state
    last_sign = ""
    consecutive_count = 0
    no_hand_count = 0
    collection_start: float | None = None
    frame_number = 0
    current_emotion = "neutral"

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "frame":
                frame_b64 = data.get("frame", "")
                if not frame_b64:
                    continue

                frame_number += 1

                # --- Step 1: MediaPipe hand detection ---
                landmarks, frame_bgr = await loop.run_in_executor(
                    _executor, _extract_landmarks_and_frame, frame_b64
                )

                if not landmarks or frame_bgr is None:
                    no_hand_count += 1
                    # No hand for ~2 seconds → build sentence if buffer has signs
                    if no_hand_count >= NO_HAND_TRIGGER:
                        if await should_build_sentence(user_id, threshold=MIN_SIGNS_FOR_SENTENCE):
                            await _build_and_speak(
                                user_id, websocket, db_session_id, current_emotion
                            )
                            collection_start = None
                            last_sign = ""
                            consecutive_count = 0
                    continue

                no_hand_count = 0

                # --- Step 2: Classify sign with HuggingFace model ---
                sign, confidence = await classify_sign(frame_bgr, landmarks)

                logger.debug(
                    "sign_classified",
                    sign=sign,
                    confidence=round(confidence, 3),
                    frame=frame_number,
                )

                # Skip "nothing" / low-confidence
                if sign in ("NOTHING", "NONE", "DEL", "") or confidence < CONFIDENCE_THRESHOLD:
                    consecutive_count = 0
                    last_sign = ""
                    continue

                # --- Step 3: Stability — same sign N consecutive frames → accept ---
                if sign == last_sign:
                    consecutive_count += 1
                else:
                    last_sign = sign
                    consecutive_count = 1

                if consecutive_count == CONFIRM_FRAMES:
                    # Accept this sign into the buffer
                    if sign == "SPACE":
                        session = await get_session(user_id)
                        buf = " ".join(session["sign_buffer"])
                        await websocket.send_json({
                            "type": "letter",
                            "letter": " ",
                            "buffer": buf + " ",
                        })
                    else:
                        session = await add_sign(user_id, sign)
                        buf = " ".join(session["sign_buffer"])
                        await websocket.send_json({
                            "type": "letter",
                            "letter": sign,
                            "buffer": buf,
                        })
                        logger.info(
                            "sign_accepted",
                            sign=sign,
                            confidence=round(confidence, 3),
                            buffer=buf,
                        )

                    # Start collection timer on first accepted sign
                    if collection_start is None:
                        collection_start = time.monotonic()

                    # Reset so same sign doesn't get accepted again immediately
                    consecutive_count = 0

                # --- Step 4: Emotion detection (every Nth frame) ---
                if frame_number % EMOTION_EVERY_N_FRAMES == 0:
                    emo, emo_conf = await loop.run_in_executor(
                        _executor, detect_emotion_from_frame, frame_b64
                    )
                    if emo != "neutral" or emo_conf > 0.4:
                        current_emotion = emo
                        await set_emotion(user_id, emo)
                        await websocket.send_json({
                            "type": "emotion",
                            "emotion": emo,
                            "confidence": emo_conf,
                        })

                # --- Step 5: Collection timeout (max 10s) ---
                if collection_start and (time.monotonic() - collection_start) >= COLLECTION_TIMEOUT:
                    if await should_build_sentence(user_id, threshold=MIN_SIGNS_FOR_SENTENCE):
                        await _build_and_speak(
                            user_id, websocket, db_session_id, current_emotion
                        )
                    collection_start = None
                    last_sign = ""
                    consecutive_count = 0

            elif msg_type == "clear_buffer":
                await clear_buffer(user_id)
                collection_start = None
                last_sign = ""
                consecutive_count = 0
                await websocket.send_json({"type": "letter", "letter": "", "buffer": ""})

            elif msg_type == "set_language":
                await set_language(user_id, data.get("language", "en"))

    except WebSocketDisconnect:
        logger.info("sign_ws_disconnected", user_id=user_id)
    except Exception:
        logger.exception("sign_ws_error", user_id=user_id)
    finally:
        try:
            s = await get_session(user_id)
            if s["sign_buffer"]:
                await _build_and_speak(
                    user_id, websocket, db_session_id, current_emotion
                )
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


async def _build_and_speak(user_id, websocket, db_session_id, emotion_override=None):
    """Collect signs from buffer → Gemini LLM for sentence → ElevenLabs TTS."""
    raw_text, session_emotion = await get_and_clear_buffer(user_id)
    if not raw_text:
        return

    emotion = emotion_override or session_emotion
    logger.info("building_sentence", raw_text=raw_text, emotion=emotion)

    if contains_sos_trigger(raw_text):
        async with async_session_factory() as db:
            sentence, audio_file = await trigger_sos(user_id, db)
            await db.commit()
        await websocket.send_json({
            "type": "sos_triggered",
            "text": sentence,
            "audio_url": f"/audio/{audio_file}",
        })
        return

    # Send collected signs to Gemini LLM for natural sentence generation
    try:
        result = await vision_manager.build_sentence(raw_text, emotion)
        sentence = result.text
        logger.info(
            "sentence_built",
            raw=raw_text,
            sentence=sentence,
            provider=result.provider,
            latency_ms=int(result.latency_ms),
        )
    except Exception:
        logger.exception("build_sentence_failed", raw=raw_text)
        sentence = raw_text

    # Translate if non-English
    lang_session = await get_session(user_id)
    lang = lang_session.get("language", "en")
    if lang != "en":
        try:
            tr = await vision_manager.translate(sentence, lang)
            sentence = tr.text
        except Exception:
            pass

    # ElevenLabs TTS with emotion
    audio_file = await speak(sentence, emotion=emotion)

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
