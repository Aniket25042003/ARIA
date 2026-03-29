import time

from app.utils.logger import get_logger

logger = get_logger(__name__)

SESSION_TTL = 3600  # 1 hour

# Stability: require the same letter N consecutive frames before accepting
STABILITY_THRESHOLD = 2  # Must see same letter 2 times in a row (at 5fps = 0.4s)
MIN_CONFIDENCE = 0.45  # Minimum confidence to consider a detection

# In-memory session store: {user_id: {"data": dict, "expires_at": float}}
_sessions: dict[str, dict] = {}


def _default_session() -> dict:
    return {
        "letter_buffer": [],
        "last_emotion": "neutral",
        "last_confidence": 0.0,
        "language": "en",
        "frame_count": 0,
        # Stability tracking
        "pending_letter": "",
        "pending_count": 0,
        "last_accepted_letter": "",
        "no_hand_frames": 0,  # Consecutive frames with no hand detected
    }


def _cleanup_expired() -> None:
    now = time.time()
    expired = [k for k, v in _sessions.items() if v["expires_at"] < now]
    for k in expired:
        del _sessions[k]


async def get_session(user_id: str) -> dict:
    _cleanup_expired()
    entry = _sessions.get(user_id)
    if entry and entry["expires_at"] > time.time():
        return entry["data"]
    session = _default_session()
    _sessions[user_id] = {"data": session, "expires_at": time.time() + SESSION_TTL}
    return session


async def update_session(user_id: str, session: dict) -> None:
    _sessions[user_id] = {"data": session, "expires_at": time.time() + SESSION_TTL}


async def increment_frame_count(user_id: str) -> dict:
    """Increment frame count (called when no hand landmarks detected)."""
    session = await get_session(user_id)
    session["frame_count"] += 1

    # Track consecutive no-hand frames to reset dedup after a pause
    session["no_hand_frames"] = session.get("no_hand_frames", 0) + 1

    # After ~1 second of no hand (5 frames at 5fps), reset last_accepted_letter
    # so the user can sign the same letter again after lowering their hand
    if session["no_hand_frames"] >= 5:
        session["last_accepted_letter"] = ""
        session["pending_letter"] = ""
        session["pending_count"] = 0

    await update_session(user_id, session)
    return session


async def append_letter(
    user_id: str, letter: str, confidence: float = 1.0
) -> dict:
    """Process a detected letter with stability filtering and smart deduplication.

    A letter is only added to the buffer when:
    1. Confidence >= MIN_CONFIDENCE
    2. Same letter detected for STABILITY_THRESHOLD consecutive frames
    3. Dedup: same letter blocked UNLESS user paused (lowered hand) in between
    """
    session = await get_session(user_id)
    session["frame_count"] += 1
    session["no_hand_frames"] = 0  # Hand is detected, reset no-hand counter

    stripped = letter.strip()
    if not stripped or confidence < MIN_CONFIDENCE:
        # Low confidence — don't reset pending streak, just skip
        # This prevents jitter from breaking a valid streak
        await update_session(user_id, session)
        return session

    # Stability check: same letter must repeat consecutively
    if stripped == session["pending_letter"]:
        session["pending_count"] += 1
    else:
        session["pending_letter"] = stripped
        session["pending_count"] = 1

    # Once stable, accept the letter
    if session["pending_count"] >= STABILITY_THRESHOLD:
        if stripped != session["last_accepted_letter"]:
            session["letter_buffer"].append(stripped)
            session["last_accepted_letter"] = stripped
            logger.info(
                "letter_accepted",
                letter=stripped,
                confidence=round(confidence, 2),
                buffer_len=len(session["letter_buffer"]),
                buffer="".join(session["letter_buffer"]),
            )
        # Keep count at threshold so we don't keep re-accepting
        session["pending_count"] = STABILITY_THRESHOLD

    await update_session(user_id, session)
    return session


async def clear_buffer(user_id: str) -> dict:
    """Clear the letter buffer and reset stability tracking."""
    session = await get_session(user_id)
    session["letter_buffer"] = []
    session["pending_letter"] = ""
    session["pending_count"] = 0
    session["last_accepted_letter"] = ""
    session["no_hand_frames"] = 0
    await update_session(user_id, session)
    return session


async def set_language(user_id: str, language: str) -> dict:
    """Set the output language for the session."""
    session = await get_session(user_id)
    session["language"] = language
    await update_session(user_id, session)
    return session


async def set_emotion(user_id: str, emotion: str, confidence: float) -> dict:
    """Update the detected emotion in the session."""
    session = await get_session(user_id)
    session["last_emotion"] = emotion
    session["last_confidence"] = confidence
    await update_session(user_id, session)
    return session


async def get_and_clear_buffer(user_id: str) -> tuple[str, str]:
    """Get raw text from buffer and clear it. Returns (raw_text, emotion)."""
    session = await get_session(user_id)
    raw_text = "".join(session["letter_buffer"]).strip()
    emotion = session["last_emotion"]
    session["letter_buffer"] = []
    session["pending_letter"] = ""
    session["pending_count"] = 0
    session["last_accepted_letter"] = ""
    session["no_hand_frames"] = 0
    await update_session(user_id, session)
    return raw_text, emotion


async def should_build_sentence(user_id: str, threshold: int = 4) -> bool:
    """Check if the buffer has enough letters to trigger sentence building."""
    session = await get_session(user_id)
    return len(session["letter_buffer"]) >= threshold


async def delete_session(user_id: str) -> None:
    """Delete the session (on disconnect)."""
    _sessions.pop(user_id, None)
