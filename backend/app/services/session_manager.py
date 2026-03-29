import time

from app.utils.logger import get_logger

logger = get_logger(__name__)

SESSION_TTL = 3600  # 1 hour

# Stability: require the same letter N consecutive frames before accepting
STABILITY_THRESHOLD = 3  # Must see same letter 3 times in a row
MIN_CONFIDENCE = 0.50  # Minimum confidence to consider a detection

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


async def append_letter(
    user_id: str, letter: str, confidence: float = 1.0
) -> dict:
    """Process a detected letter with stability filtering and deduplication.

    A letter is only added to the buffer when:
    1. Confidence >= MIN_CONFIDENCE
    2. Same letter detected for STABILITY_THRESHOLD consecutive frames
    3. It differs from the last accepted letter (deduplication)
    """
    session = await get_session(user_id)
    session["frame_count"] += 1

    stripped = letter.strip()
    if not stripped or confidence < MIN_CONFIDENCE:
        # No valid detection — reset pending streak
        session["pending_letter"] = ""
        session["pending_count"] = 0
        await update_session(user_id, session)
        return session

    # Stability check: same letter must repeat consecutively
    if stripped == session["pending_letter"]:
        session["pending_count"] += 1
    else:
        session["pending_letter"] = stripped
        session["pending_count"] = 1

    # Once stable, accept the letter (with deduplication)
    if session["pending_count"] >= STABILITY_THRESHOLD:
        if stripped != session["last_accepted_letter"]:
            session["letter_buffer"].append(stripped)
            session["last_accepted_letter"] = stripped
        # Reset count so holding the letter longer doesn't keep re-adding
        # but keep pending_letter so we can detect when user changes
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
    await update_session(user_id, session)
    return raw_text, emotion


async def should_build_sentence(user_id: str, threshold: int = 5) -> bool:
    """Check if the buffer has enough letters to trigger sentence building."""
    session = await get_session(user_id)
    return len(session["letter_buffer"]) >= threshold


async def delete_session(user_id: str) -> None:
    """Delete the session (on disconnect)."""
    _sessions.pop(user_id, None)
