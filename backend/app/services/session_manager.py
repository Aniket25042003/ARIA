import time

from app.utils.logger import get_logger

logger = get_logger(__name__)

SESSION_TTL = 3600  # 1 hour

# In-memory session store: {user_id: {"data": dict, "expires_at": float}}
_sessions: dict[str, dict] = {}


def _default_session() -> dict:
    return {
        "letter_buffer": [],
        "last_emotion": "neutral",
        "last_confidence": 0.0,
        "language": "en",
        "frame_count": 0,
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


async def append_letter(user_id: str, letter: str) -> dict:
    """Append a letter to the user's buffer and return updated session."""
    session = await get_session(user_id)
    if letter.strip():
        session["letter_buffer"].append(letter)
    session["frame_count"] += 1
    await update_session(user_id, session)
    return session


async def clear_buffer(user_id: str) -> dict:
    """Clear the letter buffer and return updated session."""
    session = await get_session(user_id)
    session["letter_buffer"] = []
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
    await update_session(user_id, session)
    return raw_text, emotion


async def should_build_sentence(user_id: str, threshold: int = 6) -> bool:
    """Check if the buffer has enough letters to trigger sentence building."""
    session = await get_session(user_id)
    return len(session["letter_buffer"]) >= threshold


async def delete_session(user_id: str) -> None:
    """Delete the session (on disconnect)."""
    _sessions.pop(user_id, None)
