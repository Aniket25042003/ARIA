import time

from app.utils.logger import get_logger

logger = get_logger(__name__)

SESSION_TTL = 3600  # 1 hour

# In-memory session store: {user_id: {"data": dict, "expires_at": float}}
_sessions: dict[str, dict] = {}


def _default_session() -> dict:
    return {
        "sign_buffer": [],  # List of recognized signs (letters or words)
        "last_emotion": "neutral",
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


async def add_sign(user_id: str, sign: str) -> dict:
    """Add a recognized sign (letter or word) to the buffer."""
    session = await get_session(user_id)
    session["sign_buffer"].append(sign)
    session["frame_count"] += 1
    logger.info(
        "sign_accepted",
        sign=sign,
        buffer_len=len(session["sign_buffer"]),
        buffer=" ".join(session["sign_buffer"]),
    )
    await update_session(user_id, session)
    return session


async def increment_frame_count(user_id: str) -> dict:
    """Increment frame count (called on frames without recognition)."""
    session = await get_session(user_id)
    session["frame_count"] += 1
    await update_session(user_id, session)
    return session


async def clear_buffer(user_id: str) -> dict:
    """Clear the sign buffer."""
    session = await get_session(user_id)
    session["sign_buffer"] = []
    await update_session(user_id, session)
    return session


async def set_language(user_id: str, language: str) -> dict:
    session = await get_session(user_id)
    session["language"] = language
    await update_session(user_id, session)
    return session


async def set_emotion(user_id: str, emotion: str) -> dict:
    session = await get_session(user_id)
    session["last_emotion"] = emotion
    await update_session(user_id, session)
    return session


async def get_and_clear_buffer(user_id: str) -> tuple[str, str]:
    """Get buffer contents and clear. Returns (signs_text, emotion)."""
    session = await get_session(user_id)
    # Join signs with spaces — single letters get grouped, words stay separate
    raw_text = " ".join(session["sign_buffer"]).strip()
    emotion = session["last_emotion"]
    session["sign_buffer"] = []
    await update_session(user_id, session)
    return raw_text, emotion


async def should_build_sentence(user_id: str, threshold: int = 3) -> bool:
    """Check if the buffer has enough signs to trigger sentence building."""
    session = await get_session(user_id)
    return len(session["sign_buffer"]) >= threshold


async def delete_session(user_id: str) -> None:
    _sessions.pop(user_id, None)
