from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import async_session_factory
from app.models.postgres.user_data import UserPreferences

DEFAULT_PREFERENCES = {
    "default_mode": "sign",
    "language": "en",
    "voice_style": "rachel",
    "tts_speed": 1.0,
    "obstacle_scan_interval_ms": 1500,
    "high_contrast": False,
    "large_text": False,
    "sign_mode": {
        "auto_speak": True,
        "buffer_threshold": 6,
    },
    "guide_mode": {
        "speak_clear_path": False,
        "haptic_on_obstacle": True,
    },
}


async def get_user_preferences(user_id: str, session: AsyncSession | None = None) -> dict | None:
    async def _query(s: AsyncSession):
        result = await s.execute(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        )
        row = result.scalar_one_or_none()
        if row:
            return {"user_id": row.user_id, **row.preferences}
        return None

    if session:
        return await _query(session)
    async with async_session_factory() as s:
        return await _query(s)


async def create_user_preferences(user_id: str, session: AsyncSession | None = None) -> dict:
    async def _create(s: AsyncSession, commit: bool):
        prefs = UserPreferences(user_id=user_id, preferences=DEFAULT_PREFERENCES)
        s.add(prefs)
        if commit:
            await s.commit()
        return {"user_id": user_id, **DEFAULT_PREFERENCES}

    if session:
        return await _create(session, commit=False)
    async with async_session_factory() as s:
        result = await _create(s, commit=True)
        return result


async def update_user_preferences(user_id: str, updates: dict, session: AsyncSession | None = None) -> dict | None:
    async def _update(s: AsyncSession, commit: bool):
        result = await s.execute(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        )
        row = result.scalar_one_or_none()
        if row:
            merged = {**row.preferences, **updates}
            row.preferences = merged
            row.updated_at = datetime.now(timezone.utc)
            if commit:
                await s.commit()

    if session:
        await _update(session, commit=False)
    else:
        async with async_session_factory() as s:
            await _update(s, commit=True)
    return await get_user_preferences(user_id)
