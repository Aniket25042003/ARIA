from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import async_session_factory
from app.models.postgres.user_data import UserProfile


async def get_user_profile(user_id: str, session: AsyncSession | None = None) -> dict | None:
    async def _query(s: AsyncSession):
        result = await s.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        row = result.scalar_one_or_none()
        if row:
            return {
                "user_id": row.user_id,
                "name": row.name,
                "avatar_url": row.avatar_url,
            }
        return None

    if session:
        return await _query(session)
    async with async_session_factory() as s:
        return await _query(s)


async def create_user_profile(user_id: str, name: str, session: AsyncSession | None = None) -> dict:
    async def _create(s: AsyncSession, commit: bool):
        profile = UserProfile(user_id=user_id, name=name)
        s.add(profile)
        if commit:
            await s.commit()
        return {"user_id": user_id, "name": name, "avatar_url": None}

    if session:
        return await _create(session, commit=False)
    async with async_session_factory() as s:
        result = await _create(s, commit=True)
        return result


async def update_user_profile(user_id: str, updates: dict, session: AsyncSession | None = None) -> dict | None:
    async def _update(s: AsyncSession, commit: bool):
        result = await s.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        row = result.scalar_one_or_none()
        if row:
            for key, value in updates.items():
                if hasattr(row, key):
                    setattr(row, key, value)
            row.updated_at = datetime.now(timezone.utc)
            if commit:
                await s.commit()

    if session:
        await _update(session, commit=False)
    else:
        async with async_session_factory() as s:
            await _update(s, commit=True)
    return await get_user_profile(user_id)
