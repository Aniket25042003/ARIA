import time

import sqlalchemy
from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["health"])

_start_time = time.time()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "version": settings.app_version,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@router.get("/health/services")
async def health_services():
    results = {}

    # Database (SQLite)
    try:
        from app.db.postgres import engine

        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        results["database"] = "ok"
    except Exception as e:
        results["database"] = f"error: {e}"

    # Vision providers
    from app.services.vision.manager import vision_manager

    results["vision_providers"] = vision_manager.get_provider_status()

    return results


@router.post("/health/reset-providers")
async def reset_providers(provider: str | None = None):
    """Force-reset unhealthy provider(s) back to healthy so they get retried."""
    from app.services.vision.manager import vision_manager

    status = vision_manager.reset_provider_health(provider)
    return {"status": "reset", "providers": status}
