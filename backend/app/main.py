import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.postgres import close_postgres, init_postgres
from app.middleware.cors import setup_cors
from app.middleware.logging import LoggingMiddleware
from app.routers import auth, guide, health, sign, user
from app.services.vision.manager import vision_manager
from app.utils.audio import cleanup_old_audio, ensure_audio_dir
from app.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(debug=settings.debug)
    logger.info("starting_aria_backend", version=settings.app_version)

    # Initialize database (SQLite — creates tables on first run)
    await init_postgres()

    # Ensure audio output directory exists
    ensure_audio_dir()

    # Initialize vision providers
    vision_manager.initialize()

    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_old_audio())
    health_task = asyncio.create_task(vision_manager.run_health_loop())

    logger.info("aria_backend_ready")
    yield

    # Shutdown
    cleanup_task.cancel()
    health_task.cancel()
    await close_postgres()
    logger.info("aria_backend_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ARIA Backend",
        description="Adaptive Real-time Intelligence Assistant — Backend API",
        version=settings.app_version,
        lifespan=lifespan,
    )

    # Middleware
    setup_cors(app)
    app.add_middleware(LoggingMiddleware)

    # Static files for audio serving
    ensure_audio_dir()
    app.mount("/audio", StaticFiles(directory=settings.audio_output_dir), name="audio")

    # Routers
    app.include_router(health.router)
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(user.router, prefix="/user", tags=["user"])
    app.include_router(sign.router, prefix="/sign", tags=["sign"])
    app.include_router(guide.router, prefix="/guide", tags=["guide"])

    return app


app = create_app()
