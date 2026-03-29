from fastapi import APIRouter, Depends

from app.dependencies import get_current_user
from app.models.postgres.user import User
from app.schemas.guide import ObstacleRequest, ObstacleResponse
from app.schemas.sign import SpeakRequest, SpeakResponse
from app.services.tts_service import speak
from app.services.vision.manager import vision_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/obstacle", response_model=ObstacleResponse)
async def detect_obstacle(
    req: ObstacleRequest,
    user: User = Depends(get_current_user),
):
    """Analyze a camera frame for obstacles. Returns warning + audio."""
    result = await vision_manager.detect_obstacle(req.frame)

    # Skip TTS for clear paths — saves ~500ms per scan
    if result.severity == "clear":
        return ObstacleResponse(
            warning=result.text,
            severity=result.severity,
            audio_url="",
            provider=result.provider,
            latency_ms=int(result.latency_ms),
        )

    # Generate TTS only for actual warnings
    audio_file = await speak(result.text, emotion="guide")

    return ObstacleResponse(
        warning=result.text,
        severity=result.severity,
        audio_url=f"/audio/{audio_file}",
        provider=result.provider,
        latency_ms=int(result.latency_ms),
    )


@router.post("/speak", response_model=SpeakResponse)
async def guide_speak(
    req: SpeakRequest,
    user: User = Depends(get_current_user),
):
    """Speak arbitrary text in guide mode voice."""
    audio_file = await speak(req.text, emotion="guide")
    return SpeakResponse(sentence=req.text, audio_url=f"/audio/{audio_file}")
