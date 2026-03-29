import hashlib
import os

from elevenlabs import ElevenLabs

from app.config import settings
from app.utils.audio import generate_audio_path
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ElevenLabs premade voice IDs mapped to emotions (free-tier compatible)
VOICE_MAP = {
    "neutral": "EXAVITQu4vr4xnSDxMaL",  # Sarah — mature, reassuring
    "happy": "cgSgspJ2msm6clMCkdW9",    # Jessica — playful, bright, warm
    "sad": "pFZP5JQG7iQjIQuC4Bku",      # Lily — velvety, soft
    "fear": "SOYHLrjzK2X1ezoPC6cr",      # Harry — fierce, urgent
    "angry": "pNInz6obpgDQGcFmaJgB",     # Adam — dominant, firm
    "surprise": "FGY2WhTYpPnrIDTdsKH5",  # Laura — enthusiastic
    "disgust": "EXAVITQu4vr4xnSDxMaL",   # Sarah
    "guide": "EXAVITQu4vr4xnSDxMaL",     # Sarah — clear for navigation
    "sos": "SOYHLrjzK2X1ezoPC6cr",       # Harry — loud and clear
}

_client: ElevenLabs | None = None
# In-memory cache: hash(text+emotion) -> filename
_tts_cache: dict[str, str] = {}


def _get_client() -> ElevenLabs:
    global _client
    if _client is None:
        _client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    return _client


def _cache_key(text: str, emotion: str) -> str:
    return hashlib.md5(f"{emotion}:{text}".encode()).hexdigest()


async def speak(text: str, emotion: str = "neutral") -> str:
    """Convert text to speech using emotion-matched voice.

    Returns the filename (not full path) of the saved MP3.
    Uses cache for repeated phrases (common in guide mode warnings).
    """
    key = _cache_key(text, emotion)

    # Check cache — return immediately if file still exists
    if key in _tts_cache:
        cached_file = _tts_cache[key]
        cached_path = os.path.join(settings.audio_output_dir, cached_file)
        if os.path.exists(cached_path):
            logger.info("tts_cache_hit", emotion=emotion, file=cached_file)
            return cached_file

    voice_id = VOICE_MAP.get(emotion, VOICE_MAP["neutral"])
    output_path = generate_audio_path(f"aria_{emotion}")

    client = _get_client()
    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_flash_v2_5",
    )

    with open(output_path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)

    filename = output_path.split("/")[-1]
    _tts_cache[key] = filename
    logger.info("tts_generated", emotion=emotion, file=filename, text_length=len(text))
    return filename
