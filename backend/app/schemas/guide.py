from pydantic import BaseModel


class ObstacleRequest(BaseModel):
    frame: str  # base64 JPEG


class ObstacleResponse(BaseModel):
    warning: str
    severity: str  # "clear", "caution", "danger"
    audio_url: str
    provider: str = ""
    latency_ms: int = 0
