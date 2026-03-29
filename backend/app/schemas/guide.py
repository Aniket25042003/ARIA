from pydantic import BaseModel


class ObstacleRequest(BaseModel):
    frame: str  # base64 JPEG


class ObstacleResponse(BaseModel):
    warning: str
    severity: str  # "clear", "caution", "danger"
    audio_url: str
    provider: str = ""
    latency_ms: int = 0


class NavigationRequest(BaseModel):
    lat: float
    lng: float
    destination: str


class NavStep(BaseModel):
    instruction: str
    distance: str
    duration: str
    lat: float | None = None
    lng: float | None = None


class NavigationResponse(BaseModel):
    route_id: str
    steps: list[NavStep]
    total_distance: str
    total_duration: str
