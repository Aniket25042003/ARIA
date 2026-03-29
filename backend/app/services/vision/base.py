from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VisionResult:
    text: str
    provider: str
    latency_ms: float
    severity: str = "clear"  # "clear", "caution", "danger"
    confidence: float | None = None


@dataclass
class SignRecognitionResult:
    sign: str  # Recognized ASL sign (letter like "A" or word like "HELLO")
    emotion: str  # Detected facial emotion
    provider: str
    latency_ms: float
    confidence: float = 0.0


class VisionProvider(ABC):
    """Abstract base class for all vision/LLM providers."""

    name: str = "base"

    @abstractmethod
    async def detect_obstacle(self, image_b64: str) -> VisionResult:
        """Analyze image for obstacles. Return short warning text."""
        ...

    @abstractmethod
    async def build_sentence(self, partial_text: str, emotion: str) -> VisionResult:
        """Complete partial ASL text into a natural sentence."""
        ...

    @abstractmethod
    async def translate(self, text: str, target_lang: str) -> VisionResult:
        """Translate text to target language."""
        ...

    async def recognize_sign(self, image_b64: str) -> SignRecognitionResult:
        """Recognize ASL sign and facial emotion from an image.

        Default: raises NotImplementedError so manager skips to next provider.
        """
        raise NotImplementedError

    async def describe_sign_sequence(self, frames_b64: list[str]) -> VisionResult:
        """Analyze a sequence of frames for ASL signs and emotion.

        Default: raises NotImplementedError so manager skips to next provider.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if this provider is currently available."""
        ...
