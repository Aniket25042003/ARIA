import base64
import time

from openai import AsyncOpenAI

from app.config import settings
from app.services.vision.base import VisionProvider, VisionResult

OBSTACLE_SYSTEM = (
    "You are a navigation assistant for a blind person. "
    "Describe ONLY immediate obstacles or hazards visible in the image. "
    "Use under 8 words. If safe: say 'Path is clear'. "
    "Format your response as: SEVERITY|description "
    "where SEVERITY is one of: clear, caution, danger."
)

SENTENCE_SYSTEM = (
    "You help deaf people communicate. Given ASL finger-spelled letters and an emotion, "
    "interpret the intended word(s) and form ONE natural sentence. "
    "The letters may contain recognition errors — similar-looking hand shapes are often confused "
    "(e.g., M/N/S/T/A/E in fist shapes, U/V/R/K with two fingers, D/G/L with one finger). "
    "Correct likely errors based on context. Return ONLY the sentence."
)

TRANSLATE_SYSTEM = "You are a translator. Return ONLY the translation, nothing else."


class OpenAIProvider(VisionProvider):
    name = "openai"

    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def detect_obstacle(self, image_b64: str) -> VisionResult:
        start = time.monotonic()
        response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": OBSTACLE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            max_tokens=50,
        )
        latency = (time.monotonic() - start) * 1000
        raw = response.choices[0].message.content.strip()

        severity, text = self._parse_obstacle_response(raw)
        return VisionResult(
            text=text, provider=self.name, latency_ms=latency, severity=severity
        )

    async def build_sentence(self, partial_text: str, emotion: str) -> VisionResult:
        start = time.monotonic()
        response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SENTENCE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Letters: '{partial_text}', Emotion: {emotion}",
                },
            ],
            max_tokens=100,
        )
        latency = (time.monotonic() - start) * 1000
        return VisionResult(
            text=response.choices[0].message.content.strip(),
            provider=self.name,
            latency_ms=latency,
        )

    async def translate(self, text: str, target_lang: str) -> VisionResult:
        if target_lang == "en":
            return VisionResult(text=text, provider=self.name, latency_ms=0)
        start = time.monotonic()
        response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM},
                {"role": "user", "content": f"Translate to {target_lang}: '{text}'"},
            ],
            max_tokens=200,
        )
        latency = (time.monotonic() - start) * 1000
        return VisionResult(
            text=response.choices[0].message.content.strip(),
            provider=self.name,
            latency_ms=latency,
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say ok"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except Exception:
            return False

    @staticmethod
    def _parse_obstacle_response(raw: str) -> tuple[str, str]:
        if "|" in raw:
            parts = raw.split("|", 1)
            severity = parts[0].strip().lower()
            text = parts[1].strip()
            if severity in ("clear", "caution", "danger"):
                return severity, text
        lower = raw.lower()
        if any(w in lower for w in ("danger", "stop", "careful")):
            return "danger", raw
        if any(w in lower for w in ("caution", "ahead", "step", "curb")):
            return "caution", raw
        return "clear", raw
