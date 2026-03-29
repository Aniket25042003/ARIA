import time

import anthropic

from app.config import settings
from app.services.vision.base import SignRecognitionResult, VisionProvider, VisionResult

OBSTACLE_PROMPT = (
    "You are a navigation assistant for a blind person. "
    "Describe ONLY immediate obstacles or hazards visible in the image. "
    "Use under 8 words. If safe: say 'Path is clear'. "
    "Format: SEVERITY|description where SEVERITY is: clear, caution, or danger."
)

SENTENCE_PROMPT = (
    "A deaf person communicated these ASL signs/letters: '{letters}'. "
    "These may be a mix of fingerspelled letters and ASL word-signs. "
    "Their emotion is: {emotion}. "
    "Interpret the intended meaning, correct likely errors, and form ONE natural English sentence. "
    "Return ONLY the sentence, nothing else."
)

SIGN_RECOGNITION_PROMPT = (
    "You are an expert ASL interpreter. Identify:\n"
    "1. The ASL sign or fingerspelled letter (uppercase A-Z or word like HELLO)\n"
    "2. The person's facial emotion (happy, sad, angry, fear, surprise, disgust, neutral)\n"
    "Format: SIGN|EMOTION (e.g., A|neutral, HELLO|happy, NONE|neutral)\n"
    "Return ONLY the SIGN|EMOTION format."
)

TRANSLATE_PROMPT = "Translate to {language}. Return ONLY the translation: '{text}'"


class ClaudeProvider(VisionProvider):
    name = "claude"

    def __init__(self):
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def detect_obstacle(self, image_b64: str) -> VisionResult:
        start = time.monotonic()
        response = await self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": OBSTACLE_PROMPT},
                    ],
                }
            ],
        )
        latency = (time.monotonic() - start) * 1000
        raw = response.content[0].text.strip()

        severity, text = self._parse_obstacle_response(raw)
        return VisionResult(
            text=text, provider=self.name, latency_ms=latency, severity=severity
        )

    async def recognize_sign(self, image_b64: str) -> SignRecognitionResult:
        start = time.monotonic()
        response = await self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": SIGN_RECOGNITION_PROMPT},
                    ],
                }
            ],
        )
        latency = (time.monotonic() - start) * 1000
        raw = response.content[0].text.strip()
        sign, emotion = self._parse_sign_response(raw)
        return SignRecognitionResult(
            sign=sign,
            emotion=emotion,
            provider=self.name,
            latency_ms=latency,
            confidence=0.80 if sign != "NONE" else 0.0,
        )

    async def build_sentence(self, partial_text: str, emotion: str) -> VisionResult:
        start = time.monotonic()
        response = await self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": SENTENCE_PROMPT.format(letters=partial_text, emotion=emotion),
                }
            ],
        )
        latency = (time.monotonic() - start) * 1000
        return VisionResult(
            text=response.content[0].text.strip(),
            provider=self.name,
            latency_ms=latency,
        )

    async def translate(self, text: str, target_lang: str) -> VisionResult:
        if target_lang == "en":
            return VisionResult(text=text, provider=self.name, latency_ms=0)
        start = time.monotonic()
        response = await self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": TRANSLATE_PROMPT.format(language=target_lang, text=text),
                }
            ],
        )
        latency = (time.monotonic() - start) * 1000
        return VisionResult(
            text=response.content[0].text.strip(),
            provider=self.name,
            latency_ms=latency,
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=5,
                messages=[{"role": "user", "content": "Say ok"}],
            )
            return bool(response.content[0].text)
        except Exception:
            return False

    @staticmethod
    def _parse_sign_response(raw: str) -> tuple[str, str]:
        if "|" in raw:
            parts = raw.split("|", 1)
            sign = parts[0].strip().upper()
            emotion = parts[1].strip().lower()
            valid = {"happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"}
            if emotion not in valid:
                emotion = "neutral"
            return sign, emotion
        return raw.strip().upper() or "NONE", "neutral"

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
