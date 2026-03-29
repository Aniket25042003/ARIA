import asyncio
import base64
import time

import google.generativeai as genai

from app.config import settings
from app.services.vision.base import VisionProvider, VisionResult

OBSTACLE_PROMPT = (
    "Navigation assistant for blind person. "
    "Immediate obstacles only, under 6 words. "
    "If safe: 'clear|Path is clear'. "
    "Format: SEVERITY|description. "
    "SEVERITY is: clear, caution, or danger."
)

SENTENCE_PROMPT_TEMPLATE = (
    "A deaf/mute person signed these letters: '{letters}'. "
    "Emotion: {emotion}. "
    "Complete into ONE natural sentence. Return ONLY the sentence."
)

TRANSLATE_PROMPT_TEMPLATE = (
    "Translate to {language}. Return ONLY the translation: '{text}'"
)


class GeminiProvider(VisionProvider):
    name = "gemini"

    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel("gemini-2.0-flash")
        self._text_model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=genai.GenerationConfig(
                max_output_tokens=60,
                temperature=0.3,
            ),
        )
        self._image_model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=genai.GenerationConfig(
                max_output_tokens=30,
                temperature=0.2,
            ),
        )
        # Set request timeout for all models
        self._request_options = {"timeout": 5}

    async def detect_obstacle(self, image_b64: str) -> VisionResult:
        start = time.monotonic()
        image_data = base64.b64decode(image_b64)
        response = await asyncio.to_thread(
            self._image_model.generate_content,
            [OBSTACLE_PROMPT, {"mime_type": "image/jpeg", "data": image_data}],
        )
        latency = (time.monotonic() - start) * 1000
        raw = response.text.strip()

        severity, text = self._parse_obstacle_response(raw)
        return VisionResult(
            text=text, provider=self.name, latency_ms=latency, severity=severity
        )

    async def build_sentence(self, partial_text: str, emotion: str) -> VisionResult:
        start = time.monotonic()
        prompt = SENTENCE_PROMPT_TEMPLATE.format(letters=partial_text, emotion=emotion)
        response = await asyncio.to_thread(self._text_model.generate_content, prompt)
        latency = (time.monotonic() - start) * 1000
        return VisionResult(
            text=response.text.strip(), provider=self.name, latency_ms=latency
        )

    async def translate(self, text: str, target_lang: str) -> VisionResult:
        if target_lang == "en":
            return VisionResult(text=text, provider=self.name, latency_ms=0)
        start = time.monotonic()
        prompt = TRANSLATE_PROMPT_TEMPLATE.format(language=target_lang, text=text)
        response = await asyncio.to_thread(self._text_model.generate_content, prompt)
        latency = (time.monotonic() - start) * 1000
        return VisionResult(
            text=response.text.strip(), provider=self.name, latency_ms=latency
        )

    async def health_check(self) -> bool:
        try:
            response = self._model.generate_content("Say ok")
            return bool(response.text)
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
        if any(w in lower for w in ("danger", "stop", "careful", "watch out")):
            return "danger", raw
        if any(w in lower for w in ("caution", "ahead", "step", "curb", "person")):
            return "caution", raw
        return "clear", raw
