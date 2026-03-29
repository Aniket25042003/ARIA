import asyncio
import base64
import time

import google.generativeai as genai

from app.config import settings
from app.services.vision.base import SignRecognitionResult, VisionProvider, VisionResult

OBSTACLE_PROMPT = (
    "Navigation assistant for blind person. "
    "Immediate obstacles only, under 6 words. "
    "If safe: 'clear|Path is clear'. "
    "Format: SEVERITY|description. "
    "SEVERITY is: clear, caution, or danger."
)

SIGN_RECOGNITION_PROMPT = (
    "You are an expert ASL (American Sign Language) interpreter.\n"
    "Look at this image carefully.\n\n"
    "1. SIGN: What ASL sign or fingerspelled letter is the person making?\n"
    "   - If fingerspelling a letter, return just the uppercase letter (A-Z)\n"
    "   - If making a word sign, return the English word in uppercase (e.g., HELLO, THANK YOU, YES, NO)\n"
    "   - If no clear sign or no hand visible, return NONE\n\n"
    "2. EMOTION: What facial emotion is the person showing?\n"
    "   Choose from: happy, sad, angry, fear, surprise, disgust, neutral\n\n"
    "Format your response EXACTLY as: SIGN|EMOTION\n"
    "Examples: A|neutral, HELLO|happy, THANK YOU|sad, B|neutral, NONE|neutral\n"
    "Return ONLY the SIGN|EMOTION format, nothing else."
)

SENTENCE_PROMPT_TEMPLATE = (
    "A deaf person communicated these ASL signs/letters in sequence: '{letters}'. "
    "These may be a mix of fingerspelled letters and ASL word-signs. "
    "Emotion: {emotion}. "
    "Interpret the intended meaning, correct likely errors, and form ONE natural English sentence. "
    "Return ONLY the sentence, nothing else."
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
        self._sign_model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=genai.GenerationConfig(
                max_output_tokens=20,
                temperature=0.1,
            ),
        )

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

    async def recognize_sign(self, image_b64: str) -> SignRecognitionResult:
        start = time.monotonic()
        image_data = base64.b64decode(image_b64)
        response = await asyncio.to_thread(
            self._sign_model.generate_content,
            [SIGN_RECOGNITION_PROMPT, {"mime_type": "image/jpeg", "data": image_data}],
        )
        latency = (time.monotonic() - start) * 1000
        raw = response.text.strip()

        sign, emotion = self._parse_sign_response(raw)
        return SignRecognitionResult(
            sign=sign,
            emotion=emotion,
            provider=self.name,
            latency_ms=latency,
            confidence=0.85 if sign != "NONE" else 0.0,
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
    def _parse_sign_response(raw: str) -> tuple[str, str]:
        """Parse 'SIGN|EMOTION' format. Returns (sign, emotion)."""
        if "|" in raw:
            parts = raw.split("|", 1)
            sign = parts[0].strip().upper()
            emotion = parts[1].strip().lower()
            valid_emotions = {"happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"}
            if emotion not in valid_emotions:
                emotion = "neutral"
            return sign, emotion
        # Fallback: treat entire response as sign
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
        if any(w in lower for w in ("danger", "stop", "careful", "watch out")):
            return "danger", raw
        if any(w in lower for w in ("caution", "ahead", "step", "curb", "person")):
            return "caution", raw
        return "clear", raw
