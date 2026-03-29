import asyncio
import base64
import io
import json
import time

import boto3
from PIL import Image

from app.config import settings
from app.services.vision.base import SignRecognitionResult, VisionProvider, VisionResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

OBSTACLE_PROMPT = (
    "You are a navigation assistant for a blind person. "
    "Describe ONLY immediate obstacles or hazards visible in the image. "
    "Use under 8 words. If safe: say 'Path is clear'. "
    "Format: SEVERITY|description where SEVERITY is: clear, caution, or danger."
)

SIGN_RECOGNITION_PROMPT = (
    "You are an expert ASL (American Sign Language) interpreter.\n"
    "Look at this image carefully and identify:\n\n"
    "1. SIGN: The ASL sign or fingerspelled letter being made.\n"
    "   - Fingerspelled letter → uppercase A-Z\n"
    "   - Word sign → uppercase English word (HELLO, THANK YOU, YES, NO)\n"
    "   - No clear sign → NONE\n\n"
    "2. EMOTION: The person's facial emotion.\n"
    "   One of: happy, sad, angry, fear, surprise, disgust, neutral\n\n"
    "Format: SIGN|EMOTION\n"
    "Examples: A|neutral, HELLO|happy, NONE|neutral\n"
    "Return ONLY the SIGN|EMOTION format."
)

SEQUENCE_ANALYSIS_PROMPT = (
    "You are an expert ASL (American Sign Language) interpreter assisting a deaf person.\n"
    "This image is a grid of sequential video frames (left-to-right, top-to-bottom) "
    "showing a person communicating in ASL. Each cell is one frame in chronological order.\n\n"
    "CRITICAL RULES:\n"
    "- You MUST identify every sign, letter, or gesture in EVERY frame. Do NOT skip any.\n"
    "- Look carefully at hand shape, hand position, palm orientation, and movement between frames.\n"
    "- If a hand is visible and forming any shape, it IS a sign — identify it. Never ignore a visible hand.\n"
    "- Distinguish between similar letters: A vs S vs T, B vs D, M vs N, I vs J, U vs V vs W.\n"
    "- If you are uncertain between two letters, pick the more likely one — do NOT skip it.\n"
    "- Only output NONE if there are truly NO hands visible in ANY frame.\n\n"
    "Identify:\n"
    "1. Every ASL sign, word-sign, or fingerspelled letter, in order:\n"
    "   - Fingerspelled letters → single uppercase letter (A, B, C...)\n"
    "   - Word signs → uppercase English word (HELLO, THANK YOU, YES, NO, HELP, PLEASE, WATER, EAT...)\n"
    "   - If the same sign is held across consecutive frames, count it ONCE.\n"
    "   - If a different sign appears even briefly in one frame, include it.\n"
    "2. The person's dominant facial emotion:\n"
    "   One of: happy, sad, angry, fear, surprise, disgust, neutral\n\n"
    "Format EXACTLY as:\n"
    "SIGNS: sign1, sign2, sign3\n"
    "EMOTION: emotion\n\n"
    "Examples:\n"
    "SIGNS: H, E, L, L, O\nEMOTION: happy\n\n"
    "SIGNS: HELLO, HOW, YOU\nEMOTION: neutral\n\n"
    "SIGNS: W, A, T, E, R\nEMOTION: neutral"
)

SENTENCE_PROMPT = (
    "A deaf person communicated these ASL signs/letters in sequence: '{signs}'.\n"
    "Their detected emotion is: {emotion}.\n\n"
    "RULES:\n"
    "- These are a mix of fingerspelled letters and ASL word-signs.\n"
    "- Group consecutive single letters into words (e.g. H,E,L,P → HELP).\n"
    "- ASL word-signs (HELLO, THANK YOU, WATER, etc.) are already whole words.\n"
    "- Fix obvious recognition errors using context (e.g. 'MATER' → 'WATER', 'HELO' → 'HELLO').\n"
    "- If some signs seem incomplete or garbled, infer the most likely intended message.\n"
    "- Produce ONE clear, natural English sentence that a text-to-speech system will read aloud.\n"
    "- The sentence should sound like natural speech, not a word list.\n"
    "- Return ONLY the sentence, nothing else."
)

TRANSLATE_PROMPT = "Translate to {language}. Return ONLY the translation: '{text}'"


class BedrockProvider(VisionProvider):
    name = "bedrock"

    def __init__(self):
        kwargs = {
            "region_name": settings.aws_region,
            "aws_access_key_id": settings.aws_access_key_id,
            "aws_secret_access_key": settings.aws_secret_access_key,
        }
        if settings.aws_session_token:
            kwargs["aws_session_token"] = settings.aws_session_token
        self._client = boto3.client("bedrock-runtime", **kwargs)
        self._model_id = settings.bedrock_model_id
        logger.info("bedrock_provider_init", model=self._model_id, region=settings.aws_region)

    def _converse_sync(self, messages: list, max_tokens: int = 50, temperature: float = 0.1) -> str:
        """Synchronous Bedrock Converse API call."""
        response = self._client.converse(
            modelId=self._model_id,
            messages=messages,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        return response["output"]["message"]["content"][0]["text"].strip()

    def _make_image_message(self, image_b64: str, prompt: str) -> list:
        """Build a Converse API message with image + text."""
        return [{
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": base64.b64decode(image_b64)},
                    }
                },
                {"text": prompt},
            ],
        }]

    def _make_text_message(self, prompt: str) -> list:
        """Build a Converse API message with text only."""
        return [{"role": "user", "content": [{"text": prompt}]}]

    @staticmethod
    def _stitch_frames(frames_b64: list[str], cols: int = 4) -> bytes:
        """Stitch multiple frames into a single grid image. Returns JPEG bytes."""
        images = []
        for b64 in frames_b64:
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            images.append(img)

        if not images:
            raise ValueError("No valid frames to stitch")

        # Resize all to same dimensions (use first frame as reference)
        w, h = images[0].size
        # Scale down for faster processing — 320px wide per cell is enough
        if w > 320:
            scale = 320 / w
            w, h = 320, int(h * scale)
            images = [img.resize((w, h), Image.LANCZOS) for img in images]

        rows = (len(images) + cols - 1) // cols
        grid = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
        for i, img in enumerate(images):
            r, c = divmod(i, cols)
            grid.paste(img, (c * w, r * h))

        buf = io.BytesIO()
        grid.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    async def detect_obstacle(self, image_b64: str) -> VisionResult:
        start = time.monotonic()
        messages = self._make_image_message(image_b64, OBSTACLE_PROMPT)
        raw = await asyncio.to_thread(self._converse_sync, messages, max_tokens=30)
        latency = (time.monotonic() - start) * 1000

        severity, text = self._parse_obstacle_response(raw)
        return VisionResult(text=text, provider=self.name, latency_ms=latency, severity=severity)

    async def recognize_sign(self, image_b64: str) -> SignRecognitionResult:
        start = time.monotonic()
        messages = self._make_image_message(image_b64, SIGN_RECOGNITION_PROMPT)
        raw = await asyncio.to_thread(self._converse_sync, messages, max_tokens=20)
        latency = (time.monotonic() - start) * 1000

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
        prompt = SENTENCE_PROMPT.format(signs=partial_text, emotion=emotion)
        messages = self._make_text_message(prompt)
        raw = await asyncio.to_thread(self._converse_sync, messages, max_tokens=100, temperature=0.3)
        latency = (time.monotonic() - start) * 1000
        return VisionResult(text=raw, provider=self.name, latency_ms=latency)

    async def translate(self, text: str, target_lang: str) -> VisionResult:
        if target_lang == "en":
            return VisionResult(text=text, provider=self.name, latency_ms=0)
        start = time.monotonic()
        prompt = TRANSLATE_PROMPT.format(language=target_lang, text=text)
        messages = self._make_text_message(prompt)
        raw = await asyncio.to_thread(self._converse_sync, messages, max_tokens=200, temperature=0.3)
        latency = (time.monotonic() - start) * 1000
        return VisionResult(text=raw, provider=self.name, latency_ms=latency)

    async def describe_sign_sequence(self, frames_b64: list[str]) -> VisionResult:
        """Stitch frames into grid → send single image to Bedrock vision."""
        start = time.monotonic()

        # Stitch frames into a single composite grid image
        grid_bytes = await asyncio.to_thread(self._stitch_frames, frames_b64)

        messages = [{
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": grid_bytes},
                    }
                },
                {"text": SEQUENCE_ANALYSIS_PROMPT},
            ],
        }]
        raw = await asyncio.to_thread(
            self._converse_sync, messages, max_tokens=150, temperature=0.2
        )
        latency = (time.monotonic() - start) * 1000

        signs, emotion = self._parse_sequence_response(raw)
        text = f"SIGNS: {signs}\nEMOTION: {emotion}"
        logger.info(
            "sequence_described",
            signs=signs,
            emotion=emotion,
            num_frames=len(frames_b64),
            latency_ms=int(latency),
        )
        return VisionResult(text=text, provider=self.name, latency_ms=latency)

    async def health_check(self) -> bool:
        try:
            messages = self._make_text_message("Say ok")
            result = await asyncio.to_thread(self._converse_sync, messages, max_tokens=5)
            return bool(result)
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
    def _parse_sequence_response(raw: str) -> tuple[str, str]:
        """Parse 'SIGNS: ...\nEMOTION: ...' response from multi-frame analysis."""
        signs = "NONE"
        emotion = "neutral"
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("SIGNS:"):
                signs = line.split(":", 1)[1].strip()
            elif line.upper().startswith("EMOTION:"):
                emotion = line.split(":", 1)[1].strip().lower()
                valid = {"happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"}
                if emotion not in valid:
                    emotion = "neutral"
        return signs, emotion

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
