import base64

import cv2
import numpy as np
from fer import FER

from app.utils.logger import get_logger

logger = get_logger(__name__)

_detector: FER | None = None


def _get_detector() -> FER:
    global _detector
    if _detector is None:
        # Use Haar cascade (mtcnn=False) — 6x faster than MTCNN on Jetson
        # Haar: ~6ms vs MTCNN: ~39ms per frame
        _detector = FER(mtcnn=False)
        logger.info("fer_detector_loaded", backend="haar_cascade")
    return _detector


def detect_emotion_from_frame(frame_b64: str) -> tuple[str, float]:
    """Detect dominant emotion from a base64-encoded JPEG frame.

    Returns:
        Tuple of (emotion_name, confidence). Defaults to ("neutral", 0.0) on failure.
    """
    try:
        img_data = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return "neutral", 0.0

        # Downscale for faster detection — emotion doesn't need high res
        h, w = frame.shape[:2]
        if w > 320:
            scale = 320 / w
            frame = cv2.resize(frame, (320, int(h * scale)), interpolation=cv2.INTER_AREA)

        detector = _get_detector()
        result = detector.detect_emotions(frame)

        if not result:
            return "neutral", 0.0

        emotions = result[0]["emotions"]
        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]

        return dominant, round(confidence, 2)
    except Exception:
        logger.exception("emotion_detection_error")
        return "neutral", 0.0
