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
        # Use MTCNN for better face detection accuracy (especially at angles)
        # ~39ms per frame on Jetson — acceptable since we only run every 5th frame
        try:
            _detector = FER(mtcnn=True)
            logger.info("fer_detector_loaded", backend="mtcnn")
        except Exception:
            # Fall back to Haar cascade if MTCNN isn't available
            _detector = FER(mtcnn=False)
            logger.info("fer_detector_loaded", backend="haar_cascade_fallback")
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

        # Use higher resolution for better face detection
        h, w = frame.shape[:2]
        if w > 480:
            scale = 480 / w
            frame = cv2.resize(frame, (480, int(h * scale)), interpolation=cv2.INTER_AREA)

        detector = _get_detector()
        results = detector.detect_emotions(frame)

        if not results:
            logger.debug("emotion_no_face_detected")
            return "neutral", 0.0

        # Use the largest face (most likely the signer)
        best = max(results, key=lambda r: r["box"][2] * r["box"][3])
        emotions = best["emotions"]
        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]

        logger.debug(
            "emotion_detected",
            emotion=dominant,
            confidence=round(confidence, 2),
            all_scores={k: round(v, 2) for k, v in emotions.items() if v > 0.05},
        )

        return dominant, round(confidence, 2)
    except Exception:
        logger.exception("emotion_detection_error")
        return "neutral", 0.0
