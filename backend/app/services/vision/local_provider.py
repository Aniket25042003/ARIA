import time

from app.services.vision.base import VisionProvider, VisionResult
from app.utils.image import decode_base64_image, resize_image
from app.utils.logger import get_logger

logger = get_logger(__name__)

# YOLO model is loaded lazily to avoid import overhead when not needed
_model = None


def _get_yolo_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO("yolov8n.pt")  # Auto-downloads ~6MB model on first run
        logger.info("yolo_model_loaded")
    return _model


# Objects that are relevant obstacles for a blind person
OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "bench", "chair", "couch",
    "dining table", "dog", "cat", "horse", "cow",
    "fire hydrant", "parking meter", "suitcase",
}

DANGER_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}


class LocalProvider(VisionProvider):
    """YOLOv8 nano running locally on Jetson GPU for obstacle detection.

    Only supports detect_obstacle. Raises NotImplementedError for
    build_sentence and translate (requires an LLM).
    """

    name = "local"

    async def detect_obstacle(self, image_b64: str) -> VisionResult:
        start = time.monotonic()

        image = decode_base64_image(image_b64)
        image = resize_image(image, max_dimension=416)

        model = _get_yolo_model()
        results = model(image, verbose=False, conf=0.35, imgsz=416)

        detections = []
        has_danger = False

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])

                if cls_name in OBSTACLE_CLASSES:
                    detections.append((cls_name, conf))
                    if cls_name in DANGER_CLASSES:
                        has_danger = True

        latency = (time.monotonic() - start) * 1000

        if not detections:
            return VisionResult(
                text="Path is clear",
                provider=self.name,
                latency_ms=latency,
                severity="clear",
            )

        # Build warning from top detections
        top_detections = sorted(detections, key=lambda x: -x[1])[:3]
        warning_parts = [name for name, _ in top_detections]
        warning = f"{', '.join(warning_parts)} ahead"
        severity = "danger" if has_danger else "caution"

        return VisionResult(
            text=warning, provider=self.name, latency_ms=latency, severity=severity
        )

    async def build_sentence(self, partial_text: str, emotion: str) -> VisionResult:
        raise NotImplementedError("Local YOLO provider cannot build sentences")

    async def translate(self, text: str, target_lang: str) -> VisionResult:
        raise NotImplementedError("Local YOLO provider cannot translate")

    async def health_check(self) -> bool:
        try:
            _get_yolo_model()
            return True
        except Exception:
            return False
