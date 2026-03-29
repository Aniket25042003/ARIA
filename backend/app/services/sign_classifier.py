import asyncio
import time

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_processor = None
_device = None

# HuggingFace model for ASL alphabet classification (ResNet50, 29 classes)
MODEL_NAME = "Abuzaid01/asl-sign-language-classifier"


def _load_model():
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    logger.info("loading_sign_model", model=MODEL_NAME)
    start = time.monotonic()

    _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    _model.eval()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _device.type == "cuda":
        _model = _model.half().to(_device)
    else:
        _model = _model.to(_device)

    latency = (time.monotonic() - start) * 1000
    labels = list(_model.config.id2label.values())
    logger.info("sign_model_loaded", device=str(_device), latency_ms=int(latency), labels=labels)
    return _model, _processor


def _crop_hand_region(frame: np.ndarray, landmarks: list[list[float]], padding: int = 40) -> np.ndarray:
    """Crop hand region from frame using MediaPipe landmark bounding box."""
    h, w = frame.shape[:2]
    xs = [lm[0] * w for lm in landmarks]
    ys = [lm[1] * h for lm in landmarks]

    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return frame
    return cropped


def _classify_sync(frame_bgr: np.ndarray, landmarks: list[list[float]] | None = None) -> tuple[str, float]:
    """Run sign classification inference. Returns (sign, confidence)."""
    import torch
    from PIL import Image

    model, processor = _load_model()

    # Crop to hand region for better accuracy
    if landmarks:
        frame_bgr = _crop_hand_region(frame_bgr, landmarks)

    # Convert BGR to RGB PIL Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    # AutoImageProcessor handles resize to model's expected input
    inputs = processor(images=image, return_tensors="pt")
    if _device.type == "cuda":
        inputs = {k: v.half().to(_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits.float(), dim=-1)
    confidence, idx = probs.max(dim=-1)
    raw_label = model.config.id2label[idx.item()]

    # Normalize label to uppercase
    sign = raw_label.strip().upper()

    return sign, float(confidence)


async def classify_sign(frame_bgr: np.ndarray, landmarks: list[list[float]] | None = None) -> tuple[str, float]:
    """Async wrapper — runs classification in thread pool to avoid blocking event loop."""
    return await asyncio.to_thread(_classify_sync, frame_bgr, landmarks)
