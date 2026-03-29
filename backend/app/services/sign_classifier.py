import asyncio
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from app.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_transform = None
_device = None

# 29 classes: A-Z + del, nothing, space
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "DEL", "NOTHING", "SPACE",
]

MODEL_REPO = "Abuzaid01/asl-sign-language-classifier"


class ASLResNet(nn.Module):
    """Exact architecture from the HuggingFace model repo."""

    def __init__(self, num_classes=29):
        super().__init__()
        self.model = models.resnet50(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def _load_model():
    global _model, _transform, _device
    if _model is not None:
        return _model, _transform

    from huggingface_hub import hf_hub_download

    logger.info("loading_sign_model", model=MODEL_REPO)
    start = time.monotonic()

    # Download weights from HuggingFace
    weights_path = hf_hub_download(MODEL_REPO, "pytorch_model.bin")

    # Build model with correct architecture
    _model = ASLResNet(num_classes=29)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    _model.load_state_dict(state_dict)
    _model.eval()

    # Use CUDA only if it actually works
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = _model.to(_device)

    # Standard ImageNet preprocessing (ResNet50 expects this)
    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    latency = (time.monotonic() - start) * 1000
    logger.info("sign_model_loaded", device=str(_device), latency_ms=int(latency))
    return _model, _transform


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
    model, transform = _load_model()

    # Crop to hand region for better accuracy
    if landmarks:
        frame_bgr = _crop_hand_region(frame_bgr, landmarks)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Transform and run inference
    tensor = transform(rgb).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = model(tensor)

    probs = torch.softmax(outputs, dim=-1)
    confidence, idx = probs.max(dim=-1)
    sign = CLASS_NAMES[idx.item()]

    return sign, float(confidence)


async def classify_sign(frame_bgr: np.ndarray, landmarks: list[list[float]] | None = None) -> tuple[str, float]:
    """Async wrapper — runs classification in thread pool to avoid blocking event loop."""
    return await asyncio.to_thread(_classify_sync, frame_bgr, landmarks)
