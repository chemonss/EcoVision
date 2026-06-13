"""
Inference pipeline for EcoVision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.utils import PROJECT_ROOT, load_image, resolve_project_path, validate_confidence


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/best.pt"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> YOLO:
    """Load a trained YOLO model from disk."""
    model_path = resolve_project_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(str(model_path))


def class_name_from_model(names: Any, class_id: int) -> str:
    """Read a class name from Ultralytics model metadata."""
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def format_detection(box: Any, names: Any) -> dict:
    """Convert one raw YOLO box to the project detection format."""
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    coordinates = [round(float(value), 2) for value in box.xyxy[0].tolist()]

    return {
        "class_id": class_id,
        "class_name": class_name_from_model(names, class_id),
        "confidence": round(confidence, 4),
        "box": coordinates,
    }


def run_inference(
    model: YOLO,
    image,
    confidence_threshold: float = 0.25,
    imgsz: int = 640,
    iou: float = 0.5,
    max_det: int = 300,
    device: str | int | None = None,
) -> list[dict]:
    """Run detection for one image and return filtered structured objects."""
    confidence_threshold = validate_confidence(confidence_threshold)
    image = load_image(image)

    results = model.predict(
        image,
        conf=confidence_threshold,
        imgsz=imgsz,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )

    if not results:
        return []

    names = getattr(model, "names", {})
    detections = [
        format_detection(box, names)
        for box in results[0].boxes
        if float(box.conf.item()) >= confidence_threshold
    ]
    return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def detect_image(
    image,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    confidence_threshold: float = 0.25,
    imgsz: int = 640,
    device: str | int | None = None,
) -> list[dict]:
    """Load a model and run inference for a single image."""
    model = load_model(model_path)
    return run_inference(
        model=model,
        image=image,
        confidence_threshold=confidence_threshold,
        imgsz=imgsz,
        device=device,
    )
