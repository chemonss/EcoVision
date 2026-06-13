"""
General utility functions for EcoVision.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_project_path(path: str | Path) -> Path:
    """Return an absolute path, treating relative paths as project-relative."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_confidence(value: float) -> float:
    """Validate a confidence threshold in the [0, 1] range."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence threshold must be between 0 and 1")
    return value


def validate_image_path(path: str | Path) -> Path:
    """Check that a path points to a supported image file."""
    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected an image file, got: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {path.suffix}")
    return path


def load_image(image: str | Path | BinaryIO | Image.Image) -> Image.Image:
    """Load an image from a path, file-like object, or PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    try:
        if isinstance(image, (str, Path)):
            image = validate_image_path(image)
        return Image.open(image).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Could not read image file") from exc


def save_image(image: Image.Image, path: str | Path) -> Path:
    """Save an image, creating the output directory if needed."""
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
