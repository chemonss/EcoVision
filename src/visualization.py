"""
Visualization utilities for EcoVision.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.utils import load_image, save_image


COLORS = [
    "#00A36C",
    "#1E88E5",
    "#F4511E",
    "#8E24AA",
    "#D81B60",
    "#6D4C41",
]


def detection_color(class_id: int) -> str:
    """Pick a stable color for a class id."""
    return COLORS[class_id % len(COLORS)]


def draw_detections(
    image,
    detections: list[dict],
    output_path: str | Path | None = None,
) -> Image.Image:
    """Draw bounding boxes and labels on an image."""
    annotated = load_image(image).copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        class_id = int(detection.get("class_id", 0))
        color = detection_color(class_id)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_height = text_box[3] - text_box[1]
        text_width = text_box[2] - text_box[0]
        text_y = max(0, y1 - text_height - 4)
        draw.rectangle([x1, text_y, x1 + text_width + 6, text_y + text_height + 4], fill=color)
        draw.text((x1 + 3, text_y + 2), label, fill="white", font=font)

    if output_path is not None:
        save_image(annotated, output_path)

    return annotated
