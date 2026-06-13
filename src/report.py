"""
Report generation logic for EcoVision.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


def detections_to_table(detections: list[dict]) -> pd.DataFrame:
    """Create a tabular view of detected objects."""
    rows = []
    for index, item in enumerate(detections, start=1):
        x1, y1, x2, y2 = item["box"]
        rows.append(
            {
                "#": index,
                "class_name": item["class_name"],
                "confidence": item["confidence"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return pd.DataFrame(rows, columns=["#", "class_name", "confidence", "x1", "y1", "x2", "y2"])


def class_statistics(detections: list[dict]) -> dict[str, int]:
    """Count detections by class name."""
    return dict(Counter(item["class_name"] for item in detections))


def pollution_level(detections: list[dict]) -> str:
    """Estimate pollution level from the number of detected waste objects."""
    count = len(detections)
    if count == 0:
        return "clean"
    if count <= 2:
        return "low"
    if count <= 6:
        return "medium"
    return "high"


def cleanup_recommendation(detections: list[dict]) -> str:
    """Generate a short cleanup recommendation."""
    level = pollution_level(detections)
    stats = class_statistics(detections)

    if level == "clean":
        return "No visible waste was detected. No cleanup action is needed."

    main_class = max(stats, key=stats.get)
    if level == "low":
        return f"A small amount of waste was detected, mostly {main_class}. Local pickup is recommended."
    if level == "medium":
        return f"Several waste objects were detected, mostly {main_class}. Cleanup with gloves and sorting bags is recommended."
    return f"High pollution was detected, mostly {main_class}. Organize a cleanup and separate recyclable waste where possible."


def build_report(detections: list[dict]) -> dict:
    """Build the structured project report for one image."""
    stats = class_statistics(detections)
    return {
        "objects": detections,
        "summary": {
            "total": len(detections),
            "by_class": stats,
        },
        "pollution_level": pollution_level(detections),
        "recommendation": cleanup_recommendation(detections),
    }
