"""
Prepare EcoVision dataset in YOLO detection format.

This script converts original TACO COCO-like annotations into the YOLO format
used by Ultralytics models: YOLO, RT-DETR and YOLO-World.

Expected input:
    data/raw/TACO/data/annotations.json
    data/raw/TACO/data/batch_*/

Generated output:
    data/processed/images/train/
    data/processed/images/val/
    data/processed/labels/train/
    data/processed/labels/val/
    configs/dataset.yaml
    results/yolo_dataset_statistics.md
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.class_mapping import COARSE_CLASSES, map_taco_category_to_coarse_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
        Parsed script arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert TACO annotations into YOLO detection format."
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "data/raw/annotations.json",
        help="Path to TACO annotations.json.",
    )

    parser.add_argument(
        "--images-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/",
        help="Root directory containing TACO image batch folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help="Output directory for YOLO-format dataset.",
    )

    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=PROJECT_ROOT / "configs/dataset.yaml",
        help="Path where Ultralytics dataset.yaml will be saved.",
    )

    parser.add_argument(
        "--stats-file",
        type=Path,
        default=PROJECT_ROOT / "results/yolo_dataset_statistics.md",
        help="Path where dataset statistics will be saved.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Target validation ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing processed images and labels before conversion.",
    )

    return parser.parse_args()



def make_safe_image_name(file_name: str) -> str:
    """
    Convert nested TACO image path into a flat safe filename.

    Example:
        batch_1/000001.jpg -> batch_1_000001.jpg

    Parameters
        file_name: Original image filename from TACO annotations.

    Returns
        Safe filename for processed YOLO dataset.
    """
    path = Path(file_name)
    stem = "_".join(path.with_suffix("").parts)
    suffix = path.suffix.lower()

    return f"{stem}{suffix}"


def resolve_image_path(images_root: Path, relative_path: Path) -> Path:
    """
    Find an image path even when the local extension case differs from annotations.
    """
    image_path = images_root / relative_path

    if image_path.exists():
        return image_path

    if image_path.parent.exists():
        target_name = image_path.name.lower()
        for candidate in image_path.parent.iterdir():
            if candidate.name.lower() == target_name:
                return candidate

    raise FileNotFoundError(f"Image file not found: {image_path}")


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """
    Convert COCO bbox into normalized YOLO bbox.

    COCO format:
        [x_min, y_min, width, height]

    YOLO format:
        [x_center, y_center, width, height], normalized to [0, 1].

    Parameters
        bbox: COCO-format bounding box.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns
        Normalized YOLO bbox, or None if bbox is invalid.
    """
    x_min, y_min, box_width, box_height = bbox

    x_max = x_min + box_width
    y_max = y_min + box_height

    # Clip bbox to image boundaries.
    x_min = max(0.0, min(float(x_min), float(image_width)))
    y_min = max(0.0, min(float(y_min), float(image_height)))
    x_max = max(0.0, min(float(x_max), float(image_width)))
    y_max = max(0.0, min(float(y_max), float(image_height)))

    clipped_width = x_max - x_min
    clipped_height = y_max - y_min

    x_center = (x_min + clipped_width / 2.0) / image_width
    y_center = (y_min + clipped_height / 2.0) / image_height
    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height

    values = (x_center, y_center, normalized_width, normalized_height)
    return values


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    """
    Create output directories for YOLO dataset.

    Parameters
        output_dir: Root output directory.
        overwrite: If True, remove previous processed images and labels.

    Returns
        Dictionary with output directory paths.
    """
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    if overwrite:
        shutil.rmtree(images_dir, ignore_errors=True)
        shutil.rmtree(labels_dir, ignore_errors=True)

    paths = {
        "train_images": images_dir / "train",
        "val_images": images_dir / "val",
        "train_labels": labels_dir / "train",
        "val_labels": labels_dir / "val",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def build_image_class_counts(
    image_id_to_annotations: dict[int, list[dict]],
    category_id_to_info: dict[int, dict],
) -> dict[int, Counter]:
    """
    Count EcoVision class occurrences inside each image.

    This is used for class-aware image-level train/validation split.

    Parameters
        image_id_to_annotations: Mapping from image ID to its annotations.
        category_id_to_info: Mapping from original TACO category ID to metadata.

    Returns
        Dictionary mapping image ID to Counter[class_id -> object count].
    """
    image_id_to_class_counts: dict[int, Counter] = {}

    for image_id, annotations in image_id_to_annotations.items():
        class_counts = Counter()

        for annotation in annotations:
            category_info = category_id_to_info[annotation["category_id"]]

            class_id = map_taco_category_to_coarse_id(
                name=category_info["name"],
                supercategory=category_info["supercategory"],
            )

            class_counts[class_id] += 1

        image_id_to_class_counts[image_id] = class_counts

    return image_id_to_class_counts


def compute_split_score(
    val_counts: Counter,
    val_image_count: int,
    target_val_counts: Counter,
    target_val_image_count: int,
    target_val_object_count: float,
) -> float:
    """
    Compute how close a validation split is to the target distribution.

    Lower score is better.

    Parameters
        val_counts: Current validation object counts by class.
        val_image_count: Current number of validation images.
        target_val_counts: Target validation object counts by class.
        target_val_image_count: Target number of validation images.
        target_val_object_count: Target total number of validation objects.

    Returns
        Split quality score.
    """
    class_score = 0.0

    for class_id in range(len(COARSE_CLASSES)):
        target_count = target_val_counts[class_id]
        actual_count = val_counts[class_id]

        if target_count <= 0:
            class_score += float(actual_count**2)
            continue

        relative_error = (actual_count - target_count) / target_count
        class_score += relative_error**2

    actual_val_object_count = sum(val_counts.values())

    object_relative_error = (
        (actual_val_object_count - target_val_object_count) / target_val_object_count
        if target_val_object_count > 0
        else 0.0
    )

    image_relative_error = (
        (val_image_count - target_val_image_count) / target_val_image_count
        if target_val_image_count > 0
        else 0.0
    )

    object_score = object_relative_error**2
    image_score = image_relative_error**2

    return class_score + 0.75 * object_score + 0.25 * image_score


def class_aware_split_image_ids(
    image_ids: list[int],
    image_id_to_class_counts: dict[int, Counter],
    val_ratio: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    """
    Split image IDs into train and validation subsets.

    The split is performed at image level, but validation images are selected
    to keep per-class object counts close to the requested validation ratio.

    Parameters
        image_ids: List of all image IDs.
        image_id_to_class_counts: Per-image object counts by EcoVision class.
        val_ratio: Target validation ratio.
        seed: Random seed.

    Returns
        Train image IDs and validation image IDs.
    """
    rng = random.Random(seed)

    shuffled_image_ids = list(image_ids)
    rng.shuffle(shuffled_image_ids)

    target_val_image_count = int(round(len(shuffled_image_ids) * val_ratio))
    target_val_image_count = max(
        1,
        min(target_val_image_count, len(shuffled_image_ids) - 1),
    )

    total_class_counts = Counter()
    for image_id in shuffled_image_ids:
        total_class_counts.update(image_id_to_class_counts.get(image_id, Counter()))

    target_val_counts = Counter(
        {
            class_id: total_class_counts[class_id] * val_ratio
            for class_id in range(len(COARSE_CLASSES))
        }
    )

    target_val_object_count = sum(total_class_counts.values()) * val_ratio

    remaining_image_ids = list(shuffled_image_ids)
    val_ids: set[int] = set()
    val_counts = Counter()

    while len(val_ids) < target_val_image_count:
        best_index = None
        best_score = float("inf")

        for index, image_id in enumerate(remaining_image_ids):
            candidate_counts = val_counts + image_id_to_class_counts.get(
                image_id,
                Counter(),
            )

            candidate_score = compute_split_score(
                val_counts=candidate_counts,
                val_image_count=len(val_ids) + 1,
                target_val_counts=target_val_counts,
                target_val_image_count=target_val_image_count,
                target_val_object_count=target_val_object_count,
            )

            if candidate_score < best_score:
                best_score = candidate_score
                best_index = index

        if best_index is None:
            break

        selected_image_id = remaining_image_ids.pop(best_index)
        val_ids.add(selected_image_id)
        val_counts.update(image_id_to_class_counts.get(selected_image_id, Counter()))

    train_ids = set(shuffled_image_ids) - val_ids

    return train_ids, val_ids


def write_dataset_yaml(dataset_yaml_path: Path) -> None:
    """
    Write Ultralytics dataset.yaml.

    The path is relative to the project root. Therefore, training commands
    should be executed from the root directory of the repository.

    Parameters
        dataset_yaml_path: Output path for dataset.yaml.
    """
    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    names_block = "\n".join(
        f"  {class_id}: {class_name}"
        for class_id, class_name in enumerate(COARSE_CLASSES)
    )

    dataset_root = (PROJECT_ROOT / "data/processed").resolve().as_posix()

    content = f"""path: {dataset_root}
train: images/train
val: images/val

names:
{names_block}
"""

    dataset_yaml_path.write_text(content, encoding="utf-8")


def write_statistics(
    stats_file: Path,
    train_image_count: int,
    val_image_count: int,
    train_object_counts: Counter,
    val_object_counts: Counter,
) -> None:
    """
    Save dataset preparation statistics.

    Parameters
        stats_file: Path to output markdown file.
        train_image_count: Number of train images.
        val_image_count: Number of validation images.
        train_object_counts: Per-class object counts for train split.
        val_object_counts: Per-class object counts for validation split.
    """
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    total_counts = Counter()
    total_counts.update(train_object_counts)
    total_counts.update(val_object_counts)

    lines = [
        "# YOLO Dataset Statistics",
        "",
        "## Split",
        "",
        f"- Train images: {train_image_count}",
        f"- Validation images: {val_image_count}",
        f"- Total images: {train_image_count + val_image_count}",
        "",
        "## Classes",
        "",
        "| class_id | class_name | train objects | val objects | total objects |",
        "|---:|---|---:|---:|---:|",
    ]

    for class_id, class_name in enumerate(COARSE_CLASSES):
        train_count = train_object_counts[class_id]
        val_count = val_object_counts[class_id]
        total_count = total_counts[class_id]

        lines.append(
            f"| {class_id} | {class_name} | "
            f"{train_count} | {val_count} | {total_count} |"
        )

    lines.extend(
        [
            "",
            "## Conversion Notes",
            "",
            "- Original TACO annotations were converted from COCO-like format to YOLO detection format.",
            "- Original TACO categories were mapped to six EcoVision coarse classes using `src/class_mapping.py`.",
            "- Bounding boxes were converted from `[x_min, y_min, width, height]` to normalized YOLO `[x_center, y_center, width, height]` format.",
            "- The train/validation split was performed at image level using a class-aware strategy.",
        ]
    )

    stats_file.write_text("\n".join(lines), encoding="utf-8")


def convert_dataset(args: argparse.Namespace) -> None:
    """
    Convert TACO dataset into YOLO format.

    The function performs a class-aware image-level train/validation split,
    then converts COCO-like TACO annotations into YOLO label files.

    Parameters
        args: Parsed command line arguments.
    """
    with args.annotations.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    images = annotations["images"]
    categories = annotations["categories"]
    object_annotations = annotations["annotations"]

    image_id_to_info = {image["id"]: image for image in images}

    category_id_to_info = {
        category["id"]: {
            "name": category["name"],
            "supercategory": category.get("supercategory", ""),
        }
        for category in categories
    }

    image_id_to_annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in object_annotations:
        image_id_to_annotations[annotation["image_id"]].append(annotation)

    image_id_to_class_counts = build_image_class_counts(
        image_id_to_annotations=image_id_to_annotations,
        category_id_to_info=category_id_to_info,
    )

    train_ids, val_ids = class_aware_split_image_ids(
        image_ids=list(image_id_to_info.keys()),
        image_id_to_class_counts=image_id_to_class_counts,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_paths = prepare_output_dirs(args.output_dir, args.overwrite)

    split_counters = {
        "train": Counter(),
        "val": Counter(),
    }

    split_image_counts = {
        "train": 0,
        "val": 0,
    }


    for image_id, image_info in image_id_to_info.items():
        split = "val" if image_id in val_ids else "train"

        image_width = int(image_info["width"])
        image_height = int(image_info["height"])

        source_relative_path = Path(image_info["file_name"])
        source_image_path = resolve_image_path(args.images_root, source_relative_path)

        output_image_name = make_safe_image_name(image_info["file_name"])
        output_label_name = Path(output_image_name).with_suffix(".txt").name

        output_image_path = output_paths[f"{split}_images"] / output_image_name
        output_label_path = output_paths[f"{split}_labels"] / output_label_name

        shutil.copy2(source_image_path, output_image_path)

        label_lines: list[str] = []

        for annotation in image_id_to_annotations.get(image_id, []):
            category_info = category_id_to_info[annotation["category_id"]]

            class_id = map_taco_category_to_coarse_id(
                name=category_info["name"],
                supercategory=category_info["supercategory"],
            )

            if not 0 <= class_id < len(COARSE_CLASSES):
                raise ValueError(f"Invalid class_id={class_id} for image_id={image_id}")

            yolo_bbox = coco_bbox_to_yolo(
                bbox=annotation["bbox"],
                image_width=image_width,
                image_height=image_height,
            )

            x_center, y_center, width, height = yolo_bbox

            label_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

            split_counters[split][class_id] += 1

        output_label_path.write_text("\n".join(label_lines), encoding="utf-8")
        split_image_counts[split] += 1

    write_dataset_yaml(args.dataset_yaml)

    write_statistics(
        stats_file=args.stats_file,
        train_image_count=split_image_counts["train"],
        val_image_count=split_image_counts["val"],
        train_object_counts=split_counters["train"],
        val_object_counts=split_counters["val"],
    )

    print("Dataset conversion finished.")
    print(f"Train images: {split_image_counts['train']}")
    print(f"Validation images: {split_image_counts['val']}")
    print(f"Dataset YAML: {args.dataset_yaml}")
    print(f"Statistics: {args.stats_file}")
    print()

    print("Class distribution after split:")
    for class_id, class_name in enumerate(COARSE_CLASSES):
        train_count = split_counters["train"][class_id]
        val_count = split_counters["val"][class_id]
        total_count = train_count + val_count
        val_share = val_count / total_count if total_count > 0 else 0.0

        print(
            f"{class_id} {class_name}: "
            f"train={train_count}, "
            f"val={val_count}, "
            f"total={total_count}, "
            f"val_share={val_share:.3f}"
        )


def main() -> None:
    args = parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()
