"""
Prepare a one-label EcoVision YOLO dataset.

This script reuses the already prepared multi-class YOLO dataset and collapses
all object labels into one class: waste. Images are symlinked by default so the
experiment does not duplicate the full image dataset.

Expected input:
    data/processed/images/train/
    data/processed/images/val/
    data/processed/labels/train/
    data/processed/labels/val/

Generated output:
    data/processed_one_label/images/train/
    data/processed_one_label/images/val/
    data/processed_one_label/labels/train/
    data/processed_one_label/labels/val/
    configs/dataset_one_label.yaml
    results/yolo_one_label_dataset_statistics.md
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse the prepared EcoVision YOLO dataset into one class."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help="Prepared multi-class YOLO dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed_one_label",
        help="Output directory for the one-label YOLO dataset.",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=PROJECT_ROOT / "configs/dataset_one_label.yaml",
        help="Path where the one-label Ultralytics dataset YAML will be saved.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=PROJECT_ROOT / "results/yolo_one_label_dataset_statistics.md",
        help="Path where dataset statistics will be saved.",
    )
    parser.add_argument(
        "--class-name",
        default="waste",
        help="Name of the single detection class.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing one-label images and labels before conversion.",
    )
    return parser.parse_args()


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    if overwrite:
        shutil.rmtree(images_dir, ignore_errors=True)
        shutil.rmtree(labels_dir, ignore_errors=True)

    paths: dict[str, Path] = {}
    for split in SPLITS:
        paths[f"{split}_images"] = images_dir / split
        paths[f"{split}_labels"] = labels_dir / split

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def link_or_copy_image(source: Path, target: Path, copy_images: bool) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()

    if copy_images:
        shutil.copy2(source, target)
        return

    try:
        relative_source = os.path.relpath(source.resolve(), target.parent.resolve())
        target.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, target)


def collapse_label_file(source: Path, target: Path) -> tuple[int, Counter]:
    original_counts: Counter = Counter()
    lines: list[str] = []

    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"{source}:{line_number} must contain 5 YOLO columns")

        original_class = int(float(parts[0]))
        original_counts[original_class] += 1
        lines.append("0 " + " ".join(parts[1:]))

    target.write_text("\n".join(lines), encoding="utf-8")
    return len(lines), original_counts


def write_dataset_yaml(dataset_yaml: Path, output_dir: Path, class_name: str) -> None:
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = output_dir.resolve().as_posix()

    content = f"""path: {dataset_root}
train: images/train
val: images/val

names:
  0: {class_name}
"""
    dataset_yaml.write_text(content, encoding="utf-8")


def write_statistics(
    stats_file: Path,
    output_dir: Path,
    class_name: str,
    split_image_counts: Counter,
    split_object_counts: Counter,
    original_class_counts: dict[str, Counter],
    used_symlinks: bool,
) -> None:
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    total_images = sum(split_image_counts.values())
    total_objects = sum(split_object_counts.values())

    lines = [
        "# One-Label YOLO Dataset Statistics",
        "",
        f"- Dataset: `{output_dir.relative_to(PROJECT_ROOT)}`",
        f"- Class 0: `{class_name}`",
        f"- Images are {'symlinked' if used_symlinks else 'copied'} from `data/processed`.",
        "",
        "## Split",
        "",
        "| split | images | objects |",
        "|---|---:|---:|",
    ]

    for split in SPLITS:
        lines.append(f"| {split} | {split_image_counts[split]} | {split_object_counts[split]} |")

    lines.extend(
        [
            f"| total | {total_images} | {total_objects} |",
            "",
            "## Original Class Counts Before Collapse",
            "",
            "| split | original class id | objects |",
            "|---|---:|---:|",
        ]
    )

    for split in SPLITS:
        for class_id, count in sorted(original_class_counts[split].items()):
            lines.append(f"| {split} | {class_id} | {count} |")

    lines.extend(
        [
            "",
            "## Conversion Notes",
            "",
            "- The train/validation split is inherited from `data/processed`.",
            "- All YOLO label class IDs are rewritten to `0`; bounding boxes are unchanged.",
            "- This dataset tests whether the detector can localize waste without fine-grained waste type classification.",
        ]
    )

    stats_file.write_text("\n".join(lines), encoding="utf-8")


def convert_dataset(args: argparse.Namespace) -> None:
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    for split in SPLITS:
        for kind in ("images", "labels"):
            path = args.source_dir / kind / split
            if not path.exists():
                raise FileNotFoundError(f"Missing source directory: {path}")

    output_paths = prepare_output_dirs(args.output_dir, args.overwrite)
    split_image_counts: Counter = Counter()
    split_object_counts: Counter = Counter()
    original_class_counts = {split: Counter() for split in SPLITS}

    for split in SPLITS:
        source_images = args.source_dir / "images" / split
        source_labels = args.source_dir / "labels" / split

        for image_path in sorted(path for path in source_images.iterdir() if path.is_file()):
            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")

            output_image = output_paths[f"{split}_images"] / image_path.name
            output_label = output_paths[f"{split}_labels"] / label_path.name

            link_or_copy_image(image_path, output_image, args.copy_images)
            object_count, class_counts = collapse_label_file(label_path, output_label)

            split_image_counts[split] += 1
            split_object_counts[split] += object_count
            original_class_counts[split].update(class_counts)

    write_dataset_yaml(args.dataset_yaml, args.output_dir, args.class_name)
    write_statistics(
        stats_file=args.stats_file,
        output_dir=args.output_dir,
        class_name=args.class_name,
        split_image_counts=split_image_counts,
        split_object_counts=split_object_counts,
        original_class_counts=original_class_counts,
        used_symlinks=not args.copy_images,
    )

    print("One-label dataset conversion finished.")
    print(f"Train images: {split_image_counts['train']}")
    print(f"Validation images: {split_image_counts['val']}")
    print(f"Train objects: {split_object_counts['train']}")
    print(f"Validation objects: {split_object_counts['val']}")
    print(f"Dataset YAML: {args.dataset_yaml}")
    print(f"Statistics: {args.stats_file}")


def main() -> None:
    args = parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()
