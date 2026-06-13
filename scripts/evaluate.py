"""
Evaluate a trained EcoVision YOLO model.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "tmp/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))

from ultralytics import YOLO  # noqa: E402


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/dataset.yaml")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/runs/yolov8n_baseline")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--n-errors", type=int, default=80)
    return parser.parse_args()


def pick_device(device: str) -> str | int:
    if device != "auto":
        return device

    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def read_dataset(path: Path) -> tuple[Path, Path, list[str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    names = data["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else names
    return root / data["val"], root / "labels/val", names


def labels_for(path: Path, size: tuple[int, int]) -> list[dict]:
    if not path.exists():
        return []

    width, height = size
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cls, xc, yc, bw, bh = map(float, line.split())
        labels.append(
            {
                "cls": int(cls),
                "box": [
                    (xc - bw / 2) * width,
                    (yc - bh / 2) * height,
                    (xc + bw / 2) * width,
                    (yc + bh / 2) * height,
                ],
            }
        )
    return labels


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def predict(model: YOLO, image: Path, args: argparse.Namespace, device: str | int) -> list[dict]:
    result = model.predict(str(image), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device, verbose=False)[0]
    return [
        {"cls": int(box.cls.item()), "conf": float(box.conf.item()), "box": [float(v) for v in box.xyxy[0]]}
        for box in result.boxes
    ]


def draw_example(image: Path, labels: list[dict], preds: list[dict], names: list[str], out: Path) -> None:
    img = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(img)

    for item in labels:
        draw.rectangle(item["box"], outline="lime", width=3)
        draw.text((item["box"][0], max(0, item["box"][1] - 12)), f"GT {names[item['cls']]}", fill="lime")

    for item in preds:
        draw.rectangle(item["box"], outline="red", width=3)
        draw.text((item["box"][0], item["box"][3] + 2), f"P {names[item['cls']]} {item['conf']:.2f}", fill="red")

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def save_curves(run_dir: Path, out: Path) -> None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return

    hist = pd.read_csv(csv_path)
    hist.columns = [c.strip() for c in hist.columns]

    plt.figure(figsize=(8, 4))
    for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"]:
        if col in hist:
            plt.plot(hist["epoch"], hist[col], label=col)
    plt.xlabel("epoch")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=160)
    plt.close()


def save_error_examples(model: YOLO, args: argparse.Namespace, device: str | int) -> Counter:
    val_images, val_labels, names = read_dataset(args.data)
    images = sorted(val_images.glob("*"))
    random.Random(42).shuffle(images)

    out_dir = args.out / "error_examples"
    shutil.rmtree(out_dir, ignore_errors=True)

    counts = Counter()
    saved = Counter()

    for image in images[: args.n_errors]:
        with Image.open(image) as img:
            image_size = img.size
            labels = labels_for(val_labels / f"{image.stem}.txt", image_size)
        preds = predict(model, image, args, device)

        matched = set()
        for pred in preds:
            best = max(range(len(labels)), key=lambda i: iou(pred["box"], labels[i]["box"]), default=None)
            if best is None or iou(pred["box"], labels[best]["box"]) < args.iou:
                counts["false_positive"] += 1
                kind = "false_positive"
            elif pred["cls"] != labels[best]["cls"]:
                counts["class_confusion"] += 1
                matched.add(best)
                kind = "class_confusion"
            else:
                matched.add(best)
                continue

            if saved[kind] < 4:
                saved[kind] += 1
                draw_example(image, labels, preds, names, out_dir / kind / f"{saved[kind]:02d}_{image.name}")

        missed = [label for i, label in enumerate(labels) if i not in matched]
        counts["missed_objects"] += len(missed)

        small = [label for label in missed if (label["box"][2] - label["box"][0]) * (label["box"][3] - label["box"][1]) < 0.01 * image_size[0] * image_size[1]]
        counts["missed_small_objects"] += len(small)

        if small and saved["missed_small"] < 4:
            saved["missed_small"] += 1
            draw_example(image, labels, preds, names, out_dir / "missed_small" / f"{saved['missed_small']:02d}_{image.name}")

        if (len(labels) >= 5 or len(missed) >= 2) and saved["hard_background"] < 4:
            saved["hard_background"] += 1
            draw_example(image, labels, preds, names, out_dir / "hard_background" / f"{saved['hard_background']:02d}_{image.name}")

    counts["checked_images"] = min(args.n_errors, len(images))
    return counts


def copy_val_plots(val_dir: Path, out: Path) -> None:
    names = {
        "confusion_matrix.png": "confusion_matrix.png",
        "confusion_matrix_normalized.png": "plots/confusion_matrix_normalized.png",
        "BoxF1_curve.png": "plots/F1_curve.png",
        "BoxP_curve.png": "plots/P_curve.png",
        "BoxR_curve.png": "plots/R_curve.png",
        "BoxPR_curve.png": "plots/PR_curve.png",
    }
    for src, dst in names.items():
        if (val_dir / src).exists():
            target = out / dst
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(val_dir / src, target)


def save_new_images(model: YOLO, args: argparse.Namespace, device: str | int) -> None:
    val_images, _, _ = read_dataset(args.data)
    images = [str(path) for path in sorted(val_images.glob("*"))[:8]]
    model.predict(images, imgsz=args.imgsz, conf=args.conf, device=device, save=True, project=str(args.out), name="new_image_tests", exist_ok=True, verbose=False)


def write_metrics(path: Path, metrics: dict, errors: Counter, device: str | int) -> None:
    device_name = f"cuda:{device}" if isinstance(device, int) else str(device)
    pd.DataFrame([{**metrics, **errors, "device": device_name}]).to_csv(path / "metrics.csv", index=False)

    rows = [
        ("mAP@50", metrics["map50"]),
        ("mAP@50-95", metrics["map50_95"]),
        ("precision", metrics["precision"]),
        ("recall", metrics["recall"]),
    ]
    lines = ["# Model Metrics", "", f"device: `{device_name}`", "", "| metric | value |", "|---|---:|"]
    lines += [f"| {name} | {value:.4f} |" for name, value in rows]
    lines += ["", "## Error Analysis", ""]
    lines += [f"- {key}: {errors[key]}" for key in ["checked_images", "false_positive", "missed_objects", "missed_small_objects", "class_confusion"]]
    lines += ["", "Artifacts: `models/best.pt`, `results/confusion_matrix.png`, `results/plots/`, `results/error_examples/`, `results/new_image_tests/`."]
    (path / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = get_args()
    args.model = args.model.resolve()
    args.data = args.data.resolve()
    args.run_dir = args.run_dir.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    model = YOLO(str(args.model))
    result = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        iou=args.iou,
        plots=True,
        project=str(args.out / "evaluation"),
        name=args.model.stem,
        exist_ok=True,
        verbose=False,
    )

    metrics = {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
    }

    copy_val_plots(Path(result.save_dir), args.out)
    shutil.copy2(args.run_dir / "results.png", args.out / "training_results.png")
    save_curves(args.run_dir, args.out)
    errors = save_error_examples(model, args, device)
    save_new_images(model, args, device)
    write_metrics(args.out, metrics, errors, device)

    print(pd.DataFrame([metrics]))


if __name__ == "__main__":
    main()
