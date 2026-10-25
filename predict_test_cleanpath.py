#!/usr/bin/env python3
"""Save annotated test-split predictions (absolute output dir, run -2 best.pt).

Selects test images that contain Platelets first, then fills with other images,
and saves conf>=0.25 annotated copies to
    runs/yolo11n_test_evaluation_cleanpath/predictions
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS = PROJECT_ROOT / "runs" / "yolo11n_txlpbc-2" / "weights" / "best.pt"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "TXL-PBC_Dataset-master" / "TXL-PBC"
OUT_DIR = PROJECT_ROOT / "runs" / "yolo11n_test_evaluation_cleanpath" / "predictions"


def gt_counts(label_file: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    if not label_file.is_file():
        return counts
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if parts:
            cls = int(float(parts[0]))
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def pick_images(limit_platelet: int = 4, limit_other: int = 4) -> list[Path]:
    """Deterministic selection: platelet-bearing images first, then others."""
    img_dir = DATASET_ROOT / "images" / "test"
    lbl_dir = DATASET_ROOT / "labels" / "test"
    images = sorted(img_dir.glob("*.png"))
    chosen: list[Path] = []
    # 1) images containing Platelets (class 2)
    for img in images:
        if len(chosen) >= limit_platelet:
            break
        if gt_counts(lbl_dir / f"{img.stem}.txt").get(2):
            chosen.append(img)
    # 2) fill with other images in sorted order
    for img in images:
        if len(chosen) >= limit_platelet + limit_other:
            break
        if img not in chosen:
            chosen.append(img)
    return chosen


def main() -> int:
    if not WEIGHTS.is_file():
        print(f"[FAIL] weights not found: {WEIGHTS}")
        return 2

    from ultralytics import YOLO

    images = pick_images()
    if not images:
        print(f"[FAIL] no test images found in {DATASET_ROOT / 'images' / 'test'}")
        return 1

    print(f"weights : {WEIGHTS}")
    print(f"outdir  : {OUT_DIR}")
    print(f"images  : {len(images)}")
    model = YOLO(str(WEIGHTS))
    results = model.predict(
        source=[str(p) for p in images],
        conf=0.25,
        imgsz=640,
        save=True,
        project=str(OUT_DIR.parent),
        name="predictions",
        exist_ok=True,
        verbose=False,
    )

    names = model.model.names
    lbl_dir = DATASET_ROOT / "labels" / "test"
    print(f"\n{'image':<46} {'GT (after dedup)':<28} predictions@0.25")
    for img, res in zip(images, results):
        gt = gt_counts(lbl_dir / f"{img.stem}.txt")
        gt_str = ", ".join(f"{names[c]}:{n}" for c, n in sorted(gt.items())) or "-"
        from collections import Counter
        cnt = Counter(names[int(c)] for c in res.boxes.cls.int().tolist())
        pred_str = ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items())) or "-"
        print(f"{img.name:<46} {gt_str:<28} {pred_str}")

    print(f"\nsaved to: {OUT_DIR}")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
