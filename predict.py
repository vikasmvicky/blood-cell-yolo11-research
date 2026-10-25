#!/usr/bin/env python3
"""Run YOLO11 inference on held-out TXL-PBC test images and save visualizations.

Usage:
    python predict.py                                        # best.pt, 6 test images
    python predict.py --weights yolo11n.pt --no-save-plots   # stats only, initial model
    python predict.py --split val --limit 4 --conf 0.5

Outputs annotated images with predicted boxes + confidence scores to
runs/debug_stage1/<split>_predictions/ and prints per-image confidence
statistics at several thresholds next to ground-truth counts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "yolo11n_txlpbc" / "weights" / "best.pt"
DATASET_ROOT = (PROJECT_ROOT / "dataset" / "TXL-PBC_Dataset-master" / "TXL-PBC")
THRESHOLDS = (0.01, 0.05, 0.25, 0.50)
CLASS_NAMES = {0: "WBC", 1: "RBC", 2: "Platelets"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TXL-PBC held-out prediction")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--limit", type=int, default=6, help="number of images")
    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence threshold used for saved visualizations")
    p.add_argument("--stats-conf", type=float, default=0.001,
                   help="low threshold for the statistics pass (matches ultralytics val)")
    p.add_argument("--project", type=Path,
                   default=PROJECT_ROOT / "runs" / "debug_stage1")
    p.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args(argv)


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


def pick_images(split: str, limit: int) -> list[Path]:
    """Deterministic selection: images with WBC and platelet GT first, then by name."""
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split
    images = sorted(img_dir.glob("*.png"))
    chosen: list[Path] = []
    for want_cls in (0, 2):  # WBC, then Platelets
        if len(chosen) >= limit:
            break
        for img in images:
            if img in chosen:
                continue
            counts = gt_counts(lbl_dir / f"{img.stem}.txt")
            if counts.get(want_cls):
                chosen.append(img)
                break
    for img in images:  # fill the rest in sorted order
        if len(chosen) >= limit:
            break
        if img not in chosen:
            chosen.append(img)
    return chosen


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.weights.is_file():
        print(f"[FAIL] weights not found: {args.weights}")
        return 2

    from ultralytics import YOLO

    images = pick_images(args.split, args.limit)
    if not images:
        print(f"[FAIL] no images in {DATASET_ROOT / 'images' / args.split}")
        return 1
    names = {int(k): v for k, v in CLASS_NAMES.items()}
    print(f"weights: {args.weights}")
    print(f"split  : {args.split}  images: {len(images)}")

    model = YOLO(str(args.weights))
    saved_names = model.model.names
    if saved_names != names:
        print(f"[WARN] model.names {saved_names} differs from expected {names}")

    # Pass 1: statistics at the same low threshold ultralytics uses for val (0.001)
    stats_results = model.predict(
        source=[str(p) for p in images], conf=args.stats_conf,
        imgsz=args.imgsz, verbose=False,
    )

    # Pass 2: annotated visualizations at the reportable threshold
    plot_results = None
    if args.save_plots:
        out_dir = args.project / f"{args.split}_predictions"
        plot_results = model.predict(
            source=[str(p) for p in images], conf=args.conf,
            imgsz=args.imgsz, save=True, project=str(args.project),
            name=f"{args.split}_predictions", exist_ok=True, verbose=False,
        )
        print(f"plots  : {out_dir}\n")

    agg = {t: 0 for t in THRESHOLDS}
    agg_cls: dict[int, dict[int, int]] = {t: {} for t in THRESHOLDS}
    print(f"{'image':<40} {'GT':<22} " + " ".join(f"n@{t:<4}" for t in THRESHOLDS)
          + "  maxconf")
    for img, res in zip(images, stats_results):
        gt = gt_counts(DATASET_ROOT / "labels" / args.split / f"{img.stem}.txt")
        gt_str = ",".join(f"{CLASS_NAMES[c]}:{n}" for c, n in sorted(gt.items())) or "-"
        confs = res.boxes.conf
        clss = res.boxes.cls.int()
        row = []
        for t in THRESHOLDS:
            mask = confs >= t
            n = int(mask.sum())
            agg[t] += n
            for c in clss[mask].tolist():
                agg_cls[t][c] = agg_cls[t].get(c, 0) + 1
            row.append(str(n))
        maxconf = float(confs.max()) if confs.numel() else 0.0
        print(f"{img.name:<40} {gt_str:<22} " + " ".join(f"{r:<8}" for r in row)
              + f"  {maxconf:.3f}")

        # top-5 detections of this image
        if confs.numel():
            k = min(5, confs.numel())
            topv, topi = confs.topk(k)
            tops = ", ".join(f"{model.model.names[int(clss[i])]}:{float(v):.3f}"
                             for v, i in zip(topv, topi))
            print(f"{'':2}  top{k}: {tops}")

    print("\naggregate detections across picked images:")
    for t in THRESHOLDS:
        per_cls = ", ".join(f"{model.model.names.get(c, c)}={n}"
                            for c, n in sorted(agg_cls[t].items())) or "none"
        print(f"  conf>={t:<5} total={agg[t]:<5} by class: {per_cls}")
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
