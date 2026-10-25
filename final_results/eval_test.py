#!/usr/bin/env python3
"""Evaluate the trained YOLO11n checkpoint on the held-out test split.

Outputs are written to an ABSOLUTE path inside this project so that the run's
working directory can never redirect results into another folder again.

Usage:
    python eval_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS = PROJECT_ROOT / "runs" / "yolo11n_txlpbc-2" / "weights" / "best.pt"
DATA = PROJECT_ROOT / "configs" / "data.yaml"
# Absolute output dir (the fix for the old capstone-path bug)
OUT_DIR = PROJECT_ROOT / "runs" / "yolo11n_test_evaluation_cleanpath"


def main() -> int:
    if not WEIGHTS.is_file():
        print(f"[FAIL] weights not found: {WEIGHTS}")
        return 2
    if not DATA.is_file():
        print(f"[FAIL] data config not found: {DATA}")
        return 2

    from ultralytics import YOLO

    print(f"weights : {WEIGHTS}")
    print(f"data    : {DATA}")
    print(f"split   : test")
    print(f"outdir  : {OUT_DIR}")

    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.7,
        project=str(OUT_DIR),
        name="val",
        exist_ok=True,
        save_json=False,
        plots=True,
        verbose=True,
    )

    print("\n--- test metrics ---")
    print(f"  precision(B)     : {metrics.box.mp:.4f} ({metrics.box.mp * 100:.2f}%)")
    print(f"  recall(B)        : {metrics.box.mr:.4f} ({metrics.box.mr * 100:.2f}%)")
    print(f"  mAP50(B)         : {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
    print(f"  mAP50-95(B)      : {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
    names = metrics.names
    for i, (p, r, ap50, ap) in enumerate(
        zip(metrics.box.p, metrics.box.r, metrics.box.ap50, metrics.box.ap)
    ):
        print(f"  {names[i]:<10}: P={p:.3f} R={r:.3f} mAP50={ap50:.3f} mAP50-95={ap:.3f}")
    print(f"\nsaved to: {OUT_DIR / 'val'}")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
