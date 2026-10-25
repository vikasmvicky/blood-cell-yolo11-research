#!/usr/bin/env python3
"""YOLO11 training entry point for the TXL-PBC blood-cell dataset (3 classes).

Usage:
    python train.py --check-only               # validate dataset + environment, no training
    python train.py                             # YOLO11n baseline, 100 epochs, seed 42
    python train.py --epochs 1 --batch 4        # short smoke run
    python train.py --model yolo11s.pt          # larger YOLO11 variant

Exit codes: 0 = OK, 1 = dataset validation failed, 2 = environment blocked.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validate_dataset import validate  # noqa: E402  (local module, same folder)

DEFAULT_DATA = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs"
INSTALL_CMD = "python -m pip install -r requirements.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLO11 on TXL-PBC (WBC / RBC / Platelets)."
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help=f"data config (default: {DEFAULT_DATA})")
    p.add_argument("--model", default="yolo11n.pt",
                   help="YOLO11 model, e.g. yolo11n.pt (pretrained, auto-downloaded "
                        "on first run) or yolo11n.yaml (from scratch)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', '0', '0,1', ...")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--patience", type=int, default=50,
                   help="early-stopping patience (epochs)")
    p.add_argument("--project", type=Path, default=DEFAULT_PROJECT,
                   help=f"runs output dir (default: {DEFAULT_PROJECT})")
    p.add_argument("--name", default=None,
                   help="run name (default: <model-stem>_txlpbc)")
    p.add_argument("--resume", action="store_true", help="resume last run")
    p.add_argument("--check-only", action="store_true",
                   help="validate dataset + environment and exit, no training")
    p.add_argument("--skip-validation", action="store_true",
                   help="skip the pre-training dataset validation")
    return p.parse_args(argv)


def check_environment() -> tuple[bool, str, bool]:
    """Return (deps_ok, report, cuda_available)."""
    lines: list[str] = []
    ok = True
    cuda = False

    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        lines.append(f"torch      : {torch.__version__}"
                     + ("" if cuda else "  (CPU build / no GPU visible)"))
    except ImportError:
        ok = False
        lines.append("torch      : MISSING")

    try:
        import ultralytics
        lines.append(f"ultralytics: {ultralytics.__version__}")
    except ImportError:
        ok = False
        lines.append("ultralytics: MISSING")

    try:
        import yaml  # noqa: F811
        lines.append(f"PyYAML     : {yaml.__version__}")
    except ImportError:
        ok = False
        lines.append("PyYAML     : MISSING")

    if cuda:
        import torch
        lines.append(f"GPU        : {torch.cuda.get_device_name(0)}")
    else:
        lines.append("GPU        : none detected -> training would run on CPU (slow)")

    return ok, "\n".join(lines), cuda


def resolve_device(requested: str, cuda: bool) -> tuple[str | None, str | None]:
    """Return (device, fatal_error)."""
    req = (requested or "auto").strip().lower()
    if req in ("auto", ""):
        return ("0" if cuda else "cpu"), None
    if req == "cpu":
        return "cpu", None
    if not cuda:
        return None, (
            f"BLOCKED: --device {requested} requested but no CUDA device is available "
            f"(torch is CPU-only on this machine). Use --device cpu to run anyway."
        )
    return requested, None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 1) dataset validation ------------------------------------------------
    if args.skip_validation:
        print("[1/3] dataset validation: SKIPPED (--skip-validation)")
    else:
        print("[1/3] dataset validation")
        if not validate(args.data):
            print("next  : fix the dataset/config errors above, then re-run "
                  "`python validate_dataset.py`")
            return 1

    # 2) environment -------------------------------------------------------
    print("\n[2/3] environment")
    deps_ok, env_report, cuda = check_environment()
    print(env_report)
    if not deps_ok:
        print("BLOCKED: required packages are not installed.")
        print(f"next  : {INSTALL_CMD}")
        if args.check_only:
            return 2

    # 3) device ------------------------------------------------------------
    print("\n[3/3] device")
    device, fatal = resolve_device(args.device, cuda)
    if fatal:
        print(fatal)
        return 2
    print(f"device : {device}")

    run_name = args.name or f"{Path(args.model).stem}_txlpbc"

    if args.check_only:
        if deps_ok:
            print(f"\nREADY TO TRAIN: YES -> python train.py")
            return 0
        print("\nREADY TO TRAIN: NO (see blocker above)")
        return 2

    if not deps_ok:
        print("\nREADY TO TRAIN: NO (see blocker above)")
        return 2

    # 4) training ----------------------------------------------------------
    from ultralytics import YOLO

    model = YOLO(args.model)  # .pt missing -> ultralytics downloads it on first run
    print(f"\nTraining {args.model} on {args.data}")
    print(f"  epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} "
          f"optimizer={args.optimizer} seed={args.seed} device={device} "
          f"run={args.project / run_name}")

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        seed=args.seed,
        optimizer=args.optimizer,
        patience=args.patience,
        project=str(args.project),
        name=run_name,
        resume=args.resume,
        exist_ok=False,
        val=True,
        plots=True,
        deterministic=True,
    )

    # 5) report (only reached if training actually ran) --------------------
    trainer = getattr(model, "trainer", None)
    save_dir = Path(trainer.save_dir) if trainer is not None else args.project / run_name
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"

    metrics = getattr(results, "results_dict", None)
    if metrics:
        print("\n--- results ---")
        for key in ("metrics/precision(B)", "metrics/recall(B)",
                    "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
    print(f"run dir : {save_dir}")
    print(f"best    : {best}{'  (exists)' if best.is_file() else '  (NOT FOUND)'}")
    print(f"last    : {last}{'  (exists)' if last.is_file() else '  (NOT FOUND)'}")
    print("TRAINING COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
