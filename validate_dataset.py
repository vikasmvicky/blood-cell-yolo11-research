#!/usr/bin/env python3
"""Validate the TXL-PBC dataset before YOLO11 training.

Checks the data config, class names/order, split directories, image/label
pairing, label format and coordinate ranges, and prints per-class box counts.

Usage:
    python validate_dataset.py
    python validate_dataset.py --data dataset/TXL-PBC_Dataset-master/TXL-PBC/data.yaml

Exit codes: 0 = PASS, 1 = FAIL, 2 = config unreadable.
Dependencies: PyYAML only (no torch / ultralytics required).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # reported cleanly in load_config()
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "configs" / "data.yaml"

# Verified from labels/*/*.txt class ids across all splits.
EXPECTED_NAMES = {0: "WBC", 1: "RBC", 2: "Platelets"}
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IGNORE_NAMES = {"thumbs.db", "desktop.ini"}
MAX_PRINTED_ERRORS = 10


def _norm(name: str) -> str:
    """Normalize a class name for comparison ('Platelets' -> 'platelet')."""
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if len(s) > 3 and s.endswith("s"):
        s = s[:-1]
    return s


def load_config(data_yaml: Path) -> dict:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data config not found: {data_yaml}")
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed. Next command: python -m pip install PyYAML"
        )
    with open(data_yaml, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{data_yaml} did not parse into a YAML mapping")
    return cfg


def class_names(cfg: dict) -> dict[int, str]:
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("data config is missing 'names'")


def dataset_root(cfg: dict, data_yaml: Path) -> Path:
    raw = cfg.get("path")
    root = Path(str(raw)) if raw else data_yaml.parent
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def _split_dir(cfg: dict, root: Path, key: str) -> Path:
    raw = Path(str(cfg.get(key, key)))
    return raw if raw.is_absolute() else (root / raw).resolve()


def _list_images(img_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in img_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and p.name.lower() not in IGNORE_NAMES
        and not p.name.startswith(".")
    )


def _list_labels(lbl_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in lbl_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
        and p.name.lower() not in IGNORE_NAMES
        and not p.name.startswith(".")
    )


def _parse_label(
    path: Path, nc: int, errors: list[str]
) -> list[tuple[int, float, float, float, float]]:
    """Return [(cls, cx, cy, w, h)] for one label file; append to errors on bad lines."""
    boxes = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{lineno}: non-numeric value")
            continue
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            errors.append(f"{path.name}:{lineno}: non-finite coordinate")
            continue
        if not 0 <= cls < nc:
            errors.append(f"{path.name}:{lineno}: class id {cls} outside [0, {nc})")
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            errors.append(f"{path.name}:{lineno}: center ({cx}, {cy}) outside [0, 1]")
            continue
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append(f"{path.name}:{lineno}: size ({w}, {h}) outside (0, 1]")
            continue
        boxes.append((cls, cx, cy, w, h))
    return boxes


def validate(data_yaml: Path) -> bool:
    data_yaml = Path(data_yaml).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    print("=== TXL-PBC dataset validation ===")
    print(f"config : {data_yaml}")

    try:
        cfg = load_config(data_yaml)
        names = class_names(cfg)
        root = dataset_root(cfg, data_yaml)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAIL] {exc}")
        return False

    nc = int(cfg.get("nc", len(names)))
    print(f"root   : {root}")
    print(f"classes: {nc} -> {names}")

    # --- config checks -------------------------------------------------
    checks: list[tuple[bool, str]] = []
    for key in ("train", "val", "test"):
        if key not in cfg:
            errors.append(f"data config missing '{key}' key")
    if nc != len(names):
        errors.append(f"nc={nc} does not match len(names)={len(names)}")
    if set(names) != set(EXPECTED_NAMES) or any(
        _norm(names.get(i, "")) != _norm(v) for i, v in EXPECTED_NAMES.items()
    ):
        errors.append(
            f"class names/order mismatch: got {names}, "
            f"expected {EXPECTED_NAMES} (must match label ids)"
        )
    if not root.is_dir():
        errors.append(f"dataset root does not exist: {root}")
    checks.append((not errors, "config keys, nc, class names/order, root path"))

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        print("RESULT: FAIL")
        return False

    # --- per-split checks ---------------------------------------------
    split_stats: dict[str, dict] = {}
    all_stems: dict[str, set[str]] = {}
    for split in SPLITS:
        img_dir = _split_dir(cfg, root, split)
        # YOLO convention: labels mirror images/ under a parallel labels/ tree
        lbl_dir = root / "labels" / img_dir.name

        stat = {
            "images": 0, "labels": 0, "boxes": 0,
            "per_class": [0] * nc, "empty": 0,
        }
        if not img_dir.is_dir():
            errors.append(f"{split}: image dir missing: {img_dir}")
            split_stats[split] = stat
            continue
        if not lbl_dir.is_dir():
            errors.append(f"{split}: label dir missing: {lbl_dir}")
            split_stats[split] = stat
            continue

        images = _list_images(img_dir)
        labels = _list_labels(lbl_dir)
        stat["images"], stat["labels"] = len(images), len(labels)
        all_stems[split] = {p.stem for p in images}

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        for stem in sorted(img_stems - lbl_stems):
            errors.append(f"{split}: image without label: {stem}")
        for stem in sorted(lbl_stems - img_stems):
            errors.append(f"{split}: label without image: {stem}")

        for lp in labels:
            parsed = _parse_label(lp, nc, errors)
            if not parsed:
                stat["empty"] += 1
            for cls, *_ in parsed:
                stat["per_class"][cls] += 1
                stat["boxes"] += 1

        split_stats[split] = stat

    # cross-split leakage
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            dup = all_stems.get(a, set()) & all_stems.get(b, set())
            for stem in sorted(dup):
                errors.append(f"duplicate image stem across {a}/{b}: {stem}")

    # --- report ---------------------------------------------------------
    print()
    print(f"{'split':<6} {'images':>7} {'labels':>7} {'boxes':>7} "
          + " ".join(f"{EXPECTED_NAMES[i]:>9}" for i in range(nc)))
    totals = [0] * nc
    for split in SPLITS:
        s = split_stats[split]
        print(f"{split:<6} {s['images']:>7} {s['labels']:>7} {s['boxes']:>7} "
              + " ".join(f"{c:>9}" for c in s["per_class"]))
        totals = [t + c for t, c in zip(totals, s["per_class"])]
    print(f"{'total':<6} {sum(s['images'] for s in split_stats.values()):>7} "
          f"{sum(s['labels'] for s in split_stats.values()):>7} "
          f"{sum(s['boxes'] for s in split_stats.values()):>7} "
          + " ".join(f"{t:>9}" for t in totals))
    print()

    checks.append((all(s["images"] == s["labels"] and s["images"] > 0
                       for s in split_stats.values()),
                   "image/label pairing and non-empty splits"))
    checks.append((all(s["empty"] == 0 for s in split_stats.values()),
                   "no empty label files"))
    checks.append((not any("duplicate image stem" in e for e in errors),
                   "no duplicate images across splits"))
    checks.append((not any(
        ("fields" in e or "class id" in e or "outside" in e or "non-" in e)
        for e in errors),
                   "label format and coordinate ranges"))

    if sum(totals) > 0:
        rbc, platelet = totals[1], totals[2]
        ratio = (rbc / platelet) if platelet else float("inf")
        print(f"[INFO] class imbalance: RBC:Platelets = {ratio:.1f}:1 "
              f"({totals[1]} vs {totals[2]} boxes) -> objective 3 input")
        if ratio > 10:
            warnings.append("severe class imbalance (>10:1) - plan re-weighting/augmentation")

    print("--- checks ---")
    for ok, label in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {label}")
    for w in warnings:
        print(f"  [WARN] {w}")

    if errors:
        print(f"--- errors ({len(errors)}, showing up to {MAX_PRINTED_ERRORS}) ---")
        for e in errors[:MAX_PRINTED_ERRORS]:
            print(f"  [FAIL] {e}")

    ok = not errors
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help=f"path to data.yaml (default: {DEFAULT_DATA})")
    args = parser.parse_args(argv)
    try:
        return 0 if validate(args.data) else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAIL] {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
