"""Shared pytest fixtures for blood-cell detection tests.

Tests are scoped to ``tests/`` so the legacy research scripts at the
project root (``eval_test.py`` etc.) are not collected.

The real YOLO11n model is loaded once per session and skipped cleanly
when the weights file is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from detector import DEFAULT_WEIGHTS, load_model

SAMPLE_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "TXL-PBC_Dataset-master"
    / "TXL-PBC"
    / "images"
    / "test"
    / "00552f60c43350a0bf516cfbc13db058.png"
)


@pytest.fixture(scope="session")
def model():
    """Load the trained YOLO11n model once for the whole test session."""
    if not DEFAULT_WEIGHTS.is_file():
        pytest.skip(f"model weights not found: {DEFAULT_WEIGHTS}")
    return load_model(DEFAULT_WEIGHTS)


@pytest.fixture(scope="session")
def sample_image():
    """A real TXL-PBC blood-smear image from the held-out test split."""
    if not SAMPLE_IMAGE.is_file():
        pytest.skip(f"sample image not found: {SAMPLE_IMAGE}")
    from PIL import Image

    return Image.open(SAMPLE_IMAGE).convert("RGB")


@pytest.fixture
def make_video(tmp_path):
    """Factory that writes a small MP4 video and returns its path.

    ``content="smear"`` draws moving cell-like circles;
    ``content="blank"`` draws a black video (no detections expected).
    """

    def _make(
        name="sample.mp4",
        size=(320, 240),
        frames=12,
        fps=24.0,
        content="smear",
        fourcc="mp4v",
    ):
        path = tmp_path / name
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            size,
        )
        if not writer.isOpened():
            pytest.skip(f"VideoWriter not available for {fourcc}")
        for i in range(frames):
            if content == "smear":
                frame = np.full((size[1], size[0], 3), 200, np.uint8)
                # Moving bright disc, roughly the size of a smear object.
                cv2.circle(
                    frame,
                    (size[0] // 3 + i * 2, size[1] // 2),
                    min(size) // 6,
                    (60, 60, 210),
                    -1,
                )
            else:
                frame = np.zeros((size[1], size[0], 3), np.uint8)
            writer.write(frame)
        writer.release()
        return path

    return _make
