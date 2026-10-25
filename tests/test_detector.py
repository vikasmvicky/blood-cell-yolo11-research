"""Tests for the image-detection functions in detector.py.

Covers: image loading/validation, inference, drawing, counting and CSV
export so the existing image mode cannot regress silently.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from detector import (
    CLASS_NAMES,
    CLASS_ORDER,
    Detection,
    count_by_class,
    detect,
    detections_to_csv,
    draw_detections,
    load_model,
    open_image,
    run_inference,
    DEFAULT_WEIGHTS,
)


class _FakeUpload:
    """Mimics Streamlit's UploadedFile (bytes + .name)."""

    def __init__(self, data: bytes, name: str = "upload.png"):
        self._data = data
        self.name = name

    def read(self) -> bytes:
        return self._data


def _png_bytes(width=64, height=48, color=(120, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# open_image
# --------------------------------------------------------------------------

def test_open_image_accepts_png():
    img = open_image(_FakeUpload(_png_bytes()))
    assert img.size == (64, 48)
    assert img.mode == "RGB"


def test_open_image_rejects_garbage():
    with pytest.raises(ValueError, match="not a valid image"):
        open_image(_FakeUpload(b"definitely not an image"))


def test_open_image_rejects_empty():
    with pytest.raises(ValueError):
        open_image(_FakeUpload(b""))


def test_open_image_converts_to_rgb():
    buf = io.BytesIO()
    Image.new("L", (10, 10), 128).save(buf, format="PNG")  # grayscale
    img = open_image(_FakeUpload(buf.getvalue()))
    assert img.mode == "RGB"


# --------------------------------------------------------------------------
# load_model
# --------------------------------------------------------------------------

def test_load_model_missing_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pt")


def test_default_weights_exist():
    assert DEFAULT_WEIGHTS.is_file(), f"missing: {DEFAULT_WEIGHTS}"
    assert DEFAULT_WEIGHTS.suffix == ".pt"


# --------------------------------------------------------------------------
# run_inference / detect (real model)
# --------------------------------------------------------------------------

def test_run_inference_rejects_bad_conf(model, sample_image):
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="[Cc]onfidence"):
            run_inference(model, sample_image, conf=bad)


def test_detect_real_smear_image(model, sample_image):
    detections, annotated = detect(model, sample_image, conf=0.25)

    assert annotated.size == sample_image.size
    assert len(detections) > 0, "expected detections on a real smear image"

    # Class ids and names must match the training configuration.
    for d in detections:
        assert d.cls_id in CLASS_NAMES
        assert d.name == CLASS_NAMES[d.cls_id]
        assert 0.0 <= d.confidence <= 1.0
        assert d.x2 > d.x1 and d.y2 > d.y1

    # Detections sorted by confidence, descending.
    confs = [d.confidence for d in detections]
    assert confs == sorted(confs, reverse=True)


def test_detect_draws_boxes(model, sample_image):
    """Annotated output must differ from the input where boxes were drawn."""
    detections, annotated = detect(model, sample_image, conf=0.25)
    assert detections, "need at least one detection for this test"

    assert annotated.size == sample_image.size
    diff = sum(
        abs(a - b)
        for a, b in zip(
            sample_image.convert("RGB").tobytes()[:5000],
            annotated.tobytes()[:5000],
        )
    )
    assert diff > 0, "annotated image is identical to the input"


def test_detect_conf_threshold_filters(model, sample_image):
    low, _ = detect(model, sample_image, conf=0.05)
    high, _ = detect(model, sample_image, conf=0.90)
    assert len(high) <= len(low)
    assert all(d.confidence >= 0.90 for d in high)


# --------------------------------------------------------------------------
# draw_detections (pure function)
# --------------------------------------------------------------------------

def test_draw_detections_no_changes_without_boxes():
    img = Image.new("RGB", (40, 40), (10, 10, 10))
    out = draw_detections(img, [])
    assert out.tobytes() == img.tobytes()


def test_draw_detections_draws_and_preserves_input():
    img = Image.new("RGB", (80, 80), (255, 255, 255))
    det = Detection(cls_id=0, name="WBC", confidence=0.9,
                    x1=10, y1=10, x2=50, y2=50)
    out = draw_detections(img, [det])
    assert out.size == img.size
    assert out.tobytes() != img.tobytes()  # something was drawn
    assert img.getpixel((10, 10)) == (255, 255, 255)  # input untouched


# --------------------------------------------------------------------------
# count_by_class / detections_to_csv
# --------------------------------------------------------------------------

def _det(cls_id, conf=0.5):
    return Detection(
        cls_id=cls_id,
        name=CLASS_NAMES[cls_id],
        confidence=conf,
        x1=0, y1=0, x2=10, y2=10,
    )


def test_count_by_class_zero_and_order():
    counts = count_by_class([])
    assert list(counts.keys()) == list(CLASS_ORDER)
    assert all(v == 0 for v in counts.values())


def test_count_by_class_totals():
    counts = count_by_class([_det(0), _det(0), _det(1), _det(2)])
    assert counts == {"WBC": 2, "RBC": 1, "Platelets": 1}


def test_detections_to_csv_format():
    csv_text = detections_to_csv([_det(1, conf=0.8765)])
    lines = csv_text.strip().splitlines()
    assert lines[0] == (
        "class_id,class_name,confidence,x1,y1,x2,y2,width,height"
    )
    assert len(lines) == 2
    fields = lines[1].split(",")
    assert fields[0] == "1"
    assert fields[1] == "RBC"
    assert fields[2] == "0.8765"
    assert fields[7:] == ["10", "10"]  # width, height


def test_detections_to_csv_empty_still_has_header():
    lines = detections_to_csv([]).strip().splitlines()
    assert len(lines) == 1
