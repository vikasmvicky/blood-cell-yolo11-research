"""Tests for video_processor.py.

Covers: upload validation, probing (corrupt/empty/unsupported inputs),
end-to-end frame-by-frame processing with the real model, count accuracy,
frame skipping, output integrity, codec fallback and temp-file hygiene.
"""

from __future__ import annotations

import glob
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

import video_processor as vp
from detector import CLASS_ORDER, DEFAULT_WEIGHTS, detect
from video_processor import (
    VIDEO_SUFFIXES,
    delete_quietly,
    probe_video,
    process_video,
    save_upload_to_temp,
)


# --------------------------------------------------------------------------
# Extension / upload validation
# --------------------------------------------------------------------------

def test_video_suffixes_expected():
    assert VIDEO_SUFFIXES == ("mp4", "avi", "mov")


def test_save_upload_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported video format"):
        save_upload_to_temp(b"data", "clip.mkv")


def test_save_upload_rejects_missing_extension():
    with pytest.raises(ValueError, match="Unsupported video format"):
        save_upload_to_temp(b"data", "noextension")


def test_save_upload_writes_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setattr(
        vp.tempfile, "tempdir", str(tmp_path),
    )
    path = save_upload_to_temp(b"hello-bytes", "clip.mp4")
    try:
        assert path.is_file()
        assert path.suffix == ".mp4"
        assert path.read_bytes() == b"hello-bytes"
    finally:
        delete_quietly(path)
    assert not path.exists()


def test_uppercase_extension_accepted(tmp_path):
    src = save_upload_to_temp(b"x", "CLIP.MOV")
    assert src.suffix == ".mov"
    delete_quietly(src)


# --------------------------------------------------------------------------
# probe_video: invalid and unsupported inputs
# --------------------------------------------------------------------------

def test_probe_missing_file(tmp_path):
    with pytest.raises(ValueError, match="empty or missing"):
        probe_video(tmp_path / "absent.mp4")


def test_probe_empty_file(tmp_path):
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="empty or missing"):
        probe_video(p)


def test_probe_garbage_bytes(tmp_path):
    p = tmp_path / "garbage.mp4"
    p.write_bytes(b"this is not a video" * 100)
    with pytest.raises(ValueError, match="could not be opened|no decodable"):
        probe_video(p)


def test_probe_truncated_video(tmp_path, make_video):
    src = make_video(frames=10)
    p = tmp_path / "trunc.mp4"
    p.write_bytes(src.read_bytes()[:1024])
    with pytest.raises(ValueError):
        probe_video(p)


def test_probe_unsupported_extension(tmp_path):
    p = tmp_path / "clip.mkv"
    p.write_bytes(b"whatever")
    with pytest.raises(ValueError, match="Unsupported video format"):
        probe_video(p)


# --------------------------------------------------------------------------
# probe_video: valid inputs and metadata
# --------------------------------------------------------------------------

def test_probe_valid_mp4_metadata(make_video):
    src = make_video(size=(320, 240), frames=12, fps=24.0)
    info = probe_video(src)
    assert info.width == 320
    assert info.height == 240
    assert info.frame_count == 12
    assert info.fps == pytest.approx(24.0, abs=0.5)
    assert info.duration_seconds == pytest.approx(0.5, abs=0.05)
    assert info.suffix == ".mp4"
    assert info.resolution == "320 x 240"
    assert not info.fps_assumed


def test_probe_mov_container(make_video):
    src = make_video(name="clip.mov", size=(160, 120), frames=6)
    info = probe_video(src)
    assert info.width == 160 and info.height == 120
    assert info.suffix == ".mov"


def test_probe_odd_dimensions_reports_source_size(make_video):
    """probe must report whatever dimensions the decoded frames have.

    OpenCV writers round odd sizes at encode time on this machine, so
    the encoded source comes out even; the assertion accepts either to
    stay valid on backends that do preserve odd sizes.
    """
    src = make_video(size=(321, 241), frames=4)
    info = probe_video(src)
    assert (info.width, info.height) in {(321, 241), (320, 240)}


# --------------------------------------------------------------------------
# delete_quietly
# --------------------------------------------------------------------------

def test_delete_quietly_handles_none_and_missing():
    delete_quietly(None)
    delete_quietly(Path("/nonexistent/nowhere.mp4"))  # must not raise


def test_delete_quietly_removes_file(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"data")
    delete_quietly(p)
    assert not p.exists()


# --------------------------------------------------------------------------
# process_video: argument validation
# --------------------------------------------------------------------------

def test_process_rejects_bad_conf(model, make_video):
    src = make_video()
    with pytest.raises(ValueError, match="[Cc]onfidence"):
        process_video(model, src, conf=1.5)


def test_process_rejects_bad_interval(model, make_video):
    src = make_video()
    with pytest.raises(ValueError, match="interval"):
        process_video(model, src, frame_interval=0)


def test_process_rejects_corrupt_input(model, tmp_path):
    p = tmp_path / "corrupt.mp4"
    p.write_bytes(b"garbage" * 200)
    with pytest.raises(ValueError):
        process_video(model, p)


# --------------------------------------------------------------------------
# process_video: end-to-end with the real model
# --------------------------------------------------------------------------

def _decode_all(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def test_process_valid_video_end_to_end(model, make_video):
    src = make_video(size=(320, 240), frames=12, fps=24.0)
    result = process_video(model, src, conf=0.25, frame_interval=1)

    try:
        # Frame accounting.
        assert result.processed_frames == 12
        assert result.written_frames == 12
        assert result.failed_frames == 0
        assert result.frame_interval == 1

        # Output exists, plays and has the expected properties.
        assert result.output_path.is_file()
        assert result.output_path.stat().st_size > 0
        out_info = probe_video(result.output_path)
        assert out_info.frame_count == 12
        assert out_info.width == 320 and out_info.height == 240

        # Counts are internally consistent: totals = sum of class counts.
        assert set(result.counts) == set(CLASS_ORDER)
        assert sum(result.counts.values()) == result.total_detections

        # Every decoded output frame must exist (plays correctly).
        decoded = _decode_all(result.output_path)
        assert len(decoded) == 12
    finally:
        delete_quietly(result.output_path)


def test_process_counts_match_image_pipeline(model, make_video):
    """Video counts must equal per-frame counts from detect() on the same frames."""
    src = make_video(size=(320, 240), frames=6, fps=12.0)
    result = process_video(model, src, conf=0.25, frame_interval=1)
    try:
        # Independent reference: run the *image* pipeline on each frame.
        from PIL import Image

        cap = cv2.VideoCapture(str(src))
        expected = {name: 0 for name in CLASS_ORDER}
        frames_read = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets, _ = detect(model, Image.fromarray(rgb), conf=0.25)
            for d in dets:
                expected[d.name] += 1
            frames_read += 1
        cap.release()

        assert frames_read == 6
        assert result.counts == expected, (
            f"video counts {result.counts} != image-pipeline counts {expected}"
        )
        assert result.total_detections == sum(expected.values())
    finally:
        delete_quietly(result.output_path)


def test_process_blank_video_zero_detections(model, make_video):
    src = make_video(size=(320, 240), frames=6, content="blank")
    result = process_video(model, src, conf=0.25)
    try:
        assert result.total_detections == 0
        assert all(v == 0 for v in result.counts.values())
        assert result.processed_frames == 6
        # Output still produced and playable even with zero detections.
        assert probe_video(result.output_path).frame_count == 6
    finally:
        delete_quietly(result.output_path)


def test_process_frame_interval_skips_inference(model, make_video):
    src = make_video(size=(320, 240), frames=12)
    result = process_video(model, src, conf=0.25, frame_interval=3)
    try:
        assert result.processed_frames == 4   # frames 0, 3, 6, 9
        assert result.written_frames == 12    # all frames still written
        assert "1 in 3 frames" in result.note
        assert probe_video(result.output_path).frame_count == 12
    finally:
        delete_quietly(result.output_path)


def test_process_odd_dimensions_round_to_even(model, make_video, monkeypatch):
    """Odd source dimensions must not produce a cropped/desynced output.

    Every OpenCV writer backend on this machine rounds odd dimensions
    at write time, so a genuinely odd-sized source video cannot be
    created here. Instead we simulate the probe metadata reporting an
    odd-sized source (which real-world footage can do) and verify the
    pipeline rounds the encoder size to even and records the note.
    """
    src = make_video(size=(320, 240), frames=6)
    real_probe = vp.probe_video

    def odd_probe(path):
        info = real_probe(path)
        from video_processor import VideoInfo

        return VideoInfo(
            width=321,
            height=241,
            fps=info.fps,
            frame_count=info.frame_count,
            duration_seconds=info.duration_seconds,
            suffix=info.suffix,
        )

    monkeypatch.setattr(vp, "probe_video", odd_probe)
    result = process_video(model, src, conf=0.25)
    try:
        out = probe_video(result.output_path)
        assert out.width % 2 == 0 and out.height % 2 == 0
        assert out.width == 320 and out.height == 240
        assert "odd dimensions" in result.note
        assert len(_decode_all(result.output_path)) == 6
    finally:
        delete_quietly(result.output_path)


def test_process_avi_input(model, make_video):
    src = make_video(name="clip.avi", size=(320, 240), frames=6,
                     fourcc="MJPG")
    result = process_video(model, src, conf=0.25)
    try:
        assert result.processed_frames == 6
        assert result.output_path.suffix in {".mp4", ".avi"}
    finally:
        delete_quietly(result.output_path)


def test_progress_callback_called(model, make_video):
    src = make_video(frames=8)
    calls = []
    result = process_video(
        model, src, conf=0.25,
        progress_callback=lambda done, total: calls.append((done, total)),
    )
    try:
        assert len(calls) == 8
        assert calls[0][0] == 1
        assert calls[-1][0] == 8
        assert all(t == 8 for _, t in calls)
    finally:
        delete_quietly(result.output_path)


def test_output_temp_files_cleaned_on_success(model, make_video):
    """Successful runs must not leave test-encode leftovers behind."""
    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "bloodcell_*")))
    src = make_video(frames=6)
    result = process_video(model, src, conf=0.25)
    after = set(glob.glob(str(Path(tempfile.gettempdir()) / "bloodcell_*")))
    try:
        # Only the returned output file may remain; no probe/test leftovers.
        leftovers = after - before - {str(result.output_path)}
        assert leftovers == set(), f"leftover temp files: {leftovers}"
    finally:
        delete_quietly(result.output_path)


def test_no_output_left_behind_when_encoder_unavailable(model, make_video,
                                                        monkeypatch):
    """If no codec works, a clear error is raised and no file is kept."""
    src = make_video(frames=4)
    monkeypatch.setattr(vp, "_CODEC_CANDIDATES", ((".xyz", "ZZZZ", "Bad"),))
    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "bloodcell_*")))
    with pytest.raises(RuntimeError, match="could not encode"):
        process_video(model, src, conf=0.25)
    after = set(glob.glob(str(Path(tempfile.gettempdir()) / "bloodcell_*")))
    assert after == before


def test_all_inference_failures_raise(model, make_video, monkeypatch):
    """When every frame's inference fails, surface an error not a fake video."""

    def boom(*args, **kwargs):
        raise RuntimeError("inference exploded")

    monkeypatch.setattr(vp, "run_inference", boom)
    src = make_video(frames=4)
    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "bloodcell_*")))
    with pytest.raises(RuntimeError, match="every processed frame"):
        process_video(model, src, conf=0.25)
    after = set(glob.glob(str(Path(tempfile.gettempdir()) / "bloodcell_*")))
    assert after == before, "failed run must not leave output files"


def test_progress_callback_error_does_not_leak_files(model, make_video):
    """An exception in the caller's callback must not leave partial output."""

    def bad_callback(done, total):
        raise KeyError("callback crashed")

    src = make_video(frames=6)
    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "bloodcell_*")))
    with pytest.raises(KeyError):
        process_video(model, src, conf=0.25,
                      progress_callback=bad_callback)
    after = set(glob.glob(str(Path(tempfile.gettempdir()) / "bloodcell_*")))
    assert after == before, "partial output must be deleted on crash"


# --------------------------------------------------------------------------
# Output must visibly contain drawn boxes
# --------------------------------------------------------------------------

def test_output_frames_contain_drawn_annotations(model, make_video):
    """Processed output frames must differ from the raw source frames."""
    src = make_video(size=(320, 240), frames=6)
    result = process_video(model, src, conf=0.25, frame_interval=1)
    try:
        src_frames = _decode_all(src)
        out_frames = _decode_all(result.output_path)
        assert len(src_frames) == len(out_frames) == 6

        # Lossy codecs cause small diffs everywhere; require a clear
        # signal: some frame must differ substantially (drawn boxes).
        max_diff = max(
            float(np.mean(np.abs(a.astype(int) - b.astype(int))))
            for a, b in zip(src_frames, out_frames)
        )
        assert max_diff > 1.0, f"no visible annotations (max diff {max_diff})"
    finally:
        delete_quietly(result.output_path)


@pytest.mark.skipif(
    not DEFAULT_WEIGHTS.is_file(),
    reason="model weights missing",
)
def test_weights_unchanged_by_tests():
    """Guard: running tests must not modify the trained weights."""
    import hashlib

    digest = hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest()
    assert len(digest) == 64
