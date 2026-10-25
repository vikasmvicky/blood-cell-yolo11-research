#!/usr/bin/env python3
"""Frame-by-frame video detection helpers for the Streamlit app.

Design notes:
    * Frames are read one at a time with OpenCV, so the whole video is
      never loaded into memory.
    * Inference reuses ``detector.run_inference`` and drawing reuses
      ``detector.draw_detections``, so video frames use exactly the same
      YOLO11n model, class ids and box style as the image mode:
          0 = WBC, 1 = RBC, 2 = Platelets
    * Inference runs on CPU (``device="cpu"`` inside detector.py).
    * The output container/codec is verified with a tiny test encode
      before any real frame is written, so a failed MP4 encode falls
      back to a compatible alternative instead of producing a corrupt
      file.

Classes:
    0 = WBC
    1 = RBC
    2 = Platelets
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from detector import CLASS_ORDER, draw_detections, run_inference

# Extensions accepted by the video uploader.
VIDEO_SUFFIXES = ("mp4", "avi", "mov")

# (container suffix, fourcc code, human-readable label)
# tried in this order: MP4 first, AVI/MJPG as the compatible fallback.
_CODEC_CANDIDATES = (
    (".mp4", "mp4v", "MP4 (mp4v)"),
    (".mp4", "avc1", "MP4 (avc1)"),
    (".avi", "MJPG", "AVI (MJPG)"),
)

# Used only when the container does not report a frame rate.
_FALLBACK_FPS = 25.0


@dataclass(frozen=True)
class VideoInfo:
    """Metadata read from an uploaded video."""

    width: int
    height: int
    fps: float
    frame_count: int  # 0 when the container does not report it
    duration_seconds: float  # 0.0 when unknown
    suffix: str
    fps_assumed: bool = False  # container did not report a frame rate

    @property
    def resolution(self) -> str:
        return f"{self.width} x {self.height}"


@dataclass
class VideoResult:
    """Outcome of one video-processing run."""

    output_path: Path
    output_label: str  # codec/container actually used
    info: VideoInfo
    processed_frames: int  # frames that ran YOLO inference
    written_frames: int  # frames written to the output video
    counts: dict[str, int]  # total detections over processed frames
    total_detections: int
    failed_frames: int  # frames where inference raised an error
    frame_interval: int
    note: str = ""  # extra explanation (fallback codec, unknown FPS, ...)

    @property
    def average_per_frame(self) -> float:
        if self.processed_frames == 0:
            return 0.0
        return self.total_detections / self.processed_frames


def save_upload_to_temp(data: bytes, filename: str) -> Path:
    """Write uploaded video bytes to a named temp file (cv2 needs a path)."""
    suffix = Path(filename).suffix.lower()
    if suffix.lstrip(".") not in VIDEO_SUFFIXES:
        raise ValueError(
            "Unsupported video format. Please upload an MP4, AVI or MOV file."
        )

    fd, name = tempfile.mkstemp(
        prefix="bloodcell_upload_",
        suffix=suffix,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise

    return Path(name)


def delete_quietly(path) -> None:
    """Best-effort temp file removal; never raises."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def probe_video(path) -> VideoInfo:
    """Validate a video file and return its metadata.

    Raises ValueError with a user-facing message when the file is
    unsupported, corrupted or contains no decodable frames.
    """
    path = Path(path)

    if path.suffix.lower().lstrip(".") not in VIDEO_SUFFIXES:
        raise ValueError(
            "Unsupported video format. Please upload an MP4, AVI or MOV file."
        )

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(
            "The uploaded video file is empty or missing. "
            "Please upload a valid MP4, AVI or MOV file."
        )

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(
                "The uploaded file could not be opened as a video. "
                "It may be corrupted or encoded in an unsupported format."
            )

        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(
                "The video could be opened but no decodable frames were "
                "found. The file may be corrupted or truncated."
            )

        height, width = frame.shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise ValueError("The video reports an invalid resolution.")

    fps_assumed = False
    if not np.isfinite(fps) or fps <= 0:
        fps = _FALLBACK_FPS
        fps_assumed = True

    duration = frame_count / fps if frame_count > 0 else 0.0

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=max(frame_count, 0),
        duration_seconds=duration,
        suffix=path.suffix.lower(),
        fps_assumed=fps_assumed,
    )


def _new_temp_path(suffix: str, prefix: str = "bloodcell_video_") -> Path:
    """Reserve a unique temp path and hand it back empty."""
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    Path(name).unlink(missing_ok=True)
    return Path(name)


def _try_codec(
    suffix: str,
    fourcc: str,
    fps: float,
    size: tuple[int, int],
) -> Optional[tuple[Path, str]]:
    """Test-encode two frames and verify the result reopens cleanly.

    Returns (path, label) on success, None when this codec cannot produce
    a readable file on this machine.
    """
    candidate = _new_temp_path(suffix)
    writer = cv2.VideoWriter(
        str(candidate),
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        size,
    )
    try:
        if not writer.isOpened():
            # The backend may still have touched the file on disk.
            delete_quietly(candidate)
            return None
        blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        writer.write(blank)
        writer.write(blank)
    finally:
        writer.release()

    # Verify the test file can actually be decoded again.
    readable = False
    if candidate.is_file() and candidate.stat().st_size > 0:
        check = cv2.VideoCapture(str(candidate))
        try:
            ok, _ = check.read()
            readable = bool(ok)
        finally:
            check.release()

    if not readable:
        delete_quietly(candidate)
        return None

    return candidate, fourcc


def _open_writer(
    fps: float,
    size: tuple[int, int],
) -> tuple[Path, "cv2.VideoWriter", str]:
    """Open the first working VideoWriter, MP4 preferred over AVI."""
    attempts: list[str] = []
    for suffix, fourcc, label in _CODEC_CANDIDATES:
        attempts.append(label)
        probe = _try_codec(suffix, fourcc, fps, size)
        if probe is None:
            continue
        test_path, _ = probe
        # Re-open the same container for the real output.
        output_path = _new_temp_path(suffix)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            size,
        )
        if writer.isOpened():
            delete_quietly(test_path)
            return output_path, writer, label
        writer.release()
        delete_quietly(test_path)
        delete_quietly(output_path)

    raise RuntimeError(
        "OpenCV could not encode an output video on this machine "
        f"(tried: {', '.join(attempts)}). Try a smaller/shorter video or "
        "reinstall OpenCV with FFmpeg support."
    )


def process_video(
    model,
    input_path,
    conf: float = 0.25,
    frame_interval: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VideoResult:
    """Detect blood cells frame by frame and write an annotated video.

    Args:
        model: Loaded YOLO model (from detector.load_model).
        input_path: Path to the uploaded video.
        conf: Confidence threshold, same meaning as in image mode.
        frame_interval: Run inference on every Nth frame (1 = every frame).
        progress_callback: Called as ``callback(frames_read, total_frames)``
            after every read frame. ``total_frames`` is 0 when unknown.

    Returns:
        VideoResult with output path, metadata and detection totals.

    Raises:
        ValueError: invalid input video or arguments.
        RuntimeError: no usable output encoder.
    """
    if not 0.0 <= conf <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    if frame_interval < 1:
        raise ValueError("Frame interval must be at least 1.")

    input_path = Path(input_path)
    info = probe_video(input_path)

    # MPEG-style encoders silently crop odd dimensions (641x479 ->
    # 640x478), which would make frames and writer disagree. Round the
    # output size down to even and resize frames to match.
    size = (info.width - info.width % 2, info.height - info.height % 2)
    output_path, writer, output_label = _open_writer(info.fps, size)

    counts = {name: 0 for name in CLASS_ORDER}
    frame_index = 0
    processed = 0
    written = 0
    total_detections = 0
    failed_frames = 0
    first_inference_error: Optional[str] = None

    capture = cv2.VideoCapture(str(input_path))
    try:
        try:
            if not capture.isOpened():
                raise ValueError(
                    "The video could not be opened for processing. "
                    "It may have been removed or corrupted."
                )

            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                if (frame.shape[1], frame.shape[0]) != size:
                    frame = cv2.resize(frame, size)

                # Only every Nth frame runs inference; skipped frames are
                # copied through unchanged (no boxes, not counted).
                if frame_index % frame_interval == 0:
                    processed += 1
                    try:
                        rgb = Image.fromarray(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        )
                        detections = run_inference(model, rgb, conf=conf)
                        annotated = draw_detections(rgb, detections)
                        frame = cv2.cvtColor(
                            np.asarray(annotated),
                            cv2.COLOR_RGB2BGR,
                        )
                        total_detections += len(detections)
                        for detection in detections:
                            if detection.name in counts:
                                counts[detection.name] += 1
                    except Exception as exc:
                        # Keep processing; the raw frame is written instead.
                        failed_frames += 1
                        if first_inference_error is None:
                            first_inference_error = (
                                f"{type(exc).__name__}: {exc}"
                            )

                writer.write(frame)
                written += 1
                frame_index += 1

                if progress_callback is not None:
                    progress_callback(frame_index, info.frame_count)
        finally:
            capture.release()
            writer.release()
    except Exception:
        # Never leave a partial output file behind on unexpected errors
        # (e.g. a failing progress callback).
        delete_quietly(output_path)
        raise

    if written == 0:
        delete_quietly(output_path)
        raise ValueError(
            "No frames could be read from the video. "
            "The file may be corrupted."
        )

    if failed_frames and processed and failed_frames == processed:
        delete_quietly(output_path)
        raise RuntimeError(
            "Detection failed on every processed frame. "
            "See the application logs for details."
        )

    # Integrity check: the output must exist and be re-decodable.
    valid = False
    if output_path.is_file() and output_path.stat().st_size > 0:
        check = cv2.VideoCapture(str(output_path))
        try:
            ok, _ = check.read()
            valid = bool(ok)
        finally:
            check.release()
    if not valid:
        delete_quietly(output_path)
        raise RuntimeError(
            "The annotated video could not be encoded correctly on this "
            "machine. No output file was produced."
        )

    notes = []
    if size != (info.width, info.height):
        notes.append(
            f"Input resolution {info.resolution} has odd dimensions; "
            f"output was rounded to {size[0]} x {size[1]} for encoder "
            "compatibility."
        )
    if info.fps_assumed:
        notes.append(
            "The video does not report a frame rate; assuming "
            f"{_FALLBACK_FPS:.0f} FPS for output timing."
        )
    if info.frame_count <= 0:
        notes.append(
            "The container does not report a frame count; "
            "processed frames were counted while reading."
        )
    if failed_frames:
        notes.append(
            f"{failed_frames} processed frame(s) failed inference and were "
            "written without bounding boxes."
        )
        if first_inference_error:
            notes.append(f"First inference error: {first_inference_error}")
    if output_label.startswith("AVI"):
        notes.append(
            "MP4 encoding was unavailable, so the output was written as "
            "AVI (MJPG) instead."
        )
    if frame_interval > 1:
        notes.append(
            f"Only 1 in {frame_interval} frames was processed; skipped "
            "frames appear without bounding boxes and are not counted."
        )

    return VideoResult(
        output_path=output_path,
        output_label=output_label,
        info=info,
        processed_frames=processed,
        written_frames=written,
        counts=counts,
        total_detections=total_detections,
        failed_frames=failed_frames,
        frame_interval=frame_interval,
        note=" ".join(notes),
    )
