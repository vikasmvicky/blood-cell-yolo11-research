
#!/usr/bin/env python3
"""YOLO11n blood-cell detection logic shared by the Streamlit app.

Classes:
    0 = WBC
    1 = RBC
    2 = Platelets

Runs inference on CPU and draws coloured bounding boxes with class labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "yolo11n_txlpbc-2"
    / "weights"
    / "best.pt"
)

CLASS_NAMES = {
    0: "WBC",
    1: "RBC",
    2: "Platelets",
}

CLASS_ORDER = ("WBC", "RBC", "Platelets")

# Colours are RGB values.
CLASS_COLORS = {
    "WBC": (30, 90, 255),
    "RBC": (0, 200, 220),
    "Platelets": (220, 50, 180),
}

TEXT_COLOR = (0, 0, 0)


@dataclass
class Detection:
    """One detected object with a pixel-space bounding box."""

    cls_id: int
    name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def load_model(weights: Path = DEFAULT_WEIGHTS):
    """Load the trained YOLO11n model."""
    from ultralytics import YOLO

    weights = Path(weights)

    if not weights.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights}"
        )

    return YOLO(str(weights))


def open_image(fileobj) -> Image.Image:
    """Open an uploaded image and return an RGB PIL image."""
    from io import BytesIO

    try:
        img = Image.open(BytesIO(fileobj.read()))
        img.load()
    except Exception as exc:
        raise ValueError(
            "The uploaded file is not a valid image. "
            "Please upload a JPG, JPEG or PNG blood-smear image."
        ) from exc

    return img.convert("RGB")


def run_inference(
    model,
    image: Image.Image,
    conf: float = 0.25,
) -> list[Detection]:
    """Run YOLO inference on a PIL image and return sorted detections.

    Shared by the image mode and the video mode so both use the same
    model settings, classes and confidence handling.
    """
    if not 0.0 <= conf <= 1.0:
        raise ValueError(
            "Confidence threshold must be between 0 and 1."
        )

    results = model.predict(
        source=image,
        conf=conf,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    result = results[0]
    detections: list[Detection] = []

    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i].item())

        x1, y1, x2, y2 = (
            int(value)
            for value in result.boxes.xyxy[i].tolist()
        )

        detections.append(
            Detection(
                cls_id=cls_id,
                name=CLASS_NAMES.get(cls_id, str(cls_id)),
                confidence=float(result.boxes.conf[i].item()),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    # Highest-confidence detections first.
    detections.sort(
        key=lambda d: d.confidence,
        reverse=True,
    )

    return detections


def draw_detections(
    image: Image.Image,
    detections: list[Detection],
) -> Image.Image:
    """Draw coloured bounding boxes and labels on a copy of the image."""
    # Draw our own boxes and labels instead of result.plot().
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    # Use a standard available PIL font.
    font = ImageFont.load_default()

    for detection in detections:
        color = CLASS_COLORS.get(
            detection.name,
            (255, 255, 0),
        )

        box = [
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
        ]

        # Draw bounding box.
        draw.rectangle(
            box,
            outline=color,
            width=2,
        )

        # Class name and confidence.
        label = (
            f"{detection.name} "
            f"{detection.confidence:.2f}"
        )

        text_bbox = draw.textbbox(
            (0, 0),
            label,
            font=font,
        )

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 3

        label_width = text_width + padding * 2
        label_height = text_height + padding * 2

        # Keep the label within the image boundaries.
        label_x = max(
            0,
            min(
                detection.x1,
                annotated.width - label_width,
            ),
        )

        label_y = detection.y1 - label_height

        # If there's no room above the box, place it inside.
        if label_y < 0:
            label_y = min(
                detection.y1,
                annotated.height - label_height,
            )

        # Solid class-coloured background.
        draw.rectangle(
            [
                label_x,
                label_y,
                label_x + label_width,
                label_y + label_height,
            ],
            fill=color,
        )

        # Black text for contrast.
        draw.text(
            (
                label_x + padding,
                label_y + padding,
            ),
            label,
            fill=TEXT_COLOR,
            font=font,
        )

    return annotated


def detect(
    model,
    image: Image.Image,
    conf: float = 0.25,
) -> tuple[list[Detection], Image.Image]:
    """Run YOLO inference and return detections plus an annotated image."""
    detections = run_inference(model, image, conf=conf)
    annotated = draw_detections(image, detections)

    return detections, annotated


def count_by_class(
    detections: list[Detection],
) -> dict[str, int]:
    """Return counts in WBC, RBC and Platelets order."""
    counts = {
        name: 0
        for name in CLASS_ORDER
    }

    for detection in detections:
        counts[detection.name] = (
            counts.get(detection.name, 0) + 1
        )

    return counts


def detections_to_csv(
    detections: list[Detection],
) -> str:
    """Export detections to CSV without requiring pandas."""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(
        [
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
        ]
    )

    for detection in detections:
        writer.writerow(
            [
                detection.cls_id,
                detection.name,
                f"{detection.confidence:.4f}",
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                detection.width,
                detection.height,
            ]
        )

    return buffer.getvalue()
