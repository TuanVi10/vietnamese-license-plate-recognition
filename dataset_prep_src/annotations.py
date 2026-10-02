"""
annotations.py
==============
Canonical in-memory representation of detection annotations plus readers and
writers for the YOLO ``*.txt`` label format.

Internal convention
-------------------
* Every box is stored as absolute pixels ``(x1, y1, x2, y2)`` in the coordinate
  frame of the image that will finally be shipped (i.e. after any resize).
* The YOLO text format is ``<class_id> <cx> <cy> <w> <h>`` with all geometry
  normalised to ``[0, 1]``.

The module also hosts the small geometry helpers used to *re-map* imported
labels onto a re-sized image and to *merge* boxes coming from different sources.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dataset_config import FilterConfig
from utils import (
    Box,
    GeoTransform,
    box_area,
    box_height,
    box_width,
    clip_xyxy,
    ensure_dir,
    iou_xyxy,
    nms_xyxy,
    read_json,
    write_json,
)

DEFAULT_DECIMALS = 6


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class BBox:
    """One detection box with its class, confidence and provenance."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    score: float = 1.0
    source: str = "manual"

    # -- basic geometry -------------------------------------------------- #
    @property
    def width(self) -> float:
        return box_width(self.to_xyxy())

    @property
    def height(self) -> float:
        return box_height(self.to_xyxy())

    @property
    def area(self) -> float:
        return box_area(self.to_xyxy())

    @property
    def aspect_ratio(self) -> float:
        height = self.height
        return (self.width / height) if height > 0 else 0.0

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_xyxy(self) -> Box:
        return (float(self.x1), float(self.y1), float(self.x2), float(self.y2))

    def is_valid(self, min_size: float = 1.0) -> bool:
        return self.width >= min_size and self.height >= min_size

    # -- transforms ------------------------------------------------------ #
    def clip(self, width: float, height: float) -> "BBox":
        """Return a copy clamped to the image rectangle."""
        x1, y1, x2, y2 = clip_xyxy(self.to_xyxy(), width, height)
        return BBox(x1, y1, x2, y2, self.class_id, self.score, self.source)

    def scaled(self, factor_x: float, factor_y: Optional[float] = None) -> "BBox":
        """Return a copy scaled around the origin (used to fix size mismatches)."""
        if factor_y is None:
            factor_y = factor_x
        return BBox(
            self.x1 * factor_x,
            self.y1 * factor_y,
            self.x2 * factor_x,
            self.y2 * factor_y,
            self.class_id,
            self.score,
            self.source,
        )

    def transformed(self, transform: GeoTransform) -> "BBox":
        """Return a copy mapped through a :class:`utils.GeoTransform`."""
        x1, y1, x2, y2 = transform.forward_box(self.to_xyxy())
        return BBox(x1, y1, x2, y2, self.class_id, self.score, self.source)

    def clone(self) -> "BBox":
        return BBox(self.x1, self.y1, self.x2, self.y2, self.class_id, self.score, self.source)

    # -- serialisation --------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(float(self.x1), 3),
            "y1": round(float(self.y1), 3),
            "x2": round(float(self.x2), 3),
            "y2": round(float(self.y2), 3),
            "class_id": int(self.class_id),
            "score": round(float(self.score), 4),
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BBox":
        return cls(
            x1=float(data.get("x1", 0.0)),
            y1=float(data.get("y1", 0.0)),
            x2=float(data.get("x2", 0.0)),
            y2=float(data.get("y2", 0.0)),
            class_id=int(data.get("class_id", 0)),
            score=float(data.get("score", 1.0)),
            source=str(data.get("source", "manual")),
        )


@dataclass
class ImageAnnotation:
    """Every box belonging to one shipped image."""

    image: str
    width: int
    height: int
    boxes: List[BBox] = field(default_factory=list)

    def class_counts(self) -> Counter:
        return Counter(int(box.class_id) for box in self.boxes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "width": int(self.width),
            "height": int(self.height),
            "boxes": [box.to_dict() for box in self.boxes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageAnnotation":
        return cls(
            image=str(data.get("image", "")),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            boxes=[BBox.from_dict(item) for item in data.get("boxes", [])],
        )


# --------------------------------------------------------------------------- #
# YOLO text format
# --------------------------------------------------------------------------- #
def bbox_to_yolo_values(
    box: BBox, width: float, height: float, min_size: float = 1e-6
) -> Optional[tuple]:
    """Convert a box to normalised ``(cx, cy, w, h)``; ``None`` if degenerate."""
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = clip_xyxy(box.to_xyxy(), width, height)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= min_size or box_h <= min_size:
        return None
    center_x = min(max((x1 + x2) / 2.0 / width, 0.0), 1.0)
    center_y = min(max((y1 + y2) / 2.0 / height, 0.0), 1.0)
    norm_w = min(max(box_w / width, 0.0), 1.0)
    norm_h = min(max(box_h / height, 0.0), 1.0)
    return (center_x, center_y, norm_w, norm_h)


def bbox_to_yolo_line(box: BBox, width: float, height: float, decimals: int = DEFAULT_DECIMALS) -> Optional[str]:
    """Render one YOLO label line, or ``None`` when the box cannot be encoded."""
    values = bbox_to_yolo_values(box, width, height)
    if values is None:
        return None
    center_x, center_y, norm_w, norm_h = values
    template = "{{0}} {{1:.{0}f}} {{2:.{0}f}} {{3:.{0}f}} {{4:.{0}f}}".format(decimals)
    return template.format(int(box.class_id), center_x, center_y, norm_w, norm_h)


def parse_yolo_line(
    line: str,
    width: float,
    height: float,
    default_class: int = 0,
) -> Optional[BBox]:
    """Parse one YOLO label line back into an absolute-pixel :class:`BBox`."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        center_x = float(parts[1])
        center_y = float(parts[2])
        norm_w = float(parts[3])
        norm_h = float(parts[4])
    except ValueError:
        return None
    score = 1.0
    if len(parts) >= 6:
        try:
            score = float(parts[5])
        except ValueError:
            score = 1.0
    if class_id < 0:
        class_id = int(default_class)
    if width <= 0 or height <= 0:
        return None
    box_w = norm_w * width
    box_h = norm_h * height
    center_x *= width
    center_y *= height
    return BBox(
        x1=center_x - box_w / 2.0,
        y1=center_y - box_h / 2.0,
        x2=center_x + box_w / 2.0,
        y2=center_y + box_h / 2.0,
        class_id=class_id,
        score=score,
        source="yolo",
    )


def load_yolo_labels(
    path: Any,
    width: float,
    height: float,
    default_class: int = 0,
) -> List[BBox]:
    """Read a YOLO ``*.txt`` file; returns an empty list when it is missing."""
    label_path = Path(path)
    if not label_path.exists():
        return []
    boxes: List[BBox] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            box = parse_yolo_line(line, width, height, default_class)
            if box is not None:
                boxes.append(box)
    return boxes


def yolo_labels_to_text(
    boxes: Sequence[BBox],
    width: float,
    height: float,
    decimals: int = DEFAULT_DECIMALS,
    num_classes: Optional[int] = None,
) -> str:
    """Render every box of one image as YOLO label text (may be an empty string).

    When ``num_classes`` is given, boxes whose ``class_id`` is outside
    ``[0, num_classes)`` are skipped so the produced file stays consistent with
    the ``nc`` declared in ``data.yaml``.
    """
    lines = []
    for box in boxes:
        if num_classes is not None and not (0 <= int(box.class_id) < int(num_classes)):
            continue
        line = bbox_to_yolo_line(box, width, height, decimals)
        if line is not None:
            lines.append(line)
    return ("\n".join(lines) + "\n") if lines else ""


def save_yolo_labels(
    path: Any,
    boxes: Sequence[BBox],
    width: float,
    height: float,
    decimals: int = DEFAULT_DECIMALS,
    num_classes: Optional[int] = None,
) -> Path:
    """Write a YOLO ``*.txt`` label file (an empty file for negative samples).

    ``num_classes`` acts as a last-line defence: any box whose ``class_id`` is
    outside ``[0, num_classes)`` is silently dropped instead of being written,
    so a label file can never reference an undeclared class.
    """
    label_path = Path(path)
    ensure_dir(label_path.parent)
    text = yolo_labels_to_text(boxes, width, height, decimals, num_classes=num_classes)
    with label_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    return label_path


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def transform_boxes(boxes: Iterable[BBox], transform: GeoTransform) -> List[BBox]:
    """Map a list of boxes through a geometric transform (letterbox / stretch)."""
    if transform.is_identity():
        return [box.clone() for box in boxes]
    return [box.transformed(transform) for box in boxes]


def scale_boxes(boxes: Iterable[BBox], factor_x: float, factor_y: float) -> List[BBox]:
    """Rescale boxes when the declared annotation size differs from the real one."""
    if abs(factor_x - 1.0) < 1e-9 and abs(factor_y - 1.0) < 1e-9:
        return [box.clone() for box in boxes]
    return [box.scaled(factor_x, factor_y) for box in boxes]


def box_issues(
    box: BBox,
    width: float,
    height: float,
    config: FilterConfig,
    tolerance: float = 0.5,
    num_classes: Optional[int] = None,
) -> List[str]:
    """Return every rule violation of ``box`` (empty list means the box is fine).

    When ``num_classes`` is provided a box whose ``class_id`` does not index
    into ``[0, num_classes)`` is reported as ``unknown_class`` so callers can
    drop it, keeping ``data.yaml`` and the label files consistent.
    """
    issues: List[str] = []
    if num_classes is not None and not (0 <= int(box.class_id) < int(num_classes)):
        issues.append("unknown_class")
    if box.width <= 0 or box.height <= 0:
        issues.append("degenerate")
        return issues
    if (
        box.x1 < -tolerance
        or box.y1 < -tolerance
        or box.x2 > float(width) + tolerance
        or box.y2 > float(height) + tolerance
    ):
        issues.append("out_of_bounds")
    if box.width < config.min_width_px or box.height < config.min_height_px:
        issues.append("too_small")
    if width > 0 and height > 0:
        ratio = box.area / float(width * height)
        if ratio < config.min_area_ratio:
            issues.append("area_too_small")
        elif ratio > config.max_area_ratio:
            issues.append("area_too_large")
    aspect = box.aspect_ratio
    if aspect < config.min_aspect_ratio:
        issues.append("aspect_too_tall")
    elif aspect > config.max_aspect_ratio:
        issues.append("aspect_too_wide")
    return issues


def filter_boxes(
    boxes: Iterable[BBox],
    config: FilterConfig,
    width: Optional[float] = None,
    height: Optional[float] = None,
    num_classes: Optional[int] = None,
) -> List[BBox]:
    """Clip, validate and keep only the boxes that survive :class:`FilterConfig`.

    When ``num_classes`` is provided, boxes whose ``class_id`` does not index
    into ``[0, num_classes)`` are dropped as well, so the emitted label files can
    never reference a class that ``data.yaml`` (``nc: num_classes``) does not
    declare - a state Ultralytics would otherwise reject at training time.
    """
    kept: List[BBox] = []
    for box in boxes:
        candidate = box.clone()
        if width and height:
            candidate = candidate.clip(width, height)
        issues = [
            issue
            for issue in box_issues(
                candidate,
                width or 0.0,
                height or 0.0,
                config,
                num_classes=num_classes,
            )
            if issue != "out_of_bounds"
        ]
        if issues:
            continue
        kept.append(candidate)
    return kept


def merge_boxes(boxes: Sequence[BBox], iou_threshold: float = 0.5) -> List[BBox]:
    """Class-aware NMS: keep the highest scoring box of every overlapping cluster."""
    if not boxes:
        return []
    grouped: Dict[int, List[BBox]] = {}
    for box in boxes:
        grouped.setdefault(int(box.class_id), []).append(box)
    merged: List[BBox] = []
    for class_id in sorted(grouped):
        group = sorted(grouped[class_id], key=lambda item: float(item.score), reverse=True)
        coords = [item.to_xyxy() for item in group]
        scores = [float(item.score) for item in group]
        for index in nms_xyxy(coords, scores, iou_threshold):
            merged.append(group[index])
    return merged


def dedupe_boxes(boxes: Sequence[BBox], iou_threshold: float = 0.6) -> List[BBox]:
    """Remove near-identical duplicates while keeping the original ordering."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda index: float(boxes[index].score), reverse=True)
    keep_indices: List[int] = []
    for index in order:
        duplicate = False
        for kept_index in keep_indices:
            if boxes[index].class_id == boxes[kept_index].class_id and iou_xyxy(
                boxes[index].to_xyxy(), boxes[kept_index].to_xyxy()
            ) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            keep_indices.append(index)
    keep_indices.sort()
    return [boxes[index] for index in keep_indices]


def count_classes(images: Iterable[ImageAnnotation]) -> Dict[int, int]:
    """Total number of boxes per class across a collection of annotations."""
    counter: Counter = Counter()
    for annotation in images:
        counter.update(annotation.class_counts())
    return {int(key): int(value) for key, value in sorted(counter.items())}


# --------------------------------------------------------------------------- #
# Disk store for the whole annotation map
# --------------------------------------------------------------------------- #
def annotations_to_dict(
    annotations: Dict[str, ImageAnnotation],
    classes: Sequence[str],
    label_mode: str,
) -> Dict[str, Any]:
    """Serialise the annotation map into a JSON-friendly structure."""
    return {
        "classes": list(classes),
        "label_mode": str(label_mode),
        "count": len(annotations),
        "images": {name: annotation.to_dict() for name, annotation in annotations.items()},
    }


def annotations_from_dict(data: Optional[Dict[str, Any]]) -> Dict[str, ImageAnnotation]:
    """Rebuild the annotation map from :func:`annotations_to_dict` output."""
    if not data:
        return {}
    images = data.get("images", {}) if isinstance(data, dict) else {}
    return {name: ImageAnnotation.from_dict(payload) for name, payload in images.items()}


def save_annotations(
    path: Any,
    annotations: Dict[str, ImageAnnotation],
    classes: Sequence[str],
    label_mode: str,
) -> Path:
    """Persist the annotation map as UTF-8 JSON."""
    return write_json(path, annotations_to_dict(annotations, classes, label_mode))


def load_annotations_file(path: Any) -> Dict[str, ImageAnnotation]:
    """Load an annotation map previously written by :func:`save_annotations`.

    Raises :class:`FileNotFoundError` when the file is missing (mirroring
    :func:`splitter.load_splits`) so the pipeline never silently continues with
    an empty label set after a partial/resumed run.
    """
    payload = read_json(path, default=None)
    if not payload:
        raise FileNotFoundError(
            "annotations not found at {0}; run the 'labels' (and 'validate') "
            "stage first".format(path)
        )
    return annotations_from_dict(payload)
