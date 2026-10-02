"""
visualize.py
============
Quick quality-control helpers: draw boxes on images and build contact sheets so
a human can confirm (in a few seconds) that the automatic pipeline did not go
mad before spending GPU hours on training.

All text drawn by OpenCV has to be ASCII, therefore captions are slugified.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from annotations import BBox, ImageAnnotation
from dataset_config import DatasetConfig
from preprocess import ImageRecord
from utils import Reporter, ascii_slug, ensure_dir, imread_unicode, imwrite_unicode

# Deterministic colours per class (BGR).
_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (0, 220, 0),
    (255, 170, 0),
    (0, 140, 255),
    (255, 0, 200),
    (0, 220, 220),
    (200, 200, 0),
)


def class_color(class_id: int) -> Tuple[int, int, int]:
    """Stable BGR colour for a class id."""
    return _PALETTE[int(class_id) % len(_PALETTE)]


def draw_boxes(
    image: np.ndarray,
    boxes: Sequence[BBox],
    class_names: Optional[Sequence[str]] = None,
    thickness: int = 2,
    show_score: bool = True,
) -> np.ndarray:
    """Return a copy of ``image`` with every box drawn."""
    canvas = image.copy()
    for box in boxes:
        color = class_color(int(box.class_id))
        point_a = (int(round(box.x1)), int(round(box.y1)))
        point_b = (int(round(box.x2)), int(round(box.y2)))
        cv2.rectangle(canvas, point_a, point_b, color, thickness)

        label = ""
        if class_names and 0 <= int(box.class_id) < len(class_names):
            label = str(class_names[int(box.class_id)])
        else:
            label = str(int(box.class_id))
        if show_score:
            label = "{0} {1:.2f}".format(label, float(box.score))
        if label:
            text_y = max(12, point_a[1] - 6)
            cv2.putText(
                canvas,
                label,
                (point_a[0], text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def fit_thumbnail(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Resize ``image`` to fit inside the given box, keeping the aspect ratio."""
    if image is None or image.size == 0:
        return np.zeros((max_height, max_width, 3), dtype=np.uint8)
    height, width = image.shape[:2]
    ratio = min(max_width / float(width), max_height / float(height))
    new_width = max(1, int(round(width * ratio)))
    new_height = max(1, int(round(height * ratio)))
    interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def make_contact_sheet(
    items: Sequence[Tuple[np.ndarray, str]],
    columns: int = 4,
    cell: int = 320,
    gap: int = 6,
    caption_height: int = 20,
    background: Tuple[int, int, int] = (28, 28, 28),
) -> Optional[np.ndarray]:
    """Tile ``(image, caption)`` pairs into a single sheet. ``None`` when empty."""
    if not items:
        return None
    columns = max(1, int(columns))
    rows = int(math.ceil(len(items) / float(columns)))
    cell_height = cell + caption_height
    sheet_width = columns * cell + (columns + 1) * gap
    sheet_height = rows * cell_height + (rows + 1) * gap
    sheet = np.full((sheet_height, sheet_width, 3), background, dtype=np.uint8)

    for index, (image, caption) in enumerate(items):
        row, column = divmod(index, columns)
        origin_y = gap + row * (cell_height + gap)
        origin_x = gap + column * (cell + gap)
        thumbnail = fit_thumbnail(image, cell, cell)
        thumb_h, thumb_w = thumbnail.shape[:2]
        offset_y = origin_y + caption_height + (cell - thumb_h) // 2
        offset_x = origin_x + (cell - thumb_w) // 2
        sheet[offset_y : offset_y + thumb_h, offset_x : offset_x + thumb_w] = thumbnail
        cv2.putText(
            sheet,
            ascii_slug(caption, 34)[:34],
            (origin_x + 2, origin_y + caption_height - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return sheet


def save_contact_sheet(
    items: Sequence[Tuple[np.ndarray, str]],
    out_path: Any,
    columns: int = 4,
    cell: int = 320,
) -> Optional[Path]:
    """Build a contact sheet and write it to ``out_path``."""
    sheet = make_contact_sheet(items, columns=columns, cell=cell)
    if sheet is None:
        return None
    path = Path(out_path)
    ensure_dir(path.parent)
    imwrite_unicode(path, sheet, quality=90)
    return path


def save_qc_samples(
    config: DatasetConfig,
    records: Sequence[ImageRecord],
    annotations: Dict[str, ImageAnnotation],
    reporter: Optional[Reporter] = None,
    limit: Optional[int] = None,
    columns: Optional[int] = None,
) -> Optional[Path]:
    """Render `limit` random-ish images (with boxes) into ``<output>/qc/preview.jpg``."""
    reporter = reporter or Reporter(quiet=True)
    limit = int(limit if limit is not None else config.preview_limit)
    columns = int(columns if columns is not None else config.preview_columns)
    if limit <= 0:
        return None

    images_dir = Path(config.work_dir) / "images"
    kept = [record for record in records if not record.dropped][: max(limit, 1)]
    if not kept:
        return None

    # Deterministic spread: take an even sample across the sorted record list.
    step = max(1, len(kept) // limit)
    selected = kept[::step][:limit]

    items: List[Tuple[np.ndarray, str]] = []
    for record in selected:
        image = imread_unicode(images_dir / record.name)
        if image is None:
            continue
        annotation = annotations.get(record.name)
        boxes = annotation.boxes if annotation else []
        drawn = draw_boxes(image, boxes, config.classes)
        items.append((drawn, "{0}/{1}:{2}".format(record.group, record.name, len(boxes))))

    out_path = Path(config.output_dir) / "qc" / "preview.jpg"
    saved = save_contact_sheet(items, out_path, columns=columns, cell=320)
    if saved is not None:
        reporter.info("preview: quality-control sheet written to {0}".format(saved))
    return saved


def save_split_contact_sheets(
    config: DatasetConfig,
    records: Sequence[ImageRecord],
    annotations: Dict[str, ImageAnnotation],
    splits: Dict[str, List[str]],
    limit_per_split: int = 12,
    columns: int = 4,
    reporter: Optional[Reporter] = None,
) -> Dict[str, str]:
    """One contact sheet per split – handy to eyeball the split balance."""
    reporter = reporter or Reporter(quiet=True)
    images_dir = Path(config.work_dir) / "images"
    produced: Dict[str, str] = {}
    records_by_name = {record.name: record for record in records}

    for split_name, names in splits.items():
        items: List[Tuple[np.ndarray, str]] = []
        for name in names[:limit_per_split]:
            record = records_by_name.get(name)
            if record is None:
                continue
            image = imread_unicode(images_dir / name)
            if image is None:
                continue
            annotation = annotations.get(name)
            boxes = annotation.boxes if annotation else []
            items.append((draw_boxes(image, boxes, config.classes), name))
        if not items:
            continue
        out_path = Path(config.output_dir) / "qc" / "preview_{0}.jpg".format(split_name)
        saved = save_contact_sheet(items, out_path, columns=columns, cell=300)
        if saved is not None:
            produced[split_name] = str(saved)
    if produced:
        reporter.info(
            "preview: split sheets -> {0}".format(", ".join(sorted(produced)))
        )
    return produced
