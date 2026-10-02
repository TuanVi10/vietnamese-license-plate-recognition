"""
visualize.py
============
Module 7 (bản rút gọn) — vẽ overlay kết quả lên ảnh tĩnh.

Với ảnh tĩnh ta chỉ cần: khung biển số + text đọc được + ảnh crop biển số để
đối chiếu khi viết báo cáo (giải thích trường hợp đọc sai).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

GREEN = (0, 200, 0)
RED = (0, 0, 220)
ORANGE = (0, 165, 255)


def draw_bbox(
    image: np.ndarray,
    bbox: Sequence[float],
    color: Tuple[int, int, int] = GREEN,
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """Vẽ 1 khung chữ nhật (và nhãn) lên bản copy của ảnh."""
    canvas = image.copy()
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, int(thickness))

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, text_thickness = 0.6, 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, text_thickness)
        top = max(y1 - 8, text_h + 4)
        cv2.rectangle(
            canvas,
            (x1, top - text_h - 4),
            (x1 + text_w + 8, top + baseline),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 4, top),
            font,
            scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return canvas


def draw_reading(
    image: np.ndarray,
    reading: Any,
    show_crop: bool = True,
    crop_size: Tuple[int, int] = (320, 48),
) -> np.ndarray:
    """Vẽ kết quả :class:`~src.plate_reader.PlateReading` lên ảnh gốc.

    Args:
        image: Ảnh gốc.
        reading: Đối tượng có ``bbox``, ``formatted``, ``text``, ``crop``, ``valid``.
        show_crop: Có dán ảnh crop biển số vào góc trên-trái hay không.
        crop_size: Kích thước ảnh crop hiển thị ``(width, height)``.

    Returns:
        Ảnh đã vẽ overlay.
    """
    canvas = image.copy()
    bbox = getattr(reading, "bbox", None)
    valid = bool(getattr(reading, "valid", False))
    color = GREEN if valid else ORANGE

    label = getattr(reading, "formatted", "") or getattr(reading, "text", "") or "?"
    confidence = float(getattr(reading, "confidence", 0.0) or 0.0)
    label = f"{label} ({confidence:.2f})"

    if bbox:
        canvas = draw_bbox(canvas, bbox, color=color, label=label)

    crop = getattr(reading, "crop", None)
    if show_crop and crop is not None and getattr(crop, "size", 0):
        thumb_w, thumb_h = int(crop_size[0]), int(crop_size[1])
        if canvas.shape[0] >= thumb_h and canvas.shape[1] >= thumb_w:
            thumb = cv2.resize(crop, (thumb_w, thumb_h))
            canvas[0:thumb_h, 0:thumb_w] = thumb
            cv2.rectangle(canvas, (0, 0), (thumb_w - 1, thumb_h - 1), color, 1)

    return canvas


def side_by_side(image: np.ndarray, other: np.ndarray, gap: int = 10) -> np.ndarray:
    """Ghép 2 ảnh cạnh nhau theo chiều ngang (tiện so sánh gốc vs overlay)."""
    height = max(image.shape[0], other.shape[0])

    def _pad(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        missing = height - img.shape[0]
        if missing <= 0:
            return img
        return cv2.copyMakeBorder(img, 0, missing, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    left, right = _pad(image), _pad(other)
    divider = np.full((height, gap, 3), 128, dtype=np.uint8)
    return np.hstack([left, divider, right])
