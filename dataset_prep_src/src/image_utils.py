"""
image_utils.py
==============
Tiện ích ảnh dùng chung: đọc/ghi an toàn với đường dẫn Unicode, chuyển đổi màu,
crop theo bbox, xoay ảnh, cắt viền, đo độ mờ/độ sáng.

Các hàm ở đây được Module 0 (lọc frame) và Module 4 (tiền xử lý biển số) dùng lại.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Đọc ảnh an toàn với đường dẫn chứa ký tự Unicode.

    ``cv2.imread`` không hỗ trợ đường dẫn Unicode trên Windows, nên ta đọc bytes
    rồi decode bằng ``cv2.imdecode``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"File ảnh rỗng: {path}")
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Không decode được ảnh: {path}")
    return image


def imwrite_unicode(path: str, image: np.ndarray) -> None:
    """Ghi ảnh an toàn với đường dẫn Unicode (dùng ``cv2.imencode`` + tofile)."""
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    extension = os.path.splitext(path)[1] or ".png"
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"Không encode được ảnh để ghi: {path}")
    buffer.tofile(path)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh về grayscale (chấp nhận ảnh xám, BGR, BGRA)."""
    if image is None:
        raise ValueError("to_gray nhận ảnh None")
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Đảm bảo ảnh ở dạng 3 kênh BGR."""
    if image is None:
        raise ValueError("ensure_bgr nhận ảnh None")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ép ảnh về ``uint8`` trong khoảng [0, 255]."""
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def binarize_otsu(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Nhị phân hoá bằng ngưỡng Otsu.

    Args:
        image: Ảnh đầu vào (BGR hoặc grayscale).
        invert: True -> chữ trắng trên nền đen; False -> chữ đen trên nền trắng.
    """
    gray = to_gray(image)
    blur = cv2.medianBlur(gray, 3)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(blur, 0, 255, flag + cv2.THRESH_OTSU)
    return binary


def clip_bbox(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Kẹp bbox ``(x1, y1, x2, y2)`` vào trong biên ảnh, đảm bảo rộng/cao >= 1px."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def expand_bbox(
    bbox: Sequence[float],
    margin: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Nới rộng bbox thêm ``margin`` (tỷ lệ theo w/h) rồi kẹp vào biên ảnh."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    dx = (x2 - x1) * float(margin)
    dy = (y2 - y1) * float(margin)
    return clip_bbox((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)


def crop_bbox(image: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    """Crop vùng ảnh theo bbox ``(x1, y1, x2, y2)`` trong hệ toạ độ ảnh."""
    if image is None:
        raise ValueError("crop_bbox nhận ảnh None")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, width, height)
    return image[y1:y2, x1:x2].copy()


def rotate_image(image: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray:
    """Xoay ảnh quanh tâm một góc ``angle`` (độ, dương = ngược chiều kim đồng hồ).

    Mở rộng canvas để không cắt mất nội dung ở 4 góc, phần thêm vào lấp bằng
    ``border_value`` (mặc định 255 — màu nền biển trắng).
    """
    if image is None or image.size == 0:
        raise ValueError("rotate_image nhận ảnh rỗng")
    if abs(angle) < 1e-6:
        return image.copy()

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(round(height * sin + width * cos))
    new_h = int(round(height * cos + width * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]

    if image.ndim == 3:
        border = tuple([int(border_value)] * image.shape[2])
    else:
        border = int(border_value)

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def trim_uniform_borders(image: np.ndarray, tolerance: int = 12) -> np.ndarray:
    """Cắt bớt các viền đồng màu (do xoay ảnh sinh ra) quanh vùng nội dung."""
    if image is None or image.size == 0:
        return image
    gray = to_gray(image)
    background = float(np.median(gray))
    mask = np.abs(gray.astype(np.float32) - background) > float(tolerance)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return image
    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    x1, x2 = int(cols[0]), int(cols[-1]) + 1
    if (y2 - y1) < 3 or (x2 - x1) < 3:
        return image
    return image[y1:y2, x1:x2].copy()


def variance_of_laplacian(image: np.ndarray) -> float:
    """Chỉ số độ mờ (blur) phổ biến: phương sai của Laplacian."""
    gray = to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(image: np.ndarray) -> float:
    """Độ sáng trung bình (0-255)."""
    return float(to_gray(image).mean())


def resize_by_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Resize ảnh theo chiều cao mục tiêu, giữ nguyên aspect ratio."""
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or target_height <= 0:
        return image
    scale = float(target_height) / float(height)
    new_w = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_w, int(target_height)), interpolation=interpolation)


def optional_float(value: Optional[float]) -> Optional[float]:
    """Chuyển giá trị có thể là ``None`` sang ``float`` an toàn."""
    if value is None:
        return None
    return float(value)
