"""
sample_data.py
==============
Sinh ảnh biển số VN giả lập để test/demo pipeline khi chưa có ảnh thật.

Chỉ dùng ký tự 0-9 và A-Z nên không cần font hỗ trợ tiếng Việt — ``cv2.putText``
với ``FONT_HERSHEY_SIMPLEX`` là đủ.

Hàm :func:`add_degradation` mô phỏng điều kiện thực tế (nghiêng, mờ, nhiễu, tối)
để kiểm tra Module 4 (deskew, CLAHE) có tác dụng.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _draw_text_centered(
    canvas: np.ndarray,
    text: str,
    center: Tuple[float, float],
    font_scale: float,
    thickness: int,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Vẽ text căn giữa tại ``center`` (toạ độ theo pixel)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    origin = (int(center[0] - text_w / 2), int(center[1] + text_h / 2))
    cv2.putText(canvas, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)


def make_single_line_plate(
    serial: str = "51F",
    numbers: str = "12345",
    width: int = 470,
    height: int = 110,
) -> np.ndarray:
    """Tạo ảnh biển ô tô 1 dòng (nền trắng, viền đen, chữ ``SERIAL-NUMBERS``)."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (3, 3), (width - 4, height - 4), (0, 0, 0), 3)
    _draw_text_centered(canvas, f"{serial}-{numbers}", (width / 2.0, height / 2.0), 1.7, 4)
    return canvas


def make_two_line_plate(
    serial: str = "29-B1",
    numbers: str = "12345",
    width: int = 260,
    height: int = 180,
) -> np.ndarray:
    """Tạo ảnh biển xe máy 2 dòng (hàng trên ``SERIAL``, hàng dưới ``NUMBERS``)."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (3, 3), (width - 4, height - 4), (0, 0, 0), 3)
    cv2.line(canvas, (10, height // 2), (width - 10, height // 2), (0, 0, 0), 1)
    _draw_text_centered(canvas, serial, (width / 2.0, height * 0.27), 1.4, 3)
    _draw_text_centered(canvas, numbers, (width / 2.0, height * 0.72), 1.4, 3)
    return canvas


def add_degradation(
    image: np.ndarray,
    angle: float = 0.0,
    blur: int = 0,
    noise: float = 0.0,
    brightness: int = 0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Thêm biến dạng mô phỏng điều kiện chụp thực tế.

    Args:
        image: Ảnh biển số sạch.
        angle: Góc xoay (độ), dương = ngược chiều kim đồng hồ.
        blur: Bán kính kernel Gaussian (0 = không làm mờ).
        noise: Độ lệch chuẩn của nhiễu Gaussian cộng thêm.
        brightness: Cộng thêm vào giá trị pixel (âm = tối đi).
        seed: Seed cho nhiễu (để kết quả tái lập được).

    Returns:
        Ảnh đã bị làm biến dạng.
    """
    result = image.copy()

    if angle:
        height, width = result.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(angle), 1.0)
        result = cv2.warpAffine(
            result,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    if blur and blur > 0:
        kernel = int(blur) * 2 + 1
        result = cv2.GaussianBlur(result, (kernel, kernel), 0)

    if brightness:
        result = np.clip(result.astype(np.int16) + int(brightness), 0, 255).astype(np.uint8)

    if noise and noise > 0:
        rng = np.random.default_rng(seed)
        noise_image = rng.normal(0.0, float(noise), result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise_image, 0, 255).astype(np.uint8)

    return result


def make_demo_images() -> dict:
    """Tạo bộ ảnh demo: biển 1 dòng sạch + biển 2 dòng nghiêng/mờ."""
    return {
        "car_plate_clean.png": make_single_line_plate("51F", "12345"),
        "car_plate_tilted.png": add_degradation(
            make_single_line_plate("29H", "1234"), angle=-6.0, noise=6.0, seed=0
        ),
        "motorbike_plate_two_line.png": make_two_line_plate("29-B1", "12345"),
        "motorbike_plate_tilted.png": add_degradation(
            make_two_line_plate("51-F1", "2345"), angle=5.0, blur=1, noise=8.0, seed=1
        ),
    }
