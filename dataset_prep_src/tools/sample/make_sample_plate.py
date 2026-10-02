"""
tools/make_sample_plate.py
==========================
CLI sinh ảnh biển số VN giả lập để test pipeline khi chưa có ảnh thật.

Ví dụ::

    python tools/make_sample_plate.py --output-dir data/samples
    python tools/make_sample_plate.py --two-line --serial 29-B1 --numbers 12345 \
        --angle 5 --blur 1
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

# Cho phép chạy trực tiếp từ thư mục gốc project.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.image_utils import imwrite_unicode  # noqa: E402
from src.sample_data import (  # noqa: E402
    add_degradation,
    make_demo_images,
    make_single_line_plate,
    make_two_line_plate,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Khai báo tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Sinh ảnh biển số VN giả lập.")
    parser.add_argument("--output-dir", type=str, default="data/samples", help="Thư mục ghi ảnh.")
    parser.add_argument("--name", type=str, default="plate.png", help="Tên file khi sinh 1 ảnh.")
    parser.add_argument("--serial", type=str, default="51F", help="Phần series (VD: 51F, 29-B1).")
    parser.add_argument("--numbers", type=str, default="12345", help="Phần số đăng ký.")
    parser.add_argument("--two-line", action="store_true", help="Sinh biển xe máy 2 dòng.")
    parser.add_argument("--angle", type=float, default=0.0, help="Góc nghiêng (độ).")
    parser.add_argument("--blur", type=int, default=0, help="Bán kính làm mờ Gaussian.")
    parser.add_argument("--noise", type=float, default=0.0, help="Độ lệch chuẩn nhiễu Gaussian.")
    parser.add_argument("--brightness", type=int, default=0, help="Cộng/trừ độ sáng pixel.")
    parser.add_argument("--all", action="store_true", help="Sinh toàn bộ bộ ảnh demo.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Sinh ảnh theo tham số rồi ghi ra đĩa."""
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        for name, image in make_demo_images().items():
            path = os.path.join(args.output_dir, name)
            imwrite_unicode(path, image)
            print(f"[i] Đã ghi: {path}")
        return 0

    if args.two_line:
        base = make_two_line_plate(args.serial, args.numbers)
    else:
        base = make_single_line_plate(args.serial, args.numbers)

    image = add_degradation(
        base,
        angle=args.angle,
        blur=args.blur,
        noise=args.noise,
        brightness=args.brightness,
        seed=0,
    )
    path = os.path.join(args.output_dir, args.name)
    imwrite_unicode(path, image)
    print(f"[i] Đã ghi: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
