"""
tools/make_demo_video.py
========================
Dựng 1 video demo ngắn từ các ảnh THẬT có biển số trong dataset (mỗi ảnh giữ vài
frame), để chạy full pipeline (Module 0->7) trên dữ liệu thật.

Chạy::

    python tools/make_demo_video.py --count 15 --hold 6 --fps 15
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np


def letterbox(image, size: int = 640, color=(114, 114, 114)) -> np.ndarray:
    """Resize giữ tỷ lệ + đệm về hình vuông ``size x size`` (không méo chữ)."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Build demo video from real plate images")
    parser.add_argument("--images-dir", default=r"S:\M_AGENT\CV\datasets\dataset\images\train")
    parser.add_argument("--labels-dir", default=r"S:\M_AGENT\CV\datasets\dataset\labels\train")
    parser.add_argument("--out", default=r"S:\M_AGENT\CV\data\samples\demo_real.mp4")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--hold", type=int, default=6)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    images = []
    for image_path in sorted(glob.glob(os.path.join(args.images_dir, "*.jpg"))):
        label_path = os.path.join(args.labels_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            images.append(image_path)
        if len(images) >= args.count:
            break

    if not images:
        print("[!] không có ảnh nào có nhãn")
        return 1

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.size, args.size))
    for image_path in images:
        image = cv2.imread(image_path)
        frame = letterbox(image, args.size)
        for _ in range(args.hold):
            writer.write(frame)
    writer.release()

    print(f"[i] Đã ghi video: {args.out} ({len(images)} ảnh x {args.hold} frame @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
