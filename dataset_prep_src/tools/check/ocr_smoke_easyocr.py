"""
tools/ocr_smoke_easyocr.py
==========================
Smoke test OCR engine EasyOCR (fallback theo spec) — xác minh:
1. import torch + easyocr OK.
2. Đọc ảnh biển tổng hợp (1 dòng + 2 dòng).
3. Đọc crop biển số THẬT từ dataset (ảnh gốc + phóng to 4x).

Chạy::

    python tools/ocr_smoke_easyocr.py
"""

from __future__ import annotations

import glob
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SAMPLES = r"S:\M_AGENT\CV\data\samples"
DATASET = r"S:\M_AGENT\CV\datasets\dataset"


def main() -> int:
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())

    from src.config import OCRConfig
    from src.ocr import available_engines, create_ocr_engine

    print("available_engines:", available_engines())
    engine = create_ocr_engine(OCRConfig(engine="easyocr", use_gpu=False))
    print("engine:", engine.name)

    print("\n=== Ảnh biển tổng hợp ===")
    for name in ("plate_1line.png", "plate_2line.png"):
        img = cv2.imread(os.path.join(SAMPLES, name))
        if img is None:
            print(f"  {name}: KHONG doc duoc anh")
            continue
        result = engine.recognize(img)
        print(f"  {name}: TEXT={result.text!r} CONF={round(result.confidence, 3)} raw={result.raw_text!r}")

    print("\n=== Crop biển số thật ===")
    for image_path in sorted(glob.glob(os.path.join(DATASET, "images", "train", "*.jpg"))):
        label_path = image_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).replace(".jpg", ".txt")
        if not os.path.exists(label_path):
            continue
        lines = [line for line in open(label_path, encoding="utf-8") if line.strip()]
        if not lines:
            continue
        tokens = lines[0].split()
        cx, cy, w, h = (float(tokens[i]) for i in (1, 2, 3, 4))
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        x1 = max(0, int((cx - w / 2.0) * width))
        y1 = max(0, int((cy - h / 2.0) * height))
        x2 = min(width, int((cx + w / 2.0) * width))
        y2 = min(height, int((cy + h / 2.0) * height))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        print(f"  {os.path.basename(image_path)} crop={crop.shape[1]}x{crop.shape[0]}")
        for label, source in (
            ("goc", crop),
            ("4x", cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)),
        ):
            result = engine.recognize(source)
            print(f"    [{label}] TEXT={result.text!r} CONF={round(result.confidence, 3)} raw={result.raw_text!r}")
        break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
