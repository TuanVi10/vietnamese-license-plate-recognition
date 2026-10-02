"""
tools/ocr_smoke.py
=================
Smoke test OCR (PaddleOCR) để xác minh engine chạy được và đọc được chữ.

Kiểm tra:
1. Ảnh biển tổng hợp (`data/samples/plate_*.png`) — gọi PaddleOCR trực tiếp.
2. Crop biển số THẬT từ dataset — gọi qua wrapper `src.ocr.create_ocr_engine`
   (cả ảnh gốc lẫn ảnh phóng to 4x để OCR dễ đọc).

Chạy::

    python tools/ocr_smoke.py
"""

from __future__ import annotations

import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SAMPLES = r"S:\M_AGENT\CV\data\samples"
DATASET = r"S:\M_AGENT\CV\datasets\dataset"


def direct_paddle(image_path: str) -> None:
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang="en",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    for result in ocr.predict(image_path):
        payload = getattr(result, "json", None)
        if payload is not None:
            payload = payload if callable(payload) else payload
        print("  [direct] res:", payload)
        if isinstance(payload, dict) and "res" in payload:
            print("  [direct] rec_texts:", payload["res"].get("rec_texts"))


def main() -> int:
    print("=== 1) Ảnh biển tổng hợp ===")
    for name in ("plate_1line.png", "plate_2line.png"):
        path = os.path.join(SAMPLES, name)
        img = cv2.imread(path)
        print(f"\n--- {name} shape={None if img is None else img.shape} "
              f"mean={None if img is None else round(float(img.mean()), 1)} ---")
        if img is not None:
            direct_paddle(path)

    print("\n=== 2) Crop biển số thật từ dataset ===")
    from src.config import OCRConfig
    from src.ocr import create_ocr_engine

    engine = create_ocr_engine(OCRConfig())
    print("engine:", engine.name)

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
        print(f"\n--- {os.path.basename(image_path)} crop={crop.shape[1]}x{crop.shape[0]} ---")
        for label, source in (("goc", crop), ("4x", cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))):
            result = engine.recognize(source)
            print(f"  [{label}] TEXT={result.text!r} CONF={round(result.confidence, 3)} raw={result.raw_text!r}")
        break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
