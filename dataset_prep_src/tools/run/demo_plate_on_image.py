"""
tools/demo_plate_on_image.py
============================
Kiểm tra nhanh: plate_detector.pt (Module 2) + EasyOCR (Module 5) trên 1 ảnh THẬT
có biển số từ dataset — xác nhận chain detect -> crop -> OCR hoạt động trên dữ liệu
thật (khác với ảnh tổng hợp).

Chạy::

    python tools/demo_plate_on_image.py
"""

from __future__ import annotations

import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import OCRConfig  # noqa: E402
from src.ocr import create_ocr_engine  # noqa: E402

DATASET = r"S:\M_AGENT\CV\datasets\dataset"
PLATE_MODEL = r"S:\M_AGENT\CV\models\plate_detector.pt"


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(PLATE_MODEL)
    engine = create_ocr_engine(OCRConfig(engine="easyocr"))
    print("plate model:", PLATE_MODEL)
    print("ocr engine:", engine.name)

    for image_path in sorted(glob.glob(os.path.join(DATASET, "images", "train", "*.jpg"))):
        label_path = image_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).replace(".jpg", ".txt")
        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        results = model.predict(image, conf=0.25, verbose=False, imgsz=640)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        box = boxes[0].xyxy[0].tolist()
        x1, y1, x2, y2 = (int(v) for v in box)
        conf = float(boxes.conf[0])
        crop = image[y1:y2, x1:x2]
        big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        reading = engine.recognize(big)
        print("=" * 60)
        print("image      :", os.path.basename(image_path), image.shape[1], "x", image.shape[0])
        print("plate_bbox :", (x1, y1, x2, y2), "detect_conf=", round(conf, 3))
        print("OCR text   :", repr(reading.text), "conf=", round(reading.confidence, 3))
        out = r"S:\M_AGENT\CV\data\samples\real_plate_demo.jpg"
        cv2.imwrite(out, big)
        print("crop saved :", out)
        break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
