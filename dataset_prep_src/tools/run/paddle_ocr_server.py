"""
tools/paddle_ocr_server.py
==========================
Worker OCR thường trú (chạy bằng .venv-paddle) — CÁCH B (tránh xung đột torch↔paddle).

Giao thức stdin/stdout:
  stdin : mỗi dòng = đường dẫn 1 ảnh biển (đã căn chỉnh).
  stdout: mỗi dòng = "<path>\\t<text>" (flush ngay), text đã lọc 0-9 A-Z.

Pipeline (bên .venv) spawn process này 1 lần, ghi path vào stdin, đọc text từ stdout.
"""
from __future__ import annotations

import base64
import sys

import cv2
import numpy as np
from paddleocr import PaddleOCR


def main() -> int:
    ocr = PaddleOCR(lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        img = cv2.imdecode(np.frombuffer(base64.b64decode(line), np.uint8),
                           cv2.IMREAD_COLOR)
        text = ""
        if img is not None:
            res = ocr.predict(img)
            texts = []
            for r in res:
                d = r.json if hasattr(r, "json") else r
                texts += d.get("res", {}).get("rec_texts", [])
            text = "".join(texts)
        print(text.replace("\t", " "), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
