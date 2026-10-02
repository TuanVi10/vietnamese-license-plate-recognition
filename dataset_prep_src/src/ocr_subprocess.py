"""
src/ocr_subprocess.py
=====================
PaddleOCR qua subprocess (.venv-paddle) — CÁCH B: tránh xung đột torch/paddle.
Spawn ``tools/paddle_ocr_server.py`` 1 lần, giao tiếp stdin/stdout.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .ocr import DEFAULT_CHARSET, BaseOCREngine, OCRItem


class SubprocessPaddleEngine(BaseOCREngine):
    """OCR engine gọi PaddleOCR chạy ở venv riêng (.venv-paddle)."""

    name = "paddle_subprocess"

    def __init__(
        self,
        python: str = r"S:\M_AGENT\CV\.venv-paddle\Scripts\python.exe",
        script: str = r"S:\M_AGENT\CV\dataset_prep_src\tools\paddle_ocr_server.py",
        charset: str = DEFAULT_CHARSET,
        drop_empty: bool = True,
    ) -> None:
        super().__init__(charset=charset, drop_empty=drop_empty)
        import subprocess

        self._proc = subprocess.Popen(
            [python, script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1)

    def _run(self, image: np.ndarray) -> List[OCRItem]:
        import base64

        import cv2

        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            return []
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        self._proc.stdin.write(b64 + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline().strip()
        if not line:
            return []
        return [OCRItem(text=line, raw_text=line, confidence=1.0, box=None)]

    def close(self) -> None:
        try:
            self._proc.stdin.close()
            self._proc.terminate()
        except Exception:
            pass
