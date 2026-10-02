"""
tools/install_paddle.py
=======================
Cài PaddleOCR vào MỘT venv RIÊNG (không có torch) để tránh xung đột
torch (numpy 2.x / CUDA 12.6) vs paddle. Chạy BẰNG python của .venv-paddle:

    S:\\M_AGENT\\CV\\.venv-paddle\\Scripts\\python.exe tools/install_paddle.py

Dùng paddlepaddle-gpu 3.0.0b2 + index cu118 (theo khuyến nghị tránh lỗi #63595/#10558).
"""

from __future__ import annotations

import subprocess
import sys

PADDLE_INDEX = "https://www.paddlepaddle.org.cn/packages/stable/cu118/"

CMDS = [
    [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
    [sys.executable, "-m", "pip", "install",
     "paddlepaddle-gpu==3.0.0b2", "-i", PADDLE_INDEX],
    [sys.executable, "-m", "pip", "install", "paddleocr"],
]


def main() -> int:
    for cmd in CMDS:
        print("> " + " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        print(f"[rc={rc}]", flush=True)
        if rc != 0:
            print("[!] dung lai do lenh tren loi", flush=True)
            return rc
    print("INSTALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
