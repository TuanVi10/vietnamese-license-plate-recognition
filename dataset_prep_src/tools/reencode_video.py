"""
tools/reencode_video.py
=======================
Gộp overlay frame (tracks_frames/frame_*.png) -> H.264 mp4 (Windows mở được)
bằng imageio + imageio-ffmpeg (bundle ffmpeg/libx264). Không cần chạy lại pipeline.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import imageio

FRAMES = Path(r"S:\M_AGENT\CV\runs\e2e_full\tracks_frames")
OUT = Path(r"S:\M_AGENT\CV\runs\e2e_full\output_demo.mp4")
FPS = 12.0  # 59.94fps / frame_interval 5 ≈ 12fps (đúng tốc độ thật)


def main() -> int:
    files = sorted(FRAMES.glob("frame_*.png"))
    if not files:
        print("[!] khong co frame")
        return 1
    writer = imageio.get_writer(str(OUT), fps=FPS, codec="libx264",
                                quality=7, macro_block_size=None)
    n = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            n += 1
    writer.close()
    print(f"[i] h264 video -> {OUT} ({n} frames, {FPS}fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
