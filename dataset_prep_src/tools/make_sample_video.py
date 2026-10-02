"""
tools/make_sample_video.py
==========================
CLI sinh **video giả lập** (không cần dữ liệu thật) để test Module 0 — Frame
Extraction & Filtering.

Video gồm nền "đường" sáng, vài vạch kẻ trắng, 1 ô tô chạy ngang trái → phải
(có dán biển 1 dòng) và 1 xe máy đứng yên ở góc (dán biển 2 dòng). Trong danh
sách ``dark_frames`` / ``blur_frames``, khung bị làm tối / làm mờ **cố ý** để
chứng minh bộ lọc chất lượng của Module 0 hoạt động.

Ví dụ::

    python tools/make_sample_video.py --output data/samples/test_video.mp4
    python tools/make_sample_video.py --frames 60 --fps 30 --dark-frames 10,11,40
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from typing import Optional, Sequence

import cv2
import numpy as np

# Cho phép chạy trực tiếp từ thư mục gốc project.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_utils import resize_by_height  # noqa: E402
from src.sample_data import (  # noqa: E402
    add_degradation,
    make_single_line_plate,
    make_two_line_plate,
)


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Dán ``patch`` lên ``canvas`` tại ``(x, y)``, tự cắt phần tràn ra ngoài."""
    if patch is None or patch.size == 0:
        return
    patch_h, patch_w = patch.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(canvas_w, int(x) + patch_w), min(canvas_h, int(y) + patch_h)
    if x2 <= x1 or y2 <= y1:
        return
    canvas[y1:y2, x1:x2] = patch[y1 - int(y) : y2 - int(y), x1 - int(x) : x2 - int(x)]


def _render_frame(
    index: int,
    total_frames: int,
    width: int,
    height: int,
    car_plate: np.ndarray,
    bike_plate: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vẽ 1 khung "sạch": đường + vạch kẻ + ô tô chạy + xe máy đứng yên."""
    # Nền đường xám sáng (~130) + texture nhiễu để khung sạch CHẮC CHẮN đủ nét:
    # codec ``mp4v`` là lossy và làm mượt bớt chi tiết tần số cao, nên cần biên độ
    # nhiễu đủ lớn để ``variance_of_laplacian`` vẫn vượt ngưỡng 100 sau khi encode.
    frame = np.full((height, width, 3), 130, dtype=np.uint8)
    noise = rng.normal(0.0, 12.0, (height, width, 1))
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Các vạch kẻ đường màu trắng (nét) — nguồn chi tiết tần số cao.
    for ratio in (0.15, 0.28, 0.72, 0.85):
        line_y = int(height * ratio)
        line_x = 0
        while line_x + 70 < width:
            cv2.rectangle(frame, (line_x, line_y), (line_x + 70, line_y + 6), (255, 255, 255), -1)
            line_x += 130

    # Ô tô chạy ngang trái -> phải: vị trí x là hàm tuyến tính của chỉ số frame.
    car_w = max(120, width // 4)
    car_h = max(70, int(car_w * 0.45))
    car_y = int(height * 0.42)
    travel = max(1, int(total_frames) - 1)
    car_x = int(-car_w + (width + car_w) * (index / travel))
    cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (200, 120, 60), -1)
    win_h = max(10, car_h // 3)
    cv2.rectangle(
        frame,
        (car_x + car_w // 5, car_y + car_h // 6),
        (car_x + car_w - car_w // 5, car_y + car_h // 6 + win_h),
        (40, 40, 40),
        -1,
    )
    plate_x = car_x + (car_w - car_plate.shape[1]) // 2
    plate_y = car_y + car_h - car_plate.shape[0] - 6
    _paste_patch(frame, car_plate, plate_x, plate_y)

    # Xe máy đứng yên ở góc dưới-trái, dán biển 2 dòng bên cạnh.
    bike_w, bike_h = 90, 64
    bike_x = 30
    bike_y = height - bike_h - 24
    cv2.rectangle(frame, (bike_x, bike_y), (bike_x + bike_w, bike_y + bike_h), (35, 35, 35), -1)
    _paste_patch(frame, bike_plate, bike_x + bike_w + 6, height - bike_plate.shape[0] - 24)

    return frame


def _path_is_ascii(path: str) -> bool:
    """True nếu toàn bộ đường dẫn (đã chuẩn hoá tuyệt đối) chỉ chứa ký tự ASCII.

    Dùng để quyết định có cần ghi video qua file tạm ASCII hay không — vì
    ``cv2.VideoWriter`` không hỗ trợ đường dẫn Unicode trên Windows.
    """
    return all(ord(char) < 128 for char in os.path.abspath(path))


def _ascii_temp_dir() -> str:
    """Trả về 1 thư mục tạm có đường dẫn ASCII (best-effort).

    ``tempfile.gettempdir()`` có thể nằm dưới hồ sơ người dùng có tên Unicode
    (VD: ``C:\\Users\\Nguyễn\\AppData\\Local\\Temp``) — khi đó file tạm vẫn là
    đường dẫn Unicode và ``cv2.VideoWriter`` lại fail đúng như điều nhánh này
    đang cố tránh. Hàm thử lần lượt vài thư mục tạm "chuẩn" và trả về cái đầu
    tiên có đường dẫn ASCII; nếu không tìm được thì đành dùng thư mục tạm hệ thống.
    """
    candidates = [
        tempfile.gettempdir(),
        os.environ.get("TEMP"),
        os.environ.get("TMP"),
    ]
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot") or os.environ.get("windir")
        if system_root:
            candidates.append(os.path.join(system_root, "Temp"))
    candidates.append(os.path.join(os.sep, "tmp"))

    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
        except OSError:
            continue
        if _path_is_ascii(candidate):
            return candidate
    return tempfile.gettempdir()


def build_synthetic_video(
    output_path: str,
    total_frames: int = 150,
    fps: float = 30.0,
    width: int = 960,
    height: int = 540,
    dark_frames: Sequence[int] = (30, 31, 32, 100),
    blur_frames: Sequence[int] = (60, 61, 62, 120),
    seed: int = 0,
) -> str:
    """Sinh video giả lập: đường + xe chạy ngang + biển số dán lên xe.

    Frame trong ``dark_frames`` bị làm tối gần như toàn khung (``mean_brightness``
    < 50) và frame trong ``blur_frames`` bị làm mờ mạnh (``variance_of_laplacian``
    < 100) — nhờ đó self_check kiểm chứng được bộ lọc của Module 0.

    Returns:
        Đường dẫn file video đã ghi.

    Raises:
        ValueError: Nếu ``total_frames``/``width``/``height`` không hợp lệ.
        RuntimeError: Nếu ``cv2.VideoWriter`` không mở được (thiếu codec ``mp4v``).
    """
    if int(total_frames) <= 0:
        raise ValueError("total_frames phải > 0 để sinh video.")
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width/height phải > 0 để sinh video.")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    dark_set = {int(value) for value in dark_frames}
    blur_set = {int(value) for value in blur_frames}

    car_plate = resize_by_height(make_single_line_plate("51F", "12345"), 40)
    bike_plate = resize_by_height(make_two_line_plate("29-B1", "12345"), 56)

    size = (int(width), int(height))

    # cv2.VideoWriter không hỗ trợ đường dẫn Unicode trên Windows -> nếu đường dẫn
    # đích chứa ký tự Unicode thì ghi ra file tạm ASCII rồi ``shutil.move`` sang đích
    # (cùng tinh thần ``imwrite_unicode``). LƯU Ý: phải kiểm tra cả THƯ MỤC CHA, vì nếu
    # thư mục cha có Unicode thì file tạm đặt trong đó vẫn không phải ASCII và codec sẽ
    # fail đúng với lỗi mà nhánh này đang cố tránh.
    ascii_safe = _path_is_ascii(output_path)
    dir_ascii_safe = _path_is_ascii(output_dir)

    temp_path = output_path
    temp_created = False
    if not ascii_safe:
        if dir_ascii_safe:
            # Chỉ tên file có Unicode, thư mục cha vẫn ASCII -> tạo tạm cùng thư mục
            # cho nhanh và để việc move là rename nguyên tố (cùng ổ đĩa).
            temp_fd, temp_path = tempfile.mkstemp(
                prefix=".tmpsamplevideo_", suffix=".mp4", dir=output_dir
            )
        else:
            # Cả đường dẫn cha cũng có Unicode -> buộc phải tạo tạm trong một thư mục
            # tạm THỰC SỰ ASCII (không dùng thẳng ``tempfile.gettempdir()`` vì nó có
            # thể nằm dưới hồ sơ người dùng tên Unicode) rồi move sang đích.
            temp_fd, temp_path = tempfile.mkstemp(
                prefix="tmpsamplevideo_", suffix=".mp4", dir=_ascii_temp_dir()
            )
        os.close(temp_fd)
        temp_created = True

    writer = None
    committed = False
    try:
        writer = cv2.VideoWriter(
            temp_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size
        )
        if not writer.isOpened():
            raise RuntimeError(
                "Không mở được cv2.VideoWriter với codec 'mp4v'. Hãy kiểm tra bản "
                "opencv-python có hỗ trợ FFMPEG/codec mp4v (xem 'cv2.getBuildInformation()')."
            )
        for index in range(int(total_frames)):
            frame = _render_frame(
                index, total_frames, width, height, car_plate, bike_plate, rng
            )
            if index in dark_set:
                frame = add_degradation(frame, brightness=-110)
            if index in blur_set:
                frame = add_degradation(frame, blur=25)
            writer.write(frame)
        writer.release()
        writer = None
        if temp_created:
            # ``shutil.move`` an toàn khi file tạm nằm khác ổ đĩa (cross-device) và tự
            # ghi đè file đích nếu đã tồn tại (khác ``os.replace`` vốn fail cross-device).
            shutil.move(temp_path, output_path)
        committed = True
    finally:
        if writer is not None:
            writer.release()
        # Dọn file tạm nếu có lỗi xảy ra giữa chừng (chưa move được sang đích).
        if temp_created and not committed and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    print(f"[i] Đã ghi video: {output_path} ({int(total_frames)} frame @ {fps} fps)")
    return output_path


def _parse_int_list(raw: str) -> list:
    """Parse chuỗi ``'30,31,32'`` (hoặc dùng ``;``) thành list int."""
    return [int(piece.strip()) for piece in str(raw).replace(";", ",").split(",") if piece.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Khai báo tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Sinh video giả lập để test Module 0 — Frame Extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/samples/test_video.mp4",
        help="File video ghi ra.",
    )
    parser.add_argument("--frames", type=int, default=150, help="Tổng số frame.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS của video.")
    parser.add_argument("--width", type=int, default=960, help="Chiều rộng khung.")
    parser.add_argument("--height", type=int, default=540, help="Chiều cao khung.")
    parser.add_argument(
        "--dark-frames",
        type=str,
        default="30,31,32,100",
        help="Danh sách chỉ số frame làm tối (ngăn cách bằng dấu phẩy).",
    )
    parser.add_argument(
        "--blur-frames",
        type=str,
        default="60,61,62,120",
        help="Danh sách chỉ số frame làm mờ (ngăn cách bằng dấu phẩy).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed để tái lập.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python tools/make_sample_video.py --output data/samples/test_video.mp4``."""
    args = parse_args(argv)
    try:
        build_synthetic_video(
            args.output,
            total_frames=args.frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
            dark_frames=_parse_int_list(args.dark_frames),
            blur_frames=_parse_int_list(args.blur_frames),
            seed=args.seed,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
