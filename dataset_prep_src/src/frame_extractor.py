"""
frame_extractor.py
==================
Module 0 — Trích & lọc frame từ video (SPEC KỸ THUẬT v2, "Module 0 — Frame
Extraction & Filtering").

Module này đọc video bằng ``cv2.VideoCapture`` (có xử lý đường dẫn Unicode trên
Windows), xét **mỗi ``frame_interval`` frame** (VD: 5 = khoảng 6 frame/giây với
video 30fps), giữ lại những frame **đủ sáng** (``mean_brightness``) và **đủ nét**
(``variance_of_laplacian``), đồng thời **lưu lại chỉ số frame GỐC** trong video
(``FrameRecord.frame_index``) để các module sau (detect biển số, tracking, OCR...)
khớp lại đúng thời điểm xe xuất hiện.

Vì là mắt xích đầu tiên của pipeline video, module chỉ phụ thuộc ``numpy`` +
``cv2`` + các tiện ích nội bộ (:mod:`src.image_utils`), **không** cần
``ultralytics`` / ``paddleocr`` / ``easyocr`` / ``torch``.

Cách dùng nhanh::

    from src.config import FrameExtractionConfig
    from src.frame_extractor import FrameExtractor

    extractor = FrameExtractor(FrameExtractionConfig(frame_interval=5))
    result = extractor.extract("data/samples/test_video.mp4")
    print(result.stats.kept_frames, "frame giữ lại")
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .config import FrameExtractionConfig
from .image_utils import mean_brightness, resize_by_height, variance_of_laplacian


#: FPS dự phòng khi OpenCV không đọc được FPS của video (trả về 0).
_DEFAULT_FPS = 25.0

#: Số lần "đọc lỗi liên tiếp" tối đa trước khi coi như đã hết video thật sự.
#: Dùng để phá vòng lặp nếu codec liên tục trả ``(ret=True, frame=None)`` mà
#: không bao giờ báo hết video (``ret=False``).
_MAX_CONSECUTIVE_READ_FAILURES = 10


def _path_is_ascii(path: str) -> bool:
    """True nếu đường dẫn tuyệt đối chỉ chứa ký tự ASCII.

    ``cv2.VideoCapture`` không hỗ trợ đường dẫn Unicode trên Windows — cùng lý do
    khiến :func:`src.image_utils.imread_unicode` phải tồn tại cho ảnh. Hàm này cho
    biết khi nào cần chuyển sang đường dẫn tạm ASCII.
    """
    return all(ord(char) < 128 for char in os.path.abspath(path))


def _safe_remove(path: Optional[str]) -> None:
    """Xóa file nếu tồn tại; bỏ qua mọi lỗi hệ thống file (best-effort)."""
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Đổi ``value`` sang ``float`` hữu hạn; trả ``default`` nếu ``NaN``/``inf``/lỗi.

    OpenCV có thể trả ``NaN`` cho ``CAP_PROP_FPS`` / ``CAP_PROP_FRAME_COUNT`` với
    một số codec. Nếu không chặn, ``NaN`` sẽ lan vào ``timestamp`` (thành NaN) và
    ``int(NaN)`` sẽ ném ``ValueError``.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _open_capture(video_path: str) -> Tuple[cv2.VideoCapture, Optional[str]]:
    """Mở ``cv2.VideoCapture`` an toàn với đường dẫn Unicode trên Windows.

    Thử mở trực tiếp trước; nếu thất bại VÀ đường dẫn có ký tự ngoài ASCII thì
    copy video sang 1 file tạm ASCII rồi mở file tạm đó (``cv2.VideoCapture``
    không mở được đường dẫn Unicode trên Windows). Nếu đường dẫn đã ASCII mà vẫn
    lỗi thì trả về capture CHƯA mở — đó là lỗi thật (file hỏng / thiếu codec).

    Returns:
        Cặp ``(capture, temp_path)``. ``temp_path`` là ``None`` nếu mở trực tiếp;
        nếu khác ``None`` thì bên gọi PHẢI xóa file tạm sau khi ``release()``.
    """
    capture = cv2.VideoCapture(video_path)
    if capture.isOpened() or _path_is_ascii(video_path):
        return capture, None

    capture.release()
    suffix = os.path.splitext(video_path)[1] or ".mp4"
    temp_path: Optional[str] = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(prefix="alpr_video_", suffix=suffix)
        os.close(temp_fd)
        with open(video_path, "rb") as source, open(temp_path, "wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    except OSError:
        _safe_remove(temp_path)
        return cv2.VideoCapture(video_path), None

    temp_capture = cv2.VideoCapture(temp_path)
    if not temp_capture.isOpened():
        temp_capture.release()
        _safe_remove(temp_path)
        return cv2.VideoCapture(video_path), None
    return temp_capture, temp_path


@dataclass
class FrameRecord:
    """Một frame đã vượt qua bộ lọc của Module 0."""

    #: Chỉ số frame GỐC trong video (0-based, tự đếm khi đọc — không dùng index
    #: của vòng lặp sau khi đã nhảy bước).
    frame_index: int
    #: Ảnh BGR đã giữ (đã resize nếu ``resize_height > 0``).
    frame: np.ndarray
    #: Thời điểm trong video tính bằng giây: ``frame_index / fps``.
    timestamp: float
    #: Độ sáng trung bình của ảnh GỐC (trước resize).
    brightness: float
    #: Chỉ số độ nét (variance of Laplacian) của ảnh GỐC (trước resize).
    blur_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata của frame (KHÔNG chứa ảnh — ảnh không JSON-able)."""
        return {
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 4),
            "brightness": round(float(self.brightness), 4),
            "blur_score": round(float(self.blur_score), 4),
        }


@dataclass
class FrameExtractionStats:
    """Thống kê quá trình trích frame (để in log và ghi manifest JSON)."""

    #: Tổng số frame đọc được từ video.
    total_frames: int = 0
    #: Số frame đã đưa vào xét (sau ``skip_start_frames`` + ``frame_interval``).
    scanned_frames: int = 0
    #: Số frame giữ lại.
    kept_frames: int = 0
    #: Số frame bị bỏ vì quá tối.
    skipped_dark: int = 0
    #: Số frame bị bỏ vì quá mờ.
    skipped_blurry: int = 0
    #: Số frame bị bỏ vì vượt ``max_frames``.
    skipped_by_limit: int = 0
    #: FPS của video (dùng giá trị dự phòng nếu OpenCV trả 0).
    fps: float = 0.0
    #: Thời lượng video tính bằng giây.
    duration: float = 0.0
    #: Chiều rộng khung (pixel) theo metadata của video.
    width: int = 0
    #: Chiều cao khung (pixel) theo metadata của video.
    height: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thống kê thành dict thuần để ghi JSON."""
        return {
            "total_frames": int(self.total_frames),
            "scanned_frames": int(self.scanned_frames),
            "kept_frames": int(self.kept_frames),
            "skipped_dark": int(self.skipped_dark),
            "skipped_blurry": int(self.skipped_blurry),
            "skipped_by_limit": int(self.skipped_by_limit),
            "fps": round(float(self.fps), 4),
            "duration": round(float(self.duration), 4),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass
class FrameExtractionResult:
    """Kết quả đầy đủ của Module 0 cho 1 video."""

    video_path: str
    frames: List[FrameRecord] = field(default_factory=list)
    stats: FrameExtractionStats = field(default_factory=FrameExtractionStats)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize kết quả (chỉ meta + stats + danh sách frame, KHÔNG ảnh)."""
        return {
            "video_path": self.video_path,
            "stats": self.stats.to_dict(),
            "warnings": list(self.warnings),
            "frames": [record.to_dict() for record in self.frames],
        }


def _iter_frames(
    video_path: str,
    config: FrameExtractionConfig,
    stats: FrameExtractionStats,
    warnings: List[str],
) -> Iterator[FrameRecord]:
    """Generator lõi: đọc video, lọc frame và yield từng :class:`FrameRecord`.

    Hàm này cập nhật trực tiếp ``stats``/``warnings`` trong lúc chạy và **luôn**
    giải phóng ``cv2.VideoCapture`` trong khối ``finally`` (kể cả khi generator
    bị đóng giữa vòng lặp do người dùng ``break``).

    Raises:
        FileNotFoundError: Nếu ``video_path`` không tồn tại.
        ValueError: Nếu không mở được video hoặc không đọc nổi frame nào.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

    capture, temp_video = _open_capture(video_path)
    try:
        if not capture.isOpened():
            raise ValueError(
                f"Không mở được video (file lỗi hoặc codec không hỗ trợ): {video_path}"
            )

        # Đọc metadata; chặn ``NaN``/``inf`` (một số codec trả về) để không lan vào
        # ``timestamp`` (thành NaN) hay ``int(NaN)`` (ném ValueError).
        raw_fps = _finite_float(capture.get(cv2.CAP_PROP_FPS))
        if raw_fps > 0.0:
            fps = raw_fps
        else:
            fps = _DEFAULT_FPS
            warnings.append(
                f"OpenCV không đọc được FPS hợp lệ (trả {capture.get(cv2.CAP_PROP_FPS)!r}); "
                f"tạm dùng {_DEFAULT_FPS:.0f} fps để tính timestamp."
            )

        stats.fps = fps
        stats.width = int(_finite_float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        stats.height = int(_finite_float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        reported_total = int(_finite_float(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

        interval = max(1, int(config.frame_interval))
        skip_start = max(0, int(config.skip_start_frames)) + max(0, int(round(config.start_seconds * fps)))
        resize_height = max(0, int(config.resize_height))
        max_frames = max(0, int(config.max_frames))
        max_duration = max(0.0, float(config.max_duration))

        frame_index = 0
        read_any = False
        consecutive_failures = 0
        while True:
            ret, frame = capture.read()
            if not ret or frame is None or frame.size == 0:
                # OpenCV trả ``ret=False`` cho CẢ "hết video" lẫn "lỗi đọc". Ta dừng
                # ở lần đọc thất bại ĐẦU TIÊN để ``total_frames`` đúng bằng số frame
                # thực đọc được (không tăng khống ở mốc EOF).
                if not ret:
                    break
                # ret=True nhưng frame rỗng/lỗi (một số codec trả về ở cuối video):
                # nếu đã chạm số frame mà metadata báo thì coi như hết video.
                if reported_total > 0 and frame_index >= reported_total:
                    break
                # Đếm số lỗi LIÊN TIẾP để thoát nếu codec trả (True, None) triền miên
                # (không để vòng lặp chạy vô hạn).
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                    break
                # Frame lỗi nhưng vẫn tính vào ``total_frames`` để giữ đúng chỉ số
                # frame gốc cho các module sau.
                stats.total_frames += 1
                frame_index += 1
                continue

            consecutive_failures = 0
            read_any = True
            stats.total_frames += 1

            current_index = frame_index
            frame_index += 1

            # 1) Bỏ N frame đầu video.
            if current_index < skip_start:
                continue
            # 2) Bước nhảy frame, tính TƯƠNG ĐỐI từ mốc skip_start (không neo cứng
            # vào frame 0), để "bỏ N frame đầu" đúng nghĩa: lấy mẫu
            # skip_start, skip_start + interval, skip_start + 2*interval, ...
            if interval > 1 and (current_index - skip_start) % interval != 0:
                continue

            stats.scanned_frames += 1

            # 2b) Giới hạn theo thời lượng: dừng sớm khi chạm ``max_duration`` giây.
            if max_duration > 0.0 and (current_index / fps) >= max_duration:
                stats.skipped_by_limit += 1
                break

            # Đo chất lượng trên ảnh GỐC (trước resize).
            brightness = mean_brightness(frame)
            blur_score = variance_of_laplacian(frame)

            # 3) Resize (giữ aspect ratio) nếu được yêu cầu.
            working = frame
            if resize_height > 0 and frame.shape[0] != resize_height:
                working = resize_by_height(frame, resize_height)

            # 4) Giới hạn số frame giữ lại. Đã đủ số cần giữ -> dừng SỚM, không giải
            # mã nốt phần còn lại của video (đúng kỳ vọng khi đặt ``--max-frames``).
            # Frame vượt giới hạn đầu tiên vẫn được đếm vào ``skipped_by_limit``.
            if max_frames > 0 and stats.kept_frames >= max_frames:
                stats.skipped_by_limit += 1
                break

            # 5) Bộ lọc chất lượng (độ sáng trước, độ nét sau).
            if config.enable_quality_filter:
                if brightness < float(config.brightness_threshold):
                    stats.skipped_dark += 1
                    continue
                if blur_score < float(config.blur_threshold):
                    stats.skipped_blurry += 1
                    continue

            stats.kept_frames += 1
            yield FrameRecord(
                frame_index=current_index,
                frame=working,
                timestamp=current_index / fps,
                brightness=brightness,
                blur_score=blur_score,
            )

        if not read_any:
            raise ValueError(f"Không đọc được frame nào từ video: {video_path}")

        total_for_duration = reported_total if reported_total > 0 else stats.total_frames
        stats.duration = total_for_duration / fps if fps > 0 else 0.0
    finally:
        capture.release()
        _safe_remove(temp_video)


def iter_valid_frames(
    video_path: str,
    config: Optional[FrameExtractionConfig] = None,
) -> Iterator[FrameRecord]:
    """Generator yield từng frame hợp lệ — không giữ toàn bộ video trong RAM.

    Đường dẫn được kiểm tra NGAY khi gọi hàm (không đợi tới lần ``next()`` đầu
    tiên) để lỗi ``FileNotFoundError`` nổi lên đúng thời điểm, trong khi phần
    đọc/lọc frame vẫn diễn ra LƯỜI (lazy) qua generator nội bộ.

    Args:
        video_path: Đường dẫn video đầu vào.
        config: Tham số Module 0. ``None`` -> dùng :class:`FrameExtractionConfig` mặc định.

    Yields:
        :class:`FrameRecord` cho từng frame vượt qua bộ lọc.

    Raises:
        FileNotFoundError: Nếu ``video_path`` không tồn tại.
        ValueError: Nếu không mở được video hoặc không đọc nổi frame nào.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")
    effective_config = config or FrameExtractionConfig()
    stats = FrameExtractionStats()
    warnings: List[str] = []
    return _iter_frames(video_path, effective_config, stats, warnings)


def extract_frames(
    video_path: str,
    config: Optional[FrameExtractionConfig] = None,
) -> FrameExtractionResult:
    """Chạy Module 0 trên 1 video và trả kết quả đầy đủ.

    Args:
        video_path: Đường dẫn video đầu vào.
        config: Tham số Module 0. ``None`` -> dùng :class:`FrameExtractionConfig` mặc định.

    Returns:
        :class:`FrameExtractionResult` gồm danh sách frame giữ lại, thống kê và
        các cảnh báo (VD: FPS không đọc được).

    Raises:
        FileNotFoundError: Nếu ``video_path`` không tồn tại.
        ValueError: Nếu không mở được video hoặc không đọc nổi frame nào.
    """
    effective_config = config or FrameExtractionConfig()
    stats = FrameExtractionStats()
    warnings: List[str] = []
    frames: List[FrameRecord] = []
    for record in _iter_frames(video_path, effective_config, stats, warnings):
        frames.append(record)
    return FrameExtractionResult(
        video_path=video_path,
        frames=frames,
        stats=stats,
        warnings=warnings,
    )


class FrameExtractor:
    """Bọc Module 0 thành object để tái sử dụng / nối pipeline về sau."""

    def __init__(self, config: Optional[FrameExtractionConfig] = None) -> None:
        """Khởi tạo với cấu hình cho trước (mặc định :class:`FrameExtractionConfig`)."""
        self._config = config or FrameExtractionConfig()
        self._last_stats: Optional[FrameExtractionStats] = None

    @property
    def config(self) -> FrameExtractionConfig:
        """Cấu hình Module 0 đang dùng."""
        return self._config

    @property
    def last_stats(self) -> Optional[FrameExtractionStats]:
        """Stats của lần :meth:`extract` gần nhất (``None`` nếu chưa chạy)."""
        return self._last_stats

    def iter_frames(self, video_path: str) -> Iterator[FrameRecord]:
        """Yield từng frame hợp lệ của ``video_path`` (không giữ toàn bộ trong RAM)."""
        return iter_valid_frames(video_path, self._config)

    def extract(self, video_path: str) -> FrameExtractionResult:
        """Chạy Module 0 trên 1 video và lưu lại stats cho lần gọi gần nhất."""
        result = extract_frames(video_path, self._config)
        self._last_stats = result.stats
        return result
