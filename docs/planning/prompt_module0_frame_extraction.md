# BRIEF: Module 0 — Frame Extraction & Filtering (trích & lọc frame từ video)

> **Đây là brief DUY NHẤT của phiên này.** Đọc hết file trước khi viết dòng code đầu tiên.
> Chỉ triển khai **MỘT module (Module 0)**. Không làm thêm module nào khác.

---

## 1. Bối cảnh dự án

Hệ thống ALPR (nhận diện biển số xe Việt Nam từ video), theo *SPEC KỸ THUẬT v2*. Pipeline:

```
video.mp4 → [Module 0: Frame Extraction & Filtering]      ← LẦN NÀY LÀM CÁI NÀY
          → Module 1: Vehicle Detection (YOLOv8n)
          → Module 2: Plate Detection (YOLO fine-tune)
          → Module 3: Tracking (SORT/ByteTrack)
          → Module 4: Tiền xử lý ảnh biển số
          → Module 5: OCR (PaddleOCR)
          → Module 6: Hậu xử lý luật biển VN + voting
          → Module 7: Output & visualize
```

**Đã xong ở lần chạy trước** (bản `CV_doc_bien_anh_20260926040600`, đã kiểm chứng chạy đúng — **không được viết lại**):

| Thành phần | File |
|---|---|
| Module 4 — Tiền xử lý ảnh biển số | `src/preprocess_plate.py` |
| Module 5 — OCR (PaddleOCR 2.x/3.x + EasyOCR fallback) | `src/ocr.py` |
| Module 6 — Hậu xử lý luật biển VN + voting | `src/postprocess.py` |
| Module 2 — Dò bbox biển số bằng YOLO (tùy chọn) | `src/detector.py` |
| Module 7 (rút gọn) — Overlay + JSON | `src/visualize.py`, `main.py` |
| Ghép pipeline **ảnh tĩnh** | `src/plate_reader.py` (`PlateReader`, `PlateReading`) |
| Nền dùng chung (config, tiện ích ảnh, sinh ảnh demo) | `src/config.py`, `src/image_utils.py`, `src/sample_data.py` |

**Lần này chỉ làm Module 0** — mắt xích đầu tiên của pipeline video (theo spec mục 3, "Module 0 — Frame Extraction & Filtering").

---

## 2. QUY TẮC BẮT BUỘC (đọc kỹ — vi phạm là hỏng việc)

| # | Quy tắc |
|---|---|
| **R1** | **TUYỆT ĐỐI KHÔNG được chạy code.** Không tạo venv, không `pip install`, không gọi tool thực thi, không chạy `python xxx.py`. Phiên này bạn **chỉ có tool đọc/ghi/liệt kê file**. Nếu bạn thấy mình đang muốn "chạy thử", hãy **dừng lại và tự soát bằng cách đọc lại code** thay vì chạy. |
| **R2** | Môi trường Python **đã được chuẩn bị SẴN bên ngoài workspace**: Python 3.12 + `numpy` + `opencv-python`. Chỉ được dùng **thư viện chuẩn Python + `numpy` + `cv2`**. KHÔNG thêm dependency mới, **KHÔNG sửa `requirements.txt`**, KHÔNG import `ultralytics` / `paddleocr` / `easyocr` / `torch` ở lần này. |
| **R3** | **Chỉ 1 vòng làm việc:** viết file → đọc lại file để tự soát tĩnh → kết thúc. **Không lặp sửa đi sửa lại.** Nếu phát hiện lỗi khi tự soát, sửa dứt điểm trong 1–2 lần ghi rồi dừng, không tiếp tục "tinh chỉnh" vô hạn. |
| **R4** | **Không tạo nhiều phiên bản** của cùng một file (`xxx_v2.py`, `xxx_new.py`, `xxx_final.py`, `xxx_fixed.py`). Mỗi đường dẫn trong mục 4 chỉ có **đúng 1 file cuối cùng**. |
| **R5** | Không sửa/xóa nội dung đã có của các module đã xong, **trừ đúng các điểm nêu ở mục 4**. Không đổi tên hàm, tên class, tham số, thứ tự tham số của bất kỳ API đã có. |
| **R6** | Code mới: docstring + comment **tiếng Việt**, có `from __future__ import annotations`, type hints đầy đủ, style giống hệt các file đính kèm. Không placeholder, không `pass` chỗ dở. |
| **R7** | Ghi file **UTF-8**. Giữ nguyên các ký tự Unicode trong docstring tiếng Việt. |
| **R8** | Viết **đầy đủ nội dung cả file** khi gọi tool ghi file (không cắt bớt, không `... (giữ nguyên phần cũ)`). |

---

## 3. INTERFACE BẮT BUỘC (chép đúng tên, đúng tham số)

### 3.1 `src/config.py` — CHỈ THÊM (không đổi gì đang có)

Thêm dataclass mới **ngay trước** `@dataclass class PipelineConfig`:

```python
@dataclass
class FrameExtractionConfig:
    """Tham số cho Module 0 — Trích & lọc frame từ video (spec v2, Module 0).

    Đây là các "điểm khởi điểm cần tinh chỉnh", không phải số cố định.
    """

    #: Bước nhảy frame. 1 = lấy mọi frame; 5 = ~6 frame/giây với video 30fps.
    frame_interval: int = 5
    #: Ngưỡng độ sáng trung bình tối thiểu (thang 0-255).
    brightness_threshold: float = 50.0
    #: Ngưỡng độ mờ tối thiểu (variance of Laplacian).
    blur_threshold: float = 100.0
    #: Bật/tắt bộ lọc chất lượng. False = giữ mọi frame theo ``frame_interval``.
    enable_quality_filter: bool = True
    #: Giới hạn số frame giữ lại. 0 = không giới hạn.
    max_frames: int = 0
    #: Bỏ qua N frame đầu video (giai đoạn camera chưa ổn định).
    skip_start_frames: int = 0
    #: Nếu > 0, resize frame đã giữ về đúng chiều cao này. 0 = giữ nguyên.
    resize_height: int = 0
```

Rồi thêm vào `PipelineConfig` **đúng một field**, đặt cạnh các field `preprocess` / `ocr` / `postprocess`:

```python
    frame_extraction: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)
```

**Ràng buộc:** giữ nguyên mọi field và thứ tự field cũ; `PipelineConfig.to_dict()` (dùng `asdict`) vẫn phải chạy được và tự động bao gồm `frame_extraction`.

### 3.2 `src/frame_extractor.py` — FILE MỚI (Module 0)

Docstring module phải nêu rõ: đọc video bằng `cv2.VideoCapture`, xét mỗi `frame_interval` frame, giữ frame đủ sáng & đủ nét, lưu lại **chỉ số frame gốc** để các module sau khớp lại thời gian xe xuất hiện.

Phải export **đúng** các tên sau:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .config import FrameExtractionConfig
from .image_utils import mean_brightness, variance_of_laplacian   # BẮT BUỘC tái dùng, KHÔNG viết lại


@dataclass
class FrameRecord:
    """Một frame đã vượt qua bộ lọc của Module 0."""

    frame_index: int      # chỉ số frame GỐC trong video (0-based)
    frame: np.ndarray     # ảnh BGR đã giữ
    timestamp: float      # giây = frame_index / fps
    brightness: float     # mean_brightness của frame
    blur_score: float     # variance_of_laplacian của frame

    def to_dict(self) -> Dict[str, Any]:   # KHÔNG chứa ảnh (không JSON-serializable)


@dataclass
class FrameExtractionStats:
    """Thống kê quá trình trích frame (để in log và ghi manifest JSON)."""

    total_frames: int = 0        # tổng số frame đọc được từ video
    scanned_frames: int = 0      # số frame đã đưa vào xét (sau skip_start + frame_interval)
    kept_frames: int = 0         # số frame giữ lại
    skipped_dark: int = 0        # bỏ vì quá tối
    skipped_blurry: int = 0      # bỏ vì quá mờ
    skipped_by_limit: int = 0    # bỏ vì vượt max_frames
    fps: float = 0.0
    duration: float = 0.0        # giây
    width: int = 0
    height: int = 0

    def to_dict(self) -> Dict[str, Any]


@dataclass
class FrameExtractionResult:
    """Kết quả đầy đủ của Module 0 cho 1 video."""

    video_path: str
    frames: List[FrameRecord] = field(default_factory=list)
    stats: FrameExtractionStats = field(default_factory=FrameExtractionStats)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]   # chỉ meta + stats + danh sách FrameRecord.to_dict(), KHÔNG chứa ảnh


def iter_valid_frames(
    video_path: str,
    config: Optional[FrameExtractionConfig] = None,
) -> Iterator[FrameRecord]:
    """Generator yield từng frame hợp lệ — không giữ toàn bộ video trong RAM."""


def extract_frames(
    video_path: str,
    config: Optional[FrameExtractionConfig] = None,
) -> FrameExtractionResult:
    """Chạy Module 0 trên 1 video và trả kết quả đầy đủ."""


class FrameExtractor:
    """Bọc Module 0 thành object để tái sử dụng / nối pipeline về sau."""

    def __init__(self, config: Optional[FrameExtractionConfig] = None) -> None: ...

    @property
    def config(self) -> FrameExtractionConfig: ...

    @property
    def last_stats(self) -> Optional[FrameExtractionStats]:
        """Stats của lần :meth:`extract` gần nhất (None nếu chưa chạy)."""

    def iter_frames(self, video_path: str) -> Iterator[FrameRecord]: ...

    def extract(self, video_path: str) -> FrameExtractionResult: ...
```

**Hành vi bắt buộc:**

* `frame_index` phải là **chỉ số frame gốc** trong video (tự đếm khi đọc, không dùng index của vòng lặp sau khi nhảy bước).
* `timestamp = frame_index / fps`; nếu `fps <= 0` (OpenCV trả 0) → dùng `25.0` và thêm 1 cảnh báo vào `warnings`.
* Thứ tự lọc trong 1 frame: `skip_start_frames` → `frame_interval` → `resize_height` → `max_frames` → `enable_quality_filter` (brightness trước, blur sau) → giữ.
* Frame đọc lỗi (`ret is False` hoặc `frame is None`) → bỏ qua, không crash, vẫn tăng `total_frames`.
* `frames` **không bao giờ chứa frame None/rỗng**; `blur_score`/`brightness` tính trên ảnh gốc (trước resize).
* `video_path` không tồn tại → `FileNotFoundError` với thông điệp tiếng Việt rõ ràng. Mở được nhưng không đọc nổi frame nào → `ValueError` (thông điệp tiếng Việt).
* **Luôn** `cap.release()` trong `finally` (cả trong generator, kể cả khi bị `break` giữa vòng lặp).
* `resize_height > 0` → dùng `src.image_utils.resize_by_height` nếu phù hợp (hoặc `cv2.resize` + `INTER_AREA`/`INTER_CUBIC`), giữ aspect ratio.
* `enable_quality_filter=False` → bỏ qua điều kiện brightness/blur, giữ frame theo `frame_interval` (dùng cho chế độ "ưu tiên độ đầy đủ hơn tốc độ" mà spec mô tả).

### 3.3 `tools/make_sample_video.py` — FILE MỚI (sinh video test, KHÔNG cần dữ liệu thật)

Mục đích: tạo được video `.mp4` giả lập để test Module 0, trong đó **cố ý** chèn frame tối / frame mờ để chứng minh bộ lọc hoạt động.

```python
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sample_data import (  # noqa: E402
    add_degradation,
    make_single_line_plate,
    make_two_line_plate,
)


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
    """


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: python tools/make_sample_video.py --output data/samples/test_video.mp4"""
```

CLI dùng `argparse` với `formatter_class=argparse.ArgumentDefaultsHelpFormatter` (giống `tools/make_sample_plate.py`):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--output` | `data/samples/test_video.mp4` | File video ghi ra |
| `--frames` | `150` | Tổng số frame |
| `--fps` | `30` | FPS của video |
| `--width` / `--height` | `960` / `540` | Kích thước khung |
| `--dark-frames` | `30,31,32,100` | Danh sách chỉ số frame làm tối |
| `--blur-frames` | `60,61,62,120` | Danh sách chỉ số frame làm mờ |
| `--seed` | `0` | Seed để tái lập |

**Ràng buộc sinh video:**

* Nền "đường" sáng (xám ~120-160) có vài vạch kẻ trắng; **1 ô tô** (rectangle) chạy ngang trái→phải, vị trí `x` là hàm tuyến tính của chỉ số frame, dán `make_single_line_plate("51F", "12345")` đã resize lên thân xe (dùng `image_utils.resize_by_height`).
* **1 xe máy** đứng yên ở góc, dán `make_two_line_plate("29-B1", "12345")` đã resize.
* Frame tối: đưa **toàn bộ khung** về vùng tối (ví dụ `add_degradation(frame, brightness=-110)`) sao cho `mean_brightness(frame) < 40`.
* Frame mờ: `add_degradation(frame, blur=25)` (hoặc `cv2.GaussianBlur` kernel ~51-61) sao cho `variance_of_laplacian(frame) < 60`.
* Ghi video bằng `cv2.VideoWriter` + codec `mp4v` (`cv2.VideoWriter_fourcc(*"mp4v")`).
* `writer.isOpened()` là `False` → `raise RuntimeError` với thông điệp tiếng Việt rõ ràng (gợi ý codec / `opencv-python`).
* Nếu đường dẫn chứa ký tự Unicode: ghi ra file tạm ASCII cùng thư mục rồi `os.replace` (vì `cv2.VideoWriter` không hỗ trợ đường dẫn Unicode trên Windows) — cùng tinh thần với `imwrite_unicode`.
* Tạo thư mục cha nếu chưa có; `writer.release()` trong `finally`; in `[i] Đã ghi video: {output_path} ({total_frames} frame @ {fps} fps)`; trả về `0` khi thành công.

### 3.4 `tools/self_check.py` — CHỈ THÊM test cho Module 0

Thêm đúng **một hàm** `test_frame_extractor()` và **một dòng gọi** trong `main()` (đặt trước `test_preprocess_helpers()`), dùng lại nguyên `check()` và biến `FAILURES` đang có. **Không được sửa logic các test cũ.**

```python
def test_frame_extractor() -> None:
    """Kiểm tra Module 0: trích frame, lọc tối/mờ, giới hạn số lượng, chỉ số gốc."""
```

Nội dung test bắt buộc (dùng `tempfile.TemporaryDirectory()` cho file tạm, tự dọn sạch, không để rác trong `data/`):

1. Sinh video test bằng API `build_synthetic_video` (import từ `tools.make_sample_video`) vào thư mục tạm: `total_frames=60`, `fps=30`, `dark_frames=(10, 11, 40)`, `blur_frames=(20, 21, 50)`.
2. `result = extract_frames(path, FrameExtractionConfig(frame_interval=5))` rồi:
   * `check("tổng frame đọc được", result.stats.total_frames, 60)`
   * `check("có giữ được frame", result.stats.kept_frames > 0, True)`
   * `check("bỏ được frame tối", result.stats.skipped_dark > 0, True)`
   * `check("bỏ được frame mờ", result.stats.skipped_blurry > 0, True)`
   * `check("fps đọc đúng", round(result.stats.fps), 30)`
3. `check("mọi frame đều là ảnh hợp lệ", all(f.frame is not None and f.frame.size > 0 for f in result.frames), True)`
4. `check("mọi frame đều đủ sáng", all(f.brightness >= 50.0 for f in result.frames), True)`
5. `check("mọi frame đều đủ nét", all(f.blur_score >= 100.0 for f in result.frames), True)`
6. `check("frame_index là bội của frame_interval", all(f.frame_index % 5 == 0 for f in result.frames), True)`
7. `check("frame_index tăng dần", [f.frame_index for f in result.frames] == sorted(f.frame_index for f in result.frames), True)`
8. `check("timestamp = frame_index / fps", all(abs(f.timestamp - f.frame_index / 30.0) < 1e-6 for f in result.frames), True)`
9. `check("các frame tối/mờ bị loại", {f.frame_index for f in result.frames} & {10, 11, 40, 20, 21, 50}, set())`
10. `check("max_frames cắt đúng", len(extract_frames(path, FrameExtractionConfig(frame_interval=5, max_frames=3)).frames), 3)`
11. `check("tắt bộ lọc chất lượng thì giữ nhiều hơn", len(extract_frames(path, FrameExtractionConfig(frame_interval=5, enable_quality_filter=False)).frames) > result.stats.kept_frames, True)`
12. `check("file không tồn tại -> FileNotFoundError", isinstance(exc, FileNotFoundError), True)` (dùng `try/except Exception as exc`).
13. `check("iter_valid_frames khớp extract_frames", len(list(iter_valid_frames(path, FrameExtractionConfig(frame_interval=5)))), result.stats.kept_frames)`

**Lưu ý codec:** nếu môi trường không có codec `mp4v`, `build_synthetic_video` sẽ `raise RuntimeError`; test phải bọc `try/except RuntimeError` → in `[SKIP]` (giống pattern `[SKIP]` đã dùng ở `test_paddle_parse_compat`) rồi `return`, **KHÔNG** tính là FAIL.

### 3.5 `main.py` — thêm nhánh `--video`

* Import thêm (gộp vào các import `src.*` đang có): `from src.config import FrameExtractionConfig` và `from src.frame_extractor import FrameExtractor`.
* Thêm tham số CLI, đặt cạnh `--image`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--video PATH` | `None` | Chạy **chỉ Module 0** trên video (không OCR) |
| `--frame-interval` | `5` | Bước nhảy frame |
| `--brightness-threshold` | `50.0` | Ngưỡng sáng tối thiểu |
| `--blur-threshold` | `100.0` | Ngưỡng nét tối thiểu |
| `--max-frames` | `0` | Giới hạn số frame giữ lại (0 = không giới hạn) |
| `--skip-start-frames` | `0` | Bỏ N frame đầu |
| `--save-frames` | off | Lưu frame giữ lại vào `<output_dir>/frames/frame_<index:06d>.png` (dùng `imwrite_unicode`) |
| `--no-quality-filter` | off | Tắt bộ lọc sáng/nét |

* Trong `build_config()`: gán `config.frame_extraction = FrameExtractionConfig(...)` từ các tham số trên.
* Trong `main()`:
  * Điều kiện báo lỗi đổi thành `if not args.image and not args.demo and not args.video:` → in `[!] Cần truyền --image, --video hoặc --demo.` và `return 2`.
  * Nhánh `--video` **chạy ĐỘC LẬP**: **không** gọi `build_reader(config)` cho nhánh này (để không cần cài PaddleOCR/EasyOCR). Tức là khối `try: reader = build_reader(config)` phải được đặt SAU khi đã xử lý xong `--video`.
  * In log: `[i] Video: <path>` / `[i] Tổng frame: X | đã xét: Y | giữ lại: Z` / `[i] Bỏ do tối: A | do mờ: B | do giới hạn: C` / `[i] fps: .., thời lượng: .. s, kích thước: WxH` / `[i] Thời điểm giữ frame (s): 0.17, 1.33, ...` (in tối đa 20 mốc đầu rồi `...`).
  * Ghi manifest JSON (mặc định `<output_dir>/frames_manifest.json`):
    `{"module": "Module 0 - Frame Extraction", "video": path, "config": asdict(config.frame_extraction), "stats": stats.to_dict(), "frames": [f.to_dict() for f in frames]}` với `ensure_ascii=False, indent=2`.
* Docstring đầu file: thêm 2 ví dụ
  `python tools/make_sample_video.py --output data/samples/test_video.mp4`
  `python main.py --video data/samples/test_video.mp4 --frame-interval 5 --save-frames`
  và ghi rõ: *"Module 0 chạy độc lập cho video; các module 1/2/3 (detect + tracking) sẽ nối ở bước sau."*
* **Không được làm hỏng** `--image` / `--demo`: khi không truyền `--video` thì luồng và output y như cũ (`run_single`, `save_outputs`, `result.json` giữ nguyên chữ ký).

### 3.6 `README.md` — CHỈ THÊM (không xóa/đổi nội dung cũ)

Thêm mục **"Module 0 — Frame Extraction & Filtering (trích frame từ video)"** gồm:
* Vị trí Module 0 trong pipeline và quan hệ với các module đã xong.
* Cách chạy 2 lệnh ở mục 3.5 + ý nghĩa `frames_manifest.json` và thư mục `frames/`.
* Bảng tham số `FrameExtractionConfig` (tên, mặc định, ý nghĩa) + ghi chú "đây là điểm khởi đầu cần tinh chỉnh trên video thật".
* Ghi chú: lần này **chưa** nối Module 1/2/3 nên chưa nhận diện được biển số từ video.

---

## 4. Danh sách file (đúng và đủ — KHÔNG tạo file nào khác)

| # | File | Hành động |
|---|---|---|
| 1 | `src/frame_extractor.py` | **TẠO MỚI** — toàn bộ Module 0 |
| 2 | `src/config.py` | **SỬA** — thêm `FrameExtractionConfig` + 1 field `frame_extraction` trong `PipelineConfig` |
| 3 | `tools/make_sample_video.py` | **TẠO MỚI** — sinh video test |
| 4 | `tools/self_check.py` | **SỬA** — thêm `test_frame_extractor()` |
| 5 | `main.py` | **SỬA** — thêm nhánh `--video` |
| 6 | `README.md` | **SỬA** — thêm tài liệu Module 0 |

* KHÔNG sửa `requirements.txt` (không thêm thư viện nào).
* KHÔNG tạo thư mục `tests/`, KHÔNG tạo file `.md` mới ngoài `README.md`.
* KHÔNG tạo file `demo`, `example`, `quickstart`, `*_v2.py`, `*_new.py`, `*_fixed.py`...

---

## 5. ĐỊNH NGHĨA HOÀN THÀNH (DoD — tự kiểm tra trước khi kết thúc)

1. Đúng **6 file** ở mục 4, đúng đường dẫn, code **đầy đủ** (không placeholder, không `pass` chỗ dở, không `TODO`).
2. `src/frame_extractor.py` export **đúng** các tên ở mục 3.2 với chữ ký y hệt.
3. Chỉ import: thư viện chuẩn, `numpy`, `cv2`. **Không** có `ultralytics`, `paddleocr`, `easyocr`, `torch`, `yaml`, `sortedcontainers`.
4. `frame_index` là chỉ số frame gốc; `timestamp = frame_index / fps`; `cap.release()` nằm trong `finally`.
5. `tools/self_check.py` có `test_frame_extractor()` và **không** làm hỏng các test Module 4/5/6/PlateReader đang có (không sửa `check()`, `FAILURES`, `main()` ngoài việc thêm 1 dòng gọi test mới).
6. `main.py --image ...` và `main.py --demo` giữ nguyên hành vi; `main.py --video ...` chạy **không cần** OCR.
7. Không có file rác / file phiên bản / file tạm bị bỏ lại trong repo.
8. Báo cáo cuối cùng (ngắn, dạng gạch đầu dòng):
   * File đã tạo/sửa (kèm mục đích 1 dòng).
   * API đã export của Module 0.
   * Giá trị mặc định của các ngưỡng.
   * Phần nào bạn **không kiểm chứng được** vì không được phép chạy code (nói thẳng, đừng giả vờ đã kiểm thử).

---

## 6. PHỤ LỤC — API ĐANG CÓ (dùng lại, KHÔNG viết lại)

### 6.1 `src/image_utils.py` — đã có sẵn, **bắt buộc dùng lại**

```python
def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray        # raise FileNotFoundError/ValueError
def imwrite_unicode(path: str, image: np.ndarray) -> None                          # tự tạo thư mục cha
def to_gray(image: np.ndarray) -> np.ndarray
def ensure_bgr(image: np.ndarray) -> np.ndarray
def ensure_uint8(image: np.ndarray) -> np.ndarray
def binarize_otsu(image: np.ndarray, invert: bool = False) -> np.ndarray
def clip_bbox(bbox, width: int, height: int) -> Tuple[int, int, int, int]
def expand_bbox(bbox, margin: float, width: int, height: int) -> Tuple[int, int, int, int]
def crop_bbox(image: np.ndarray, bbox) -> np.ndarray
def rotate_image(image: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray
def trim_uniform_borders(image: np.ndarray, tolerance: int = 12) -> np.ndarray
def variance_of_laplacian(image: np.ndarray) -> float      # <-- Module 0 dùng cho blur_score
def mean_brightness(image: np.ndarray) -> float            # <-- Module 0 dùng cho brightness
def resize_by_height(image: np.ndarray, target_height: int) -> np.ndarray   # giữ aspect ratio (INTER_CUBIC/AREA)
def optional_float(value) -> Optional[float]
```

### 6.2 `src/config.py` — cấu trúc hiện tại (chỉ được THÊM, không đổi)

```python
DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def filter_charset(text: str, charset: str = DEFAULT_CHARSET, keep_separator: bool = False) -> str

@dataclass class PreprocessConfig: ...   # target_height=48, target_width=320, ...
@dataclass class OCRConfig: ...          # engine="auto", lang="en", use_gpu=False, ...
@dataclass class PostprocessConfig: ...  # min_confidence=0.35, min_readings=2, ...

@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    detection_model: Optional[str] = None
    detection_conf: float = 0.25
    detection_device: Optional[str] = None
    detection_imgsz: int = 640
    output_dir: str = "results"

    def to_dict(self) -> Dict[str, Any]:   # return asdict(self)
```

### 6.3 `src/sample_data.py` — dùng để sinh video test

```python
def make_single_line_plate(serial="51F", numbers="12345", width=470, height=110) -> np.ndarray
def make_two_line_plate(serial="29-B1", numbers="12345", width=260, height=180) -> np.ndarray
def add_degradation(image, angle=0.0, blur=0, noise=0.0, brightness=0, seed=None) -> np.ndarray
def make_demo_images() -> dict[str, np.ndarray]
```

### 6.4 `tools/self_check.py` — khung hiện tại

```python
FAILURES = []
def check(description: str, actual, expected) -> None: ...      # in "[OK  ]"/"[FAIL]"
def test_normalize() -> None: ...
def test_validate() -> None: ...
def test_format() -> None: ...
def test_voting() -> None: ...
def test_postprocessor() -> None: ...
def test_interpolate() -> None: ...
def test_preprocess_helpers() -> None: ...
def test_plate_reader_joined_text() -> None: ...
def test_paddle_parse_compat() -> None: ...   # có pattern: print("[SKIP] ...") rồi return
def main() -> int: ...                        # gọi lần lượt các test, in "[KẾT QUẢ] Tất cả kiểm tra ĐỀU ĐẠT."
if __name__ == "__main__":
    raise SystemExit(main())
```

### 6.5 `main.py` — khung hiện tại (cần thêm nhánh `--video`)

```python
def parse_args(argv=None) -> argparse.Namespace          # đang có --image, --bbox, --detector, --engine, --output, --output-dir, --demo, ...
def build_config(args) -> PipelineConfig
def build_reader(config) -> PlateReader
def parse_bbox(value) -> Optional[List[int]]
def save_outputs(reader, image, reading, config, stem, save_crop=False) -> dict
def run_single(reader, image, stem, config, bbox=None, save_crop=False) -> dict
def main(argv=None) -> int
```

### 6.6 `src/plate_reader.py`, `src/postprocess.py`, `src/preprocess_plate.py`, `src/ocr.py`

**Không được sửa** các file này (chi tiết API không cần cho Module 0).

---

## 7. NHẮC LẠI LẦN CUỐI (3 câu quan trọng nhất)

1. **KHÔNG chạy code, KHÔNG tạo venv, KHÔNG `pip install`.** Bạn chỉ có tool đọc/ghi file trong phiên này. Muốn "kiểm thử" thì **đọc lại code và tự suy luận**.
2. **Chỉ 1 vòng, chỉ 6 file ở mục 4, không tạo bản `v2`/`_fixed`.** Ghi file 1 lần cho đủ và đúng; nếu phát hiện lỗi thì sửa dứt điểm rồi DỪNG.
3. **Không đổi chữ ký API đã có.** Module 0 chỉ *thêm* file mới và *thêm* vào các file cũ, không phá vỡ Module 4/5/6 đã chạy đúng.
