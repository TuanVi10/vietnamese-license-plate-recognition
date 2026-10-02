

---

## session_e9d3d5c5-779a-4650-aac9-d7de4274d8cd (session_e9d3d5c5-779a-4650-aac9-d7de4274d8cd)
*2026-09-26 08:16:11*

## Muc tieu

Phiên làm việc được giao **triển khai đúng MỘT module duy nhất: Module 0 — Frame Extraction & Filtering** (trích & lọc frame từ video) cho hệ thống ALPR nhận diện biển số xe Việt Nam theo *SPEC KỸ THUẬT v2*.

Module 0 là mắt xích **đầu tiên** của pipeline video:

```
video.mp4 → [Module 0]  ← phiên này
          → Module 1 (Vehicle Detection)
          → Module 2 (Plate Detection) → Module 3 (Tracking)
          → Module 4 (tiền xử lý) → Module 5 (OCR)
          → Module 6 (hậu xử lý + voting) → Module 7 (output)
```

Yêu cầu hành vi cốt lõi:
- Đọc video bằng `cv2.VideoCapture`, xét **mỗi `frame_interval` frame**, giữ lại frame **đủ sáng** (`mean_brightness`) và **đủ nét** (`variance_of_laplacian`).
- **Lưu chỉ số frame GỐC** (`frame_index`) để các module sau khớp lại thời điểm xe xuất hiện; `timestamp = frame_index / fps`.

Sản phẩm gồm **đúng 6 file** (2 tạo mới, 4 sửa): `src/frame_extractor.py` (mới), `src/config.py` (sửa), `tools/make_sample_video.py` (mới), `tools/self_check.py` (sửa), `main.py` (sửa), `README.md` (sửa).

Ràng buộc bắt buộc (vi phạm là hỏng việc):
- **R1:** TUYỆT ĐỐI không chạy code / không tạo venv / không `pip install` — chỉ có tool đọc/ghi/liệt kê file.
- **R2:** Chỉ dùng thư viện chuẩn Python + `numpy` + `cv2`; không thêm dependency, không sửa `requirements.txt`.
- **R3:** Chỉ 1 vòng làm việc; **R4:** không tạo bản `_v2`/`_fixed`; **R5:** không sửa API của các module đã xong (4/5/6/PlateReader); **R6:** docstring/comment tiếng Việt + type hints; **R7:** UTF-8; **R8:** ghi đầy đủ nội dung file.

## Van de gap phai & cach giai quyet

| # | Vấn đề | Phát hiện bởi | Cách giải quyết | Kết quả |
|---|---|---|---|---|
| 1 | `cv2.VideoCapture` không mở được đường dẫn Unicode trên Windows (mã **A1**, HIGH) | Review findings (Programmer xử lý) | Thêm `_path_is_ascii()` + `_open_capture()`: thử mở trực tiếp; nếu lỗi **và** đường dẫn non-ASCII thì copy video sang file tạm ASCII rồi mở; `_iter_frames` release capture **và** xóa file tạm trong `finally` | Module 0 an toàn với đường dẫn Unicode, nhất quán với `imread_unicode` của `image_utils` |
| 2 | OpenCV có thể trả `NaN`/`inf` cho `CAP_PROP_FPS` / `FRAME_COUNT` / `WIDTH` / `HEIGHT` (mã **A2**) | Review findings (Programmer) | Thêm `_finite_float()`; `NaN`/`inf`/lỗi → dùng FPS dự phòng `25.0` + thêm cảnh báo | Không lọt `NaN` vào `timestamp`; không ném `ValueError` ở `int(NaN)` |
| 3 | Vòng lặp đọc có thể chạy vô hạn khi codec trả `(ret=True, frame=None)` liên tục; `total_frames` bị tăng khống ở mốc EOF (mã **A3**) | Review findings (Programmer) | Dừng ở lần đọc lỗi **đầu tiên**; frame `(True, None)` tăng `consecutive_failures`, giới hạn bởi `_MAX_CONSECUTIVE_READ_FAILURES = 10` | `total_frames` = số frame thực đọc; chặn được lặp vô hạn |
| 4 | Vẫn giải mã toàn bộ phần còn lại của video dù đã đủ `max_frames` (mã **A4**) | Review findings (Programmer) | `break` sớm khi đạt giới hạn (vẫn đếm vào `skipped_by_limit`) | Hiệu năng tốt hơn khi dùng `--max-frames` |
| 5 | `FileNotFoundError` bị trễ tới lần `next()` đầu tiên của generator (mã **A5**) | Review findings (Programmer) | Kiểm tra `os.path.isfile` **ngay khi gọi hàm**, rồi trả generator lười nội bộ | Lỗi nổi lên đúng thời điểm gọi, vẫn streaming lazy (không nạp cả video) |
| 6 | Fallback Unicode của `make_sample_video` **vô hiệu** vì `tempfile.gettempdir()` cũng có thể chứa Unicode (mã **B6**, HIGH) | Review findings (Programmer) | Thêm `_ascii_temp_dir()`: thử lần lượt `gettempdir` → `TEMP` → `TMP` → `%SystemRoot%\Temp` → `/tmp`, chọn thư mục ASCII đầu tiên | Nhánh fallback thực sự hoạt động |
| 7 | Codec `mp4v` (lossy) có thể kéo độ nét frame "sạch" xuống dưới ngưỡng blur sau round-trip (mã **B7**) | Review findings (Programmer) | Tăng σ nhiễu sinh frame 6→12 và thêm 2 vạch kẻ đường tương phản cao | Frame "sạch" giữ `variance_of_laplacian` > 100 sau mã hóa (lý luận tĩnh, chưa chạy) |
| 8 | `main()` của `make_sample_video` chỉ bắt `RuntimeError` (mã **B8**) | Review findings (Programmer) | Bắt thêm `ValueError`, `OSError` | CLI báo lỗi thân thiện hơn |
| 9 | `test_frame_extractor` dùng `except Exception` che mất lỗi import Module 0 thật (mã **C1**) | Review findings (Programmer) | Thu hẹp về `except ImportError`; chuyển `import tempfile` lên cấp module | Import hỏng của Module 0 → FAIL rõ ràng; chỉ thiếu `cv2`/`numpy` mới `[SKIP]` |
| 10 | Công cụ sửa file theo dòng (`apply_text_edits`) làm hỏng newline: file từ 346 → 702 dòng với dòng trống xen kẽ; `tools/self_check.py` bị ghi bằng CRLF | Programmer (tự phát hiện) | Viết lại **toàn bộ** `src/frame_extractor.py` bằng `save_file` (giữ encoding `utf-8-sig`) | File sạch trở lại; ghi nhận `self_check.py` còn CRLF (chỉ ảnh hưởng hiển thị, Python đọc bình thường) |

**Lưu ý về kiểm chứng (do agent báo cáo trung thực):** Vì bị **cấm chạy code**, mọi thay đổi chỉ được suy luận tĩnh. Agent **không thể xác nhận runtime**: `total_frames == 60`, `fps == 30`, việc frame tối/mờ thực sự rơi dưới ngưỡng 50/100 sau mã hóa `mp4v`, hay frame sạch giữ được ≥ 100.

## Quyet dinh quan trong

- **Tái dùng `src/image_utils.py` thay vì viết lại**: dùng `mean_brightness`, `variance_of_laplacian`, `resize_by_height` (đúng yêu cầu bắt buộc tái dùng của brief) để đo chất lượng trên ảnh GỐC trước resize.
- **Chỉ THÊM vào `config.py`**: thêm dataclass mới `FrameExtractionConfig` và **đúng một field** `frame_extraction` trong `PipelineConfig`, giữ nguyên mọi field và thứ tự cũ để `to_dict()` qua `asdict` vẫn chạy và tự bao gồm config mới.
- **`frame_index` = chỉ số frame GỐC** (tự đếm khi đọc, không dùng index sau khi nhảy bước) và `timestamp = frame_index / fps` — để các module detect/tracking/OCR sau này khớp đúng thời điểm xe xuất hiện.
- **Thiết kế generator lười (lazy)**: `iter_valid_frames` / `_iter_frames` yield từng frame, không giữ toàn bộ video trong RAM; `cap.release()` luôn nằm trong `finally` (kể cả khi bị `break` giữa vòng lặp).
- **Module 0 chạy ĐỘC LẬP trong `main.py --video`**: nhánh `--video` được xử lý **TRƯỚC** khi khởi tạo `reader`, để không cần cài PaddleOCR/EasyOCR; `--image` / `--demo` giữ nguyên hành vi và output.
- **FPS dự phòng = 25.0** khi OpenCV trả 0/NaN, kèm cảnh báo ghi vào `warnings` (thay vì crash).
- **Ngưỡng mặc định coi là "điểm khởi đầu cần tinh chỉnh"**: `frame_interval=5`, `brightness_threshold=50.0`, `blur_threshold=100.0`, `enable_quality_filter=True`, `max_frames=0`, `skip_start_frames=0`, `resize_height=0`.
- **Thứ tự lọc trong 1 frame** được chốt: `skip_start_frames` → `frame_interval` → `resize_height` → `max_frames` → `enable_quality_filter` (brightness trước, blur sau).
- **Xử lý Unicode cho video** bằng cơ chế file tạm ASCII, mô phỏng theo tinh thần `imread_unicode` / `imwrite_unicode`.
- **`tools/self_check.py` chỉ được THÊM** `test_frame_extractor()` + 1 dòng gọi trong `main()`; không sửa `check()`, `FAILURES` hay các test cũ.

## Bai hoc rut ra

- **Đường dẫn Unicode trên Windows là bẫy xuyên suốt**: không chỉ `cv2.imread`/`cv2.imwrite` mà cả `cv2.VideoCapture` và `cv2.VideoWriter`. Nên có một cơ chế dùng chung cho mọi thao tác I/O.
- **Metadata từ OpenCV không đáng tin**: FPS/FRAME_COUNT/WIDTH/HEIGHT có thể trả 0, `NaN` hoặc `inf` → luôn chuẩn hóa (`_finite_float`) trước khi tính toán, nếu không `NaN` sẽ lan vào kết quả và `int(NaN)` sẽ ném lỗi.
- **Vòng lặp đọc video phải luôn có "van an toàn"**: giới hạn số lần lỗi liên tiếp để phá vòng lặp vô hạn, và luôn release tài nguyên trong `finally`.
- **Dữ liệu test sinh tự động có thể mâu thuẫn với ngưỡng lọc**: codec lossy (`mp4v`) có thể kéo độ nét của frame "sạch" xuống dưới ngưỡng blur — cần sinh dữ liệu có **biên an toàn** (tăng nhiễu/vạch kẻ tương phản cao) để test không giòn.
- **Exception clause nên thu hẹp**: dùng `except ImportError` thay vì `except Exception` để không che mất lỗi thật (import hỏng phải FAIL rõ ràng, chỉ thiếu thư viện mới `[SKIP]`).
- **Công cụ sửa file theo dòng có thể làm hỏng newline/cấu trúc file**: khi thay đổi lớn, viết lại **toàn bộ file** để đảm bảo nội dung và encoding đúng (giữ `utf-8-sig`).
- **Không chạy được code thì phải kiểm tra tĩnh và báo cáo trung thực**: nói thẳng phần **không kiểm chứng được** thay vì giả vờ đã kiểm thử.
- **Ràng buộc chặt (đúng 6 file, không tạo bản `v2`, không đổi API cũ)** giúp giữ ổn định: các module 4/5/6 đã chạy đúng cần được bảo vệ khỏi thay đổi ngoài phạm vi.
- **Nguyên tắc "chỉ thêm, không sửa"** trong `config.py` / `main.py` / `README.md` bảo toàn `to_dict()`, `--image`, `--demo` — thay đổi mới không phá vỡ hành vi cũ.
- **Tận dụng cơ chế `[SKIP]` có sẵn** (như ở `test_paddle_parse_compat`) khi môi trường thiếu codec/thư viện, để bài self-check không tính là FAIL oan.
