# PROJECT_CODE_GUIDE.md — Hướng dẫn đọc & tiếp tục code project

> Mục tiêu: đọc xong là bạn có thể **tự đọc, hiểu, debug và viết tiếp** project "Nhận diện
> biển số xe Việt Nam qua video/ảnh". Không tóm tắt kiểu "file này xử lý database" — mỗi
> file đều có: **tồn tại để làm gì, nằm đâu trong kiến trúc, ai gọi nó, nó gọi ai,
> class/function nào, input→output, dữ liệu chảy thế nào, vì sao viết vậy, đổi thì ảnh
> hưởng gì, edge case + lỗi đáng chú ý.**

## MỤC LỤC
1. [Bản đồ thư mục & kiến trúc](#1-bản-đồ-thư-mục--kiến-trúc)
2. [FILE NÀO CHẠY ĐƯỢC + vai trò](#2-file-nào-chạy-được-entry-points--vai-trò)
3. [Core library `src/` — chi tiết từng file](#3-core-library-src--chi-tiết-từng-file)
4. [Tools `tools/` — chi tiết từng script](#4-tools-tools--chi-tiết-từng-script)
5. [Toàn bộ tham số Config](#5-toàn-bộ-tham-số-config)
6. [Cách chạy & tái tạo từ đầu](#6-cách-chạy--tái-tạo-từ-đầu)
7. [Cheat-sheet mọi "điểm đặc biệt"](#7-cheat-sheet-mọi-điểm-đặc-biệt)

---

## 1. Bản đồ thư mục & kiến trúc

```
S:\M_AGENT\CV\
├── main.py                  # ENTRY POINT: CLI chạy pipeline VIDEO (Module 0→7)
├── train_detector.py        # ENTRY POINT: fine-tune YOLO11n dò biển số
├── download_datasets.py     # ENTRY POINT: tải 3 dataset Roboflow (cần API key)
├── requirements.txt         # deps toàn pipeline (torch/ultralytics/paddle/...)
├── videodemo2.mp4           # video test thật 13 phút (bị gitignore)
├── models/                  # weights suy luận: yolo11n.pt + plate_detector.pt
├── datasets/                # data train/eval (dataset, ccpd_pose, corner_regression)
├── labeling/                # ground-truth gán nhãn tay (CSV + ảnh crop)
├── runs/                    # kết quả train/eval + weights checkpoint
├── docs/                    # tài liệu (file này + 20 file giải thích module)
└── dataset_prep_src/
    ├── src/                 # ★ 18 module CORE (thư viện, KHÔNG chạy trực tiếp)
    └── tools/               # ★ 41 script, gom 7 nhóm:
        ├── data/   (10)     #   tải/chuyển/nhãn dataset
        ├── train/  (4)      #   huấn luyện model
        ├── eval/   (13)     #   đánh giá & chẩn đoán
        ├── run/    (6)      #   chạy pipeline/demo + paddle_ocr_server
        ├── sample/ (2)      #   sinh ảnh/video mẫu
        ├── check/  (5)      #   self-check + smoke test + cài đặt
        └── utils/  (1)      #   geometry_utils (thư viện)
```

### Kiến trúc pipeline video (luồng dữ liệu)

```
VideoCapture ──► M0 FrameExtractor ──► M1 VehicleDetector ──► M3 SortTracker
   (frame_extractor.py)            (detect_vehicle.py)        (tracker.py)
                                                                  │ track_id
                                                                  ▼
                                        M2 PlateDetector (detector.py, tùy chọn)
                                        │  └ source="detection" / "fallback"
                                                                  ▼
                                   Crop biển → TopKBuffer (best_frame.py, KHÔNG OCR ngay)
                                                                  ▼
                                        ★ CornerRegressor (corner_regressor.py)
                                        │  |góc|>15° → warpPerspective(4 góc)
                                                                  ▼
                                   M4 Preprocess (preprocess_plate.py): deskew→tách dòng→CLAHE
                                                                  ▼
                                   M5 OCR (ocr.py / ocr_subprocess.py): PaddleOCR/EasyOCR
                                                                  ▼
                                   M6 Postprocess/Fusion (postprocess.py / fusion.py): vote
                                                                  ▼
                                   M7 Output (visualize.py + main.py): JSON/CSV/txt/video
```

**2 kiểu chạy:**
- **Ảnh tĩnh**: `plate_reader.py` nối M2→M4→M5→M6→M7 cho 1 ảnh. Voting = nhiều biến thể của cùng ảnh.
- **Video**: `video_pipeline.py` nối M0→M1→M3→(M2)→M4/5/6→M7. Voting = nhiều frame của cùng `track_id`.

---

## 2. FILE NÀO CHẠY ĐƯỢC (Entry Points) + vai trò

Quy ước: file **chạy được** = có `if __name__ == "__main__"` hoặc là script top-level.
File **thư viện** = chỉ được `import`, không tự chạy.

### 2.1 Ba file gốc (root)

| File | Vai trò | Chạy bằng |
|---|---|---|
| `main.py` | **Cổng chính.** CLI chạy toàn bộ pipeline video (M0→M7), xuất JSON/CSV/txt/crop/video | `python main.py --video <mp4> [--ocr-engine ...]` |
| `train_detector.py` | Fine-tune YOLO11n phát hiện biển số trên `datasets/dataset/`, chép `best.pt` → `models/plate_detector.pt` | `python train_detector.py --epochs 3 --batch 8` |
| `download_datasets.py` | Tải 3 dataset biển VN từ Roboflow (script top-level, chạy ngay khi import) | `python download_datasets.py` (cần `ROBOFLOW_API_KEY`) |

### 2.2 Bốn mươi script trong `tools/` (chạy `python tools/<nhóm>/<tên>.py`)

> Các nhóm A→F tương ứng thư mục con: A=`data/`, B=`train/`, C=`eval/`, D=`run/`,
> E=`sample/`, F=`check/`. Thư viện `geometry_utils.py` nằm ở `utils/`.

**A. Xây dataset (10):**

| Script | Vai trò |
|---|---|
| `download_ccpd.py` | Tải + giải nén CCPD2019 từ HuggingFace |
| `ccpd_to_pose.py` | CCPD2019 → YOLO-pose 4 keypoint (TL,TR,BR,BL), sinh `data_pose.yaml` + `manifest.csv` |
| `ccpd_warped_to_yolo.py` | CCPD-warped → YOLO (bbox) để train corner regressor |
| `gen_ccpd_warped.py` | Sinh ảnh "biển đã warp" bằng homography ngẫu nhiên (data huấn luyện corner) |
| `unify_labeling.py` | Chuẩn hoá 4 nguồn nhãn tay (3 chuẩn toạ độ) → 1 CSV pixel + `split.csv` chia theo track |
| `normalize_annguyen_labels.py` | Chuẩn hoá riêng nguồn nhãn annguyen |
| `extract_labeling_crops.py` | Trích crop biển số (không OCR) để gán nhãn Label Studio |
| `make_labeling_xlsx.py` | Tạo xlsx gán nhãn biển số từ `manifest.csv` |
| `qa_labeled.py` | Kiểm tra chất lượng `labeled.csv` |
| `qa_ccpd_pose.py` | QA dataset YOLO-pose (đếm ảnh/label, kiểm tra keypoint) |

**B. Train (4):**

| Script | Vai trò |
|---|---|
| `train_pose_ccpd.py` | Train YOLO-pose 4 keypoint trên CCPD (2 chế độ `baseline`/`augmented`) |
| `train_corner_regressor.py` | Train corner regressor (yolo11n-pose) trên CCPD-warped, imgsz=160 |
| `stop_at_epoch.py` | Callback dừng train ở epoch chỉ định |
| `verify_pose_augment.py` | Kiểm chứng augment keypoint của ultralytics (flip_idx có đúng) |

**C. Đánh giá & chẩn đoán (13):**

| Script | Vai trò |
|---|---|
| `eval_corner_regressor.py` | Đo mAP/IoU của corner regressor |
| `eval_corner_ocr.py` | Đo end-to-end corner → OCR |
| `eval_fusion.py` | So sánh vote cả chuỗi (cũ) vs fusion theo vị trí (mới) |
| `eval_compare.py` | So sánh model vs human label |
| `eval_lowconf.py` | Đánh giá ngưỡng confidence thấp |
| `eval_pose_bands.py` | Đánh giá pose theo từng dải góc nghiêng |
| `compare_model_vs_human.py` | So sánh model detect (Module 2) vs nhãn tay |
| `diagnose_pose_errors.py` | Chẩn đoán lỗi pose (đọc, không ghi) |
| `ceiling_test.py` | Ceiling test (đo giới hạn trên của pipeline) |
| `ceiling_test_paddle.py` | Ceiling test riêng PaddleOCR |
| `ceiling_test_paddle_align.py` | Ceiling test PaddleOCR + align góc |
| `analyze_angle_dist.py` | Thống kê phân bố góc nghiêng dataset |
| `viz_ccpd_pose.py` | Vẽ overlay 4 keypoint CCPD để kiểm tra mắt |

**D. Chạy / demo (6):**

| Script | Vai trò |
|---|---|
| `run_full.py` | **Chạy pipeline FULL video** `videodemo2.mp4` (CornerRegressor + PaddleOCR subprocess) → `runs/e2e_full/` |
| `test_e2e.py` | Test end-to-end nhanh (đoạn video ngắn) |
| `demo_plate_on_image.py` | Demo plate_detector + EasyOCR trên 1 ảnh thật |
| `make_demo_video.py` | Ghép ảnh biển thật thành video demo |
| `reencode_video.py` | Re-encode video (sửa codec) |
| `paddle_ocr_server.py` | Server PaddleOCR qua stdin/stdout (được `ocr_subprocess.py` spawn) |

**E. Sinh mẫu (2):**

| Script | Vai trò |
|---|---|
| `make_sample_plate.py` | Sinh ảnh biển VN giả lập |
| `make_sample_video.py` | Sinh video mẫu có xe + biển di chuyển |

**F. Self-check / cài đặt (5):**

| Script | Vai trò |
|---|---|
| `self_check.py` | Self-check toàn pipeline (ảnh tĩnh) |
| `self_check_tracking.py` | Self-check riêng phần tracking |
| `ocr_smoke.py` | Smoke test PaddleOCR khởi tạo + đọc 1 ảnh |
| `ocr_smoke_easyocr.py` | Smoke test EasyOCR |
| `install_paddle.py` | Cài PaddleOCR/PaddlePaddle vào venv đúng cách |

### 2.3 Thư viện (KHÔNG tự chạy, chỉ để import)

| File | Vai trò |
|---|---|
| `src/*.py` (18 file) | Core modules — xem mục 3 |
| `tools/utils/geometry_utils.py` | Sắp 4 góc + kiểm tra lồi + bỏ điểm trùng. **Import bởi `data/ccpd_to_pose.py` và `data/unify_labeling.py`** (KHÔNG nằm trong `src/`) |

---

## 3. Core library `src/` — chi tiết từng file

### 3.1 `src/__init__.py` — khai báo gói
- **Vai trò**: đánh dấu `src/` là package, đặt `__version__="1.1.0"` và `__all__`.
- **Điểm đáng chú ý**: `__all__` đang **thiếu 4 module mới** `best_frame`, `corner_regressor`,
  `fusion`, `ocr_subprocess` (thêm sau, chưa cập nhật). Docstring chỉ liệt kê Module 0-7 cũ.
  `import src.fusion` vẫn chạy bình thường; chỉ `from src import *` là không thấy 4 module này.

### 3.2 `src/config.py` — mọi ngưỡng/trọng số gom 1 chỗ
- **Vai trò**: toàn bộ cấu hình dạng `@dataclass`, không hard-code rải rác.
- **Hàm**: `filter_charset(text, charset, keep_separator)` — giữ ký tự thuộc charset, uppercase;
  `keep_separator=True` giữ thêm `-`, `.`, khoảng trắng.
- **Hằng**: `DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"`.
- **10 dataclass**: `PreprocessConfig`, `OCRConfig`, `PostprocessConfig`, `FrameExtractionConfig`,
  `VehicleDetectionConfig`, `TrackingConfig`, `VideoPipelineConfig`, `BestFrameConfig`,
  `FusionConfig`, `PipelineConfig` (giá trị mặc định xem mục 5).
- **`PipelineConfig`** là gốc: gom 9 config con + `detection_model/conf/device/imgsz` + `output_dir`.
  Có `to_dict()` (dùng `asdict`) để serialize JSON.
- **Ai gọi nó**: gần như MỌI module (`from .config import ...`). `main.py` tạo `PipelineConfig()` rồi
  override từng field theo CLI args.
- **Đổi thì ảnh hưởng**: đổi mặc định → ảnh hưởng toàn pipeline; đổi tên field → phải sửa mọi nơi dùng.

### 3.3 `src/image_utils.py` — tiện ích ảnh dùng chung
- **Vai trò**: I/O Unicode, chuyển màu, crop/xoay/resize, đo mờ/độ sáng. Dùng bởi M0 và M4.
- **Hàm chính**:
  - `imread_unicode` / `imwrite_unicode` — **điểm đặc biệt**: `cv2.imread/imwrite` KHÔNG hỗ trợ
    đường dẫn Unicode trên Windows → đọc bằng `np.fromfile`+`imdecode`, ghi bằng `imencode`+`tofile`.
  - `to_gray` / `ensure_bgr` / `ensure_uint8` — chuẩn hoá kênh (2/3/4 kênh).
  - `binarize_otsu(image, invert)` — nhị phân Otsu.
  - `clip_bbox` — kẹp bbox vào biên, đảm bảo w/h ≥ 1px.
  - `expand_bbox` / `crop_bbox`.
  - `rotate_image(image, angle, border_value=255)` — **điểm đặc biệt**: mở rộng canvas
    `new_w=h·sin+w·cos` để KHÔNG cắt góc; phần thêm lấp màu nền (255 = trắng).
  - `trim_uniform_borders` — cắt viền đồng màu do xoay sinh ra.
  - `variance_of_laplacian` (độ mờ), `mean_brightness` (độ sáng), `resize_by_height`.
- **Edge case**: mọi hàm nhận ảnh `None` đều raise `ValueError` sớm.

### 3.4 `src/frame_extractor.py` — Module 0 (trích & lọc frame)
- **Vai trò**: đọc video, xét mỗi `frame_interval` frame, giữ frame đủ sáng+nét, **lưu chỉ số frame
  GỐC** + timestamp để module sau khớp đúng thời điểm.
- **Chỉ phụ thuộc**: numpy + cv2 + image_utils (không cần ultralytics/paddle/torch).
- **Class/function**:
  - `FrameRecord`: `frame`, `frame_index` (gốc), `timestamp`.
  - `FrameExtractionStats` / `FrameExtractionResult`.
  - `_finite_float(value, default)` — **điểm đặc biệt**: OpenCV trả `NaN` cho `CAP_PROP_FPS`/
    `FRAME_COUNT` ở vài codec; chặn để `NaN` không lan vào timestamp (gây `ValueError` khi `int(NaN)`).
  - `_open_capture(path)` — **Unicode**: mở trực tiếp; nếu lỗi + path có ký tự ngoài ASCII thì copy
    video sang file tạm ASCII rồi mở file tạm.
  - `_iter_frames(...)` — generator lazy; phá vòng lặp khi đọc lỗi liên tiếp
    `_MAX_CONSECUTIVE_READ_FAILURES=10`.
  - `iter_valid_frames` / `extract_frames` / `FrameExtractor` — 3 mức API (generator/list/object).
- **Edge case**: `frame_interval=5` + 30fps ≈ 6 frame/giây. Frame bị lọc vẫn giữ `frame_index` gốc.

### 3.5 `src/detector.py` — Module 2 (dò biển số, ảnh tĩnh)
- **Vai trò**: bọc `ultralytics.YOLO` dò biển số trong 1 ảnh. Không truyền model → coi cả ảnh là biển
  (tiện khi đầu vào đã crop sẵn). `ultralytics` là phụ thuộc TÙY CHỌN.
- **Class**: `Detection` (bbox, confidence, class_id, label) + `PlateDetector`.
- **`detect(image)`**: `model.predict(conf, imgsz, verbose=False)` → lọc `boxes` → tạo `Detection` →
  sort giảm dần confidence. `best_bbox()` trả bbox tự tin nhất.
- **Lưu ý**: thiếu `ultralytics` thì `__init__` raise `RuntimeError` (không tự fallback — vì M2 trong
  video đã có fallback riêng ở `video_pipeline`).

### 3.6 `src/detect_vehicle.py` — Module 1 (dò phương tiện)
- **Vai trò**: dò bbox PHƯƠNG TIỆN (car/motorcycle/bus/truck) bằng YOLO. Bbox xe = vùng tìm kiếm M2
  và đầu vào M3.
- **Hằng**: `COCO_VEHICLE_CLASSES = {2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}` (KHÔNG bicycle).
  `FULL_FRAME_LABEL="vehicle"`.
- **`clip_bbox(bbox,w,h)`** — **KHÁC image_utils.clip_bbox**: chịu bbox méo (mảng `(1,4)`/`(4,1)`,
  thừa phần tử, chứa `NaN`/`inf`) bằng `np.nan_to_num` → làm sạch dữ liệu bẩn thay vì sập pipeline.
- **Chế độ "toàn khung" (`full_frame_fallback`)**: `model_path is None` → trả đúng 1 detection phủ
  toàn frame, để pipeline video vẫn chạy khi chưa có model.
- **`detect()`**: có model → `model.predict(conf, iou, imgsz, max_det, classes)`; `classes==()` thì
  KHÔNG lọc lớp (trả mọi lớp COCO). Sort giảm dần confidence.
- **Còn có**: `ultralytics_available()`, `build_vehicle_detector()`, `detect_vehicles()`, `best()`.
- **Ai gọi nó**: `video_pipeline.py` trong `process_frame`.

### 3.7 `src/tracker.py` — Module 3 (SORT: Kalman + Hungarian + IoU)
- **Vai trò**: gán `track_id` ổn định cho mỗi xe qua nhiều frame → gom nhiều lần đọc cùng xe để vote.
- **Thuật toán SORT**:
  1. Mỗi track giữ Kalman filter (vận tốc không đổi) trên state `[u, v, s, r, u', v', s']`
     (tâm, diện tích, tỉ lệ w/h + vận tốc tương ứng).
  2. `predict` ước lượng bbox hiện tại của mọi track.
  3. Ghép detection↔track bằng **IoU + Hungarian**.
  4. Track không cập nhật quá `max_age` frame bị xoá; đủ `min_hits` lần khớp liên tiếp mới "xác nhận".
- **Chỉ dùng numpy + math** (không cần scipy/filterpy/motpy).
- **Class/function**:
  - `box_iou`, `iou_matrix` (vector hoá numpy, tránh nghẽn O(n_det·n_trk) khi đông xe).
  - `KalmanBoxTracker` (giữ `hit_streak`, `time_since_update`, `age`, `confidence`).
  - `SortTracker.update(detections, frame_index, timestamp)` — predict → associate → update →
    tạo track mới → gom kết quả "confirmed" → xoá track chết.
  - `TrackedObject`, `Tracker = SortTracker` (alias), `track_detections()`.
- **Điểm đáng chú ý**: "confirmed" xét theo `hit_streak >= min_hits` (KHÔNG theo `frame_count`) —
  nếu không, mọi track mới ở frame đầu sẽ bị trả về dù chưa từng cập nhật.

### 3.8 `src/best_frame.py` — chọn top-K crop tốt nhất mỗi track
- **Vai trò**: trong pha thu thập, mỗi frame cắt crop biển, chấm điểm, đẩy vào `TopKBuffer` (KHÔNG
  OCR). Cuối track chỉ OCR `top-K` crop điểm cao nhất → tránh OCR frame mờ/nhỏ/nghiêng.
- **Điểm crop** = tổ hợp có trọng số (mọi thành phần chuẩn hoá về [0,1]):
  - sharpness (var of Laplacian, thang log trên crop resize theo chiều cao cố định)
  - height (chiều cao biển px), aspect (độ lệch w/h so với loại biển kỳ vọng)
  - confidence (detection), trừ exposure (phạt cháy sáng/quá tối).
  - Công thức: `w_sharp·sharp + w_height·height + w_aspect·aspect + w_conf·conf − w_exposure·exposure`.
- **Class**: `ScoreDetail`, `PlateCandidate` (crop + score + source + components), `TopKBuffer`.
- **`TopKBuffer`** — ràng buộc quan trọng:
  - Giữ tối đa `k` crop, khoảng cách frame ≥ `min_frame_gap`.
  - Xếp hạng theo CẶP `(is_detection, score)` — **fallback KHÔNG BAO GIỜ vượt detection thật** dù
    điểm cao hơn (vì `_rank_key` đặt `is_detection` trước `score`).
  - Khi detection thật xuất hiện → xoá hết crop fallback còn sót.
- **Ai gọi nó**: `video_pipeline.py` (thu thập crop + flush top-K cuối video).

### 3.9 `src/corner_regressor.py` — hồi quy 4 góc biển (yolo11n-pose)
- **Vai trò**: module MỚI (bước 4). Dùng `yolo11n-pose` (4 keypoint = TL, TR, BR, BL) để đo chính xác
  4 góc biển TRÊN CROP (đã biết chắc có biển). Kết hợp gate góc: `|angle|>15°` → warpPerspective(4 góc),
  ngược lại dùng bbox + affine deskew như cũ.
- **Class**: `CornerResult` (corners (4,2), conf, angle) + `CornerRegressor`.
- **`predict(crop)`**: `model.predict(conf, imgsz=160)` → lấy `result.keypoints.xy` → chọn instance có
  `confs.mean()` cao nhất → `angle = atan2(tr.y−tl.y, tr.x−tl.x)` (góc cạnh trên TL→TR).
- **Lưu ý**: `imgsz=160` (crop nhỏ, không cần 640). `result.keypoints.conf` có thể None → thay bằng
  mảng ones.

### 3.10 `src/preprocess_plate.py` — Module 4 (tiền xử lý biển)
- **Vai trò**: chuẩn hoá ảnh biển trước OCR. Thứ tự **bắt buộc** theo spec 3.4:
  1. Crop theo bbox (nếu có) → 2. Deskew (LUÔN trước tách dòng) → 3. Tách dòng (biển 2 dòng) + ghép
  ngang → 4. CLAHE tăng tương phản → 5. Resize giữ aspect ratio (padding, KHÔNG kéo dãn) → 6. Gate kích
  thước tối thiểu (quá nhỏ → `too_small`, KHÔNG đưa vào OCR).
- **Class**: `PreprocessResult` (image, lines, is_two_line, skew_angle, too_small, warnings) +
  `PlatePreprocessor`.
- **Hàm chính**:
  - `estimate_skew_angle(image, max_angle, step)` — **điểm đặc biệt**: projection profile — khi deskew
    đúng, histogram chiếu dọc có phương sai cao nhất (đỉnh/đáy ký tự thẳng hàng). Quét góc
    `[-max_angle, max_angle]`, chọn góc cho variance max, tinh chỉnh bước nhỏ + nội suy parabol.
  - `split_lines(image, gap_ratio)` — cắt 2 dòng theo thung lũng (gap) của projection ngang.
  - `merge_lines_horizontally` — ghép 2 dòng ngang (chèn khoảng trắng `merge_gap`).
  - `apply_clahe` — CLAHE tăng tương phản cục bộ (chống chói/tối một phần).
  - `resize_with_padding` — resize giữ aspect + padding trắng.
  - `_looks_two_line` — đoán 2 dòng: aspect < `two_line_min_aspect` VÀ tồn tại thung lũng giữa 2 dòng.
  - `iter_variants` — sinh nhiều biến thể (merged, merged_binary, từng line) để OCR nhiều lần rồi vote.
- **Ai gọi nó**: `plate_reader.py` (ảnh tĩnh) + `video_pipeline.py` (qua PlateReader).

### 3.11 `src/ocr.py` — Module 5 (OCR)
- **Vai trò**: engine chính PaddleOCR (PP-OCRv5), fallback EasyOCR. Hỗ trợ **cả API 2.x lẫn 3.x**.
- **Class**: `OCRItem` (text, confidence, box, raw_text, char_probs), `OCRResult`,
  `BaseOCREngine` (abstract), `PaddleOCREngine`, `EasyOCREngine`.
- **Hàm chính**:
  - `sort_reading_order(items)` — sắp item theo thứ tự đọc (trên→dưới, trái→phải).
  - `_box_array(box)` — chuẩn hoá box về `(N,2)`.
  - `create_ocr_engine(config, prefer)` — factory, tự fallback Paddle→Easy; raise `RuntimeError` nếu
    không engine nào khởi tạo được (kèm json chi tiết lỗi).
  - `available_engines()` — kiểm tra nhanh paddle/easyocr đã cài chưa.
- **Điểm đáng chú ý**: CPU phải `enable_mkldnn=False` (tránh crash PaddleOCR trên vài cấu hình).
  `raw_text` GIỮ NGUYÊN dấu phân cách (`-`, `.`, khoảng trắng) trước khi lọc charset — cần cho M6 suy
  đúng series 1 hay 2 chữ.

### 3.12 `src/ocr_subprocess.py` — PaddleOCR qua subprocess (CÁCH B)
- **Vai trò**: chạy PaddleOCR ở venv RIÊNG (`.venv-paddle`) qua subprocess để tránh xung đột
  torch/paddle. Spawn `tools/paddle_ocr_server.py` 1 lần, giao tiếp stdin/stdout.
- **Class**: `SubprocessPaddleEngine` (name=`paddle_subprocess`).
- **`_run(image)`**: encode ảnh JPEG → base64 → ghi 1 dòng vào stdin → đọc 1 dòng text từ stdout.
- **`close()`**: đóng stdin + terminate process.
- **Khi nào dùng**: khi không muốn/nên cài paddle chung venv với torch (`run_full.py` dùng cách này).

### 3.13 `src/plate_reader.py` — pipeline ẢNH TĨNH
- **Vai trò**: nối M2→M4→M5→M6→M7 cho 1 ảnh tĩnh. "Voting theo track_id" thay bằng voting trên nhiều
  biến thể của cùng ảnh.
- **Class**: `PlateReading` (text, formatted, confidence, valid, reliable, vote_count, is_two_line,
  skew_angle, too_small, engine, bbox, readings, vote, crop...) + `PlateReader`.
- **`read(image, bbox=None)`**: detector (tuỳ chọn) → preprocess → OCR nhiều biến thể → postprocess
  (vote) → format. `read_file(path)` đọc từ file; `preprocess_only()` chạy riêng M4.
- **Điểm đáng chú ý**: `_joined_raw_text` ghép `raw_text` theo ĐÚNG thứ tự item (không theo thứ tự
  engine trả về) để dấu phân cách khớp vị trí với `text` — cần cho M6 suy series.

### 3.14 `src/postprocess.py` — Module 6 (luật biển VN + voting)
- **Vai trò**: chuẩn hoá + sửa ký tự dễ nhầm + kiểm định định dạng + voting theo nhóm.
- **Hằng**:
  - `LETTER_TO_DIGIT = {"O":"0","I":"1","L":"1","B":"8","S":"5"}` — chữ bị OCR nhầm thành số.
  - `DIGIT_TO_LETTER = {"0":"O","1":"I","2":"Z","5":"S","6":"G","8":"B"}`.
  - `PLATE_PATTERN = ^[0-9]{2}[A-Z]{1,2}[0-9]{3,6}$` — biển trắng dân sự.
  - `MIN_PLATE_LENGTH=6`, `MAX_PLATE_LENGTH=10`.
- **Hàm chính**:
  - `build_position_template(length)` — sinh "khuôn" `['D','D','L','A'/'D',...]` (D=số, L=chữ, A=cả hai).
  - `_apply_position_template` — **điểm đặc biệt**: sửa ký tự theo VỊ TRÍ kỳ vọng (vd vị trí số thì O→0,
    B→8). Cố tình KHÔNG map mơ hồ (A→4, D→0, Q→0) vì dễ biến chuỗi rác thành "trông hợp lệ".
  - `normalize_plate_text` / `validate_plate` / `format_plate`.
  - `VotingAggregator` — lọc confidence < `min_confidence` + lọc sai định dạng TRƯỚC khi vote; bỏ phiếu
    cho chuỗi chuẩn hoá xuất hiện nhiều nhất (hoà thì ưu tiên confidence trung bình cao hơn).
  - `VoteResult`, `PostProcessor.process(readings, group_id)`.
  - `group_readings_by_track` / `vote_by_track` — mở rộng sang video.
  - `interpolate_linear` — nội suy frame thiếu (giống `add_missing_data.py`).
- **Điểm đáng chú ý**: sửa ký tự theo vị trí = cách giảm lỗi OCR hiệu quả **không cần train lại model**.

### 3.15 `src/fusion.py` — gộp nhiều lần đọc theo VỊ TRÍ ký tự
- **Vai trò**: khác `VotingAggregator` (vote CẢ CHUỖI), fusion bỏ phiếu TỪNG VỊ TRÍ ký tự → 1 ký tự sai
  lẻ tẻ (5→6) không chia nhỏ phiếu.
- **Tầng A (mặc định)**: normalize → lọc độ dài `[MIN,MAX]` + confidence ≥ `min_confidence` → vote độ
  dài (đa số) → gom cụm (chống trộn 2 biển: cùng độ dài, khác ≤ `max_cluster_distance` ký tự) → vote
  từng vị trí (trọng số `conf^a · quality^b`, mỗi crop chia đều trọng số) → validate_plate; sai định
  dạng thì lùi về vote cả chuỗi.
- **Tầng B (chưa chạy)**: nếu lần đọc có `char_probs` → cộng log-prob theo vị trí.
- **Class**: `FusionResult`, `_Reading`, `fuse_readings(...)`, `load_province_codes(path)`.
- **Điểm đáng chú ý**: `vote_count`/`supporting_frames` tính theo CROP (`frame_index`) không theo số
  lần đọc — 1 crop nhiều reading không bỏ nhiều phiếu. Phát hiện "nghi trộn 2 biển" khi cụm lớn thứ 2
  chiếm ≥ `cluster_min_share` → đánh dấu `reliable=False`.

### 3.16 `src/video_pipeline.py` — FILE GLUE (quan trọng nhất)
- **Vai trò**: nối M0→M1→M2→M3→M4/5/6→M7 cho video. Giữ state (tracker, bộ gom reading, top-K buffer)
  trên instance → phải gọi tuần tự theo frame.
- **Class/function**:
  - `TrackReading`, `FramePipelineResult`, `VideoPipelineResult` (có `to_dict`).
  - `VideoPipeline(cfg, ...)` — khởi tạo VehicleDetector, PlateDetector, SortTracker, PlateReader,
    PostProcessor; lazy-load OCR (`_ensure_reader`).
  - `process_frame(frame, frame_index, timestamp)` — M1 (bọc try/except, lỗi → fallback toàn khung) →
    M3 (bọc try/except, lỗi → reset tracker) → M2 `_locate_plate_candidate` → crop → best-frame →
    M4/5/6 → gom reading theo track.
  - `_locate_plate_candidate` — **điểm đặc biệt**: ưu tiên detection M2 nằm trong/chồng bbox xe
    (`source="detection"`), dùng `used_plate_indices` để mỗi plate detection chỉ thuộc ĐÚNG 1 track;
    không có → fallback nửa dưới bbox xe (`_fallback_plate_bbox`, `plate_region_ratio`).
  - `_containment_score` — biển nằm TRONG xe +1.0 (ưu tiên) so với chỉ chồng lấn.
  - `run(video_path, keep_frame_images)` — extract frames → process từng frame → `_flush_all_tracks`
    (OCR top-K) → `_finalize_tracks` (vote/fusion) → trả `VideoPipelineResult`.
  - **OOM protection**: `keep_frame_images=False` thì set `frame_result.frame=None` + `record.frame=None`
    để không giữ cả video trong RAM.
- **Ai gọi nó**: `main.py` (root) và `tools/run_full.py`, `tools/test_e2e.py`.

### 3.17 `src/visualize.py` — Module 7 (overlay ảnh tĩnh)
- **Vai trò**: vẽ khung + text biển số + dán crop biển vào góc để đối chiếu khi viết báo cáo.
- **Hàm**: `draw_bbox`, `draw_reading` (màu GREEN nếu valid, ORANGE nếu không), `side_by_side`.
- **Màu**: `GREEN=(0,200,0)`, `RED=(0,0,220)`, `ORANGE=(0,165,255)`.

### 3.18 `src/sample_data.py` — sinh biển giả lập để test
- **Vai trò**: sinh ảnh biển VN giả (1 dòng + 2 dòng) + `add_degradation` (nghiêng/mờ/nhiễu/tối) để
  kiểm tra Module 4 (deskew, CLAHE) có tác dụng khi chưa có ảnh thật.
- **Hàm**: `make_single_line_plate`, `make_two_line_plate`, `add_degradation`, `make_demo_images`.
- **Lưu ý**: chỉ dùng 0-9/A-Z nên không cần font tiếng Việt (`cv2.putText` + `FONT_HERSHEY_SIMPLEX`).

---

## 4. Tools `tools/` — chi tiết các script quan trọng

(Danh sách đầy đủ 40 script + vai trò đã ở mục 2.2. Dưới đây đào sâu các script phức tạp.
Các path dưới đây là **tương đối so với `tools/`**: `data/`, `train/`, `eval/`, `run/`,
`sample/`, `check/`, `utils/`.)

### `run_full.py` — chạy FULL video thật
- **Môi trường**: chạy bằng `.venv-datasets` (chứa torch + ultralytics).
- **Input**: `videodemo2.mp4` + `models/yolo11n.pt` + `models/plate_detector.pt` +
  `runs/corner_regression/corner_reg100/weights/best.pt`.
- **Cách nối**: tạo `PipelineConfig()` → set model path + `bbox_margin=0.15` +
  `frame_interval=5` + `save_frame_overlays=True` → tạo `SubprocessPaddleEngine()` (OCR ở venv
  riêng) → `PlateReader(cfg, ocr_engine=engine)` → `VideoPipeline(cfg, reader=reader)` →
  `pipeline.set_corner_regressor(CornerRegressor(best, imgsz=160), angle_threshold=15.0)` →
  `pipeline.run(VIDEO, keep_frame_images=False)`.
- **Output**: `runs/e2e_full/results.{json,csv,txt}` + `plate_crops/` + `tracks_frames/` + gộp
  `output_demo.mp4`.

### `unify_labeling.py` — gộp 4 nguồn nhãn về 1 CSV pixel
- **Vấn đề**: 4 file nhãn tay dùng NHIỀU chuẩn toạ độ khác nhau:
  1. `labels201-400.csv` — pixel, đúng schema, nhưng cột `image` là URL Label Studio cần decode.
  2. `401-600.csv` — Label Studio thô: toạ độ PHẦN TRĂM (0-100) trong cột `label` (JSON), polygon 5
     điểm (điểm cuối = điểm đầu) cần bỏ bớt.
  3. `labels_anh_601_800.csv` — pixel, đúng.
  4. `801-1701.xlsx` — SAI tên cột `unreadable_text` + toạ độ phần trăm.
- **Đầu ra**: `labeled.csv` (10 cột `image,plate_text,tl_x...bl_y`, toạ độ PIXEL, góc sắp lại TL,TR,BR,BL)
  + `split.csv` (chia theo `track_id`, KHÔNG chia theo ảnh — vì 1 track = nhiều ảnh cùng xe, chia theo
  ảnh sẽ rò rỉ).
- **Import**: `geometry_utils.order_corners_geometric`, `dedupe_closing_point`.

### `ccpd_to_pose.py` — CCPD2019 → YOLO-pose
- **Format tên file CCPD** (7 field cách nhau `-`): `<area>-<tiltH>_<tiltV>-<bbox>-<4 góc>-<7 ký tự>-<bright>-<blur>`.
- **Điểm đặc biệt**: 4 đỉnh CCPD bắt đầu từ góc PHẢI-DƯỚI theo thứ tự (BR,BL,TL,TR). Script KHÔNG dùng
  thứ tự gốc mà **sắp lại hình học** về TL→TR→BR→BL, rồi **đối chiếu** với thứ tự gốc để phát hiện ca
  bất đồng (biển xoay gần 45°). Đã kiểm: 0 bất đồng / 17 774 ảnh.
- **Đầu ra**: `images/{train,val}/*.jpg` + `labels/{train,val}/*.txt` (kpt_shape [4,3]) +
  `data_pose.yaml` (có `flip_idx: [1,0,3,2]`) + `manifest.csv` (chứa angle_deg, tilt, bright, blur).

### `train_pose_ccpd.py` — train YOLO-pose 4 keypoint
- **2 chế độ**: `baseline` (degrees=0, perspective=0) vs `augmented` (degrees=20, perspective=0.001)
  để bù việc 82% ảnh CCPD có |góc| < 5°.
- **Hyperparam chung**: imgsz=640, epochs=100, patience=25, batch=8, **fliplr=0.0** (TẮT lật ngang —
  lật sẽ đảo thứ tự ký tự), translate=0.1, scale=0.5, mosaic=1.0, close_mosaic=10.

### `train_corner_regressor.py` — train corner trên CCPD-warped
- **Khác pose**: dùng `datasets/corner_regression/data.yaml`, imgsz=160 (crop nhỏ), batch=64, patience=30.
  Model gốc `yolo11n-pose.pt`. Output `runs/corner_regression/corner_reg*/weights/best.pt`.

### `geometry_utils.py` (thư viện) — sắp 4 góc
- `order_corners_geometric(pts)` → `(TL,TR,BR,BL)` bằng công thức lồi:
  `TL=min(x+y)`, `BR=max(x+y)`, `TR=max(x−y)`, `BL=min(x−y)`.
- `is_convex(pts)` — kiểm tra tứ giác lồi (cross product không đổi dấu qua 4 cạnh).
- `dedupe_closing_point(points)` — bỏ điểm cuối trùng điểm đầu (polygon khép kín của Label Studio).

### `paddle_ocr_server.py` — server OCR stdin/stdout
- Đọc từng dòng base64 (ảnh JPEG) từ stdin, chạy PaddleOCR, ghi 1 dòng text ra stdout. Được
  `ocr_subprocess.py` spawn với venv `.venv-paddle`.

---

## 5. Toàn bộ tham số Config

> Giá trị chính xác nằm trong `src/config.py`. Bảng dưới tóm tắt ý nghĩa từng nhóm.

| Dataclass | Điều khiển | Vài giá trị quan trọng (mặc định) |
|---|---|---|
| `PreprocessConfig` | M4 tiền xử lý | `target 320×48`, `max_skew_angle=20`, `clahe_clip_limit=2.0`, `two_line_min_aspect=1.8`, `pad_color=255` |
| `OCRConfig` | M5 OCR | `engine="auto"`, `lang="en"`, `use_gpu=False`, `enable_mkldnn=False` (CPU), `use_angle_cls=True` |
| `PostprocessConfig` | M6 vote | `min_confidence=0.35`, `min_readings`, `require_valid_format`, `apply_position_fix` |
| `FrameExtractionConfig` | M0 | `frame_interval=5`, `max_duration`, ngưỡng blur/brightness |
| `VehicleDetectionConfig` | M1 | `model_path`, `conf=0.25`, `iou`, `classes` (COCO xe), `full_frame_fallback` |
| `TrackingConfig` | M3 SORT | `max_age=30`, `min_hits=2`, `iou_threshold` |
| `VideoPipelineConfig` | glue video | `read_plates`, `plate_fallback_to_vehicle`, `plate_region_ratio`, `plate_min_confidence`, `save_frame_overlays`, `max_reading_frames` |
| `BestFrameConfig` | chọn top-K | `k=5`, `min_frame_gap`, `w_sharp=0.4`, `w_height=0.25`, `w_aspect=0.10`, `w_conf=0.25`, `w_exposure=0.10`, `dark_threshold=30`, `bright_threshold=225` |
| `FusionConfig` | fusion vị trí | `enabled=False`, `min_confidence=0.35`, `max_cluster_distance=2`, `cluster_min_share=0.3`, `confidence_exponent=1.0`, `quality_exponent=1.0` |
| `PipelineConfig` | gom tất cả | 9 config con + `detection_model/conf/device/imgsz` + `output_dir` |

---

## 6. Cách chạy & tái tạo từ đầu

### 6.1 Môi trường (3 venv)
| venv | Chứa gì | Dùng cho |
|---|---|---|
| `.venv-datasets` | torch + ultralytics + cv2 + PIL | train/eval + chạy pipeline (detect/track/preprocess) |
| `.venv-paddle` | paddlepaddle + paddleocr | OCR riêng (qua subprocess `ocr_subprocess.py`) |
| `.venv` | base (có cv2/numpy...) | script nhẹ, self-check |

### 6.2 Chạy nhanh (inference video)
```powershell
.venv-datasets\Scripts\Activate.ps1
python main.py --video videodemo2.mp4 --ocr-engine easyocr --save-overlays
```
Cần có: `models/yolo11n.pt` + `models/plate_detector.pt` (+ corner `best.pt` nếu muốn warp góc).

### 6.3 Tái tạo toàn bộ (từ data → model → eval)
1. **Tải data**: `python tools/download_ccpd.py` (CCPD2019) + `python download_datasets.py` (Roboflow).
2. **Xây dataset pose**: `python tools/ccpd_to_pose.py --src .../extracted --out datasets/ccpd_pose`.
3. **Train detector biển**: `python train_detector.py --epochs 50 --batch 8`.
4. **Train corner/pose**: `python tools/train_pose_ccpd.py --mode augmented` +
   `python tools/train_corner_regressor.py --epochs 100`.
5. **Eval**: `python tools/eval_corner_regressor.py`, `python tools/eval_fusion.py`, ...
6. **Chạy full**: `python tools/run_full.py`.

### 6.4 Quy ước weights quan trọng
- `models/yolo11n.pt` — base YOLO11n (dò xe, COCO pretrained).
- `models/plate_detector.pt` — fine-tune dò biển (đầu ra của `train_detector.py`).
- `runs/corner_regression/corner_reg*/weights/best.pt` — corner regressor (yolo11n-pose fine-tune).
- `dataset_prep_src/yolo11n-pose.pt` — base pose (được `train_pose_ccpd.py` / `train_corner_regressor.py`
  tham chiếu bằng tên `"yolo11n-pose.pt"`).

---

## 7. Cheat-sheet mọi "điểm đặc biệt"

| Chỗ | Điểm đặc biệt |
|---|---|
| Unicode I/O | `imread_unicode`/`imwrite_unicode` dùng `imdecode`/`imencode` vì cv2 không chịu path Unicode trên Windows |
| Chống NaN | `_finite_float` chặn `NaN`/`inf` từ `CAP_PROP_FPS`/`FRAME_COUNT` |
| Bbox bẩn | `detect_vehicle.clip_bbox` dùng `np.nan_to_num` chịu bbox méo/NaN thay vì sập |
| Deskew | projection profile: variance histogram dọc max + nội suy parabol |
| Sắp 4 góc | `TL=min(x+y)`, `TR=max(x−y)`, `BR=max(x+y)`, `BL=min(x−y)` |
| Kiểm tra lồi | cross product không đổi dấu qua 4 cạnh |
| Best-frame rank | cặp `(is_detection, score)` — fallback KHÔNG bao giờ vượt detection thật |
| Sửa ký tự | theo VỊ TRÍ kỳ vọng (O↔0, I↔1, B↔8, S↔5); KHÔNG map mơ hồ A/D/Q→0 |
| Fusion | vote theo VỊ TRÍ ký tự + phát hiện "trộn 2 biển" qua cụm Hamming |
| Voting | lọc conf thấp + sai định dạng TRƯỚC khi vote (không nhiễu kết quả) |
| OOM | `keep_frame_images=False` → set `frame=None` để không giữ video trong RAM |
| Bọc lỗi | M1/M2/M3/M5 đều try/except + fallback (thiếu model vẫn chạy tracking) |
| OCR 2 venv | PaddleOCR chạy `.venv-paddle` qua subprocess stdin/stdout tránh xung đột torch |
| fliplr=0 | BẮT BUỘC tắt lật ngang khi train (lật đảo thứ tự ký tự) |
| flip_idx | `[1,0,3,2]` trong `data_pose.yaml` để keypoint đúng khi lật |
| Gán biển→xe | `_containment_score`: biển nằm TRONG xe +1.0; `used_plate_indices` đảm bảo 1 biển 1 track |

---

*File sinh tự động khi dọn code — khớp với trạng thái 62 file Python (src/ 18 + tools/ 41 + root 3).*

