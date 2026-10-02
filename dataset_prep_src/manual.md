# Hướng dẫn sử dụng — Hệ thống đọc biển số xe Việt Nam (ALPR)

Tài liệu này mô tả cách cài đặt và sử dụng phần mềm **ALPR từ ảnh tĩnh và video**.
Phần mềm được tổ chức theo pipeline nhiều module, trong đó:

- Đọc biển số từ **1 ảnh tĩnh** đã hoàn thiện.
- Trích và lọc frame từ **video** (Module 0) hoạt động độc lập.
- Pipeline video **đầy đủ** đã nối **Module 1 — Vehicle Detection**,
  **Module 3 — Tracking** và **file glue** `src/video_pipeline.py` để chạy
  end-to-end từ video tới biển số.

---

## 1. Giới thiệu chung

Pipeline tổng thể của phần mềm:

```
video.mp4
  → Module 0: Frame Extraction & Filtering          src/frame_extractor.py
  → Module 1: Vehicle Detection (YOLOv8n)           src/detect_vehicle.py
  → Module 2: Plate Detection (YOLO, tùy chọn)      src/detector.py
  → Module 3: Tracking (SORT)                       src/tracker.py
  → Module 4: Tiền xử lý ảnh biển số                src/preprocess_plate.py
  → Module 5: OCR (PaddleOCR + EasyOCR)             src/ocr.py
  → Module 6: Hậu xử lý luật biển VN + voting       src/postprocess.py
  → Module 7: Output & visualize                    src/video_pipeline.py + src/visualize.py
```

Điểm quan trọng:

- **Module 1** (`src/detect_vehicle.py`) tìm bounding box phương tiện trong từng
  frame. Nếu chưa có model YOLO, module tự chạy ở chế độ **toàn khung** — coi cả
  frame là một phương tiện — để pipeline vẫn hoạt động.
- **Module 3** (`src/tracker.py`) gán `track_id` ổn định cho mỗi phương tiện qua
  nhiều frame bằng thuật toán **SORT** (Kalman filter + Hungarian + IoU).
- **File glue** (`src/video_pipeline.py`) nối tất cả module, gom các lần đọc biển
  số theo `track_id`, bỏ phiếu và xuất **một kết quả biển số cho mỗi xe**.

---

## 2. Các chức năng chính

| Chức năng | Cách dùng | Ghi chú |
| --- | --- | --- |
| Đọc biển số từ 1 ảnh đã crop | `python main.py --image <ảnh>` | Coi toàn bộ ảnh là biển số |
| Đọc biển số từ 1 ảnh lớn có bbox | `python main.py --image <ảnh> --bbox x1,y1,x2,y2` | Chỉ rõ vùng biển số |
| Tự dò biển số bằng YOLO (tùy chọn) | `python main.py --image <ảnh> --detector <model.pt>` | Cần `ultralytics` |
| Chạy thử nhanh không cần ảnh thật | `python main.py --demo` | Tự sinh ảnh biển số giả lập |
| Sinh ảnh biển số giả lập | `python tools/make_sample_plate.py` | Dùng để kiểm thử |
| Sinh video giả lập | `python tools/make_sample_video.py` | Tự chèn frame tối/mờ |
| Trích & lọc frame từ video | `python main.py --video <video>` | Chỉ chạy Module 0, không cần OCR |
| Chạy pipeline video đầy đủ | `python main.py --video <video> --track` | Module 0 → 1 → 2 → 3 → 4/5/6 → 7 |
| Chạy video chỉ tới tracking | `python main.py --video <video> --track --no-plate-reading` | Không cần OCR |
| Lưu ảnh overlay theo dõi xe | `python main.py --video <video> --track --save-track-frames` | Bbox xe + `track_id` + biển số |
| Tự kiểm tra logic Module 4/5/6/0 | `python tools/self_check.py` | Không cần OCR engine |
| Tự kiểm tra Module 1/3 + glue | `python tools/self_check_tracking.py` | Không cần `ultralytics`/OCR |

---

## 3. Yêu cầu môi trường

- **Python**: 3.10 trở lên.
- **Thư viện lõi** (bắt buộc cho mọi chức năng):
  - `numpy`
  - `opencv-python`
- **OCR — Module 5**:
  - `paddleocr` + `paddlepaddle` (engine chính)
  - `easyocr` (engine dự phòng)
- **YOLO — Module 1 và Module 2** (tùy chọn):
  - `ultralytics` — chỉ cần khi truyền model `.pt`.
    Nếu chưa cài, Module 1 vẫn chạy ở chế độ **toàn khung**, Module 2 được thay
    bằng vùng bbox xe xấp xỉ.

File `requirements.txt` đã khai báo đầy đủ các dependency trên.

---

## 4. Cài đặt

### 4.1 Cài đặt đầy đủ

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Cài toàn bộ dependency
pip install -r requirements.txt
```

### 4.2 Cài đặt tối thiểu theo nhu cầu

| Nhu cầu | Lệnh cài |
| --- | --- |
| Chỉ chạy Module 0 (trích/lọc frame video) | `pip install numpy opencv-python` |
| Chạy video tracking nhưng không đọc biển số | `pip install numpy opencv-python` |
| Dùng model YOLO cho Module 1/2 | `pip install ultralytics` |
| Đọc biển số bằng EasyOCR | `pip install easyocr` |
| Đọc biển số bằng PaddleOCR | `pip install paddleocr paddlepaddle` |

---

## 5. Sử dụng nhanh

### 5.1 Chạy thử không cần dữ liệu thật

```bash
# Tự sinh ảnh biển số giả lập rồi đọc thử
python main.py --demo

# Tự sinh video giả lập để kiểm thử pipeline video
python tools/make_sample_video.py --output data/samples/test_video.mp4
```

### 5.2 Đọc biển số từ ảnh tĩnh

```bash
# Ảnh đã crop sẵn — toàn bộ ảnh là biển số
python main.py --image data/plate.png

# Ảnh lớn — chỉ rõ bbox biển số
python main.py --image anh.jpg --bbox 120,340,420,430

# Dùng YOLO tự dò biển số trong ảnh
python main.py --image anh.jpg --detector data/models/plate_detector.pt

# Ép dùng EasyOCR thay vì PaddleOCR
python main.py --image anh.jpg --engine easyocr

# Lưu thêm ảnh crop biển số
python main.py --image anh.jpg --save-crop
```

Kết quả ảnh tĩnh được ghi vào thư mục `results/`:

```
results/
├── result.json            # kết quả đọc biển số
├── <tên_ảnh>_overlay.png  # ảnh gốc + khung biển số + chữ đọc được
└── plate_crops/           # ảnh biển số đã tiền xử lý (khi dùng --save-crop)
```

### 5.3 Trích & lọc frame từ video (chỉ Module 0)

```bash
# Chạy chỉ Module 0, lưu frame giữ lại và manifest
python main.py --video data/samples/test_video.mp4 --frame-interval 5 --save-frames
```

Kết quả:

```
results/
├── frames_manifest.json   # metadata frame + thống kê + cấu hình
└── frames/                # frame_<chỉ_số_gốc>.png
```

### 5.4 Chạy pipeline video đầy đủ (Module 0 → 7)

```bash
# Không cần model/OCR: chỉ chạy Module 0 → 1 → 3
python main.py --video data/samples/test_video.mp4 --track --no-plate-reading

# Dùng model YOLO dò xe (Module 1) và đọc biển số nếu đã cài OCR
python main.py --video video.mp4 --track --vehicle-model yolov8n.pt

# Full pipeline với cả model dò biển số (Module 2)
python main.py --video video.mp4 --track \
    --vehicle-model yolov8n.pt \
    --detector data/models/plate_detector.pt

# Lưu thêm frame overlay để xem bbox xe + track_id + biển số
python main.py --video video.mp4 --track --save-track-frames
```

Kết quả pipeline video đầy đủ:

```
results/
├── video_tracks.json      # kết quả theo từng track_id (đã voting)
└── tracks_frames/         # ảnh overlay (khi dùng --save-track-frames)
```

---

## 6. Chi tiết Module 1, Module 3 và file glue

### 6.1 Module 1 — `src/detect_vehicle.py`

Module này phát hiện phương tiện trong một frame.

Các thành phần chính:

| Tên | Vai trò |
| --- | --- |
| `VehicleDetection` | Một detection: `bbox`, `confidence`, `class_id`, `label` |
| `VehicleDetector` | Lớp bọc `ultralytics.YOLO`, có `detect(frame)` và `best(frame)` |
| `build_vehicle_detector(config)` | Tạo `VehicleDetector` từ cấu hình |
| `detect_vehicles(frame, detector=None, config=None)` | Hàm tiện dụng dò xe trong 1 frame |
| `COCO_VEHICLE_CLASSES` | Nhóm lớp COCO mặc định: `2=car`, `3=motorcycle`, `5=bus`, `7=truck` |

Cấu hình `VehicleDetectionConfig` trong `src/config.py`:

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `model_path` | `None` | Đường dẫn model YOLO `.pt`. `None` = chế độ toàn khung |
| `conf` | `0.25` | Ngưỡng confidence tối thiểu |
| `iou` | `0.45` | Ngưỡng IoU cho NMS của YOLO |
| `device` | `None` | Thiết bị suy luận: `cpu`, `cuda`, `0`... |
| `imgsz` | `640` | Kích thước ảnh đầu vào cho YOLO |
| `max_det` | `100` | Số detection tối đa mỗi frame |
| `classes` | `(2, 3, 5, 7)` | Các lớp COCO được giữ lại |
| `full_frame_fallback` | `True` | Khi không có model, trả về 1 bbox phủ toàn khung |

Ví dụ sử dụng API hiện có:

```python
from src.detect_vehicle import VehicleDetector

detector = VehicleDetector("yolov8n.pt")   # hoặc VehicleDetector() cho chế độ toàn khung
detections = detector.detect(frame_bgr)
for det in detections:
    print(det.label, det.confidence, det.bbox)
```

### 6.2 Module 3 — `src/tracker.py`

Module này theo dõi phương tiện qua nhiều frame bằng thuật toán **SORT**:

```
Kalman filter (mô hình vận tốc không đổi)
  → dự đoán bbox hiện tại
  → ghép detection với track bằng IoU + Hungarian
  → cập nhật track / tạo track mới / xóa track quá hạn
```

Các thành phần chính:

| Tên | Vai trò |
| --- | --- |
| `SortTracker` | Tracker chính; phương thức `update(detections, frame_index=None, timestamp=None)` |
| `Tracker` | Bí danh của `SortTracker` |
| `TrackedObject` | Kết quả track: `track_id`, `bbox`, `confidence`, `hits`, `age`, `state` |
| `track_detections(detection_frames, config=None)` | Chạy tracker qua một chuỗi frame detection |
| `box_iou`, `iou_matrix` | Các hàm tiện ích tính IoU |

Cấu hình `TrackingConfig` trong `src/config.py`:

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `enabled` | `True` | Tắt tracking thì mỗi detection là một track độc lập |
| `max_age` | `30` | Số frame giữ track không cập nhật trước khi xóa |
| `min_hits` | `3` | Số lần khớp liên tiếp tối thiểu để xác nhận track |
| `iou_threshold` | `0.3` | Ngưỡng IoU tối thiểu để ghép detection với track |

Ví dụ sử dụng API hiện có:

```python
from src.tracker import SortTracker

tracker = SortTracker()
for frame_index, detections in enumerate(list_detections_theo_frame):
    tracks = tracker.update(detections, frame_index=frame_index)
    for track in tracks:
        print(track.track_id, track.bbox)
```

### 6.3 File glue — `src/video_pipeline.py`

File glue nối các module lại thành pipeline hoàn chỉnh:

```
Module 0 → Module 1 → Module 2 (nếu có) → Module 3 → Module 4/5/6 → Module 7
```

Các thành phần chính:

| Tên | Vai trò |
| --- | --- |
| `VideoPipeline` | Lớp glue chính; có `run(video_path)`, `process_frame(...)`, `iter_frames(...)` |
| `VideoPipelineResult` | Kết quả đầy đủ: `frames`, `tracks`, `readings`, `stats`, `warnings` |
| `TrackResult` | Kết quả đã voting cho một `track_id` |
| `TrackReading` | Một lần đọc biển số gắn với một `track_id` tại một frame |
| `run_video_pipeline(video_path, config=None)` | Hàm tiện dụng chạy pipeline một lần |

Logic glue:

- Module 0 trích và lọc frame.
- Module 1 dò phương tiện trong frame.
- Module 3 gán `track_id` ổn định.
- Với mỗi track, glue xác định vùng biển số:
  - Dùng **Module 2** nếu có model dò biển số.
  - Nếu không có, xấp xỉ bằng **phần dưới bbox phương tiện** theo
    `VideoPipelineConfig.plate_region_ratio`.
- Module 4/5/6 đọc biển số và bỏ phiếu theo `track_id`.
- Module 7 ghi JSON và (tùy chọn) lưu ảnh overlay.

Cấu hình `VideoPipelineConfig` trong `src/config.py`:

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `read_plates` | `True` | `False` = chỉ chạy Module 0/1/3, không OCR |
| `save_frame_overlays` | `False` | Lưu frame overlay vào `tracks_frames/` |
| `plate_fallback_to_vehicle` | `True` | Không có model biển số thì dùng bbox xe |
| `plate_region_ratio` | `0.5` | Tỷ lệ chiều cao vùng biển số tính từ đáy bbox xe |
| `plate_min_confidence` | `0.25` | Ngưỡng confidence cho detection biển số |
| `max_reading_frames` | `0` | Giới hạn số frame đọc biển số (0 = không giới hạn) |

Ví dụ sử dụng API hiện có:

```python
from src.config import PipelineConfig
from src.video_pipeline import VideoPipeline

pipeline = VideoPipeline(PipelineConfig())
result = pipeline.run("data/samples/test_video.mp4")
for track in result.tracks:
    print(track.track_id, track.formatted)
```

---

## 7. Tham số dòng lệnh của `main.py`

### 7.1 Tham số chung và cho ảnh tĩnh

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--image PATH` | `None` | Đọc biển số từ 1 ảnh tĩnh |
| `--video PATH` | `None` | Chạy trên video. Không có `--track` = chỉ Module 0 |
| `--demo` | off | Sinh ảnh demo rồi đọc thử |
| `--output PATH` | `None` | File JSON kết quả (mặc định theo từng chế độ) |
| `--output-dir PATH` | `results` | Thư mục lưu kết quả |
| `--bbox x1,y1,x2,y2` | `None` | Bbox biển số trong ảnh lớn |
| `--detector PATH` | `None` | Model YOLO dò biển số (Module 2) |
| `--engine auto/paddle/easyocr` | `auto` | OCR engine sử dụng |
| `--gpu` | off | Dùng GPU cho OCR |
| `--save-crop` | off | Lưu ảnh biển số đã tiền xử lý |

### 7.2 Tham số cho Module 0 (khi dùng `--video`)

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--frame-interval N` | `5` | Bước nhảy frame (1 = lấy mọi frame) |
| `--brightness-threshold F` | `50.0` | Ngưỡng độ sáng tối thiểu (0-255) |
| `--blur-threshold F` | `100.0` | Ngưỡng độ nét tối thiểu (variance of Laplacian) |
| `--max-frames N` | `0` | Giới hạn số frame giữ lại (0 = không giới hạn) |
| `--skip-start-frames N` | `0` | Bỏ N frame đầu video |
| `--save-frames` | off | Lưu frame giữ lại vào `frames/` |
| `--no-quality-filter` | off | Tắt bộ lọc sáng/nét, giữ mọi frame theo `--frame-interval` |

### 7.3 Tham số cho pipeline video đầy đủ (Module 1/3 + glue)

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--track` | off | Bật pipeline video đầy đủ thay vì chỉ Module 0 |
| `--vehicle-model PATH` | `None` | Model YOLO dò phương tiện (Module 1). Trống = toàn khung |
| `--vehicle-conf F` | `0.25` | Ngưỡng confidence dò phương tiện |
| `--tracker-max-age N` | `30` | Số frame giữ track không cập nhật (Module 3) |
| `--tracker-min-hits N` | `3` | Số lần khớp liên tiếp để xác nhận track (Module 3) |
| `--tracker-iou F` | `0.3` | Ngưỡng IoU ghép detection ↔ track (Module 3) |
| `--no-plate-reading` | off | Chỉ chạy Module 0/1/3, không đọc biển số |
| `--save-track-frames` | off | Lưu frame overlay vào `tracks_frames/` |

### 7.4 Tham số cho hậu xử lý và voting (Module 6)

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--min-confidence F` | `0.35` | Ngưỡng confidence tối thiểu khi voting |
| `--min-readings N` | `2` | Số lần đọc hợp lệ tối thiểu để kết quả đáng tin cậy |
| `--no-position-fix` | off | Tắt sửa ký tự dễ nhầm theo vị trí |
| `--allow-invalid` | off | Cho phép chuỗi không khớp định dạng biển VN vào voting |

---

## 8. Kết quả đầu ra

### 8.1 Ảnh tĩnh

`results/result.json` chứa `config` và `results`, mỗi kết quả gồm `plate_text`,
`confidence`, `vote_count`, `total_readings`, `valid`, `reliable`...

### 8.2 Video — chỉ Module 0

`results/frames_manifest.json` chứa:

- `module`: `"Module 0 - Frame Extraction"`.
- `video`: đường dẫn video.
- `config`: cấu hình `FrameExtractionConfig`.
- `stats`: tổng frame, đã xét, giữ lại, bỏ do tối/mờ, fps, thời lượng, kích thước.
- `frames`: metadata từng frame giữ lại.

### 8.3 Video — pipeline đầy đủ

`results/video_tracks.json` chứa:

- `module`: `"Video Pipeline (Module 0 -> 1 -> 2 -> 3 -> 4/5/6 -> 7)"`.
- `config`: toàn bộ `PipelineConfig`.
- `stats`: gồm `vehicle_detector`, `plate_detector`, `tracker`, số frame, số track,
  số lần đọc biển số...
- `tracks`: kết quả mỗi `track_id` sau khi voting.
- `frames`: metadata từng frame (số track, danh sách track, danh sách reading).
- `readings`: mọi lần đọc biển số gắn `track_id`.

Mỗi phần tử trong `tracks` gồm:

| Trường | Ý nghĩa |
| --- | --- |
| `track_id` | Mã track ổn định của xe |
| `plate_text` | Chuỗi biển số đã chuẩn hóa |
| `plate_display` | Chuỗi hiển thị có dấu phân cách |
| `confidence` | Confidence trung bình sau voting |
| `valid_format` | Có khớp định dạng biển VN hay không |
| `reliable` | Có đủ số lần đọc tối thiểu hay không |
| `vote_count` / `total_readings` | Số phiếu / tổng số lần đọc |
| `first_frame` / `last_frame` | Khoảng frame xuất hiện của xe |
| `vehicle_bbox` | Bbox phương tiện mới nhất |
| `readings` | Danh sách chi tiết các lần đọc |

---

## 9. Tự kiểm tra

```bash
# Kiểm tra Module 4/5/6/0 — không cần OCR engine
python tools/self_check.py

# Kiểm tra Module 1/3 + file glue — không cần ultralytics/OCR
python tools/self_check_tracking.py
```

`tools/self_check_tracking.py` kiểm tra các logic quan trọng của Module 1, Module 3
và `src/video_pipeline.py` mà không yêu cầu cài model YOLO hay OCR engine.

---

## 10. Cấu trúc thư mục

```
.
├── main.py                     # CLI chính (ảnh tĩnh + video)
├── requirements.txt            # dependency
├── README.md                   # ghi chú kỹ thuật dự án
├── manual.md                   # hướng dẫn sử dụng (file này)
├── src/
│   ├── config.py               # toàn bộ cấu hình, gồm VehicleDetectionConfig, TrackingConfig
│   ├── image_utils.py          # đọc/ghi ảnh Unicode, đo sáng, đo nét, resize
│   ├── frame_extractor.py      # Module 0 — trích & lọc frame từ video
│   ├── detect_vehicle.py       # Module 1 — phát hiện phương tiện (YOLOv8n/toàn khung)
│   ├── detector.py             # Module 2 (tùy chọn) — dò biển số bằng YOLO
│   ├── tracker.py              # Module 3 — tracking SORT
│   ├── video_pipeline.py       # file glue — nối Module 0 → 7
│   ├── preprocess_plate.py     # Module 4 — tiền xử lý ảnh biển số
│   ├── ocr.py                  # Module 5 — PaddleOCR 2.x/3.x + EasyOCR fallback
│   ├── postprocess.py          # Module 6 — luật biển VN + voting
│   ├── plate_reader.py         # ghép pipeline cho 1 ảnh tĩnh
│   ├── visualize.py            # Module 7 (overlay)
│   └── sample_data.py          # sinh ảnh/video giả lập
└── tools/
    ├── make_sample_plate.py    # CLI sinh ảnh biển số demo
    ├── make_sample_video.py    # CLI sinh video giả lập
    ├── self_check.py           # self-test Module 4/5/6/0
    └── self_check_tracking.py  # self-test Module 1/3 + glue
```

---

## 11. Xử lý sự cố thường gặp

### 11.1 Thiếu OCR engine

Khi chạy ảnh tĩnh mà thiếu PaddleOCR/EasyOCR, `main.py` in thông báo và gợi ý:

```bash
pip install paddleocr paddlepaddle
# hoặc
pip install easyocr
```

Khi chạy `--video --track`, nếu thiếu OCR engine, pipeline vẫn chạy Module 0/1/3
và ghi cảnh báo; kết quả track sẽ không có `plate_text`.

### 11.2 Thiếu `ultralytics`

Nếu truyền `--vehicle-model` hoặc `--detector` nhưng chưa cài `ultralytics`, phần
mềm ghi cảnh báo và:

- Module 1 fallback về chế độ **toàn khung**.
- Module 2 bị bỏ qua, glue dùng bbox phương tiện làm vùng biển số xấp xỉ.

Cài đặt:

```bash
pip install ultralytics
```

### 11.3 Không ghi được video test (`cv2.VideoWriter`)

Thông báo lỗi sẽ gợi ý kiểm tra codec `mp4v`. Nếu môi trường thiếu codec, các bài
tự kiểm tra video sẽ in `[SKIP]` thay vì bị tính là thất bại.

### 11.4 Đường dẫn chứa ký tự Unicode

Phần mềm dùng `imread_unicode`/`imwrite_unicode` cho ảnh và có cơ chế đọc/ghi video
qua file tạm ASCII rồi `os.replace`/`shutil.move`, nên đường dẫn Unicode trên
Windows vẫn hoạt động.

### 11.5 File video không tồn tại hoặc không đọc được

- File không tồn tại → thông báo `FileNotFoundError` tiếng Việt.
- Mở được nhưng không đọc nổi frame nào → thông báo `ValueError` tiếng Việt.

### 11.6 OpenCV không đọc được FPS

Phần mềm dùng tạm `25.0` fps để tính `timestamp` và thêm cảnh báo vào manifest.

---

## 12. Ghi chú về các cấu hình cần tinh chỉnh

Các ngưỡng quan trọng nằm trong `src/config.py` và nên được hiệu chỉnh trên dữ
liệu thật:

| Class | Tham số | Mặc định | Ghi chú |
| --- | --- | --- | --- |
| `FrameExtractionConfig` | `frame_interval` | `5` | Mật độ frame đưa vào pipeline |
| `FrameExtractionConfig` | `brightness_threshold` | `50.0` | Lọc frame quá tối |
| `FrameExtractionConfig` | `blur_threshold` | `100.0` | Lọc frame quá mờ |
| `VehicleDetectionConfig` | `conf` | `0.25` | Ngưỡng nhận diện xe |
| `VehicleDetectionConfig` | `classes` | `(2, 3, 5, 7)` | Car, motorcycle, bus, truck |
| `TrackingConfig` | `max_age` | `30` | Số frame giữ track mất dấu |
| `TrackingConfig` | `min_hits` | `3` | Ngưỡng xác nhận track |
| `TrackingConfig` | `iou_threshold` | `0.3` | Ngưỡng ghép detection ↔ track |
| `VideoPipelineConfig` | `plate_region_ratio` | `0.5` | Vùng tìm biển số khi thiếu Module 2 |
| `PostprocessConfig` | `min_confidence` | `0.35` | Ngưỡng lọc trước khi voting |
| `PostprocessConfig` | `min_readings` | `2` | Ngưỡng kết quả đáng tin cậy |
