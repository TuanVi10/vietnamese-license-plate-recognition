# PLAN — Hoàn thành dự án Nhận diện Biển số xe qua Video

> Ngày lập: 2026-09-26. Nguồn căn cứ: `spec-ke-hoach-du-an-nhan-dien-bien-so-xe-v2_updated.md`
> (kiến trúc 8 module 0→7, 7 tuần) + `spec-tai-dataset-bien-so-vn.md` (dataset thô đã tải xong).
>
> **Quy tắc vận hành:** chỉ thực thi đúng phase đã được duyệt. Sau mỗi phase phải DỪNG,
> báo cáo kết quả và chờ duyệt trước khi chuyển phase tiếp theo.

---

## 1. Phạm vi dự án

Hệ thống ALPR video: `video → [0] Frame Extract/Filter → [1] Vehicle Detect (YOLOv8n)
→ [2] Plate Detect (YOLO11n fine-tuned) → [3] Tracking (SORT/ByteTrack) → [4] Preprocess
plate → [5] OCR (PaddleOCR PP-OCRv5) → [6] Postprocess/Voting → [7] Output/Visualize`.

## 2. Gap list (phần còn thiếu để hoàn thành)

### 2.1 Data + Train (Module 2) — đường găng
1. Dataset gộp cuối `datasets/images/{train,val,test}` + `datasets/labels/...` + `data.yaml` (80/10/10).
2. `train_detector.py` + config training (§2.2 spec v2).
3. `plate_detector.pt` (fine-tuned).
4. Báo cáo train (results.png, mAP@0.5 val).

### 2.2 Model artifacts + môi trường
5. `models/` + `yolov8n.pt` (detect xe) — `yolo11n.pt` dùng tạm được cho cả Module 1 & pretrain Module 2.
6. Runtime OCR: PaddleOCR PP-OCRv5 (CPU, `enable_mkldnn=False`) + EasyOCR fallback.
7. `requirements.txt` đầy đủ cho pipeline video.

### 2.3 Tích hợp pipeline video
8. `main.py` CLI chạy pipeline 0→7 (argparse, xuất `results.csv/json`, `plate_crops/`, `output_demo.mp4`).
9. Video test mẫu (`data/raw/*.mp4`).
10. Chạy end-to-end + tinh chỉnh ngưỡng (Module 0, `max_age`, conf OCR, luật biển, min-width).

### 2.4 Đánh giá & báo cáo (tuần 6–7)
11. Ground-truth gán nhãn tay tập video test nhỏ.
12. Script đo 5 metric: mAP@0.5, Plate Detection Rate, Recognition Accuracy, ID Switch Rate, FPS.
13. Báo cáo/slide + video demo.

### 2.5 Lặt vặt
14. Đồng nhất tên file theo spec §4 (`extract_frames.py`/`detect_plate.py`/`add_missing_data.py`) — không bắt buộc.
15. Dọn `yolo11n.pt` vào `models/`; dọn session dir cũ sau khi backup.

## 3. Kế hoạch theo phase (mỗi phase có Definition of Done)

### Phase 0 — Baseline & cấu trúc (½ ngày)
1. Tạo `models/`, `data/raw/`, `results/`; dời `yolo11n.pt` → `models/`.
2. Xác nhận `src/` (Module 0,1,3,4,5,6,7) import sạch bằng `.venv-datasets`;
   chạy `tools/self_check.py` + `tools/self_check_tracking.py`.
3. Viết `requirements.txt` đầy đủ.
- **DoD:** mọi module import không lỗi; 2 self-check pass; cấu trúc thư mục sẵn sàng.

### Phase 1 — Chuẩn bị dataset (½–1 ngày)
1. Chạy pipeline dataset đã cứu trên dữ liệu thật: `--labels-format yolo --labels-path datasets/raw --label-mode import --val-ratio 0.1 --test-ratio 0.1 --seed 42` (VOC/COCO thì convert trước).
2. Sinh `datasets/images|labels/{train,val,test}` + `data.yaml`; verify `nc: 1`, class `license_plate`, preview.
- **DoD:** dataset sẵn sàng train; báo cáo số ảnh thật sau lọc trùng.

### Phase 2 — Train Module 2 (2–4h GPU)
1. Viết `train_detector.py` bám §2.2: YOLO11n, `epochs=100`, `imgsz=640`, `batch=16` (Colab)/`8` (RTX 3050), `AdamW`, `box=7.5`, **`fliplr=0.0`**, mosaic + hsv/degrees nhẹ, early stopping 20, `project=runs/detect`.
2. Train trên Colab T4; smoke 1 epoch trên RTX 3050.
3. `model.val()` → mAP@0.5; lưu `best.pt` → `models/plate_detector.pt`.
- **DoD:** có `plate_detector.pt` + `results.png` + mAP@0.5 val.

### Phase 3 — Setup OCR (½ ngày)
1. Cài `paddlepaddle` (CPU) + `paddleocr` (ưu tiên tách `.venv-ocr` nếu xung đột torch); `enable_mkldnn=False`.
2. Smoke OCR 1 biển mẫu; fallback EasyOCR nếu lỗi.
- **DoD:** OCR đọc đúng 1 biển mẫu; charset 0-9 A-Z hoạt động.

### Phase 4 — CLI + tích hợp pipeline (1–2 ngày)
1. Viết `main.py` CLI nối `VideoPipeline`.
2. Sinh/lấy video giao thông thật; chạy end-to-end → `results.csv/json` + `plate_crops/` + `output_demo.mp4`.
3. Tinh chỉnh ngưỡng Module 0, `max_age`, conf OCR, luật biển.
- **DoD:** chạy xong 1 video, đủ output; track_id ổn định; biển đọc hợp lý.

### Phase 5 — Đánh giá & báo cáo (1–2 ngày)
1. Gán nhãn tay ground truth video test nhỏ.
2. Viết script đo 5 metric.
3. Tinh chỉnh theo metric đo được.
- **DoD:** bảng metric thực tế.

### Phase 6 — Hoàn thiện (½ ngày)
1. README + requirements + dọn session dir cũ.
2. (Tùy chọn) đổi tên file khớp spec §4 hoặc cập nhật spec.

## 4. Rủi ro chính
- **PaddlePaddle + torch cùng venv** → tách `.venv-ocr` nếu xung đột; fallback EasyOCR.
- **Colab T4 sẵn có / giới hạn** → local RTX 3050 (batch 8) làm dự phòng.
- **Không có video thật** → sớm xin/quay 1 clip; không thì demo bằng video tổng hợp (yếu hơn).
- **Dataset trùng ảnh giữa nguồn** → bắt buộc dedupe bằng hash trước khi chia split.

## 5. Quyết định cấu trúc (Phase 0)
- Đặt `models/`, `data/raw/`, `results/` ở **gốc `S:\M_AGENT\CV`** (song song `datasets/` đã có sẵn ở gốc).
  Spec §4 ghi `data/models/`, nhưng `datasets/` thực tế đang ở gốc CV nên giữ nhất quán tại gốc;
  sẽ chuẩn hoá lại nếu cần ở phase sau.
- Video pipeline `src/` hiện đang ở `dataset_prep_src/src/` (bản đã cứu). Chưa di chuyển trong Phase 0;
  vị trí chuẩn sẽ chốt ở phase sau (đưa ra gốc CV hoặc giữ nguyên).

## 6. Trạng thái
- [x] Dataset thô: 18,727 ảnh + nhãn tại `datasets/raw/` (4 nguồn).
- [x] Code dataset-prep đã cứu về `dataset_prep_src/`; smoke test pass.
- [x] Code pipeline video 0→7 đã cứu về `dataset_prep_src/src/` (chưa có CLI `main.py`).
- [x] `.venv-datasets`: ultralytics 8.4.163 + torch 2.14.0+cu126 (GPU RTX 3050 6GB verified).
- [x] Phase 0: HOÀN THÀNH (2026-09-26).
- [x] Phase 1: HOÀN THÀNH (2026-09-26) — dataset tại `datasets/dataset/`.
- [x] Phase 2: HOÀN THÀNH (2026-09-26) — `models/plate_detector.pt` (demo 3 epoch).
- [x] Phase 3: HOÀN THÀNH (2026-09-26) — OCR = EasyOCR (fallback spec).
- [x] Phase 4: HOÀN THÀNH (2026-09-26) — demo end-to-end chạy được trên ảnh thật.
- [ ] Phase 5–6: chưa bắt đầu (chờ duyệt từng phase).

## 11. Kết quả Phase 4 (CLI + demo end-to-end)
- CLI: `S:\M_AGENT\CV\main.py` (nối pipeline 0→7; xuất `results.csv/json` + `plate_crops/`
  + `output_demo.mp4`).
- Demo: dựng video từ 15 ảnh THẬT có biển số (`tools/make_demo_video.py`) rồi chạy full
  pipeline (yolo11n.pt dò xe + plate_detector.pt dò biển + SORT track + EasyOCR + voting).
- Kết quả: **8 xe tracked**, **3 xe đọc được biển đúng format VN**:
  `61-G6 8882` (0.89), `61-G7 9984` (0.90), `67S-213` (0.65) — 5 xe còn lại không đọc được
  (biển nhỏ/mờ — bình thường).
- Sản phẩm demo: `results/{results.csv, results.json, plate_crops/, tracks_frames/, output_demo.mp4}`.
- Lưu ý: video tổng hợp `make_sample_video.py` chỉ để test Module 0 (hình xe chữ nhật, COCO
  không nhận ra) → demo dùng ảnh thật để detect/OCR thực sự hoạt động.

## 10. Kết quả Phase 3 (OCR)
- Engine: **EasyOCR 1.7.2** (fallback theo spec), cài trong `.venv-datasets` (torch cu126 —
  không xung đột, có thể dùng GPU; mặc định chạy CPU cho crop nhỏ).
- PaddleOCR đã thử nhưng bị chặn bởi yếu tố ngoài tầm kiểm soát:
  - **CPU**: PaddlePaddle 3.3.1 crash `oneDNN/PIR` (bug backend CPU Windows), `FLAGS_use_mkldnn=False`
    và `FLAGS_enable_pir_api=False` đều không chữa được.
  - **GPU**: tải `paddlepaddle-gpu==3.2.2` (cu126) từ `paddlepaddle.org.cn` bị **stall** giữa chừng
    (server TQ không ổn định từ mạng này) — đã dừng + dọn rác.
- Smoke test `tools/ocr_smoke_easyocr.py`: đọc được biển mẫu + crop biển thật (conf ~0.65–0.76),
  charset 0-9 A-Z hoạt động (không còn ký tự `-`/`.`/khoảng trắng).
- Ghi chú: EasyOCR chạy **in-process** (load model 1 lần, tái dùng cho mọi crop) → không cần
  service HTTP riêng; nếu vẫn muốn service tách biệt có thể làm ở Phase 4.

## 9. Kết quả Phase 2 (train demo)
- Script: `train_detector.py` (bám §2.2: YOLO11n, AdamW, box=7.5, **fliplr=0.0**,
  mosaic=1.0, degrees=5.0, imgsz=640).
- Chạy demo: **3 epoch**, batch 8, trên **RTX 3050 6GB** (theo yêu cầu "epoch thấp",
  thay vì 100 epoch trên Colab).
- Kết quả val (epoch cuối): **mAP50=0.977, mAP50-95=0.647, P=0.967, R=0.953**.
- Sản phẩm: `models/plate_detector.pt` (5.21 MB) + `runs/detect/vn_plate_demo/`
  (`results.png`, `results.csv`, `args.yaml`, `train_batch*.jpg`, `weights/`).

## 8. Kết quả Phase 1
- Dataset: 18,727 ảnh thô → **18,621 ảnh** (106 trùng MD5 bị loại), **18,983 box**.
- Split **80/10/10**: train 14,897 / val 1,862 / test 1,862; class `license_plate` (nc=1).
- Sản phẩm: `datasets/dataset/{data.yaml, images/, labels/, DATASET_CARD.md, dataset_stats.json, qc/preview.jpg}`.
- Cấu hình: `datasets/pipeline_config.json` (nới `filter.min_aspect_ratio` 1.0 → 0.1 để
  giữ biển nghiêng / biển xe máy 2 dòng); effective config lưu tại `datasets/work/effective_config.json`.
- Xử lý dữ liệu lệch: nguồn `raw_annguyen` có **84 file nhãn polygon** → đã chuyển
  thành bbox bằng `dataset_prep_src/tools/normalize_annguyen_labels.py`; bản gốc lưu tại
  `datasets/_backup_raw_annguyen_labels/`.
- Trung gian `datasets/work/` (~1.3 GB) có thể xoá sau khi đã duyệt dataset.

## 7. Ghi chú môi trường (rút ra từ Phase 0)
- Mọi script pipeline/self-check in tiếng Việt ra console Windows phải bật UTF-8,
  nếu không sẽ dính `UnicodeEncodeError` (console PowerShell mặc định cp1252):
  ```powershell
  $env:PYTHONUTF8 = '1'
  # hoặc: $env:PYTHONIOENCODING = 'utf-8'
  ```
- Tên file/đường dẫn có dấu tiếng Việt: dùng `imread_unicode`/`imwrite_unicode`
  (đã có trong `src/image_utils.py`) thay vì `cv2.imread` trực tiếp.
