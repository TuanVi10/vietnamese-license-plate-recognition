# Tài liệu giải thích chi tiết từng module

Mỗi file dưới đây giải thích **một module** trong pipeline nhận diện biển số xe Việt Nam:
mục đích, input/output, cách xử lý từng bước, và quan trọng nhất là **điểm đặc biệt trong
logic tính toán** (những chỗ dễ sai, mẹo kỹ thuật, quyết định thiết kế).

## FILE NÀO CHẠY ĐƯỢC (Entry Points)

**3 file gốc (root):**

| File | Vai trò |
|---|---|
| `main.py` | CLI chạy pipeline VIDEO (M0→M7) — `python main.py --video <mp4>` |
| `train_detector.py` | Fine-tune YOLO11n dò biển số — `python train_detector.py --epochs 3` |
| `download_datasets.py` | Tải 3 dataset Roboflow (cần `ROBOFLOW_API_KEY`) |

**41 script trong `dataset_prep_src/tools/`** (chạy `python tools/<nhóm>/<tên>.py`) — gom 7 nhóm:
`data/` (10), `train/` (4), `eval/` (13), `run/` (6), `sample/` (2), `check/` (5),
`utils/` (1).

**Thư viện (không tự chạy, chỉ import):** `dataset_prep_src/src/*.py` (18 module core) +
`dataset_prep_src/tools/utils/geometry_utils.py`.

> Ghi chú sửa lỗi: docs cũ chưa có 2 module `src/ocr_subprocess.py` (PaddleOCR qua subprocess)
> và `src/sample_data.py` (sinh biển giả). `geometry_utils.py` nằm ở **`tools/`** (không phải `src/`).

## Thứ tự đọc gợi ý

| # | File | Module | Nội dung chính |
|---|---|---|---|
| 0 | [00-pipeline-tong-quan.md](00-pipeline-tong-quan.md) | Toàn pipeline | Luồng tổng thể M0→M7 |
| 1 | [01-config.md](01-config.md) | Config | Mọi ngưỡng/trọng số gom 1 chỗ |
| 2 | [02-frame-extractor.md](02-frame-extractor.md) | M0 | Trích + lọc frame (sáng/nét) |
| 3 | [03-detect-vehicle.md](03-detect-vehicle.md) | M1 | Dò phương tiện YOLO11n |
| 4 | [04-tracker-sort.md](04-tracker-sort.md) | M3 | SORT: Kalman + Hungarian + IoU |
| 5 | [05-detector-plate.md](05-detector-plate.md) | M2 | Định vị biển số |
| 6 | [06-best-frame.md](06-best-frame.md) | Best-frame | Chọn top-K crop tốt nhất/track |
| 7 | [07-corner-regressor.md](07-corner-regressor.md) | Corner | 4 góc + gate >15° → warp |
| 8 | [08-preprocess-plate.md](08-preprocess-plate.md) | M4 | Deskew, tách 2 dòng, CLAHE |
| 9 | [09-ocr.md](09-ocr.md) | M5 | PaddleOCR vs EasyOCR |
| 10 | [10-plate-reader.md](10-plate-reader.md) | Ảnh tĩnh | Đọc 1 ảnh biển |
| 11 | [11-postprocess-voting.md](11-postprocess-voting.md) | M6 | Luật biển VN + voting |
| 12 | [12-fusion.md](12-fusion.md) | Fusion | Gộp theo vị trí ký tự |
| 13 | [13-video-pipeline.md](13-video-pipeline.md) | Glue | Nối mọi module trong video |
| 14 | [14-visualize-output.md](14-visualize-output.md) | M7 | Overlay + xuất kết quả |
| 15 | [15-geometry-utils.md](15-geometry-utils.md) | Hình học | Sắp 4 góc, kiểm tra lồi |
| 16 | [16-image-utils.md](16-image-utils.md) | Tiện ích | Unicode I/O, resize, xoay |
| 17 | [17-data-pipeline-labeling.md](17-data-pipeline-labeling.md) | Dữ liệu | Xây dataset + gán nhãn |
| 18 | [18-evaluation.md](18-evaluation.md) | Đánh giá | mAP, IoU, detect-rate, OCR |
| 19 | [19-diem-dac-biet-tong-hop.md](19-diem-dac-biet-tong-hop.md) | Tổng hợp | Bảng tóm tắt mọi "điểm đặc biệt" |

## Bắt đầu nhanh

1. Đọc [00-pipeline-tong-quan.md](00-pipeline-tong-quan.md) để nắm bức tranh toàn cảnh.
2. Muốn hiểu **logic tính toán đặc biệt**, đọc thẳng [19-diem-dac-biet-tong-hop.md](19-diem-dac-biet-tong-hop.md).
3. Đào sâu từng module theo bảng trên.
