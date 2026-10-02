# Tài liệu giải thích chi tiết từng module

Mỗi file dưới đây giải thích **một module** trong pipeline nhận diện biển số xe Việt Nam:
mục đích, input/output, cách xử lý từng bước, và quan trọng nhất là **điểm đặc biệt trong
logic tính toán** (những chỗ dễ sai, mẹo kỹ thuật, quyết định thiết kế).

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
