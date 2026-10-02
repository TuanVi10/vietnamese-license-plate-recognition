# PLAN — PIPELINE TỔNG ĐÃ CHỐT (chờ duyệt)

Ngày: 30/09/2026. Đây là bản tổng hợp cuối để duyệt; sau khi duyệt tôi tự chạy tuần tự.

---

## 1. Pipeline mục tiêu (luồng cuối)

```
video.mp4
  │
  ├─ M0 FrameExtractor                      (giữ nguyên)
  ├─ M1 VehicleDetector (yolo11n.pt)        (giữ nguyên)
  ├─ M3 SortTracker → track_id              (giữ nguyên)
  │
  ├─ M2 PlateDetector (plate_detector.pt)   (giữ nguyên — ĐỊNH VỊ biển)
  │      └─ bbox + det_conf
  │
  ├─ crop với bbox_margin = 0.15            (đã chốt)
  │
  ├─ ★ CornerRegressor (yolo11n-pose, 4 keypoint)  [MỚI]
  │      └─ 4 góc (TL,TR,BR,BL) + suy góc nghiêng
  │           └─ gate: |góc| > 15° ?
  │                ├─ CÓ  → warpPerspective(4 góc) → ảnh biển phẳng
  │                └─ KHÔNG → bbox + affine deskew (cách cũ)
  │
  ├─ ★ PaddleOCR (PP-OCRv6) → text         [MỚI, thay EasyOCR]
  ├─ M6 PostProcessor / fusion              (giữ nguyên)
  └─ M7 Output                              (giữ nguyên)
```

## 2. Quyết định đã chốt

| Hạng mục | Quyết định |
|---|---|
| OCR engine | **PaddleOCR (PP-OCRv6) pretrained**, KHÔNG fine-tune trên data thật |
| Định vị biển | **`plate_detector.pt`** (detector.py), KHÔNG dùng pose model |
| Corner regressor | `yolo11n-pose`, `kpt_shape=[4,3]`, train trên **CCPD-warped (35k)** — KHÔNG dùng data thật VN để train |
| `bbox_margin` | **0.15** |
| `conf_corner` | **gate theo góc**: chỉ warp khi `|góc| > 15°`, còn lại fallback affine |

## 3. Đã hoàn thành (evidence)

- ✅ PaddleOCR cài + chạy GPU (paddle 3.3.1 + paddleocr 3.7.0, PP-OCRv6), đọc biển VN **74.6% exact / 0.917 sim** ở trần (so EasyOCR 6.6%/0.614).
- ✅ Ceiling test xác định corner-warp chỉ lợi ở **>15°** (+37 điểm exact; hại nhẹ ở <5°).
- ✅ Sinh **35 443** mẫu CCPD-warped (`gen_ccpd_warped.py`).
- ⏳ Chuyển đổi sang YOLO-pose đang chạy (`ccpd_warped_to_yolo.py`).

## 4. KẾ HOẠCH tự chạy (sau khi duyệt)

1. **Hoàn tất chuyển đổi dữ liệu** (đang chạy nền).
2. **Train corner regressor**: `train_corner_regressor.py --epochs 100 --batch 64 --imgsz 160` (yolo11n-pose, 4 keypoint).
3. **Đánh giá corner regressor**:
   - IoU: `iou_quad`, `iou_kpbox` (trên crop, tách dải góc).
   - Đóng vòng OCR: 213 ảnh ceiling test qua regressor thật → so `sim`/`exact` với mốc **OLD / CEILING** (đặc biệt dải >15°).
4. **Tích hợp vào pipeline** (`src/`):
   - Thêm `CornerRegressor` vào `video_pipeline.py`/`plate_reader.py` (crop → 4 góc → gate → warp/fallback).
   - Nối `PaddleOCR` vào `src/ocr.py` (wrapper đã có sẵn).
5. **Test end-to-end** trên video thật.

## 5. Điểm chưa chốt (xử lý ở bước 4–5, không chặn bước 1–3)

- **Cách chạy PaddleOCR trong pipeline** → **CHỐT: cách B** — giữ `.venv-paddle` riêng, pipeline gọi OCR qua **worker subprocess** (tránh xung đột torch cu126/cuDNN 9.10 vs paddle cu118/cuDNN 8.9).
- **Ngưỡng góc gate**: mặc định 15°; có thể tune sau khi có kết quả regressor.
- **2 nhãn GT lỗi** (`001106/001107` tiny_area): bỏ qua, không ảnh hưởng.

---
*Sau khi bạn gõ "duyệt", tôi chạy tuần tự bước 1→5, dừng báo cáo khi tới điểm cần quyết định (mục 5).*
