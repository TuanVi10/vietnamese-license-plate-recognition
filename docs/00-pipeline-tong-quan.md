# 00 — Tổng quan pipeline (M0 → M7)

## Bức tranh toàn cảnh

Pipeline đọc biển số xe Việt Nam từ **video** chia thành 8 module, mỗi module làm một
nhiệm vụ nhỏ, dễ cô lập lỗi và đánh giá riêng:

```
VideoCapture
   │
   ▼
M0 FrameExtractor ── trích frame mỗi `frame_interval` frame, lọc frame mờ/tối
   │  (giữ LẠI chỉ số frame GỐC + timestamp để khớp thời điểm xe xuất hiện)
   ▼
M1 VehicleDetector ─ dò bbox PHƯƠNG TIỆN (yolo11n, 4 lớp COCO: car/motorcycle/bus/truck)
   ▼
M3 SortTracker ───── gán track_id ổn định cho từng xe qua nhiều frame (SORT)
   ▼
M2 PlateDetector ─── định vị biển số TRONG vùng xe (plate_detector.pt)
   │  ├─ có detection → source="detection", bbox biển
   │  └─ không có     → source="fallback", xấp xỉ nửa dưới bbox xe
   ▼
Crop biển (lề 15%) → chấm điểm → TopKBuffer (best-frame, KHÔNG OCR ngay)
   ▼
★ CornerRegressor ── hồi quy 4 góc TL/TR/BR/BL trên crop (yolo11n-pose)
   │  ├─ |góc| > 15° → warpPerspective(4 góc) → biển phẳng
   │  └─ ngược lại   → bbox + affine deskew (cách cũ)
   ▼
M4 Preprocess ─────── deskew (projection profile) → tách 2 dòng → CLAHE → resize
   ▼
M5 OCR ────────────── PaddleOCR (PP-OCRv6), fallback EasyOCR
   ▼
M6 PostProcessor ──── chuẩn hoá + sửa ký tự dễ nhầm + kiểm định dạng + voting
   │                   (hoặc Fusion: gộp theo VỊ TRÍ ký tự)
   ▼
M7 Output ─────────── JSON + CSV + txt + video overlay
```

## Vì sao chia module?

1. **Cô lập lỗi** — biết chính xác tầng nào sai (detect? align? OCR?).
2. **Đo đạc riêng** — mỗi tầng có metric riêng (mAP cho detect, IoU cho corner, exact/sim cho OCR).
3. **Thay thế linh hoạt** — đổi OCR engine, đổi model detect mà không đụng phần còn lại.

## Dữ liệu đi qua pipeline (điểm quan trọng)

- Mỗi frame đi qua M1+M3 được gán **track_id**; mọi lần đọc biển gom theo track_id.
- Cuối video, **mỗi track chỉ trả 1 biển số** (voting/fusion nhiều lần đọc).
- Best-frame giữ **top-K crop** mỗi track để OCR (tránh OCR 300 frame của 1 xe).
