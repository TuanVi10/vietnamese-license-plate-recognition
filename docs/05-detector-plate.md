# 05 — Plate Detector (`src/detector.py`, Module 2)

## Vai trò

Định vị **biển số** trong vùng phương tiện (đầu ra của Module 1) bằng YOLO11n đã fine-tune
(`plate_detector.pt`). Nếu không truyền model, coi toàn bộ ảnh là biển số (tiện khi đầu vào
đã là crop sẵn).

## Cách hoạt động

1. `model.predict(image, conf, imgsz)`.
2. Với mỗi box, đọc `xyxy`, `conf`, `class_id`, map sang `label` từ `model.names`.
3. Sắp giảm dần theo confidence.
4. `best_bbox()` trả bbox tự tin nhất (hoặc None nếu không detect được).

## Điểm đặc biệt trong logic

### 1. Đây chỉ là "định vị", KHÔNG phải "đọc"
Module 2 trả **bbox + confidence**, không trả text. Text là việc của M4/M5/M6. Việc tách
"tìm biển" và "đọc biển" cho phép thay riêng từng model.

### 2. Source "detection" vs "fallback" (do video_pipeline quyết định)
Khi module này **có** detection nằm trong/chồng lấn bbox xe → `source="detection"` (biển thật).
Khi **không** → `source="fallback"` (xấp xỉ nửa dưới bbox xe). Đây là tín hiệu quan trọng cho
best-frame (fallback không bao giờ được ưu tiên hơn detection thật) và cho báo cáo
(model detect = số crop có source=detection).

### 3. Confidence là "xác suất có biển", KHÔNG phải "độ chính xác 4 góc"
Model có thể tự tin "đây là biển" nhưng vẽ bbox lệch. Vì vậy cần tách `det_conf` (Module 2)
khỏi "điểm chất lượng crop" (Module best-frame).
