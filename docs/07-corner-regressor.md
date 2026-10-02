# 07 — Corner Regressor (`src/corner_regressor.py`)

## Vai trò

Hồi quy **4 góc** của biển (TL, TR, BR, BL) trên crop đã biết chắc có biển, dùng
`yolo11n-pose` (`kpt_shape=[4,3]`). Từ 4 góc suy **góc nghiêng** để quyết định có warp hay không.

## Cách hoạt động

1. `model.predict(crop, conf, imgsz=160)`.
2. Lấy keypoint của detection có **conf trung bình 4 keypoint cao nhất**.
3. `angle = degrees(atan2(tr.y − tl.y, tr.x − tl.x))` — góc từ cạnh TRÊN TL→TR.

## Điểm đặc biệt trong logic

### 1. Gate theo góc: chỉ warp khi |angle| > 15°
Đã đo thực nghiệm (ceiling test): corner-warp **chỉ có lợi ở biển nghiêng >15°** (+37 điểm
exact), còn **vô hại/hại nhẹ** ở dải <5°. Vì vậy thay vì warp mọi ảnh (tốn + rủi ro làm hỏng
ảnh thẳng), pipeline dùng **gate**:
- `|angle| > 15°` → `warpPerspective(4 góc)` (chính xác cho biển nghiêng mạnh).
- ngược lại → bbox + affine deskew (cách cũ, đủ tốt cho biển gần thẳng).

### 2. Chọn detection theo conf keypoint (không phải conf box)
Có thể model trả nhiều detection; chọn detection có **mean conf của 4 keypoint** cao nhất
(`argmax(confs.mean(axis=1))`) — vì ta quan tâm độ tin cậy của 4 góc, không phải của bbox.

### 3. Góc nghiêng suy từ cạnh TRÊN (TL→TR), không dùng trục ảnh
Biển nghiêng theo phối cảnh; cạnh trên TL→TR là tín hiệu ổn định nhất cho góc roll (xoay trong
mặt phẳng ảnh), dùng `atan2` để đúng cả hướng âm/dương và tránh lỗi chia-0.
