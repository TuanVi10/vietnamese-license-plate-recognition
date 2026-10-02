# 03 — Vehicle Detector (`src/detect_vehicle.py`, Module 1)

## Vai trò

Dò bbox **phương tiện** trong frame bằng YOLO11n — đây là **vùng tìm kiếm** cho Module 2
(dò biển số) và là đầu vào cho Module 3 (tracking).

## Cách hoạt động

1. Chạy `model.predict(frame)` với các lớp COCO coi là phương tiện: `{2: car, 3: motorcycle, 5: bus, 7: truck}`.
2. Chỉ giữ detection thuộc các lớp này (KHÔNG có bicycle).
3. `clip_bbox` kẹp bbox vào biên ảnh.
4. Sắp giảm dần theo confidence.

## Điểm đặc biệt trong logic

### 1. Chế độ "toàn khung" (full-frame fallback)
Nếu `model_path = None` (chưa có model), detector trả về **1 bbox phủ toàn frame** coi cả
frame là 1 phương tiện → toàn pipeline (M3, M4…) vẫn chạy được để test logic tracking mà
chưa cần model. Đây là cách "không để thiếu model chặn tiến độ".

### 2. `clip_bbox` "làm sạch" dữ liệu bẩn
Chấp nhận bbox méo (mảng `(1,4)`/`(4,1)`, thừa phần tử, chứa `NaN`/`inf`) mà **không ném
ValueError**: lấy 4 giá trị đầu, thay NaN/inf bằng 0, rồi kẹp vào biên ảnh, đảm bảo rộng/cao ≥ 1px.
Nhờ vậy dữ liệu bẩn từ Module 2 không làm sập pipeline.

### 3. `classes == ()` là "không lọc" (chứ không phải "lọc rỗng")
Nếu người dùng cố ý truyền `classes=()` thì KHÔNG ghi đè bằng mặc định → YOLO trả về MỌI lớp
COCO. Đây là quy ước tinh tế để phân biệt "chưa set" (dùng mặc định) và "cố ý rỗng" (lấy hết).
