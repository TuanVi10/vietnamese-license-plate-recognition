# 15 — Geometry Utils (`tools/geometry_utils.py`)

## Vai trò

Tiện ích hình học cho các script gán nhãn 4 góc biển số (`ccpd_to_pose.py`, `unify_labeling.py`).

## Hàm chính

- `order_corners_geometric` — sắp 4 điểm về TL→TR→BR→BL.
- `is_convex` — kiểm tra tứ giác lồi.
- `dedupe_closing_point` — bỏ điểm cuối trùng điểm đầu (polygon khép kín của Label Studio).

## Điểm đặc biệt trong logic

### 1. Sắp 4 góc bằng công thức kinh điển (không cần ML)
```
TL = min(x + y)      BR = max(x + y)
TR = max(x - y)      BL = min(x - y)
```
Với tứ giác lồi, góc trên-trái luôn có `x+y` nhỏ nhất, dưới-phải lớn nhất; trên-phải có `x-y`
lớn nhất, dưới-trái nhỏ nhất. Thuần hình học, đúng cả khi biển nghiêng/phối cảnh.

### 2. Kiểm tra lồi bằng tích có hướng (cross product)
Duyệt 4 cạnh, tính `cross = (b-a)×(c-b)`; tứ giác lồi khi dấu KHÔNG đổi qua 4 cạnh. Dùng để
đảm bảo nhãn 4 góc hợp lệ trước khi train.

### 3. Xử lý polygon khép kín của Label Studio
Label Studio hay trả polygon 5 điểm (điểm cuối = điểm đầu). `dedupe_closing_point` bỏ điểm cuối
nếu trùng điểm đầu (trong sai số 1e-6) → về đúng 4 góc.

### 4. Kiểm chứng CCPD: 0 bất đồng trên 17 774 ảnh
Sắp theo hình học vs quy ước gốc CCPD (BR,BL,TL,TR) khớp **17 774/17 774**, tứ giác lồi 0 lỗi,
lệch keypoint round-trip max 0.0006 px — nhờ vậy dataset pose đáng tin.
