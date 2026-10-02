# 16 — Image Utils (`src/image_utils.py`)

## Vai trò

Tiện ích ảnh dùng chung: đọc/ghi Unicode, chuyển màu, crop, xoay, cắt viền, đo mờ/sáng.

## Điểm đặc biệt trong logic

### 1. Đọc/ghi ảnh an toàn với đường dẫn Unicode
`cv2.imread`/`cv2.imwrite` **không hỗ trợ Unicode trên Windows**. Giải pháp:
- `imread_unicode`: đọc file bằng `np.fromfile` → `cv2.imdecode`.
- `imwrite_unicode`: `cv2.imencode` → `buffer.tofile(path)`.
Đây là lỗi "kinh điển" của OpenCV trên Windows, phải xử lý ở mọi chỗ I/O ảnh.

### 2. `rotate_image` mở rộng canvas để không cắt góc
Xoay quanh tâm với `getRotationMatrix2D`, nhưng **tính lại kích thước canvas** theo
`new_w = h·sin + w·cos` để không cắt mất nội dung 4 góc. Phần thêm lấp bằng nền trắng (255).

### 3. `trim_uniform_borders` cắt viền đồng màu
Sau khi xoay, xuất hiện viền đồng màu (nền). Hàm đo màu nền (median) rồi cắt các hàng/cột có
pixel "khác nền" ngoài ngưỡng `tolerance` — loại viền thừa trước khi OCR.

### 4. `resize_by_height` chọn interpolation theo hướng
Phóng to (`scale > 1`) → `INTER_CUBIC` (mượt); thu nhỏ → `INTER_AREA` (tránh aliasing). Chi tiết
nhỏ nhưng ảnh hưởng chất lượng OCR.

### 5. `binarize_otsu` — nhị phân hoá bằng Otsu + median blur
`medianBlur` trước khi Otsu để giảm nhiễu hạt trước khi tách ngưỡng (chữ vs nền).
