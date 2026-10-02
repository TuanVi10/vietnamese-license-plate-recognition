# 02 — Frame Extractor (`src/frame_extractor.py`, Module 0)

## Vai trò

Đọc video, trích frame theo bước nhảy, lọc frame mờ/tối, và **giữ lại chỉ số frame GỐC**
+ timestamp để các module sau khớp đúng thời điểm xe xuất hiện.

## Cách hoạt động

1. Mở `cv2.VideoCapture` (an toàn với đường dẫn Unicode trên Windows).
2. Xét mỗi `frame_interval` frame (VD 5 → ~6 frame/giây với video 30fps).
3. Lọc chất lượng: giữ frame có `mean_brightness ≥ brightness_threshold` **và**
   `variance_of_laplacian ≥ blur_threshold`.
4. Trả về `FrameRecord(frame_index, timestamp, frame)`.

## Điểm đặc biệt trong logic

### 1. Xử lý đường dẫn Unicode (`_open_capture`)
`cv2.VideoCapture` **không mở được đường dẫn Unicode trên Windows**. Hàm thử mở trực tiếp;
nếu thất bại và đường dẫn có ký tự ngoài ASCII thì **copy video sang 1 file tạm ASCII** rồi
mở file tạm. Đây là lý do `image_utils.imread_unicode` cũng phải tồn tại cho ảnh.

### 2. Chống NaN từ OpenCV (`_finite_float`)
OpenCV có thể trả `NaN` cho `CAP_PROP_FPS` / `CAP_PROP_FRAME_COUNT` với vài codec.
Nếu không chặn, `NaN` lan vào timestamp → `int(NaN)` ném `ValueError`. Hàm ép mọi giá trị
về `float` hữu hạn (fallback `0.0`).

### 3. Phá vòng lặp khi codec "treo" (`_MAX_CONSECUTIVE_READ_FAILURES`)
Một số codec liên tục trả `(ret=True, frame=None)` mà không bao giờ báo `ret=False` (hết video).
Đếm số lần đọc lỗi liên tiếp; vượt quá 10 thì coi như hết video.

### 4. Độ mờ = variance of Laplacian
Đây là chỉ số blur kinh điển: ảnh nét → biến đổi Laplacian có phương sai lớn; ảnh mờ → nhỏ.
Rẻ (chỉ 1 phép tích chập), đủ tốt để loại frame mờ trước khi tốn công detect/OCR.

### 5. Lọc frame chỉ dùng numpy+cv2
Module 0 **không cần** ultralytics/paddleocr/torch → chạy được trong môi trường tối thiểu,
đúng tinh thần "mắt xích đầu tiên phải nhẹ".
