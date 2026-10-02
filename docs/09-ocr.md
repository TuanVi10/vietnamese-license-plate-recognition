# 09 — OCR (`src/ocr.py` + `src/ocr_subprocess.py`, Module 5)

## Vai trò

Đọc text từ ảnh biển đã tiền xử lý. Engine chính **PaddleOCR (PP-OCRv6)**, fallback EasyOCR.

## Cấu trúc

- `BaseOCREngine` — interface chung (charset, drop_empty).
- `PaddleOCREngine` — hỗ trợ cả API 2.x lẫn 3.x.
- `EasyOCREngine` — fallback (chỉ dùng `allowlist` charset).
- `SubprocessPaddleEngine` — gọi PaddleOCR ở **venv riêng** qua stdin/stdout.
- `create_ocr_engine` — tự fallback PaddleOCR → EasyOCR.

## Điểm đặc biệt trong logic

### 1. Giữ `raw_text` (có dấu phân cách) tách biệt với `text` (đã lọc charset)
`OCRItem.raw_text` giữ nguyên `-`, `.`, khoảng trắng **trước** khi lọc charset. Vì **vị trí
dấu phân cách** giúp Module 6 suy ra series có 1 hay 2 chữ cái (VD `51F-12345` → series 1 chữ
"F"; `29-B1 12345` → series 2 chữ "B1"). Nếu lọc charset trước sẽ mất thông tin này.

### 2. Charset chỉ `0-9 A-Z`
Biển VN chỉ dùng số + chữ Latin in hoa. Giới hạn charset (EasyOCR dùng `allowlist`) giảm nhiễu
kết quả (không ra ký tự `@`, `#`, tiếng Trung…).

### 3. Subprocess cách B — tránh xung đột torch/paddle
PyTorch (cu126/cuDNN 9.10) và PaddlePaddle (cu118/cuDNN 8.9) **không import chung 1 process**.
`SubprocessPaddleEngine` spawn `paddle_ocr_server.py` 1 lần, giao tiếp stdin/stdout: gửi ảnh
dạng base64, nhận text. `plate_reader` gọi OCR qua subprocess → pipeline chạy PaddleOCR mà
không cần import paddle trong process chính.

### 4. `sort_reading_order` — sắp thứ tự đọc
OCR trả nhiều item (dòng/khối) không theo thứ tự. Sắp lại theo thứ tự đọc (trên→dưới,
trái→phải) trước khi ghép thành chuỗi cuối, nếu không chuỗi biển bị đảo vị trí.
