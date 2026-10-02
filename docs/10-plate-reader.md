# 10 — Plate Reader (`src/plate_reader.py`, ảnh tĩnh)

## Vai trò

Đọc biển số từ **1 ảnh tĩnh**: nối Module 2 (detect) → M4 (tiền xử lý) → M5 (OCR) → M6 (voting).
Với ảnh tĩnh, "voting theo track_id" được thay bằng **voting nhiều lần đọc của cùng 1 ảnh**
(nhiều biến thể tiền xử lý + nhiều item OCR).

## Luồng

1. (tùy chọn) dò bbox biển bằng `PlateDetector`.
2. crop + tiền xử lý → `PreprocessResult`.
3. sinh nhiều biến thể (`iter_variants`) → OCR từng biến thể.
4. gom mọi `OCRItem` → `PostProcessor.process()` (voting).
5. đóng gói `PlateReading` (text, formatted, confidence, valid, reliable…).

## Điểm đặc biệt trong logic

### 1. `_joined_raw_text` — ghép theo ĐÚNG thứ tự đọc
`OCRResult.raw_text` ghép theo thứ tự engine trả về, có thể khác thứ tự `items` (đã sort theo
thứ tự đọc). Ghép lại từ `items` đảm bảo **dấu phân cách khớp vị trí** với `text` — cần để
Module 6 suy đúng series 1 hay 2 chữ.

### 2. `reliable` = vote đủ + ảnh không quá nhỏ
`reliable = vote.reliable and not pre.too_small`. Ảnh crop quá nhỏ (dưới ngưỡng min width/height)
đánh dấu "không đáng tin cậy" và **không đưa vào OCR** — tránh kết quả bốc phét từ ảnh vô nghĩa.

### 3. Tách "valid" (đúng định dạng) khỏi "reliable" (đủ tin)
- `valid` = chuỗi khớp `[2 số][1-2 chữ][3-6 số]`.
- `reliable` = đủ số lần đọc hợp lệ + ảnh đủ lớn.
Một chuỗi có thể valid nhưng không reliable (đọc 1 lần duy nhất, ảnh nhỏ).

### 4. Trả `crop` (ảnh đã xử lý) để đối chiếu
`PlateReading.crop` giữ ảnh biển đã tiền xử lý — không serialize ra JSON nhưng dùng để vẽ
overlay/đối chiếu khi viết báo cáo (giải thích ca đọc sai).
