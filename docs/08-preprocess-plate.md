# 08 — Tiền xử lý biển số (`src/preprocess_plate.py`, Module 4)

## Vai trò

Biến crop biển số thô thành ảnh "sạch" để OCR đọc chính xác.

## Thứ tự thao tác (BẮT BUỘC, theo spec)

```
1. crop theo bbox (nới lề bbox_margin)
2. deskew (chỉnh nghiêng)          ← LUÔN trước khi tách dòng
3. tách 2 dòng → ghép ngang
4. CLAHE (tăng tương phản cục bộ)
5. upscale nếu quá nhỏ
6. resize giữ aspect + padding
```

> Làm ngược deskew ↔ tách dòng sẽ **tách sai dòng** khi ảnh còn nghiêng.

## Điểm đặc biệt trong logic

### 1. Deskew bằng projection profile (không phải model)
`estimate_skew_angle`: nhị phân hoá → quét góc trong `[-20°, 20°]` → chọn góc làm **phương sai
histogram chiếu dọc lớn nhất** (khi thẳng, các đỉnh/đáy ký tự thẳng hàng → histogram "gồ ghề"
nhất) → tinh chỉnh bước nhỏ + nội suy parabol. Rẻ, không cần model, đủ cho biển gần thẳng.

### 2. Tách 2 dòng bằng "thung lũng" giữa các dòng
`split_lines` tìm vùng trống theo chiều ngang (projection profile) chia ảnh thành 2 dòng.
`_looks_two_line` chỉ xét khi **aspect w/h < 1.8** (biển 2 dòng xe máy gần vuông) — tránh cắt
nhầm biển 1 dòng dài.

### 3. Ghép ngang 2 dòng với `merge_gap`
Biển 2 dòng đọc là: dòng trên (mã tỉnh + series) + dòng dưới (số đăng ký). Ghép ngang 2 dòng
thành 1 chuỗi, chèn khoảng trắng `merge_gap=12px` giữa 2 dòng để OCR không dính chữ.

### 4. Resize giữ aspect + padding (KHÔNG kéo dãn)
Kéo dãn ảnh biển (stretch) làm méo ký tự → OCR sai. `resize_with_padding` giữ tỉ lệ, chèn nền
trắng (255) vào phần thừa. Nền trắng vì biển trắng dân sự.

### 5. Sinh nhiều biến thể để voting (`iter_variants`)
Với ảnh tĩnh chỉ có 1 ảnh, "voting" dựa trên **nhiều lần đọc của cùng ảnh**: bản merged,
bản nhị phân (`merged_binary`), và từng dòng riêng (`line_0`, `line_1`). Nếu `too_small` thì
KHÔNG sinh biến thể nào (tránh OCR tốn compute vô ích).
