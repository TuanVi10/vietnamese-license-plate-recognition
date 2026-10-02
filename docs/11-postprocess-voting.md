# 11 — Post-process + Voting (`src/postprocess.py`, Module 6)

## Vai trò

Áp **luật biển số Việt Nam** + **voting** để chốt 1 chuỗi biển từ nhiều lần đọc.

## Các bước

1. Chuẩn hoá chuỗi (uppercase, bỏ ký tự ngoài charset).
2. **Sửa ký tự dễ nhầm theo vị trí kỳ vọng** (O↔0, I↔1, B↔8, S↔5, L→1…).
3. Kiểm tra định dạng `^[0-9]{2}[A-Z]{1,2}[0-9]{3,6}$`.
4. Lọc confidence thấp + sai định dạng **TRƯỚC** khi voting.
5. Vote chuỗi xuất hiện nhiều nhất (hoà → ưu tiên confidence TB cao hơn).
6. Đánh dấu "reliable thấp" nếu số lần đọc < `min_readings`.

## Điểm đặc biệt trong logic

### 1. Sửa ký tự theo "khuôn vị trí" (position template) — giảm lỗi OCR không cần train lại
`build_position_template(length)` sinh khuôn: vị trí 0,1 = `D` (số — mã tỉnh), vị trí 2 = `L`
(chữ — series đầu), vị trí 3 = `A` (cả 2 — series thứ 2, có thể không có), còn lại = `D` (số).
Rồi áp bảng `LETTER_TO_DIGIT`/`DIGIT_TO_LETTER` **đúng vị trí**:
- Ở vị trí phải là số mà thấy `O` → sửa thành `0`.
- Ở vị trí phải là chữ mà thấy `0` → sửa thành `O`.

> Cố tình KHÔNG map các cặp mơ hồ như `A→4`, `D→0`, `Q→0` vì dễ biến chuỗi rác thành "trông hợp lệ".

### 2. Lọc TRƯỚC khi vote
Nếu để lần đọc sai định dạng / conf thấp lọt vào vote, nó sẽ làm nhiễu kết quả "xuất hiện nhiều
nhất". Lọc trước → vote sạch hơn.

### 3. Dùng `raw_text` để biết series 1 hay 2 chữ
Vị trí dấu phân cách trong `raw_text` cho biết series 1 hay 2 ký tự → sinh đúng "khuôn" trước
khi sửa ký tự. (Xem `_joined_raw_text` ở plate_reader.)

### 4. `interpolate_linear` — nội suy frame thiếu
Tiện ích cho video: các frame model bỏ sót detect được lấp bằng nội suy tuyến tính giữa 2 mốc
có dữ liệu gần nhất.
