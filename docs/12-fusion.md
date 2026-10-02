# 12 — Fusion (`src/fusion.py`)

## Vai trò

Gộp nhiều lần đọc của 1 track thành 1 biển số cuối theo **vị trí ký tự** — khác với
`VotingAggregator` (vote theo CẢ CHUỖI).

## Tại sao vote theo vị trí tốt hơn vote cả chuỗi?

- Vote cả chuỗi: 5 lần đọc "30A12345", 1 lần "30A12346" (sai 1 ký tự) → chuỗi đúng 5 phiếu,
  chuỗi sai 1 phiếu → vẫn đúng. Nhưng nếu mỗi lần sai ở 1 vị trí KHÁC nhau (vd "30A12345",
  "30A12346", "30A22345"...) thì **mỗi chuỗi chỉ 1-2 phiếu** → không có chuỗi nào thắng rõ.
- Vote theo vị trí: mỗi vị trí được bỏ phiếu riêng → ký tự đúng ở mỗi vị trí vẫn thắng dù
  tổng thể chuỗi hiếm khi trùng khớp hoàn toàn.

## Các bước (Tầng A)

1. Chuẩn hoá + lọc (độ dài ∈ [6,10], confidence ≥ ngưỡng).
2. **Vote độ dài** (đa số) — vì fusion theo vị trí cần biết độ dài chung.
3. **Gom cụm** chống trộn nhiều biển: cùng độ dài, khác ≤ `max_cluster_distance` ký tự; chỉ
   fuse trong cụm lớn nhất.
4. Vote từng vị trí, trọng số `conf^a × quality^b`; mỗi crop (frame_index) đóng góp trọng số
   bằng nhau (chia đều).
5. Chuỗi ghép phải qua `validate_plate`; sai định dạng → **lùi về vote cả chuỗi cũ**.

## Điểm đặc biệt trong logic

### 1. Trọng số `confidence^a × quality^b`
Đọc tự tin hơn / từ crop chất lượng hơn có tiếng nói lớn hơn. Số mũ `a`, `b` điều chỉnh mức độ
"tin" vào mỗi yếu tố.

### 2. Mỗi crop đóng góp trọng số BẰNG NHAU
Một crop có nhiều reading (nhiều biến thể OCR) không được bỏ nhiều phiếu — `vote_count`/
`supporting_frames` tính theo **crop (frame_index)**, không theo số reading. Tránh 1 frame
"đè" các frame khác.

### 3. Phát hiện "trộn nhiều biển" trong 1 track
Nếu cụm lớn thứ 2 chiếm ≥ `cluster_min_share` (30%) tổng lần đọc → nghi track chứa **2 xe/biển
khác nhau** bị gộp chung → đánh dấu `reliable=False` (không tin kết quả).

### 4. Fallback an toàn về vote cả chuỗi
Fusion theo vị trí có thể ghép ra chuỗi sai định dạng (vd độ dài đúng nhưng vị trí số/chữ sai).
Khi đó tự động **lùi về vote cả chuỗi cũ** thay vì trả chuỗi sai định dạng.

### 5. Tầng B (chỗ cắm, chưa chạy)
Nếu lần đọc có `char_probs` (xác suất từng ký tự từ OCR) thì cộng log-prob theo vị trí thay cho
vote — chính xác hơn, nhưng cần OCR cung cấp char_probs.
