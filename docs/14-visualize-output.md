# 14 — Visualize & Output (`src/visualize.py`, Module 7)

## Vai trò

Vẽ overlay kết quả (khung biển + text + crop đối chiếu) và đóng gói đầu ra JSON/CSV/txt/video.

## Hàm chính

- `draw_bbox` — vẽ khung + nhãn.
- `draw_reading` — vẽ `PlateReading` lên ảnh gốc (khung biển + text + dán crop biển vào góc).
- `side_by_side` — ghép 2 ảnh cạnh nhau để so sánh gốc vs overlay.

## Điểm đặc biệt trong logic

### 1. Màu theo trạng thái
`valid` (đúng định dạng) → **xanh lá**; không valid → **cam**. Giúp soi nhanh kết quả nào đáng tin.

### 2. Dán crop biển vào góc trên-trái để đối chiếu
Khi viết báo cáo / debug ca đọc sai, cần nhìn thấy ảnh crop biển ngay trên frame. `draw_reading`
dán thumbnail crop vào góc trên-trái, kèm khung màu trạng thái.

### 3. Đầu ra đầy đủ 3 định dạng (do main.py / run_full.py ghi)
- `results.json` — đầy đủ (mọi trường, đã serialize, bỏ ảnh crop).
- `results.csv` — bảng track (plate_text, confidence, valid_format, reliable…).
- `results.txt` — bản dễ đọc (danh sách track có text).
- `plate_crops/` — crop biển từng track (để đối chiếu).
