# 18 — Đánh giá model (`tools/eval_*.py`)

## Triết lý đánh giá (quan trọng nhất)

> **mAP trên tập train/val KHÔNG nói lên chất lượng thật.** Corner model đạt 0.995 trên CCPD
> nhưng gần như fail trên ảnh VN → luôn đánh giá trên **dữ liệu thật của bài toán**.

## Các chỉ số

| Chỉ số | Ý nghĩa | Dùng cho |
|---|---|---|
| `mAP50-95` | AP trung bình IoU 0.5→0.95 | detect biển, pose (trên val train) |
| `det%` (detect_rate) | % ảnh dò được biển | pose trên ảnh thật |
| `corner_px` / `corner_norm` | sai số 4 góc (px / chuẩn hoá) | corner |
| `quad_IoU` | IoU tứ giác dự đoán vs nhãn tay | corner |
| `angle_err` | sai số góc nghiêng (độ) | corner |
| `exact` | % chuỗi OCR khớp đúng | OCR end-to-end |
| `sim` (similarity) | độ tương đồng chuỗi (0-1) | OCR |

## Điểm đặc biệt trong logic

### 1. Quét nhiều ngưỡng conf để chẩn đoán domain gap
`eval_lowconf.py` / `eval_pose_bands.py` đánh giá ở conf = 0.001 / 0.05 / 0.25 / 0.5. Nếu conf
cao ~0% mà conf thấp lại cao → đó là **domain gap (confidence lệch xuống)**, không phải model
kém. Đây là chẩn đoán then chốt giúp không kết luận sai.

### 2. Phân tích theo DẢI GÓC nghiêng
`eval_pose_bands.py` chia ảnh thật thành `<5°`, `5-15°`, `>15°` và đo riêng từng dải → phát hiện
model yếu đúng dải biển nghiêng (mong đợi det% giảm, angle_err tăng khi sang >15°).

### 3. `corner_norm` = `corner_px / √(diện tích biển)` — bất biến kích thước
Sai số góc 20px là tốt với biển lớn nhưng tệ với biển nhỏ. Chia cho √diện tích để so sánh công
bằng giữa các ảnh kích thước khác nhau.

### 4. So sánh baseline vs augmented song song
`eval_compare.py` in bảng song song 2 model trên cùng tập → thấy rõ augmentation `degrees=20`
có ích (det% 27.2% → 83.9% ở conf 0.001).

### 5. Đóng vòng OCR với "ceiling" (mốc lý tưởng)
`eval_corner_ocr.py` / `ceiling_test_paddle_align.py` so 3 mốc:
- `OLD` (bbox+affine), `REG` (corner regressor), `CEIL` (OCR trên ảnh GT đã căn chỉnh hoàn hảo).
→ Biết được "trần" (ceiling) để đo xem corner regressor đã tiến gần lý tưởng chưa.
