# 06 — Best-Frame Selection (`src/best_frame.py`)

## Vai trò

Mỗi xe xuất hiện trong nhiều frame; thay vì OCR mọi frame (tốn + nhiễu), giữ lại **top-K crop
tốt nhất** mỗi track rồi mới OCR.

## Điểm crop = tổ hợp có trọng số (mọi thành phần chuẩn hoá về [0,1])

```
score = w_sharp·sharpness + w_height·height + w_aspect·aspect
      + w_conf·confidence − w_exposure·exposure
```

| Thành phần | Ý nghĩa | Cách chuẩn hoá |
|---|---|---|
| sharpness | độ nét (variance of Laplacian) | thang `log1p` / `sharpness_ref` |
| height | chiều cao biển (px) | `/ height_ref` |
| aspect | độ lệch w/h so với loại biển kỳ vọng | `1 − |log-ratio|` |
| confidence | conf detection biển (0 nếu fallback) | [0,1] sẵn |
| exposure | phạt vùng cháy sáng/quá tối | tỉ lệ pixel ≤ dark / ≥ bright |

## Điểm đặc biệt trong logic

### 1. Xếp hạng theo CẶP `(is_detection, score)` — fallback KHÔNG bao giờ vượt detection
`_rank_key = (1 if source != "fallback" else 0, score)`. Dù crop fallback có điểm cao hơn
cũng không được ưu tiên hơn crop detection thật. Điều này đảm bảo ta luôn OCR crop **do model
thực sự thấy biển**, không OCR vùng "đoán mò".

### 2. Khi detection thật xuất hiện → xoá hết fallback còn sót
Rule (a) trong `TopKBuffer.add`: crop fallback chỉ là "chỗ tạm" khi chưa có detection; ngay khi
có detection thật, fallback bị xoá sạch.

### 3. `min_frame_gap` chống giữ nhiều crop gần như trùng nhau
Nếu giữ 5 crop nhưng cả 5 đều là các frame liền kề (cùng 1 góc nhìn) thì voting không thêm
thông tin. `min_frame_gap` ép các crop giữ lại cách nhau đủ xa (theo frame đã lấy mẫu).

### 4. `expected_aspect` được ước lượng THEO TRACK
Tỉ lệ w/h kỳ vọng của biển 1 dòng (~4.3) khác biển 2 dòng (~1.35). Pipeline ước lượng loại
biển từ median aspect các detection thật của track rồi dùng đúng loại để chấm điểm aspect.
