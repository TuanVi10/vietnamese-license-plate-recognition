# 13 — Video Pipeline (`src/video_pipeline.py`, glue)

## Vai trò

File "glue" nối mọi module thành pipeline video hoàn chỉnh (M0 → M7). Đây là file lớn nhất vì
chứa toàn bộ điều phối giữa các module.

## Luồng `run(video_path)`

```
extract (M0) → reset → lặp từng frame:
   process_frame(frame):
     M1 dò xe → M3 tracking → M2 dò biển (1 lần/frame)
     → với mỗi track: định vị crop → best-frame buffer (KHÔNG OCR ngay)
     → (nếu bật) lưu overlay frame
   → cuối video: flush top-K từng track → OCR → voting/fusion → finalize tracks
   → (nếu bật) annotate overlay bằng text cuối → xuất kết quả
```

## Điểm đặc biệt trong logic

### 1. Mỗi plate detection chỉ gán cho ĐÚNG 1 track/frame
`used_plate_indices` chia sẻ giữa các track → 1 biển không bị đọc lặp cho nhiều xe. Ưu tiên
gán detection cho track có `_best_plate_score` cao nhất (chồng lấn bbox tốt nhất).

### 2. Fallback bị chặn khi track đã từng có detection thật
`self._track_seen_detection`: nếu track đã có detection thật thì **không dùng fallback** về sau
(tránh crop "đoán mò" lẫn vào khi đã có biển thật).

### 3. Best-frame: thu thập mọi frame nhưng chỉ OCR top-K
Trong pha thu thập, mỗi frame chỉ crop + chấm điểm + đẩy vào `TopKBuffer` (KHÔNG OCR). Cuối
video `_flush_all_tracks()` mới OCR top-K crop điểm cao nhất → tiết kiệm compute khổng lồ
(1 xe xuất hiện 300 frame chỉ OCR ~5 crop).

### 4. Bọc lỗi mọi module (fallback thay vì sập)
M1, M2, M3, M4/5/6 đều được bọc `try/except`: lỗi model/tracker/OCR ghi cảnh báo và fallback
(VD lỗi dò xe → coi cả frame là 1 xe; lỗi tracker → reset; lỗi dò biển → dùng fallback).
1 lỗi không làm sập cả video dài.

### 5. Giải phóng ảnh frame để tránh OOM
`keep_frame_images=False` → sau mỗi frame, xoá tham chiếu ảnh (`frame_result.frame = None`,
`record.frame = None`) để không giữ cả video trong RAM (overlay đã lưu trước đó nếu bật).
