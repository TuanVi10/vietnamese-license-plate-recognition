# 04 — Tracker SORT (`src/tracker.py`, Module 3)

## Vai trò

Gán **track_id ổn định** cho từng xe qua nhiều frame, để gom nhiều lần đọc của cùng 1 xe
rồi bỏ phiếu (voting theo track_id).

## Thuật toán: SORT

```
1. Mỗi track giữ 1 Kalman filter (mô hình vận tốc không đổi)
   trạng thái [u, v, s, r, u', v', s']  (tâm, diện tích, tỉ lệ w/h + vận tốc)
2. predict  →  ước lượng bbox hiện tại của mọi track
3. ghép detection ↔ track bằng IoU + Hungarian (gán tối ưu)
4. track không cập nhật > max_age frame → xoá
   track khớp liên tiếp đủ min_hits lần → "đã xác nhận"
```

Chỉ dùng `numpy + math` (không cần scipy/filterpy).

## Điểm đặc biệt trong logic

### 1. Ma trận IoU vector hoá bằng numpy (`iou_matrix`)
Thay 2 vòng lặp Python `O(n_det × n_trk)` bằng tính vector hoá — tránh nghẽn cổ chai mỗi
frame trên cảnh đông xe.

### 2. `min_hits` dựa trên `hit_streak` (số lần khớp LIÊN TIẾP), không phải `frame_count`
Nếu dùng `frame_count`, mọi track mới ở các frame đầu sẽ bị "xác nhận" dù chưa từng được
cập nhật → làm mất tác dụng của `min_hits`. Dùng `hit_streak` (reset khi track lỡ 1 frame)
mới đúng ý "phải khớp liên tiếp đủ số lần".

### 3. Cập nhật NGAY khi tạo track mới
Khi tạo track mới cho detection chưa ghép được, gọi `update()` ngay để `hits`/`hit_streak`
phản ánh đúng detection đầu tiên → `min_hits=1` xác nhận tức thì.

### 4. `first_frame` phải bắt từ track CHƯA xác nhận
`SortTracker.update()` chỉ trả track ĐÃ xác nhận, nên nếu lấy `first_frame` từ `tracks`
thì sẽ bị lệch về đúng frame xác nhận (VD min_hits=3 → first_frame=2 thay vì 0). Video pipeline
phải đọc `self.tracker.trackers` (giữ cả track chưa xác nhận) để bắt đúng frame xuất hiện đầu tiên.

### 5. Bọc lỗi tracker (Kalman suy biến → LinAlgError)
`video_pipeline` bọc `try/except` quanh `tracker.update()`; khi lỗi thì reset tracker và coi
frame đó không có track nào — 1 lỗi Kalman không làm sập cả video.
