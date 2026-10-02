# 17 — Data Pipeline & Labeling

## Tổng quan dữ liệu

| Dataset | Số ảnh | Mục đích |
|---|---|---|
| `datasets/dataset` (biển VN) | 18 619 | train detect biển số (YOLO) |
| `datasets/ccpd_pose` (CCPD2019) | 17 774 | train corner (YOLO-pose 4 keypoint) |
| `datasets/corner_regression/ccpd_warped` | 35 443 | corner + homography augment |
| `labeling` (crop thật VN) | 1 701 | ground-truth đánh giá |

## Các script chính (trong `tools/`)

- `download_ccpd.py` — tải CCPD2019 từ HuggingFace.
- `ccpd_to_pose.py` — chuyển CCPD → YOLO-pose (sắp 4 góc TL→TR→BR→BL).
- `gen_ccpd_warped.py` — sinh 35 443 mẫu CCPD-warped (homography + lọc góc ra ngoài khung).
- `ccpd_warped_to_yolo.py` — chuyển sang định dạng YOLO-pose để train.
- `extract_labeling_crops.py` — trích 1 701 crop thật từ video để gán nhãn.
- `unify_labeling.py` — gộp 5 file nhãn (1-200, 201-400, 401-600, 601-800, 801-1701) → 1 CSV pixel.
- `qa_labeled.py`, `qa_ccpd_pose.py` — kiểm tra chất lượng nhãn.

## Điểm đặc biệt trong logic

### 1. Nhãn CCPD nằm TRONG tên file
Mỗi tên file CCPD chứa 7 field (area, tilt, bbox, 4 đỉnh, 7 ký tự, brightness, blur). Phải parse
tên file để lấy 4 đỉnh. Thứ tự gốc là `BR,BL,TL,TR` (bắt đầu góc phải-dưới) → phải sắp lại.

### 2. `gen_ccpd_warped` — augment bằng homography thật
Mỗi ảnh CCPD được áp homography ngẫu nhiên (rotation + perspective jitter + scale) lên CẢ crop,
rồi biến đổi 4 góc qua CÙNG homography → GT mới. **Lọc bỏ** ca có góc rơi ra ngoài khung. Đây là
cách tạo mẫu biển nghiêng (CCPD gốc 82% có |góc| < 5°).

### 3. `unify_labeling` — 5 nguồn dùng 3 chuẩn toạ độ KHÁC NHAU
| Nguồn | Chuẩn toạ độ | Xử lý |
|---|---|---|
| 1-200 | pixel | dùng thẳng |
| 201-400 | pixel, image là URL | decode URL |
| 401-600 | % (0-100), JSON trong `label` | `x/100 × width` |
| 601-800 | pixel | dùng thẳng |
| 801-1701 | % (0-100), cột tên sai `unreadable_text` | `/100 × size` + sửa cột |

Tất cả gom về 1 CSV `labeled.csv` (10 cột, toạ độ pixel), sắp lại 4 góc theo hình học.

### 4. Chia split theo TRACK_ID (không theo ảnh)
Mỗi track = nhiều frame liên tiếp của cùng 1 xe → chia theo ảnh sẽ làm **rò rỉ** (ảnh cùng xe
rơi vào cả train lẫn test). Chia theo track_id đảm bảo val/test không chứa xe đã thấy khi train.
