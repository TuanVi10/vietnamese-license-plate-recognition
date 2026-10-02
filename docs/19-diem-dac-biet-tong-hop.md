# 19 — Tổng hợp các ĐIỂM ĐẶC BIỆT trong logic tính toán

Bảng "cheat-sheet" mọi kỹ thuật/mẹo/quyết định thiết kế đáng chú ý — dùng để ôn nhanh trước
phỏng vấn.

## Kỹ thuật xử lý ảnh / hình học

| Điểm | Nơi | Ý tưởng |
|---|---|---|
| Đọc/ghi ảnh Unicode | image_utils | `np.fromfile` + `imdecode` (OpenCV không hỗ trợ Unicode) |
| Xoay không cắt góc | image_utils | tính lại canvas `h·sin + w·cos` |
| Deskew projection profile | preprocess_plate | quét góc → chọn góc có variance histogram dọc max + nội suy parabol |
| Tách 2 dòng | preprocess_plate | aspect < 1.8 + "thung lũng" projection ngang |
| Sắp 4 góc lồi | geometry_utils | `TL=min(x+y)`, `TR=max(x−y)`… |
| Kiểm tra lồi | geometry_utils | dấu cross product không đổi qua 4 cạnh |
| Warp theo gate góc | corner_regressor | chỉ warp khi \|angle\| > 15° (đo được +37 exact) |

## Kỹ thuật ML / model

| Điểm | Nơi | Ý tưởng |
|---|---|---|
| Domain gap ≠ model kém | eval | quét nhiều conf → "confidence lệch xuống" |
| Augment đúng chỗ | gen_ccpd_warped | homography thật (bù 82% ảnh CCPD <5°) |
| Chọn detect theo conf keypoint | corner_regressor | `argmax(mean conf 4 keypoint)` |
| Gate chỉ warp biển nghiêng | video_pipeline | precision/recall cân bằng |

## Kỹ thuật tracking / voting / fusion

| Điểm | Nơi | Ý tưởng |
|---|---|---|
| IoU vector hoá | tracker | numpy thay 2 vòng lặp O(n²) |
| min_hits theo hit_streak | tracker | reset khi lỡ frame, không theo frame_count |
| Fallback không vượt detection | best_frame | rank theo cặp `(is_detection, score)` |
| Mỗi crop 1 phiếu | fusion | vote theo frame_index, không theo reading |
| Fusion theo vị trí ký tự | fusion | 1 ký tự sai không chia nhỏ phiếu |
| Phát hiện trộn 2 biển | fusion | cụm lớn thứ 2 ≥ 30% → unreliable |
| Sửa ký tự theo khuôn vị trí | postprocess | `DD L A D…` + O↔0, I↔1, B↔8, S↔5 |

## Kỹ thuật hệ thống / kỹ thuật phần mềm

| Điểm | Nơi | Ý tưởng |
|---|---|---|
| 2 venv + subprocess | ocr_subprocess | tránh xung đột torch cu126 vs paddle cu118 |
| Chống NaN | frame_extractor | OpenCV trả NaN cho FPS/frame_count |
| Chống OOM video dài | video_pipeline | xoá ref ảnh frame sau khi xử lý |
| Bọc lỗi mọi module | video_pipeline | 1 lỗi không sập cả video |
| Chia split theo track_id | unify_labeling | tránh rò rỉ val/test |

## Cải thiện baseline (số liệu đo 213 ảnh thật)

| Hạng mục | Trước → Sau |
|---|---|
| OCR | EasyOCR 6.6% → PaddleOCR 74.6% exact |
| Căn chỉnh >15° | bbox+affine 46.3% → corner-warp 83.3% |
| End-to-end | 70.4% → 79.8% exact / 0.923 sim |
| Detection | mAP50-95 = 0.73 |
