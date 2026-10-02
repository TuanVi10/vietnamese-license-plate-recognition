# BÁO CÁO DỰ ÁN — NHẬN DIỆN BIỂN SỐ XE VIỆT NAM (ALPR)

> Tài liệu tường thuật toàn bộ hành trình: từ bài toán, các khó khăn gặp phải,
> cho tới các cải tiến baseline (kèm số liệu) — dùng làm tư liệu cho CV xin thực tập.

---

## 1. Tổng quan dự án

Xây dựng pipeline **đọc biển số xe Việt Nam từ video** (video giao thông thật, 13 phút,
~60 fps) — đầu ra là **danh sách biển số theo từng xe** (track) kèm độ tin cậy.

Dự án trải đủ vòng đời một hệ thống thị giác máy tính:

```
thu thập dữ liệu → gán nhãn → huấn luyện model → đánh giá → tích hợp → end-to-end
```

---

## 2. Kiến trúc pipeline (8 module)

```
video.mp4
  │
  ├─ M0  FrameExtractor            lọc/trích frame (giữ frame gốc + timestamp)
  ├─ M1  VehicleDetector (YOLO11n)  phát hiện phương tiện
  ├─ M3  SortTracker → track_id     theo dõi từng xe qua các frame
  │
  ├─ M2  PlateDetector (YOLO11n finetune)  ĐỊNH VỊ biển số trong vùng xe
  │        └─ bbox + conf
  │
  ├─ crop biển với lề 15% (bbox_margin = 0.15)
  │
  ├─ ★ CornerRegressor (YOLO11n-pose, 4 keypoint)  [MỚI]
  │        └─ 4 góc TL/TR/BR/BL + suy góc nghiêng
  │             ├─ |góc| > 15°  → warpPerspective(4 góc) → biển phẳng
  │             └─ ngược lại    → bbox + affine deskew (cách cũ)
  │
  ├─ ★ PaddleOCR (PP-OCRv6) → text   [MỚI, thay EasyOCR]
  ├─ M6  PostProcessor / fusion       gộp nhiều lần đọc theo track, voting
  └─ M7  Output                       JSON + CSV + txt + video overlay
```

---

## 3. Hành trình & các khó khăn (tường thuật theo thứ tự thời gian)

### 3.1. Không có dataset biển số Việt Nam chuẩn → phải tự gom + tự gán nhãn
- Bài toán cần model **detect biển số VN**, nhưng dataset công khai chuẩn cho biển VN gần như không có.
- **Giải pháp**: gom ảnh biển số VN từ nhiều nguồn → chuẩn hoá về định dạng YOLO, bỏ ảnh trùng (dedup),
  tách train/val/test có phân tầng. Kết quả **18 619 ảnh** (train 14 897 / val 1 862 / test 1 862).
- Đồng thời tự xây **bộ ground-truth thật 1 701 crop** (836 track xe) bằng tay để đánh giá — tốn công
  nhưng là thứ duy nhất cho phép đo được "model tốt đến đâu trên dữ liệu THẬT".

### 3.2. OCR "quốc tế" đọc biển Việt Nam rất tệ (domain gap đầu tiên)
- Thử **EasyOCR** (engine phổ biến) lên 213 ảnh biển VN có text GT → chỉ **6.6 % exact / 0.614 sim**.
- Nguyên nhân: EasyOCR train chủ yếu trên text Latin thông thường, kém với font biển số VN
  (ký tự đặc thù, 2 dòng, nhiễu, nghiêng).
- **Giải pháp**: chuyển sang **PaddleOCR PP-OCRv6** → nhảy lên **74.6 % exact / 0.917 sim**
  (gấp ~11 lần độ chính xác exact).

### 3.3. Biển bị nghiêng / phối cảnh → cần "chỉnh góc" bằng 4 góc (keypoint)
- Đọc OCR trên biển nghiêng >15° rất kém (chỉ ~46 % exact). Cần một bước **căn chỉnh hình học**
  trước khi OCR.
- **Giải pháp**: thêm **CornerRegressor** (YOLO11n-pose, `kpt_shape=[4,3]`) dự đoán **4 góc**
  TL/TR/BR/BL rồi `warpPerspective` để "là phẳng" biển.
- **Khó khăn phụ**: nhãn 4 góc của CCPD có thứ tự gốc là `BR,BL,TL,TR` (bắt đầu từ góc phải-dưới) —
  phải **sắp lại theo hình học** TL→TR→BR→BL và **kiểm chứng 17 774/17 774 ảnh không lệch** (max 0.0006 px).

### 3.4. Domain gap thứ hai — model chỉnh góc train trên dữ liệu Trung Quốc
- Dùng **CCPD2019** (biển Trung Quốc, 17 774 ảnh) để train corner model vì không có dữ liệu VN tương đương.
- **Sự cố**: trên val CCPD model "tuyệt vời" (`mAP50-95(P) = 0.995`, bão hoà từ epoch ~10), nhưng
  trên **ảnh thật VN gần như thất bại** — ở ngưỡng `conf=0.25` chỉ bắt được **1.4 %**.
- **Chẩn đoán** (quan trọng): đây không phải "model mất khả năng" mà là **confidence bị lệch xuống
  khi đổi domain** (domain gap). Ở `conf=0.001` model vẫn bắt được **27 %**.
- **Giải pháp**: **augmentation** mạnh hơn cho góc nghiêng (`degrees=20`, `perspective=0.001`) vì
  82 % ảnh CCPD có |góc| < 5° (thiếu mẫu nghiêng). Kết quả det% ở `conf=0.001` tăng
  **27.2 % → 83.9 %**.
- **Giới hạn còn lại (đã nhận diện)**: muốn dùng được ở ngưỡng conf cao cần **fine-tune trên data thật
  201–1701** (đã có 664 biển gán tay) thay vì chỉ train trên CCPD.

### 3.5. Xung đột môi trường CUDA giữa PyTorch và PaddlePaddle
- Pipeline dùng **PyTorch** (torch cu126/cuDNN 9.10) cho YOLO, còn **PaddleOCR** cần paddle cu118/cuDNN 8.9
  → **không thể import chung một process**.
- **Giải pháp**: giữ 2 môi trường ảo riêng (`.venv-datasets`, `.venv-paddle`) và cho pipeline gọi OCR
  qua **worker subprocess** (tránh crash do xung đột thư viện GPU).

### 3.6. Các vấn đề kỹ thuật khác
- **Tracking**: dùng SORT; xử lý ID switch, chọn **best-frame** cho mỗi track (ưu tiên crop có
  detection thật, chặn fallback lẫn rác), voting/fusion nhiều lần đọc để ra 1 biển/track.
- **Codec video đầu ra**: `mp4v` (MPEG-4 cũ) khiến Windows không mở được → phải re-encode sang
  **H.264** (`avc1`) bằng `imageio-ffmpeg` (OpenCV không có OpenH264 DLL).

---

## 4. Cải thiện baseline (có số liệu đo trên 213 ảnh thật)

| Hạng mục | Trước (baseline) | Sau (cải tiến) | Cách làm |
|---|---|---|---|
| **OCR engine** | EasyOCR **6.6 %** exact / 0.614 sim | PaddleOCR **74.6 %** exact / 0.917 sim | đổi engine → PP-OCRv6 |
| **Căn chỉnh biển nghiêng (>15°)** | bbox + affine: **46.3 %** exact | corner-warp: **83.3 %** exact | CornerRegressor + warpPerspective |
| **End-to-end (detect → align → OCR)** | **70.4 %** exact / 0.886 sim | **79.8 %** exact / 0.923 sim | thêm stage chỉnh góc |
| **Detection (biển số)** | — | **mAP50-95 = 0.73**, mAP50 = 0.99 | YOLO11n, 50 epoch, 18.6k ảnh |
| **Corner model — det% trên ảnh thật** | baseline **27.2 %** @ conf 0.001 | augmented **83.9 %** @ conf 0.001 | degrees=20 + perspective=0.001 |

**Điểm mấu chốt** — phân tích theo dải góc nghiêng (213 ảnh có text GT):

| Dải góc | EasyOCR exact | PaddleOCR exact | End-to-end (corner-warp) |
|---|---|---|---|
| < 5° | 8.3 % | 70.2 % | 78.6 % |
| 5°–15° | 6.7 % | 73.3 % | 78.7 % |
| > 15° | 3.7 % | 83.3 % | **83.3 %** (từ 46.3 % của cách cũ) |

→ Việc **thêm stage chỉnh góc** đặc biệt **cứu các biển nghiêng >15°**: từ 46.3 % lên 83.3 %,
đồng thời đẩy điểm tổng từ 70.4 % lên 79.8 %.

---

## 5. Kết quả cuối cùng & đo đạc đối chiếu

- **Chạy end-to-end** trên video thật 13 phút: **1 248 track xe**, trong đó **142 biển đọc được text**
  (đã qua voting/fusion, có kiểm tra định dạng + độ tin cậy).
- **Đối chiếu model vs người gán nhãn** trên 1 701 crop (836 track):
  - Model detect: **575 biển** | Người label: **664 biển** → model **thiếu 89 biển** (recall ~86.6 %).
  - 169 crop model báo nhầm (human đánh `no_plate`), 258 crop model bỏ sót.
- **Bộ dữ liệu tự xây**: 18 619 ảnh detect + 1 701 crop ground-truth + 17 774 CCPD + 35 443 CCPD-warped.

---

## 6. Tech stack & kỹ năng đã dùng

- **Python**, **PyTorch**, **Ultralytics YOLO11** (detect + pose).
- **PaddlePaddle / PaddleOCR** (PP-OCRv6), **EasyOCR**.
- **OpenCV** (homography, `warpPerspective`, affine, crop/overlay, video writer).
- **SORT** tracking, xử lý track/ID, best-frame selection, fusion/voting.
- **NumPy**, pandas, openpyxl, imageio/imageio-ffmpeg.
- Quản lý **đa môi trường ảo + subprocess** (xử lý xung đột CUDA torch/paddle).
- **Đánh giá model**: mAP, IoU, sai số góc/keypoint, detect-rate theo dải góc, exact/similarity OCR.
- **Xây dựng & QA dataset**: dedup, phân tầng split, kiểm chứng nhãn (round-trip, tứ giác lồi).

---

## 7. Bài học rút ra (điểm nhấn cho phỏng vấn)

1. **mAP trên tập train/val KHÔNG nói lên chất lượng thật.** Corner model đạt 0.995 trên CCPD
   nhưng gần như fail trên ảnh VN → luôn **đánh giá trên dữ liệu thật của bài toán**.
2. **Domain gap ≠ model kém.** Biểu hiện "confidence lệch xuống" cần chẩn đoán đúng (quét nhiều ngưỡng)
   chứ không kết luận vội.
3. **Chất lượng dữ liệu quyết định trần hiệu năng.** Thiếu mẫu biển nghiêng → model yếu đúng dải đó;
   augmentation đúng chỗ (degrees/perspective) cải thiện rõ rệt.
4. **Chia nhỏ bài toán theo module** (detect → align → OCR → fusion) giúp cô lập lỗi và đo đạc
   từng tầng riêng biệt.
5. **Kỹ thuật "gate theo góc"** (chỉ warp khi >15°) là cách cân bằng precision/recall thực dụng
   thay vì áp dụng biến đổi đắt đỏ cho mọi ảnh.

