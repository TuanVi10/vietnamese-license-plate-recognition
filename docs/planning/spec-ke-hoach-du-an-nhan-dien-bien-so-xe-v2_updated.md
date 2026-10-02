# SPEC KỸ THUẬT v2: Hệ thống Nhận diện Biển số xe qua Video

**Dựa trên tham khảo chính:**

- Kiến trúc lõi (detect + track): [Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8)
- Bản địa hóa cho biển số VN (OCR + hậu xử lý + web app): [hieunguyen2604/alpr-do-an-mon-hoc](https://github.com/hieunguyen2604/alpr-do-an-mon-hoc)

**So với v1:** bổ sung cấu trúc dataset cụ thể, config training YOLO, pipeline extract frame, chốt rõ lựa chọn OCR, chi tiết công thức từng metric, và bảng rủi ro mở rộng. Các con số trong bản này được đánh dấu rõ **[ước tính]** ở đâu chưa có dữ liệu thật để đo — tránh nhầm là số đã đo được.

---

## 1. Mục tiêu & phạm vi

**Đầu vào:** 1 file video (.mp4) quay cảnh giao thông có nhiều xe chạy qua lại.

**Đầu ra:**

- Video/frame có vẽ bounding box quanh xe và biển số.
- Với mỗi xe xuất hiện trong video: 1 bản ghi gồm `track_id`, biển số đọc được, khung thời gian xuất hiện (frame_start–frame_end), ảnh crop biển số, độ tin cậy (confidence).
- File kết quả dạng CSV/JSON để dễ kiểm tra và làm báo cáo.

**Trong phạm vi (in-scope):**

- Xử lý video có sẵn (offline), chưa cần real-time streaming từ camera trực tiếp.
- Biển số xe máy và ô tô Việt Nam (1 dòng và 2 dòng), chủ yếu **biển trắng dân sự** (xem lưu ý ở mục 3.6 về các loại biển khác).
- Một camera, một góc quay cố định (giả định video kiểu giám sát bãi xe/tuyến đường).

**Ngoài phạm vi (out-of-scope, để dành làm mở rộng nếu còn thời gian):**

- Nhận diện đa camera, ghép dữ liệu giữa các camera.
- Biển số nước ngoài.
- Chống gian lận biển số giả/che biển.
- Các loại biển số đặc thù (biển xanh cơ quan nhà nước, biển đỏ quân đội, biển vàng kinh doanh vận tải) — luật hậu xử lý ở Module 6 sẽ không tối ưu cho các loại này trừ khi bổ sung riêng.

---

## 2. Kiến trúc hệ thống

```
                         ┌─────────────────────────┐
  video.mp4  ───────────▶│  Module 0: Frame          │
                         │  Extraction & Filtering   │
                         └────────────┬─────────────┘
                                      │ frame hợp lệ (đủ sáng, không quá mờ)
                                      ▼
                         ┌─────────────────────────┐
                         │  Module 1: Vehicle       │
                         │  Detection (YOLOv8n)     │
                         └────────────┬─────────────┘
                                      │ bbox xe + class (car/motorbike)
                                      ▼
                         ┌─────────────────────────┐
                         │  Module 2: Plate         │
                         │  Detection (YOLO fine-   │
                         │  tuned trên dataset      │
                         │  biển số)                │
                         └────────────┬─────────────┘
                                      │ bbox biển số (trong bbox xe)
                                      ▼
                         ┌─────────────────────────┐
                         │  Module 3: Tracking      │
                         │  (SORT/ByteTrack) — gán  │
                         │  track_id cho từng xe    │
                         └────────────┬─────────────┘
                                      │ track_id + bbox theo frame
                                      ▼
                         ┌─────────────────────────┐
                         │  Module 4: Tiền xử lý    │
                         │  ảnh biển số (crop,      │
                         │  deskew, tách dòng)      │
                         └────────────┬─────────────┘
                                      │ ảnh biển số đã chuẩn hóa
                                      ▼
                         ┌─────────────────────────┐
                         │  Module 5: OCR           │
                         │  (PaddleOCR PP-OCRv5)    │
                         └────────────┬─────────────┘
                                      │ chuỗi ký tự đọc được + confidence
                                      ▼
                         ┌─────────────────────────┐
                         │  Module 6: Hậu xử lý &   │
                         │  Voting theo track_id    │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Module 7: Output &      │
                         │  Visualize                │
                         └─────────────────────────┘
```

**Thay đổi so với v1:** thêm Module 0 (Frame Extraction) đứng trước Module 1, vì với video dài, chạy detect trên *mọi* frame vừa tốn compute vừa dư thừa (xe gần như không đổi vị trí giữa 2 frame liên tiếp ở 30fps). Lọc bớt frame ngay từ đầu giúp pipeline nhanh hơn đáng kể mà không mất nhiều thông tin.

---

## 3. Chi tiết từng module

### Module 0 — Frame Extraction & Filtering (mới)

**Mục đích:** giảm số frame cần xử lý, đồng thời loại sớm các frame chất lượng kém (quá tối/quá mờ) để không lãng phí compute ở các bước sau.

**Thuật toán:**

```
Đọc video bằng cv2.VideoCapture
Với mỗi frame, cách nhau `frame_interval` frame:
    - Tính độ sáng trung bình (mean pixel value)
    - Tính độ mờ (variance of Laplacian — chỉ số blur phổ biến với OpenCV)
    - Nếu brightness >= brightness_threshold VÀ blur_score >= blur_threshold:
        giữ frame, lưu lại kèm frame_index gốc (để khớp lại thời gian xuất hiện xe)
    - Ngược lại: bỏ qua frame này
```

**Tham số khởi điểm (cần tinh chỉnh sau khi thử trên video thật, không phải số cố định):**

| Tham số                 | Giá trị khởi điểm      | Ghi chú                                                                                                                                                                                      |
| ------------------------ | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `frame_interval`       | 5                           | Với video 30fps → xử lý ~6 frame/giây. Giảm xuống 3 nếu xe di chuyển nhanh, tăng lên 10 nếu cần tốc độ xử lý cao hơn.                                                      |
| `brightness_threshold` | 50 (thang 0-255)            | Frame tối hơn mức này thường cho OCR kém, cân nhắc bỏ qua.                                                                                                                          |
| `blur_threshold`       | 100 (variance of Laplacian) | Ngưỡng này phụ thuộc nhiều vào camera/độ phân giải cụ thể —**nên tự đo thử trên vài frame rõ/mờ của video thật để hiệu chỉnh**, không dùng cứng số 100. |

**Lưu ý quan trọng:** việc bỏ qua frame không đạt threshold có thể làm mất dấu 1 xe đi qua nhanh trong điều kiện thiếu sáng — cần cân nhắc đánh đổi giữa tốc độ xử lý và độ đầy đủ dữ liệu. Nếu mục tiêu ưu tiên độ chính xác hơn tốc độ, có thể tắt bộ lọc này (`frame_interval=1`, không lọc brightness/blur) và chỉ dựa vào bước voting ở Module 6 để xử lý nhiễu.

### Module 1 — Vehicle Detection

- **Model:** YOLOv8n pretrained trên COCO, dùng thẳng qua `ultralytics`, không cần train lại.
- **Input:** frame đã qua lọc ở Module 0.
- **Output:** danh sách bbox `[x1, y1, x2, y2, class_id, confidence]`.
- **Lọc class:** giữ `car`, `motorcycle`, `bus`, `truck` — đừng chỉ lọc `car` vì sẽ bỏ sót xe máy.
- **Mở rộng bbox:** nới rộng bbox xe thêm ~10-15% mỗi cạnh trước khi crop, tránh cắt mất biển số nằm ở rìa.

### Module 2 — Plate Detection

- **Model:** YOLO fine-tune riêng trên dataset biển số (khuyến nghị YOLO11n thay vì YOLOv8n nếu muốn bản mới hơn — cả hai đều dùng được, YOLO11n nhẹ và nhanh hơn một chút trên CPU theo benchmark của Ultralytics).
- **Input:** vùng ảnh đã crop theo bbox xe từ Module 1.
- **Output:** bbox biển số trong tọa độ gốc của frame.
- **Baseline nhanh:** có thể dùng thẳng `license_plate_detector.pt` từ repo Zeerak-Khan để chạy thử nghiệm tuần 1, sau đó fine-tune lại bằng dataset VN.

#### 2.1 Dataset structure

```
datasets/
├── images/
│   ├── train/        # ảnh training
│   ├── val/           # ảnh validation
│   └── test/          # ảnh test (giữ riêng, không đụng đến cho tới khi đánh giá cuối)
├── labels/
│   ├── train/         # nhãn YOLO format, tên file khớp ảnh tương ứng
│   ├── val/
│   └── test/
└── data.yaml           # YOLO data configuration
```

**Format nhãn YOLO (mỗi dòng 1 object trong ảnh):**

```
<class_id> <x_center> <y_center> <width> <height>
0 0.512 0.487 0.15 0.08
```

(toạ độ đã chuẩn hóa 0-1 theo kích thước ảnh, class_id=0 cho "license_plate" nếu chỉ có 1 class)

**Về số lượng ảnh:** ở file dataset trước, các nguồn gộp lại có thể lên tới hàng chục nghìn ảnh, nhưng đây là **[ước tính tổng gộp]** — nhiều dataset con có thể trùng ảnh gốc (cùng lấy từ 1 nguồn Roboflow public), hoặc chất lượng nhãn không đồng đều. **Việc cần làm trước khi lập kế hoạch train chính xác:** tải thật các dataset về, kiểm tra trùng lặp (có thể dùng hash ảnh để lọc trùng), rồi mới đếm số ảnh thật và chia tỷ lệ 80/10/10 cho train/val/test dựa trên con số đã lọc — không dùng tỷ lệ ước tính từ đầu để lập kế hoạch cứng.

**Format khác nhau giữa các dataset nguồn:** một số dataset Roboflow xuất theo YOLO, một số theo Pascal VOC (XML) hoặc COCO (JSON) — cần convert đồng nhất về YOLO format trước khi gộp (Roboflow có tool export lại theo format khác nếu bạn re-export).

#### 2.2 YOLO Training Configuration

| Tham số                         | Giá trị đề xuất                                   | Lý do                                                                                                                                                                                                             |
| -------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| model                            | YOLO11n (hoặc YOLOv8n)                                | Nhẹ, phù hợp chạy trên CPU/Colab free tier                                                                                                                                                                    |
| pretrained                       | yolo11n.pt (COCO)                                      | Transfer learning, hội tụ nhanh hơn train từ đầu                                                                                                                                                             |
| epochs                           | 100 (khởi điểm)                                     | Theo dõi validation loss, dừng sớm (early stopping) nếu không cải thiện sau ~15-20 epoch                                                                                                                    |
| batch                            | 16                                                     | Phù hợp GPU Colab T4 (VRAM ~15GB); giảm xuống 8 nếu bị out-of-memory                                                                                                                                         |
| imgsz                            | 640                                                    | Biển số là object nhỏ trong ảnh gốc, giữ độ phân giải đủ lớn để giữ chi tiết                                                                                                                     |
| optimizer                        | AdamW                                                  | Hội tụ ổn định hơn SGD trên dataset vừa/nhỏ                                                                                                                                                               |
| box loss gain                    | 7.5 (mặc định Ultralytics) hoặc tăng nhẹ lên ~8 | Ưu tiên bbox khít hơn → crop biển số sạch hơn cho OCR.**Đây là tham số cần thử nghiệm thực tế (chạy vài giá trị so sánh mAP), không nên đặt cố định mà không kiểm chứng.** |
| fliplr (augmentation lật ngang) | 0.0                                                    | **Bắt buộc tắt** — lật ngang sẽ làm đảo ngược thứ tự ký tự trên biển số, sinh dữ liệu train sai hoàn toàn                                                                              |
| mosaic                           | có thể giữ mặc định hoặc giảm nhẹ             | Mosaic augmentation thường ổn với object detection, nhưng nếu thấy ảnh ghép làm biển số bị cắt vụn quá nhiều, giảm tỷ lệ dùng                                                                 |

> **Lưu ý:** các giá trị epochs/batch/box-loss-gain ở trên là điểm khởi đầu hợp lý dựa trên kinh nghiệm chung với bài toán tương tự, **không phải con số đã benchmark riêng cho dataset của bạn**. Sau khi train lần đầu, xem đường cong loss/mAP trong `results.png` của Ultralytics để quyết định có cần chỉnh lại không.

### Module 3 — Tracking

- **Baseline:** SORT (`abewley/sort`) — Kalman Filter + Hungarian algorithm, đúng theo repo tham khảo gốc, dễ tích hợp.
- **Nâng cấp nếu cần:** nếu video test có nhiều xe che khuất nhau (giao lộ đông, bãi xe chật), chuyển sang **ByteTrack** (tích hợp sẵn trong `ultralytics`, chỉ cần đổi tracker config, không cần thêm thư viện ngoài) — ByteTrack xử lý tốt hơn các detection có confidence thấp, giảm tình trạng mất track khi bbox bị che một phần.
- **Input:** danh sách bbox xe theo từng frame.
- **Output:** track_id cố định xuyên suốt các frame xe xuất hiện.
- **Tham số quan trọng cần tinh chỉnh:** `max_age` (số frame cho phép track "sống" dù không có detection mới) — đặt quá thấp sẽ tạo track_id mới liên tục cho cùng 1 xe khi detection bị chớp tắt; đặt quá cao sẽ gộp nhầm 2 xe khác nhau đi qua cùng vị trí gần thời điểm nhau.

### Module 4 — Tiền xử lý ảnh biển số

- Crop ảnh biển số theo bbox từ Module 2.
- **Thứ tự bắt buộc:** deskew (chỉnh nghiêng) trước → tách dòng (nếu biển 2 dòng) sau → ghép ngang. Làm ngược thứ tự sẽ tách sai dòng khi ảnh còn nghiêng.
- Resize chuẩn hóa cho OCR — **giữ nguyên tỷ lệ khung hình (aspect ratio)**, dùng padding thay vì kéo dãn, để tránh méo ký tự.
- Tăng tương phản (CLAHE — Contrast Limited Adaptive Histogram Equalization) cho các trường hợp ảnh biển số bị chói/tối một phần.
- **Ngưỡng kích thước tối thiểu:** nếu ảnh biển số crop ra nhỏ hơn ngưỡng (ví dụ chiều rộng < 40px — **cần tự đo thử, không phải số cố định**), đánh dấu "không đủ tin cậy", không đưa vào OCR để tránh tốn compute vô ích.

### Module 5 — OCR

**Quyết định: dùng PaddleOCR (PP-OCRv5) làm engine chính**, không để "tùy chọn giữa 2 engine" như bản v1 — lý do:

- PP-OCRv5 được train với tỷ lệ ảnh biển 2 dòng cao hơn, phù hợp đặc thù biển xe máy VN.
- Hỗ trợ tốt ký tự Latin (chữ cái + số) dùng trên biển số VN.

**Lưu ý triển khai quan trọng:** trên máy CPU (không có GPU), cần set `enable_mkldnn=False` khi khởi tạo PaddleOCR để tránh crash trên một số cấu hình — đây là vấn đề đã được nhiều người dùng PaddleOCR trên CPU báo cáo, nên kiểm tra kỹ khi setup môi trường tuần 1.

**Về con số hiệu năng cụ thể (VD: "X% accuracy trên Y biển số"):** những con số benchmark publicly có sẵn thường đến từ dataset/điều kiện khác với dataset của bạn, nên **không nên trích dẫn số cụ thể vào báo cáo nếu chưa tự đo lại trên dataset của mình** — nếu cần trích dẫn, phải tìm và ghi rõ nguồn gốc bài nghiên cứu cụ thể, tránh dùng số không rõ xuất xứ.

**Giới hạn charset:** nếu PaddleOCR cho phép cấu hình charset/dictionary riêng, giới hạn chỉ nhận `0-9` và `A-Z` (bỏ ký tự đặc biệt, dấu tiếng Việt) để giảm nhiễu kết quả.

**Fallback nếu PaddleOCR gặp khó khi setup:** nếu việc cài đặt PaddlePaddle trên máy gặp vấn đề (dependency phức tạp, không tương thích Python version...) và tốn quá nhiều thời gian, có thể tạm dùng EasyOCR để có pipeline baseline chạy được trước, rồi quay lại xử lý PaddleOCR sau khi các module khác đã ổn định — đừng để việc setup OCR chặn tiến độ toàn bộ dự án.

### Module 6 — Hậu xử lý & Voting theo track_id

- **Áp luật định dạng biển số VN** (biển trắng dân sự) để loại kết quả sai rõ ràng, ví dụ biển ô tô dạng `[2 số]-[1 chữ]-[3 hoặc 4 số]` (VD: `51F-12345`), biển xe máy 2 dòng dạng tương tự chia 2 hàng.
- **Sửa lỗi ký tự thường gặp dựa vào vị trí:** nếu luật biển quy định vị trí đó bắt buộc là số nhưng OCR đọc ra chữ cái dễ nhầm (O↔0, I↔1, B↔8, S↔5), tự động sửa lại theo vị trí kỳ vọng — đây là cách giảm lỗi OCR hiệu quả mà không cần train lại model.
- **Lọc trước khi voting:** loại các kết quả OCR rõ ràng sai định dạng (quá ngắn, không đúng số ký tự kỳ vọng, không đúng vị trí chữ/số) *trước khi* đưa vào voting, để không làm nhiễu kết quả "xuất hiện nhiều nhất".
- **Ngưỡng confidence tối thiểu:** bỏ qua các lần đọc OCR có confidence quá thấp trước khi voting.
- **Nội suy frame thiếu:** dùng logic tương tự `add_missing_data.py` của repo tham khảo, nội suy vị trí bbox ở các frame model bỏ sót detect.
- **Track quá ngắn:** nếu track chỉ có dưới N lần đọc OCR hợp lệ (ví dụ N=3 — cần tự xác định qua thử nghiệm), đánh dấu kết quả "độ tin cậy thấp" thay vì đưa ra kết quả voting có thể sai.

### Module 7 — Output & Visualize

- Xuất `results.csv`/`.json` gồm: `frame_number, track_id, car_bbox, plate_bbox, plate_text, plate_confidence, vote_count`.
- Lưu kèm ảnh crop biển số ứng với mỗi track_id (phục vụ đối chiếu khi viết báo cáo, giải thích trường hợp đọc sai).
- Vẽ overlay: khung xe, khung biển số, text biển số đọc được, lên video xuất ra — dùng dữ liệu đã nội suy để overlay mượt, tránh giật/nhấp nháy.

---

## 4. Cấu trúc thư mục project đề xuất

```
license-plate-video/
├── data/
│   ├── raw/                     # video gốc dùng để test/demo
│   ├── datasets/                # dataset ảnh biển số để train (theo cấu trúc mục 3.1)
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── data.yaml
│   └── models/
│       ├── yolov8n.pt           # model detect xe (pretrained COCO)
│       └── plate_detector.pt    # model detect biển số (fine-tuned)
├── sort/                        # clone từ abewley/sort (nếu dùng SORT thay vì ByteTrack)
├── src/
│   ├── extract_frames.py        # Module 0 — mới
│   ├── detect_vehicle.py        # Module 1
│   ├── detect_plate.py          # Module 2
│   ├── tracker.py               # Module 3 — wrap SORT/ByteTrack
│   ├── preprocess_plate.py      # Module 4 — crop, deskew, tách dòng
│   ├── ocr.py                   # Module 5 — PaddleOCR wrapper
│   ├── postprocess.py           # Module 6 — luật biển số VN + voting
│   ├── add_missing_data.py      # nội suy frame thiếu
│   └── visualize.py             # Module 7 — vẽ overlay lên video
├── main.py                      # chạy toàn bộ pipeline trên 1 video
├── configs/
│   └── yolo_train_config.yaml   # config training (mục 3.2.2)
├── requirements.txt
├── results/
│   ├── results.csv
│   ├── plate_crops/             # ảnh crop biển số theo track_id
│   └── output_demo.mp4
└── report/                      # báo cáo, slide đồ án
```

---

## 5. Công nghệ / thư viện chính

| Thành phần        | Lựa chọn                       | Ghi chú                                                                       |
| ------------------- | -------------------------------- | ------------------------------------------------------------------------------ |
| Detect xe           | YOLOv8n (`ultralytics`)        | Dùng thẳng pretrained COCO                                                   |
| Detect biển số    | YOLO11n hoặc YOLOv8n fine-tuned | Train trên dataset biển số VN đã gộp và lọc trùng                     |
| Tracking            | SORT hoặc ByteTrack             | SORT để bắt đầu nhanh; chuyển ByteTrack nếu video có nhiều che khuất |
| OCR                 | PaddleOCR (PP-OCRv5)             | EasyOCR làm fallback tạm thời nếu setup PaddleOCR gặp khó khăn          |
| Xử lý ảnh        | OpenCV                           | Crop, deskew, resize, CLAHE                                                    |
| Ngôn ngữ          | Python 3.10                      | Đúng theo môi trường trong repo tham khảo                                |
| Môi trường train | Google Colab (GPU T4 free tier)  | Đủ cho epochs=100, batch=16 với dataset vài chục nghìn ảnh              |

---

## 6. Kế hoạch triển khai theo tuần

| Tuần | Công việc                                                                                                                                                                              | Đầu ra                                                                                       |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 1     | Setup môi trường (bao gồm test PaddleOCR trên CPU trước), clone repo tham khảo, chạy thử pipeline gốc trên video mẫu, viết Module 0 (extract frame)                        | Pipeline baseline chạy end-to-end; xác định các threshold Module 0 dựa trên video thật |
| 2     | Thu thập + gộp dataset biển số VN, lọc trùng lặp, convert đồng nhất format, chia train/val/test theo tỷ lệ 80/10/10 dựa trên số ảnh thật (không dùng số ước tính) | Dataset sẵn sàng để train, có`data.yaml`                                                |
| 3     | Fine-tune YOLO detect biển số, thử vài cấu hình (epochs, box loss gain), đánh giá mAP trên tập val                                                                            | Model`plate_detector.pt`; chọn cấu hình tốt nhất                                        |
| 4     | Tích hợp tracking (SORT baseline, thử ByteTrack nếu cần) + module tiền xử lý (deskew, tách dòng)                                                                               | Pipeline xử lý đúng cả biển 1 dòng và 2 dòng, track ổn định trên video nhiều xe  |
| 5     | Tích hợp PaddleOCR, viết hậu xử lý luật biển số VN + voting theo track_id                                                                                                       | Kết quả đọc biển số ổn định qua nhiều frame                                          |
| 6     | Gán nhãn tay ground truth cho vài video test, đo các metric ở mục 7, tinh chỉnh tham số dựa trên kết quả đo được                                                        | Bảng kết quả đánh giá thực tế (không phải số giả định)                           |
| 7     | Viết báo cáo, làm slide, chuẩn bị demo                                                                                                                                             | Tài liệu đồ án hoàn chỉnh                                                               |

---

## 7. Tiêu chí đánh giá (metrics)

| Metric                               | Công thức                                                                                          | Cách đo                                                                                               | Ghi chú                                                                                           |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **mAP@0.5**                    | Average Precision trung bình tại IoU threshold 0.5, tính tự động bởi Ultralytics khi validate | Chạy`model.val()` trên tập test đã tách riêng                                                  | Chỉ đánh giá chất lượng detect biển số, không liên quan OCR                             |
| **Plate Detection Rate**       | (số biển số model detect được) / (số biển số thật có trong ground truth)                  | Cần gán nhãn tay tập video test                                                                     | Đo khả năng "tìm thấy" biển số, kể cả khi đọc sai                                       |
| **Plate Recognition Accuracy** | (số biển đọc đúng toàn bộ ký tự) / (số biển đã detect được)                         | So khớp text OCR (sau voting) với nhãn ground truth                                                  | Đây là metric quan trọng nhất, phản ánh chất lượng end-to-end                            |
| **Tracking ID Switch Rate**    | (số lần track_id bị đổi giữa chừng cho cùng 1 xe) / (tổng số track)                        | Quan sát thủ công trên video overlay, hoặc theo dõi khi 1 xe biết trước xuất hiện liên tục | Không có công cụ đo tự động đơn giản — cần review video bằng mắt cho tập test nhỏ |
| **FPS xử lý trung bình**    | (số frame xử lý) / (thời gian xử lý, giây)                                                    | Đo thời gian chạy toàn pipeline trên video test                                                    | Biết pipeline có khả năng gần real-time hay không                                            |

**Về việc đặt target số cụ thể (VD: "mAP ≥ 0.95"):** không đặt trước khi có baseline. Cách làm đúng: chạy xong tuần 3 (train model detect) và tuần 5 (tích hợp OCR), ghi nhận con số baseline thật, rồi đặt mục tiêu cải thiện cho tuần 6 dựa trên baseline đó (ví dụ "cải thiện Recognition Accuracy thêm 10-15 điểm % so với baseline" thay vì con số tuyệt đối áp đặt từ đầu không có căn cứ).

---

## 8. Rủi ro & phương án dự phòng

| Rủi ro                                                         | Ảnh hưởng                                                               | Phương án dự phòng                                                                                                                |
| --------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset biển số VN không đủ đa dạng góc quay/ánh sáng | Model detect/OCR kém trên điều kiện thực tế demo                    | Bổ sung synthetic data, augmentation qua Roboflow, tự quay thêm video bổ sung                                                      |
| SORT bị mất track_id khi xe che khuất nhau                   | Một xe bị tính thành nhiều track, sai số liệu và voting            | Nâng cấp lên ByteTrack (đã tích hợp sẵn trong`ultralytics`, không cần thư viện ngoài)                                   |
| OCR đọc sai với biển mờ/xa camera                          | Giảm Recognition Accuracy                                                 | Ngưỡng confidence tối thiểu + ngưỡng kích thước ảnh crop tối thiểu + ưu tiên kết quả voting đa số                    |
| Biển 2 dòng bị tách sai dòng                               | OCR đọc lẫn ký tự giữa 2 dòng                                       | Đảm bảo đúng thứ tự deskew → tách dòng → ghép ngang (Module 4)                                                             |
| Video quay ban đêm / thiếu sáng                             | Cả detect lẫn OCR đều giảm chất lượng                              | CLAHE tăng tương phản ở Module 4; cân nhắc loại video quá tối khỏi phạm vi demo nếu không đủ thời gian xử lý riêng |
| Góc nghiêng camera quá lớn (>30°)                          | Deskew không xử lý đủ, biển bị méo nặng                           | Đặt ngưỡng góc nghiêng tối đa, bỏ qua hoặc đánh dấu "độ tin cậy thấp" cho các bbox nghiêng quá ngưỡng            |
| Dataset gộp bị trùng lặp giữa các nguồn                  | Model overfit, mAP báo cáo cao giả tạo do trùng ảnh giữa train/test | Lọc trùng bằng hash ảnh trước khi chia train/val/test (mục 3.1)                                                                 |
| Setup PaddleOCR trên máy gặp lỗi dependency/crash trên CPU | Chặn tiến độ tuần 1                                                   | Set`enable_mkldnn=False`; nếu vẫn lỗi, dùng EasyOCR tạm thời làm fallback                                                     |
| Không có ground truth để đo accuracy thật                 | Không thể báo cáo số liệu đáng tin cậy                            | Dành thời gian gán nhãn tay 1 tập video test nhỏ ở tuần 6, trước khi viết báo cáo                                         |

---

*Spec v2 được xây dựng dựa trên spec v1, gộp thêm các đề xuất cải tiến (dataset structure, YOLO config, frame extraction, chốt OCR engine, metrics chi tiết, risk mở rộng), có điều chỉnh các con số chưa kiểm chứng thành "ước tính/khởi điểm cần tinh chỉnh" thay vì số cố định, ngày 14/09/2026.*
