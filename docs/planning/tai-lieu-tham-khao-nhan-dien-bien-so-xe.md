# Tài liệu tham khảo: Dự án Nhận diện biển số xe qua Video

**Mục tiêu dự án:** Đưa vào 1 video có nhiều xe chạy qua lại → hệ thống tự động phát hiện, theo dõi (tracking) và đọc được biển số của từng xe.

---

## 1. Pipeline đề xuất

```
Video đầu vào
   │
   ▼
[1] Phát hiện xe (Vehicle Detection) — YOLOv8/YOLOv11, tùy chọn
   │
   ▼
[2] Phát hiện biển số (Plate Detection) — YOLO model train riêng cho biển số
   │
   ▼
[3] Tracking qua các frame — SORT / DeepSORT / ByteTrack
   │       (gán ID cho từng xe để không đọc trùng, không mất dấu xe)
   ▼
[4] Crop + tiền xử lý ảnh biển số — deskew, tách dòng (biển 2 dòng), tăng tương phản
   │
   ▼
[5] OCR đọc ký tự — Tesseract / CRNN / PaddleOCR / EasyOCR
   │
   ▼
[6] Hậu xử lý — áp luật định dạng biển số VN, voting kết quả qua nhiều frame theo từng ID
   │
   ▼
Kết quả: danh sách xe + biển số + thời điểm xuất hiện trong video
```

> **Lưu ý quan trọng cho video nhiều xe:** Bước [3] Tracking là bắt buộc, không thể bỏ qua. Nếu chỉ detect + OCR từng frame riêng lẻ, kết quả sẽ nhấp nháy, một xe có thể bị đọc thành nhiều biển số khác nhau qua các frame. Gán ID theo dõi (tracking ID) cho từng xe rồi gộp nhiều lần đọc OCR của cùng 1 ID (lấy kết quả xuất hiện nhiều nhất hoặc confidence cao nhất) sẽ cho kết quả ổn định hơn nhiều.

---

## 2. Dataset gợi ý

### 2.1 Dataset biển số Việt Nam (ưu tiên nếu làm biển số VN)

| Tên dataset | Nguồn | Ghi chú |
|---|---|---|
| Vietnamese Car License Plate Detection | Roboflow Universe (Cuong Ta) | Dataset object detection, 1 class "plate", định dạng YOLO. |
| Vietnam-License-Plate-Recognition | Roboflow Universe | Có nhãn từng ký tự (0-9, A-X), phù hợp train OCR character-level thay vì chỉ detect vùng biển. |
| vietnamese-license-plate (school) | Roboflow Universe | ~8.397 ảnh đã gán nhãn, dataset khá lớn, tốt để tăng độ đa dạng dữ liệu. |
| Vietnam license-plate (Tran Ngoc Xuan Tin) | Roboflow Universe | ~1.005 ảnh, có kèm model pretrained và API demo để tham khảo. |
| VNLicensePlate_yolov7 | Kaggle (bomaich) | 1.000 ảnh biển số VN (cả 1 dòng và 2 dòng), đã chia sẵn train/valid/test, định dạng YOLOv7. |
| Vietnam License Plate Segment Datasets | Kaggle (duydieunguyen) | Phân vùng biển 1 dòng (LpD) và 2 dòng (LpV), gán nhãn polygon, chụp ở nhiều điều kiện ánh sáng/góc độ — rất hữu ích để tăng độ bền của model với điều kiện thực tế. |
| license-plate-dataset | Kaggle (raidendg) | Dataset biển số Việt Nam bổ sung. |
| Vietnamese License Plate Detection | Kaggle (miahuynh04) | Dataset bổ sung, có thể gộp thêm để tăng số lượng ảnh train. |
| Bộ dữ liệu biển số xe máy VN (1.750 ảnh) | nttuan8.com (bài viết hướng dẫn train YOLO-Tiny v4) | Ảnh chụp từ camera thực tế tại điểm kiểm soát ra/vào hầm gửi xe — sát với ứng dụng thực tế (bãi xe, trạm kiểm soát). |

### 2.2 Dataset quốc tế (bổ sung nếu cần thêm dữ liệu / benchmark)

| Tên dataset | Ghi chú |
|---|---|
| CCPD (Chinese City Parking Dataset) | ~250.000 ảnh biển số Trung Quốc, rất lớn, tốt để pretrain model detect trước khi fine-tune trên dữ liệu VN. |
| OpenALPR benchmark datasets | Nhiều bộ ảnh biển số các nước, dùng để test độ tổng quát của pipeline. |
| Roboflow Universe "license-plate" (đa quốc gia) | Hàng chục dataset public, có thể gộp ảnh biển số dạng chữ-số tương tự để tăng dữ liệu train cho phần detect khung biển (vùng biển số có hình dạng chữ nhật khá giống nhau giữa các nước). |

### 2.3 Vì sao lo dataset không đủ — và cách khắc phục

Nếu gộp tất cả các dataset ở mục 2.1, bạn có thể có tổng cộng **hàng chục nghìn ảnh** cho bài toán detect vùng biển số — con số này thường đủ để train YOLO detect biển số đạt độ chính xác tốt (vì bài toán detect biển số là bài toán tương đối đơn giản, object có hình dạng đặc trưng rõ ràng).

Phần dễ thiếu dữ liệu hơn thường là **OCR đọc ký tự trên biển số VN** (vì phông chữ, cách sắp chữ 1 dòng/2 dòng của VN khác các nước khác). Một số cách khắc phục nếu thấy thiếu:

- **Tự sinh dữ liệu tổng hợp (synthetic data):** viết script tạo ảnh biển số giả theo đúng font, layout biển số VN (nhiều repo như `dotrungkien/plate-recognition` có sẵn script `gen_vn.py` làm việc này) — sinh ra hàng chục nghìn ảnh biển số with nhãn chính xác 100%, không cần gán nhãn tay.
- **Dùng OCR tổng quát đã pretrain sẵn** (PaddleOCR, EasyOCR) thay vì tự train CRNN từ đầu — các OCR này đã học trên lượng dữ liệu khổng lồ, chỉ cần fine-tune nhẹ hoặc dùng thẳng, giảm đáng kể nhu cầu dữ liệu riêng.
- **Tự quay video/chụp ảnh bổ sung** tại khu vực bạn sống (bãi xe, đường phố) để tăng dữ liệu sát với điều kiện thực tế bạn sẽ demo (góc quay, ánh sáng, loại camera).
- **Data augmentation**: xoay nhẹ, thay đổi độ sáng/tương phản, thêm nhiễu, blur — Roboflow hỗ trợ sẵn tính năng này khi export dataset.

---

## 3. Source code / GitHub tham khảo

### 3.1 Repo dành riêng cho biển số Việt Nam

| Repo | Điểm nổi bật |
|---|---|
| [hieunguyen2604/alpr-do-an-mon-hoc](https://github.com/hieunguyen2604/alpr-do-an-mon-hoc) | Đồ án hoàn chỉnh: YOLO11 detect biển → PaddleOCR đọc ký tự → hậu xử lý theo luật biển số VN. Có xử lý ảnh/video/webcam, backend FastAPI + frontend React, kèm báo cáo và slide. Phù hợp nếu bạn muốn tham khảo kiến trúc đầy đủ của 1 đồ án. |
| [mrzaizai2k/License-Plate-Recognition-YOLOv7-and-CNN](https://github.com/mrzaizai2k/License-Plate-Recognition-YOLOv7-and-CNN) | YOLOv7 detect biển + Hough transform chỉnh nghiêng + CNN đọc ký tự. Có sẵn `main_video.py` chạy trực tiếp trên file video, có `vid2img.py` để cắt frame từ video làm dữ liệu train. |
| [dotrungkien/plate-recognition](https://github.com/dotrungkien/plate-recognition) | Code gọn nhẹ, có script `gen_vn.py` sinh dữ liệu biển số VN tổng hợp — hữu ích để bù dữ liệu OCR như đã nói ở mục 2.3. |

### 3.2 Repo quốc tế — pipeline cho video có nhiều xe (có tracking)

| Repo | Điểm nổi bật |
|---|---|
| [Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8) | **Rất sát với mục tiêu của bạn**: detect xe bằng YOLOv8 → detect biển số bằng model riêng → dùng module SORT để tracking từng xe qua các frame video → có script nội suy (interpolate) dữ liệu ở các frame bị mất dấu. Đây là repo tham khảo tốt nhất để học cách xử lý "nhiều xe chạy qua lại trong video". |
| [theos-ai/license-plate-recognition](https://github.com/theos-ai/license-plate-recognition) | Dùng YOLOv7, có kèm blog post và video hướng dẫn giải thích rõ từng bước. |
| [ThorPham/License-plate-detection](https://github.com/ThorPham/License-plate-detection) | Dùng YOLOv3, đơn giản, có chế độ chạy webcam trực tiếp — dễ đọc hiểu nếu bạn mới bắt đầu. |
| License Plate Recognition YOLOv4 + OpenCV + Tesseract ([hướng dẫn](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2021-11-05--license-plates-yolov4-opencv-tesseract/2021-11-05/)) | Hướng dẫn chi tiết từng lệnh chạy trên video (`detect_video.py`), có ví dụ cụ thể để crop biển số theo từng khoảng frame. |

### 3.3 Bài viết hướng dẫn tiếng Việt (giải thích thuật toán)

- [Nhận diện biển số xe Việt Nam — Viblo](https://viblo.asia/p/nhan-dien-bien-so-xe-viet-nam-Do754P9L5M6): tutorial giải thích từng bước, kèm link dataset thô.
- [Bài toán phát hiện biển số xe máy Việt Nam — nttuan8.com](https://nttuan8.com/bai-toan-phat-hien-bien-so-xe-may-viet-nam/): hướng dẫn gán nhãn bằng LabelImg và train YOLO-Tiny v4 trên Google Colab, kèm dataset 1.750 ảnh.
- [Nhận diện và trích xuất biển số xe VN — Viblo](https://viblo.asia/p/nhan-dien-va-trich-xuat-thong-tin-bien-so-xe-viet-nam-identify-and-extract-vietnamese-license-plate-information-5pPLkP08VRZ): dùng YOLOv7, có link code và dataset "GreenParking".

---

## 4. Gợi ý lộ trình làm việc

1. **Tuần 1–2:** Thu thập + gộp dataset (mục 2.1), thử chạy thử 1 repo có sẵn (gợi ý: `Automatic-License-Plate-Recognition-using-YOLOv8`) trên video mẫu để hiểu luồng xử lý.
2. **Tuần 3:** Train lại model detect biển số trên dataset Việt Nam đã gộp (fine-tune từ YOLOv8/YOLOv11 pretrained).
3. **Tuần 4:** Tích hợp tracking (SORT/ByteTrack) để xử lý video nhiều xe, tránh đọc trùng/nhấp nháy kết quả.
4. **Tuần 5:** Tích hợp OCR (khuyến nghị PaddleOCR vì hỗ trợ tốt biển 2 dòng), viết hậu xử lý theo luật biển số VN.
5. **Tuần 6:** Test trên video thực tế, đánh giá độ chính xác, viết báo cáo/demo.

---

*Tài liệu tổng hợp ngày 14/09/2026 dựa trên tìm kiếm các nguồn công khai trên GitHub, Roboflow Universe, Kaggle và các bài viết kỹ thuật tiếng Việt.*
