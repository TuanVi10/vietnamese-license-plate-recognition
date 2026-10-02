# Lessons Learned - session_e9d3d5c5-779a-4650-aac9-d7de4274d8cd

> Session: `session_e9d3d5c5-779a-4650-aac9-d7de4274d8cd`  |  Generated: 2026-09-26 10:26:50

## Muc tieu

- **Task giao cho nhóm (node `USER`):** "Xây dựng Module 1 (`detect_vehicle.py`), Module 3 (`tracker.py`) và file glue".
- **Bối cảnh thực tế trong log:** workspace **đã có sẵn** các file này cùng phần lớn hệ thống khác — `src/frame_extractor.py` (Module 0), `src/detector.py`, `src/ocr.py`, `src/plate_reader.py`, `src/postprocess.py`, `src/preprocess_plate.py`, `src/visualize.py`, `src/config.py`, `main.py`, `README.md`, `manual.md`, `tools/self_check.py`, `tools/self_check_tracking.py`. Vì vậy công việc thực chất là **hoàn thiện / sửa lỗi** Module 1 + Module 3 + glue theo một **danh sách bug tĩnh B1–B15** do một vòng "static review" đưa ra, không phải viết mới từ đầu.
- **Kiến trúc mục tiêu (thể hiện trong docstring của glue đã đọc):** Module 0 (trích & lọc frame) → Module 1 (dò phương tiện, YOLOv8n) → Module 2 (dò biển số, tùy chọn) → Module 3 (tracking SORT) → Module 4/5/6 (tiền xử lý + OCR + voting) → Module 7 (output/overlay), với **voting theo `track_id`**.
- **Cách kết thúc phiên:** node `FINAL` báo `Loop limit reached (1)` với `loop_counter` `count: 1 / max: 1` — tức phiên dừng vì chạm giới hạn vòng lặp, sau **1 vòng sửa**, và báo cáo tổng kết của agent là điểm dừng.

## Van de gap phai & cach giai quyet

Nguồn phát hiện: **static review** (bug list B1–B15). Agent thực hiện sửa và viết báo cáo là **`Programmer Test Modification`** — agent này bắt đầu bằng việc liệt kê file (`describe_available_files`) và đọc các source hiện có (`detect_vehicle.py`, `tracker.py`, `video_pipeline.py`, `self_check_tracking.py`, `config.py`, `frame_extractor.py`, `main.py`) để "hiểu trạng thái hiện tại", rồi sửa theo danh sách có sẵn. Bảng dưới đây bám sát nội dung báo cáo cuối của agent:

| Mã | Vấn đề | Ai phát hiện | Cách giải quyết | Kết quả |
|---|---|---|---|---|
| B1 | `linear_assignment` (tracker) dùng điều kiện `if size > n_rows and size > n_cols:` — **không bao giờ đúng** (vì `size = max(...)`), khiến vùng đệm "giả" giữ sentinel `1e6` | static review | Đệm ma trận vuông bằng hàng/cột giả **chi phí 0** | Sentinel không còn bóp méo phép gán detection ↔ track |
| B2 (tracker) | `KalmanFilter.update` gọi thẳng `np.linalg.inv(innovation_cov)` → `LinAlgError` khi ma trận suy biến (box trùng nhau/thoái hoá) | static review | Fallback sang `np.linalg.pinv` khi `innovation_cov` suy biến | Không còn crash khi có box trùng lặp |
| B2 (glue) | `tracker.update` trong glue **không được bọc lỗi** → 1 frame lỗi có thể làm sập cả video | static review | Bọc `try/except`, ghi cảnh báo + reset tracker | Một frame xấu không giết cả video (chỉ mất trạng thái tracking) |
| B3 | `iou_matrix` (tracker) cài bằng vòng lặp Python **O(n²)** | static review | Vector hoá bằng numpy, thêm helper `_boxes_to_array` | Tính IoU nhanh hơn, bỏ vòng lặp thuần Python |
| B4 | Kết quả pipeline **ghim toàn bộ ảnh frame** trong RAM | static review | Thêm `run(..., keep_frame_images=False)`, gán `frame_result.frame = None` và `record.frame = None`; gom readings **tăng dần** thay vì dựng lại từ danh sách `frames` | Kết quả trả về không còn giữ mọi full frame |
| B5 | `plate_region_ratio` mặc định `1.0` (lấy cả bbox xe) **không khớp hành vi được tài liệu mô tả**, khiến OCR chạy trên cả xe | static review | Đổi mặc định `1.0 → 0.5` (nửa dưới xe — nơi thường đặt biển) | Fallback vùng biển số khớp tài liệu, giảm OCR sai vùng |
| B6 | Gán biển → track **phụ thuộc thứ tự duyệt** track của tracker | static review | Thêm `_best_plate_score`, xử lý track theo **điểm khớp giảm dần** | Biển được "nhận" bởi xe khớp nhất, bất kể thứ tự tracker trả về |
| B7 | Một lần miss là reset ngay `hit_streak` | static review | **Không sửa** — giữ như "đánh đổi đã ghi nhận của SORT" | Semantics đã kiểm chứng được bảo toàn |
| B8 | `clip_bbox` (Module 1) unpack cứng 4 phần tử → `ValueError` với mảng `(1,4)`/`(4,1)`, phần tử thừa, hay `NaN`/`inf` | static review | Chấp nhận nhiều shape và clip an toàn thay vì ném lỗi | Đường xử lý bbox không còn nổ ngoại lệ |
| B9 | `SortTracker.update` không xử lý `detections=None` | static review | Chuẩn hoá `None` → `[]` | Gọi với `None` không lỗi |
| B10 | Doc drift: không nêu rõ nghĩa của `classes=()` (rỗng = **không lọc lớp**, tức mọi lớp COCO) | static review | Ghi rõ trong docstring + comment inline | Tài liệu khớp hành vi thật |
| B11 | Doc drift ở glue: module docstring không mô tả đúng luồng voting thực tế | static review | Cập nhật docstring mô tả đúng `PostProcessor.process`/`format` | Tài liệu khớp code |
| B12 | Import `field` không dùng trong `tracker.py`; tham số `frame` không dùng trong `_save_overlay` (glue) | static review | Xoá import thừa, xoá tham số `frame` | Dọn dẹp, giảm nhiễu |
| B13 | Thuộc tính `last_bbox` được gán nhưng **không bao giờ đọc**; doc `state` mơ hồ | static review | Xoá `last_bbox`; làm rõ doc `state` | Dọn dẹp, rõ nghĩa hơn |
| B14 | `max_age`/`min_hits` tính theo **số frame** | static review | **Không sửa** — giữ làm đánh đổi SORT đã ghi nhận | Không thay đổi semantics đã kiểm chứng |
| B11 (ngoài source) | Văn xuôi trong `README.md` / `manual.md` lệch với code | static review | **Không sửa** — agent nêu lý do "không thể viết lại an toàn các file không phải source" | Còn tồn đọng, được ghi nhận rõ |
| B15 | Nằm trong dải được nêu (`B1–B15`) nhưng **không được mô tả riêng** trong báo cáo | static review | — | Không đủ dữ kiện trong log để kết luận |

**Kiểm chứng:** báo cáo khẳng định các assertion happy-path trong `tools/self_check_tracking.py` vẫn thoả — nhưng bằng cách **"verified by tracing"** (suy luận thủ công qua hình học, Hungarian, full-frame fallback, vòng đời SORT, `min_hits`, `first_frame`, và đường lỗi Module 1). Log **không ghi lại việc chạy script self-check**.

## Quyet dinh quan trong

1. **SORT thuần numpy, không phụ thuộc ngoài.** Module 3 cài tự Kalman filter + Hungarian + IoU, chỉ dùng `numpy`/`math` — không cần `scipy`/`filterpy`/`motpy`, để chạy được trong môi trường tối thiểu. Kéo theo quyết định tự cài Hungarian O(n³) (biến thể e-maxx với thế vị `u`/`v`) trong `_hungarian_square`.
2. **Chế độ "toàn khung" (`full_frame_fallback`) là xương sống của khả năng test.** Khi `model_path is None`, Module 1 coi cả frame là 1 phương tiện → toàn bộ pipeline video chạy được mà chưa cần `ultralytics`, rất tiện kiểm thử logic Module 3.
3. **Fail-soft ở biên module.** Mọi module có rủi ro (1 dò phương tiện, 2 dò biển, 4/5/6 OCR, và tracker) đều được bọc lỗi: ghi cảnh báo (`_warn`, `_warn_setup`) rồi fallback, thay vì để ngoại lệ làm sập cả video. Đây là lý do B2 phía glue phải sửa cho đồng nhất với các module khác.
4. **Bảo toàn semantics đã kiểm chứng.** B7 và B14 **cố ý không sửa** dù nằm trong bug list, để không phá các hành vi đang được self-check xác nhận — thay vào đó ghi nhận là "đánh đổi có chủ đích của SORT".
5. **Không đụng vào tài liệu văn xuôi không kiểm chứng được.** `README.md`/`manual.md` bị bỏ qua có chủ đích; chỉ sửa doc drift trong source (`docstring`/comment).
6. **Đổi default thay vì đổi tài liệu.** Với B5, chọn sửa `plate_region_ratio` mặc định về `0.5` để khớp hành vi được mô tả, thay vì viết lại mô tả.
7. **Không ghim ảnh frame mặc định.** Kết quả pipeline mặc định bỏ tham chiếu ảnh (`keep_frame_images=False`) để tránh giữ RAM; giữ ảnh là tuỳ chọn (phục vụ overlay).
8. **Tính `first_frame`/`first_timestamp` theo lần xuất hiện thật.** Glue dùng `self.tracker.trackers` (gồm cả track chưa xác nhận) qua `_track_first_seen`, thay vì lấy frame mà tracker bắt đầu trả về — tránh lệch khi `min_hits > 1`. Điều này được cố định bằng test `test_first_frame_lifecycle`.
9. **Mỗi plate detection chỉ thuộc một track trong một frame** (`used_plate_indices`) để voting theo `track_id` không bị "một biển tính cho nhiều xe".
10. **Đổi mặc định độ ưu tiên thay vì tin vào thứ tự duyệt** (B6): sắp xếp theo điểm khớp giảm dần để kết quả không phụ thuộc thứ tự tracker.

## Bai hoc rut ra

- **Có danh sách bug đánh số (B1–B15) giúp việc sửa mạch lạc và kiểm chứng được.** Mỗi mục được xử lý + báo cáo riêng, kèm tuyên bố rõ cái nào sửa, cái nào cố ý không sửa và vì sao. Đây là định dạng nên tái dùng cho các vòng review tĩnh sau.
- **Luôn hỏi "lỗi giá trị sentinel có lọt vào kết quả không?"** B1 cho thấy một class bug điển hình: đệm ma trận chi phí bằng giá trị lớn (`1e6`) rồi để vùng giả ảnh hưởng tới phép gán. Vùng giả phải có chi phí trung tính (0).
- **Số học có thể suy biến.** Box trùng nhau/thoái hoá làm `innovation_cov` suy biến → `inv` ném `LinAlgError`; giải pháp là fallback `pinv`. Tương tự, đầu vào số phải chống `NaN`/`inf` và shape bất thường (B8), và chặn `NaN` ngay từ metadata codec (`_finite_float` trong Module 0).
- **Cẩn thận với logic phụ thuộc thứ tự duyệt.** B6 là ví dụ: cùng một tập dữ liệu, kết quả gán biển ↔ xe khác nhau chỉ vì thứ tự vòng lặp. Cách chữa tổng quát: sắp theo điểm khớp / tính điểm tốt nhất trước khi cam kết.
- **Fail-soft ở ranh giới module là yêu cầu kiến trúc, không phải tính năng phụ.** Một video dài hàng nghìn frame không nên chết vì một model hỏng hay một frame lỗi; hãy bọc + cảnh báo + có đường fallback rõ ràng, và ghi cảnh báo vào kết quả (`warnings`) để người dùng thấy.
- **Đừng "sửa" các đánh đổi đã được kiểm chứng nếu không có test mới.** B7/B14 bị từ chối sửa có chủ đích — bảo toàn hành vi đang được self-check chốt quan trọng hơn việc làm đẹp theo cảm tính.
- **Doc drift cũng là bug.** B10/B11 cho thấy chú thích sai (`classes=()`), docstring mô tả sai luồng voting, và default không khớp mô tả (`plate_region_ratio`) đều gây hiểu nhầm thật. Cách xử lý có kỷ luật: sửa source + docstring trong code; với tài liệu văn xuôi không kiểm chứng được thì **ghi nhận và không tự ý viết lại**.
- **Mặc định nên tiết kiệm RAM.** Giữ ảnh full-frame trong kết quả trả về là quả bom bộ nhớ; hãy để việc giữ ảnh là opt-in và tích luỹ kết quả tăng dần thay vì dựng lại từ danh sách frame.
- **Thiết kế "model tùy chọn + fallback" làm tăng khả năng test rõ rệt.** Vì YOLO/OCR là optional, có thể viết harness `tools/self_check_tracking.py` chỉ cần `numpy` + `opencv` với stub extractor/reader, vẫn phủ được hình học, Hungarian, vòng đời SORT, `min_hits`, `first_frame`, và đường lỗi Module 1.
- **Kiểm chứng bằng suy luận không thay thế được việc chạy test.** Báo cáo chỉ khẳng định "verified by tracing"; lần sau nên **chạy `python tools/self_check_tracking.py`** và trích kết quả thực tế, đồng thời bổ sung test tường minh cho các fix mới (B1, B3, B4, B5, B6) — hiện chưa thấy test riêng cho chúng trong log.
- **Ngân sách vòng lặp là ràng buộc thật.** Phiên kết thúc ở `Loop limit reached (1)` (1/1) — hãy lập kế hoạch sao cho khối lượng sửa vừa một vòng, và luôn kết thúc bằng bản tổng kết rõ ràng (như báo cáo B1–B15) để tri thức không bị mất khi bị cắt vòng.
- **Ghi rõ phần chưa làm.** B15 không được mô tả, README/manual chưa sửa — việc nêu minh bạch các tồn đọng này giúp vòng sau tiếp nhận đúng chỗ thay vì tưởng mọi thứ đã xong.