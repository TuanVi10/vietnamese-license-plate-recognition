# Các lỗi & khó khăn thường gặp khi làm ALPR trên video (và cách tránh)

**Bối cảnh:** Dự án nhận diện biển số xe qua video có nhiều xe chạy qua lại, pipeline gồm 7 module (xem file spec). File này liệt kê các lỗi/pitfall theo từng module, để bạn biết trước và tránh khi code, thay vì phát hiện ra sau khi đã làm xong mới phải sửa lại toàn bộ.

---

## 1. Lỗi liên quan đến Vehicle Detection (Module 1)

- **Bỏ qua bước detect xe, detect thẳng biển số trên cả khung hình lớn:** làm vậy sẽ tăng false positive (model nhận nhầm bảng quảng cáo, biển báo giao thông, cửa sổ hình chữ nhật... thành biển số) và chậm hơn vì phải quét toàn bộ ảnh độ phân giải cao. Nên detect xe trước, crop vùng xe rồi mới detect biển số bên trong.
- **Dùng class COCO sai:** YOLOv8n pretrained COCO có class `car`, `motorcycle`, `bus`, `truck` riêng biệt — nếu chỉ lọc class `car` sẽ bỏ sót xe máy (rất phổ biến ở VN). Nhớ lọc đủ các class liên quan.
- **Bounding box xe quá sát/quá rộng:** nếu crop quá sát, biển số nằm rìa ảnh dễ bị cắt mất một phần khi truyền sang Module 2. Nên nới rộng bbox thêm 10-15% mỗi cạnh trước khi crop.

## 2. Lỗi liên quan đến Plate Detection (Module 2)

- **Train/test dataset không đại diện điều kiện demo thực tế:** dataset toàn ảnh chụp ban ngày, góc thẳng, nhưng video demo của bạn quay buổi tối hoặc góc nghiêng → model detect kém hẳn dù mAP lúc train cao. Nên kiểm tra dataset có đủ đa dạng về ánh sáng/góc quay/khoảng cách giống với điều kiện bạn sẽ demo hay không, trước khi train.
- **Gộp nhiều dataset nhưng không thống nhất format nhãn:** một số dataset dùng YOLO format, một số dùng Pascal VOC/COCO, một số gán nhãn theo polygon thay vì bbox — nếu gộp mà không convert đồng nhất, model sẽ học sai hoặc lỗi khi load.
- **Không chia lại train/val/test sau khi gộp:** nếu mỗi dataset con đã tự chia sẵn train/val/test riêng rồi bạn gộp nguyên theo thư mục, có thể dẫn đến mất cân bằng (ví dụ tập test quá ít) hoặc rò rỉ dữ liệu (data leakage) nếu cùng 1 ảnh gốc xuất hiện ở cả tập train lẫn test do trùng nguồn.
- **Quên augment dữ liệu:** biển số trong thực tế có thể bị mờ, xa, nghiêng, ánh sáng chói — nếu dataset train "quá sạch" (ảnh rõ, gần, thẳng), model sẽ không tổng quát tốt khi gặp điều kiện xấu hơn lúc demo.

## 3. Lỗi liên quan đến Tracking (Module 3)

- **ID switch khi 2 xe che khuất nhau (occlusion):** SORT chỉ dựa vào vị trí + Kalman Filter, không có đặc trưng ngoại hình (appearance), nên khi 2 xe đi cạnh nhau/che khuất tạm thời, rất dễ bị đổi ID giữa chừng → 1 xe bị tính thành 2 track khác nhau, làm sai số liệu và voting OCR. Nếu video có traffic dày, cân nhắc dùng DeepSORT hoặc ByteTrack thay vì SORT thuần.
- **Track ID bị tạo mới liên tục dù cùng 1 xe:** xảy ra khi Module 1/2 detect bị "chớp tắt" (frame này detect được, frame sau lại miss) — tracker hiểu nhầm là xe biến mất rồi có xe mới xuất hiện. Cần đặt tham số `max_age` (số frame cho phép track "sống" dù không có detection mới) hợp lý, không quá thấp.
- **Không đồng bộ tọa độ giữa Module 1 và Module 3:** nếu Module 3 track theo bbox xe nhưng Module 2/4 lại xử lý theo bbox biển số, cần đảm bảo ánh xạ đúng track_id của xe sang biển số tương ứng của xe đó qua từng frame — nhầm lẫn ở bước này sẽ làm voting gộp sai biển số giữa các xe khác nhau.
- **Test tracking chỉ trên video ít xe:** tracker hoạt động tốt trên video 2-3 xe nhưng dễ lỗi khi video có 10+ xe cùng lúc (giao lộ đông). Nên test sớm với video có mật độ xe cao để phát hiện vấn đề trước khi làm các bước sau.

## 4. Lỗi liên quan đến Tiền xử lý ảnh biển số (Module 4)

- **Tách sai dòng ở biển số 2 dòng (xe máy VN):** nếu ảnh biển bị nghiêng mà chưa deskew trước khi tách dòng, thuật toán tách dòng theo tọa độ ngang sẽ cắt lẫn ký tự giữa 2 dòng. Thứ tự đúng phải là: deskew trước → tách dòng sau → ghép ngang.
- **Resize sai tỷ lệ (aspect ratio) làm méo ký tự:** nếu resize ảnh biển số về kích thước cố định mà không giữ tỷ lệ gốc, chữ số có thể bị kéo dãn/méo, làm giảm độ chính xác OCR đáng kể. Nên resize có giữ tỷ lệ + pad thêm nếu cần.
- **Ảnh biển số quá nhỏ (xe ở xa camera):** sau khi crop, ảnh biển số có thể chỉ còn vài chục pixel chiều rộng — quá nhỏ để OCR đọc chính xác. Cần đặt ngưỡng kích thước tối thiểu, bỏ qua hoặc đánh dấu "không đủ tin cậy" cho các bbox quá nhỏ, thay vì cố ép đọc.
- **Bỏ qua bước tăng tương phản/khử nhiễu:** biển số ban đêm hoặc dưới nắng gắt thường bị chói/tối một phần — thêm bước cân bằng histogram (CLAHE) trước khi đưa vào OCR có thể cải thiện đáng kể độ chính xác.

## 5. Lỗi liên quan đến OCR (Module 5)

- **Kỳ vọng OCR pretrain đọc đúng 100% ngay từ đầu:** EasyOCR/PaddleOCR pretrain trên dữ liệu tổng quát (không riêng biển số VN), nên sẽ có tỷ lệ đọc sai nhất định, đặc biệt với phông chữ biển số VN. Đừng thiết kế hệ thống theo hướng "tin tuyệt đối 1 lần đọc" — luôn cần bước voting qua nhiều frame ở Module 6.
- **Nhầm lẫn ký tự dễ gây lỗi (0/O, 1/I, 8/B, 5/S):** đây là lỗi OCR rất phổ biến với biển số. Không thể loại bỏ hoàn toàn, nhưng có thể giảm bằng cách áp luật định dạng biển số VN ở Module 6 (biết vị trí nào bắt buộc là số, vị trí nào là chữ) để tự động sửa các trường hợp rõ ràng.
- **Không giới hạn ngôn ngữ/ký tự cho OCR engine:** một số OCR mặc định nhận diện đa ngôn ngữ, có thể trả về ký tự lạ (dấu, ký tự Latin mở rộng...) không có trong biển số VN. Nên giới hạn charset OCR chỉ gồm 0-9 và A-Z nếu engine hỗ trợ, để giảm nhiễu.
- **Chạy OCR trên mọi frame thay vì lọc trước:** nếu chạy OCR trên toàn bộ frame của mọi track (có thể hàng trăm frame/xe), tốc độ xử lý sẽ rất chậm. Nên lọc: chỉ chạy OCR trên các frame có bbox biển số đủ lớn/đủ rõ (dùng độ nét hoặc kích thước làm tiêu chí), hoặc lấy mẫu cách quãng (ví dụ mỗi 3-5 frame).

## 6. Lỗi liên quan đến Hậu xử lý & Voting (Module 6)

- **Voting theo majority nhưng không loại kết quả rác trước:** nếu không lọc bỏ các chuỗi ký tự rõ ràng sai định dạng trước khi voting (ví dụ OCR trả về chuỗi 2 ký tự hoặc toàn ký tự đặc biệt), các kết quả rác này có thể làm nhiễu việc chọn "kết quả xuất hiện nhiều nhất". Nên áp luật định dạng biển số VN để lọc trước, rồi mới voting trên các kết quả hợp lệ.
- **Không có ngưỡng confidence tối thiểu:** nếu nhận mọi kết quả OCR bất kể confidence, các lần đọc confidence rất thấp (gần như đoán bừa) vẫn được tính vào voting, làm giảm độ chính xác cuối cùng. Nên đặt ngưỡng confidence tối thiểu để loại bỏ.
- **Track quá ngắn (xe xuất hiện vài frame rồi đi mất, ví dụ đi ngang qua nhanh):** với track chỉ có 2-3 lần OCR, voting sẽ kém tin cậy. Cần định nghĩa rõ: track ngắn hơn N frame thì xử lý thế nào (báo "không đủ dữ liệu" thay vì đưa ra kết quả có thể sai).
- **Luật định dạng biển số VN viết cứng (hard-code) không đủ tổng quát:** biển số VN có nhiều loại (biển trắng dân sự, biển xanh cơ quan nhà nước, biển đỏ quân đội, biển vàng kinh doanh vận tải...) với format hơi khác nhau. Nếu chỉ code luật cho 1 loại phổ biến nhất, các biển số khác sẽ luôn bị loại nhầm là "sai định dạng". Nên xác định rõ phạm vi biển số nào dự án sẽ hỗ trợ, ghi rõ trong phần out-of-scope nếu không hỗ trợ hết.

## 7. Lỗi liên quan đến Output & Visualize (Module 7)

- **Không lưu lại ảnh crop biển số kèm kết quả:** khi chấm điểm/viết báo cáo, nếu chỉ có text kết quả mà không có ảnh gốc để đối chiếu, sẽ khó giải thích tại sao model đọc sai ở trường hợp cụ thể nào đó. Nên lưu kèm ảnh crop biển số ứng với mỗi track_id.
- **Overlay video bị giật/nhấp nháy do bbox không được làm mượt:** nếu vẽ trực tiếp bbox thô từng frame (có frame miss detect), video demo sẽ trông giật. Dùng lại kỹ thuật nội suy (interpolate) như trong spec sẽ giúp overlay mượt hơn.

## 8. Vấn đề chung về hiệu năng & triển khai

- **Pipeline chạy được trên ảnh tĩnh nhưng quá chậm trên video dài:** chạy YOLO + OCR trên từng frame của video vài phút có thể tốn rất nhiều thời gian nếu không tối ưu. Cân nhắc: chỉ detect biển số mỗi vài frame thay vì mọi frame (vì xe không di chuyển quá nhanh giữa các frame liên tiếp), dùng model nhẹ hơn (YOLOv8n thay vì YOLOv8x) nếu cần tốc độ.
- **Không test sớm trên video thật, chỉ test trên ảnh đơn lẻ:** nhiều lỗi (tracking, voting, hiệu năng) chỉ lộ ra khi chạy trên video thật có nhiều xe — nên chạy thử pipeline end-to-end (dù còn thô) càng sớm càng tốt (gợi ý trong spec: ngay tuần 1), thay vì hoàn thiện từng module riêng lẻ rồi mới ráp lại.
- **Thiếu log/thống kê trung gian để debug:** khi kết quả cuối sai, nếu không có log ở từng bước (bao nhiêu xe detect được, bao nhiêu biển số detect được, bao nhiêu lần OCR thành công...) sẽ rất khó xác định lỗi nằm ở module nào. Nên in log số liệu cơ bản ở mỗi module trong lúc phát triển.

## 9. Vấn đề khi đánh giá (evaluation)

- **Không có tập ground truth để tính accuracy:** nếu không tự gán nhãn thủ công một số video test (biển số chính xác của từng xe), sẽ không có cách nào đo được Plate Recognition Accuracy thực sự — chỉ đánh giá bằng cảm quan "nhìn có vẻ đúng". Nên dành thời gian gán nhãn tay một tập video test nhỏ (ví dụ 5-10 video, vài chục xe) trước khi vào giai đoạn tuần 6.
- **Đánh giá chỉ dựa trên 1 video demo duy nhất:** dễ dẫn đến overfit theo đúng điều kiện của video đó (góc quay, ánh sáng cụ thể) mà không biết pipeline có tổng quát hay không. Nên test trên ít nhất vài video có điều kiện khác nhau (ban ngày/tối, ít xe/đông xe, góc quay khác nhau).

## 10. Lưu ý quan trọng khi kỳ vọng kết quả

- Đừng kỳ vọng độ chính xác OCR đạt gần 100% như hệ thống thương mại (ANPR chuyên dụng thường dùng camera hồng ngoại chuyên biệt + góc quay cố định tối ưu) — với dataset public + camera thường, mức chính xác 70-90% tùy điều kiện là hợp lý cho một đồ án/dự án cá nhân.
- Việc pipeline "chạy được" (không lỗi runtime) khác với "chạy đúng" (kết quả chính xác) — nên luôn kiểm tra bằng mắt một số kết quả cụ thể (ảnh crop + text đọc được) ở mỗi module, đừng chỉ tin vào việc code không báo lỗi.

---

*Tài liệu tổng hợp dựa trên các vấn đề thường gặp khi triển khai pipeline ALPR (Automatic License Plate Recognition) trên video nhiều xe, đối chiếu với kiến trúc pipeline đã đề xuất trong file spec, ngày 14/09/2026.*
