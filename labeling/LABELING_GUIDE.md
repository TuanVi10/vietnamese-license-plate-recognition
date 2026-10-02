# Hướng dẫn gán nhãn biển số (labeling) — 4 góc + nội dung biển

## 1. Mục đích

Mỗi ảnh là 1 **crop biển số** cắt từ video. Với mỗi ảnh, bạn phải gán **2 thứ**:

1. **4 góc** của tấm biển (để huấn luyện/đánh giá model chỉnh góc bằng keypoint).
2. **Nội dung biển số thật** (ground-truth text, để đo độ chính xác OCR cuối cùng).

> Làm **cả hai cùng lúc** trên mỗi ảnh — khi đã mở ảnh để click 4 góc thì gõ luôn số biển, chỉ tốn vài giây.

## 2. Cấu trúc thư mục

```
labeling/
├── images/           # ảnh crop, tên tăng dần: 000001.jpg, 000002.jpg, ...
├── manifest.csv      # danh sách ảnh (image, track_id, frame_index, source, score, w, h)
└── LABELING_GUIDE.md # file này
```

- `manifest.csv` để biết mỗi ảnh thuộc `track_id`/`frame_index` nào (dùng khi đối chiếu kết quả).
- Sau khi label xong, xuất ra 1 file nhãn theo **định dạng ở mục 5**.

## 3. Quy ước 4 góc (QUAN TRỌNG NHẤT)

### 3.1 Đặt điểm ở đâu

Đặt điểm **đúng mép góc của tấm biển** (viền ngoài tấm kim loại), **không phải** góc của khung ảnh hay góc của bbox dò (bbox có thể loe ra nền).

### 3.2 Thứ tự cố định (theo chiều kim đồng hồ)

Mọi ảnh phải theo đúng thứ tự này:

| Chỉ số | Tên              | Vị trí          |
| -------- | ----------------- | ----------------- |
| 0        | TL (Top-Left)     | góc trên-trái  |
| 1        | TR (Top-Right)    | góc trên-phải  |
| 2        | BR (Bottom-Right) | góc dưới-phải |
| 3        | BL (Bottom-Left)  | góc dưới-trái |

```
TL ──────────── TR
│               │
│   TẤM BIỂN    │
│               │
BL ──────────── BR
```

**Sai thứ tự = model học sai = vô nghĩa.** Đây là lỗi phổ biến nhất, kiểm tra kỹ từng ảnh.

### 3.3 Biển bị nghiêng / phối cảnh

Vẫn đặt 4 điểm theo 4 góc vật lý của biển (điểm nào "cao hơn" là TR/TL tuỳ ảnh — dùng **thứ tự chiều kim đồng hồ quanh biển**, không dùng trục ảnh).

## 4. Quy ước nội dung biển (text)

- Gõ **chuỗi biển số**, bỏ dấu gạch/dấu chấm/khoảng trắng, **chỉ chữ số 0-9 và chữ A-Z**.
- Thứ tự: `mã tỉnh (2 số) + series (1-2 chữ) + số đăng ký (3-6 số)`.

Ví dụ:

| Hiển thị trên biển    | Ghi vào nhãn |
| ------------------------- | -------------- |
| `51F-12345`             | `51F12345`   |
| `29-B1 12345` (2 dòng) | `29B112345`  |
| `61-G7 9984` (2 dòng)  | `61G79984`   |

### Biển mờ / bị che / không rõ

- **Mắt người vẫn đọc được** (dù mờ, nghiêng): cố gắng ghi đúng — đây chính là ca khó muốn đo.
- **Hoàn toàn không đọc nổi** bằng mắt: ghi `unreadable` (và vẫn đánh 4 góc nếu còn thấy viền biển).
- **Không thấy viền biển** (crop lệch, trống): bỏ qua ảnh đó, đánh dấu `skip`.

## 5. Định dạng file nhãn (khi hoàn tất)

Gửi lại 1 file **CSV** gồm các cột (toạ độ góc là **pixel**, gốc ở góc trên-trái ảnh):

```csv
image,plate_text,tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y
000001.jpg,51F12345,10,8,290,7,291,95,9,96
000002.jpg,29B112345,5,4,180,3,181,120,4,119
000003.jpg,unreadable,0,0,0,0,0,0,0,0
```

- `image`: đúng tên file trong `images/`.
- `plate_text`: chuỗi biển (hoặc `unreadable`).
- `tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y`: toạ độ 4 góc theo đúng thứ tự TL→TR→BR→BL.

*(Nếu dùng công cụ có export riêng — CVAT/LabelMe/COCO — thì cứ xuất 4 điểm theo đúng thứ tự + text, tôi sẽ tự convert.)*

## 6. Công cụ gợi ý

1. **CVAT** (tốt nhất cho keypoint): tạo task dạng *points*, template 4 điểm gán nhãn `TL/TR/BR/BL` — tự ép thứ tự.
2. **Make Sense** (`makesense.ai`, chạy trình duyệt): chọn *Points*, thêm 4 điểm.
3. **Label Studio**: dùng *Keypoint labels*.
4. **Roboflow**: vẽ *polygon* 4 đỉnh rồi tôi trích 4 đỉnh thành 4 góc.

## 7. Checklist chất lượng

- [ ] 4 điểm theo đúng thứ tự TL→TR→BR→BL trên **mọi** ảnh.
- [ ] Điểm đặt đúng **mép góc biển**, không lệch vào trong ký tự.
- [ ] Text viết hoa, bỏ dấu phân cách, đúng `mã tỉnh + series + số`.
- [ ] Đã ghi đủ cả ảnh khó (nghiêng, mờ, 2 dòng) — không chỉ ảnh dễ.
- [ ] Ảnh không thấy biển → no_plate; không đọc được text → `unreadable`.
