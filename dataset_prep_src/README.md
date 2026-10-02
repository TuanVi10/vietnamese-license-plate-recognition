# Bộ script chuẩn bị dataset — YOLO detect biển số Việt Nam

`prepare_vn_plate_dataset` biến một thư mục ảnh thô lộn xộn thành một dataset
**đúng chuẩn YOLO** để fine-tune model detect biển số xe Việt Nam
(1 class `license_plate`).

Chỉ dùng **Python 3.12 + numpy + opencv-python** (không cần PIL, ultralytics hay PyYAML).

```
raw/                       work/                          dataset/
├── clip_a/                ├── manifest.json              ├── data.yaml
│   ├── frame_001.jpg      ├── annotations.json           ├── dataset_stats.json
│   └── frame_002.jpg      ├── report.json                ├── DATASET_CARD.md
└── clip_b/                ├── splits.json                ├── images/{train,val,test}/*.jpg
    └── ...                └── images/*.jpg              └── labels/{train,val,test}/*.txt
```

---

## 1. Cài đặt

Môi trường đã có sẵn (Python 3.12 + numpy + opencv-python):

```bash
pip install -r requirements.txt   # chỉ để tham chiếu
```

## 2. Chạy nhanh

```bash
# Tự sinh nhãn bằng heuristic thị giác máy tính (pseudo-label), chia split và xuất dataset
python main.py --raw-dir data/raw --output-dir dataset --label-mode auto

# Hoặc xuất phát từ file cấu hình
python main.py --config config.default.json
```

## 3. Kịch bản sử dụng

### a) Không có nhãn → auto-label (bootstrap)

```bash
python main.py --raw-dir data/raw --output-dir dataset \
    --label-mode auto --resize-mode letterbox --image-size 640 \
    --val-ratio 0.15 --test-ratio 0.10 --seed 42
```

Bộ auto-label dựa trên tiền nghiệm rất mạnh của biển số VN: **biển luôn rộng hơn cao**
(aspect ratio ~1.2–4.5 cho cả biển dài 1 hàng, biển 2 hàng và biển vuông xe máy),
kèm mật độ cạnh cao (nhiều ký tự) và độ "đặc" của khối contour. Pipeline:

`bilateral filter → Sobel/adaptive threshold → morphological closing ngang
(ghép các ký tự thành 1 khối) → lọc contour theo diện tích/tỉ lệ/độ đặc/mật độ cạnh
→ chấm điểm → NMS → tối đa 4 box/ảnh`.

> ⚠️ Đây là **pseudo-label**: luôn review lại bằng LabelImg/CVAT trước khi tin vào
> metric production.

### b) Đã có nhãn → import

```bash
# Pascal VOC (LabelImg / CVAT export)
python main.py --raw-dir data/raw --labels-format voc --labels-path data/labels_voc --label-mode import

# COCO (1 file json)
python main.py --raw-dir data/raw --labels-format coco --labels-path data/instances_train.json --label-mode import

# YOLO (thư mục .txt, cần biết kích thước ảnh gốc — pipeline tự lấy từ manifest)
python main.py --raw-dir data/raw --labels-format yolo --labels-path data/labels_txt --label-mode import
```

### c) Có nhãn + bù thêm bằng auto-label

```bash
python main.py --raw-dir data/raw --labels-format voc --labels-path data/labels_voc --label-mode merge
```

### d) Chạy từng stage (resume, không tính lại)

```bash
python main.py --raw-dir data/raw --stages preprocess           # quét, xoá trùng, resize
python main.py --raw-dir data/raw --stages labels validate      # sinh/kiểm tra nhãn
python main.py --raw-dir data/raw --stages split write preview  # chia + xuất + ảnh QC
```

Mỗi stage lưu artifact JSON riêng trong `work/`, nên job dài có thể dừng/chạy lại.

## 4. Các module

| File | Vai trò |
|------|---------|
| `dataset_config.py` | Dataclass cấu hình (`DatasetConfig`, `PreprocessConfig`, `AutoLabelConfig`, `FilterConfig`, `SplitConfig`) + load/save/validate JSON |
| `utils.py` | Đọc/ghi ảnh an toàn với tên file tiếng Việt (unicode), letterbox/stretch + `GeoTransform`, hình học (clip/IOU/NMS), MD5 + pHash để xoá trùng, đặt tên file |
| `annotations.py` | Model `BBox` / `ImageAnnotation`, đọc–ghi nhãn YOLO `.txt`, biến đổi toạ độ, lọc, merge (class-aware NMS) |
| `preprocess.py` | Quét ảnh → kiểm tra → xoá trùng → đổi tên ASCII → resize → `work/images` + `manifest.json` |
| `converters.py` | Nhập nhãn có sẵn: VOC XML, COCO JSON, YOLO txt → `ImportedLabels` |
| `autolabel.py` | Pseudo-label biển số VN bằng thị giác máy tính cổ điển |
| `validator.py` | Kiểm tra/clip/lọc box, thống kê dataset (percentile kích thước, aspect ratio…) |
| `splitter.py` | Chia train/val/test **theo group** (không tách frame cùng video) và **stratified** |
| `writer.py` | Xuất cây dataset YOLO + `data.yaml` + `dataset_stats.json` + `DATASET_CARD.md` |
| `visualize.py` | Vẽ box, tạo contact sheet QC (`dataset/qc/*.jpg`) |
| `pipeline.py` | Orchestrator các stage, resume qua JSON manifest |
| `main.py` | CLI |
| `tests/test_smoke.py` | Smoke test end-to-end tự sinh ảnh biển số giả |

## 5. Điểm quan trọng về dữ liệu biển số VN

1. **Xoá trùng là bắt buộc.** Video cắt frame ra sẽ cho ra hàng nghìn ảnh gần giống
   nhau; nếu không xoá, val/test sẽ "rò rỉ" và metric ảo.
   `preprocess.dedupe_exact` (MD5) bật mặc định, `dedupe_near` (pHash) bật khi cần.
2. **Chia split theo group** (`split.group_aware = true`): toàn bộ frame của một
   clip/video luôn nằm trong cùng một split.
3. **Giữ ảnh không có biển** (`filter.keep_empty_images = true`) làm negative sample
   để giảm false positive.
4. **Tỉ lệ khung biển VN**: biển dài ~4.3:1, biển 2 hàng ~1.4:1, biển xe máy ~1.36:1.
   `filter.min_aspect_ratio / max_aspect_ratio` dùng để bắt lỗi nhãn.
5. **Ảnh tiếng Việt**: `cv2.imread` lỗi với path có dấu trên Windows — project dùng
   `imread_unicode` / `imwrite_unicode` (`np.fromfile` + `cv2.imdecode`).
6. **Toạ độ nhãn luôn ở hệ toạ độ ảnh cuối cùng.** Nếu bật resize, các nhãn import
   được map qua `GeoTransform` (scale + pad) nên không bị lệch.

## 6. Huấn luyện

```bash
yolo detect train model=yolo11n.pt data=dataset/data.yaml imgsz=640 epochs=100 batch=16
```

## 7. Kiểm tra nhanh

```bash
python tests/test_smoke.py     # sinh ảnh giả, chạy full pipeline, assert output
```

## 8. Gợi ý tinh chỉnh auto-label

| Hiện tượng | Tham số cần sửa |
|-----------|-----------------|
| Bỏ sót biển nhỏ / xa | giảm `autolabel.min_area_ratio` (vd 0.001) |
| Bắt nhầm cửa sổ, biển quảng cáo | tăng `autolabel.score_threshold` (0.55–0.65) và `min_fill_ratio` |
| Biển bị cắt viền | tăng `AUTO_LABEL_EXPAND` trong `pipeline.py` (0.02 → 0.04) |
| Nhiều box trùng trên 1 biển | giảm `autolabel.nms_iou` (0.4 → 0.3) |
| Ảnh quá tối / ngược sáng | bật `autolabel.use_adaptive` (mặc định đã bật) |
