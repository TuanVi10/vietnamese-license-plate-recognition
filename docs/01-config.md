# 01 — Config (`src/config.py`)

## Vai trò

Gom **toàn bộ ngưỡng + trọng số** về một chỗ, thay vì hard-code rải rác trong code.
Tất cả là các `dataclass`, dễ sửa, dễ serialize ra JSON (cho báo cáo/đối chiếu).

## Cấu trúc

```
PipelineConfig (gom tất cả)
  ├── PreprocessConfig      # M4: deskew, 2 dòng, CLAHE, resize
  ├── OCRConfig             # M5: engine, charset, gpu
  ├── PostprocessConfig     # M6: min_confidence, min_readings, sửa ký tự
  ├── FrameExtractionConfig # M0: frame_interval, brightness/blur threshold
  ├── VehicleDetectionConfig# M1: model, conf, classes (2,3,5,7)
  ├── TrackingConfig        # M3: max_age, min_hits, iou_threshold
  ├── VideoPipelineConfig   # glue: read_plates, plate_fallback, plate_region_ratio
  ├── BestFrameConfig       # top-K: k, weights (w_sharp, w_height...)
  └── FusionConfig          # fusion: exponents, max_cluster_distance
```

## Điểm đặc biệt trong logic

### 1. `filter_charset` — lọc ký tự về đúng charset biển VN
Chỉ giữ `0-9 A-Z` (bỏ dấu, ký tự đặc biệt). `keep_separator=True` giữ thêm `-`, `.`,
khoảng trắng — **cần thiết để Module 6 suy ra series 1 hay 2 chữ cái** từ vị trí dấu phân cách.

### 2. `plate_region_ratio = 0.5` (fallback khi không có detector biển)
Khi Module 2 không tìm thấy biển, pipeline lấy **chỉ nửa DƯỚI bbox xe** làm vùng biển
(chứ không lấy cả xe). Lý do: biển số thường nằm nửa dưới; lấy cả bbox xe sẽ khiến OCR
chạy trên toàn ô tô → sinh hàng loạt "reading rác" lọt vào voting.

### 3. Các trọng số best-frame phải chuẩn hoá về `[0,1]`
`w_sharp=0.4, w_height=0.25, w_aspect=0.10, w_conf=0.25, w_exposure=0.10` — tổng = 1.1
(phần exposure bị **TRỪ**, không cộng). Mỗi thành phần phải cùng thang đo trước khi cộng
trọng số, nếu không một thành phần sẽ "lấn át" các thành phần khác.

### 4. Mặc định an toàn cho môi trường tối thiểu
- `VehicleDetectionConfig.model_path = None` → chạy chế độ "toàn khung" (chưa cần model).
- `OCRConfig.enable_mkldnn = False` trên CPU để tránh crash PaddleOCR.
- Mọi ngưỡng đều có comment "điểm khởi điểm cần tinh chỉnh" — không phải số cố định.
