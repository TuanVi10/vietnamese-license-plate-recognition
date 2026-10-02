# Nhận diện biển số xe Việt Nam (ALPR) từ video

Pipeline **đọc biển số xe Việt Nam** từ video giao thông thật, đầu ra là danh sách
biển số theo từng xe (track) kèm độ tin cậy. Dự án trải đủ vòng đời một hệ thống
thị giác máy tính: thu thập dữ liệu → gán nhãn → huấn luyện → đánh giá → tích hợp → e2e.

## Kiến trúc (8 module)

```
video.mp4
  ├─ M0  FrameExtractor             trích/lọc frame
  ├─ M1  VehicleDetector (YOLO11n)  dò phương tiện
  ├─ M3  SortTracker → track_id     theo dõi từng xe
  ├─ M2  PlateDetector (YOLO11n)    định vị biển số
  ├─ ★ CornerRegressor (YOLO11n-pose, 4 keypoint) → warpPerspective khi nghiêng >15°
  ├─ ★ PaddleOCR (PP-OCRv6)         đọc text (thay EasyOCR)
  ├─ M6  PostProcessor / fusion     voting theo track + fusion theo vị trí ký tự
  └─ M7  Output                     JSON + CSV + txt + video overlay
```

## Kết quả nổi bật (đo trên 213 ảnh thật)

| Hạng mục | Trước | Sau |
|---|---|---|
| OCR engine | EasyOCR **6.6%** exact | PaddleOCR **74.6%** exact |
| Căn chỉnh biển nghiêng >15° | bbox+affine **46.3%** | corner-warp **83.3%** |
| End-to-end | **70.4%** exact | **79.8%** exact / 0.923 sim |
| Detection biển số | — | **mAP50-95 = 0.73** |

## Cấu trúc thư mục

```
main.py                      # CLI chạy pipeline
train_detector.py            # train model detect biển số
dataset_prep_src/
  ├── src/                   # code core (18 module)
  └── tools/                 # script train/eval/chuẩn hoá dataset
docs/                        # tài liệu giải thích từng module (đọc từ docs/README.md)
labeling/                    # ground-truth (CSV nhãn) + hướng dẫn gán nhãn
REPORT.md                    # báo cáo tổng hành trình + khó khăn + cải tiến
```

## Cài đặt & chạy

Pipeline dùng **2 môi trường ảo** (PyTorch cho YOLO, PaddlePaddle cho OCR — tránh xung đột CUDA):

```powershell
# venv 1: PyTorch / ultralytics
python -m venv .venv-datasets
.venv-datasets\Scripts\pip install -r requirements.txt

# venv 2: PaddleOCR
python -m venv .venv-paddle
.venv-paddle\Scripts\pip install paddlepaddle-gpu paddleocr
```

Chạy pipeline video:

```powershell
.venv-datasets\Scripts\python main.py --video videodemo2.mp4 --device 0 --save-overlays
```

## Tài liệu chi tiết

Xem **[docs/README.md](docs/README.md)** — giải thích **từng module**, cách xử lý và các
**điểm đặc biệt trong logic tính toán** (SORT, fusion theo vị trí ký tự, gate góc, projection-profile deskew…).
