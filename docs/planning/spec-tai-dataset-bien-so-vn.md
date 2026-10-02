# SPEC: Tải & tổ chức dataset thô cho Module 2 — Plate Detection

**Mục tiêu:** Tải về 4 dataset biển số xe công khai (3 nguồn Roboflow + 1 nguồn
Kaggle), lưu đúng cấu trúc thư mục quy định, và xác nhận (verify) đã tải đủ.
**KHÔNG** thực hiện dedupe, convert format, chia train/val/test, hay train
model ở bước này. Đó là công việc của bước kế tiếp (sẽ giao riêng cho
ChatDev), nằm ngoài phạm vi spec này.

*Bản gốc của spec ghi 5 nguồn (4 Roboflow + 1 Kaggle); nguồn Roboflow
`hr-gamma` đã bị loại bỏ ngày 2026-09-26 vì dataset nguồn bị xoá — xem mục 8.1.
Trạng thái hiện tại: cả 4 nguồn còn lại đã tải xong, tổng **18,727 ảnh** —
xem bảng thực tế ở mục 6.*

**Bối cảnh dự án:** Đây là bước chuẩn bị dữ liệu cho Module 2 (Plate
Detection — fine-tune YOLO) của dự án đọc biển số xe Việt Nam từ video. Tham
khảo file `spec-ke-hoach-du-an-nhan-dien-bien-so-xe-v2_updated.md` (đính kèm
cùng thư mục dự án) nếu cần đối chiếu cấu trúc `datasets/` tổng thể — spec
này chỉ tập trung vào bước tải dữ liệu thô.

---

## 1. Yêu cầu môi trường trước khi bắt đầu

- Python đã cài sẵn (dùng đúng `.venv` của project ChatDev nếu có, hoặc venv
  riêng cho việc chuẩn bị dataset — không bắt buộc phải chung venv với
  ChatDev).
- Cài 2 thư viện cần thiết:
  ```powershell
  pip install roboflow kaggle
  ```
- **Ghi chú môi trường cho máy này (đã thực hiện ngày 2026-09-26):** KHÔNG cài
  `roboflow`/`kaggle` vào `S:\M_AGENT\CV\.venv` (đây là venv của Module 0, chỉ
  có `numpy` + `opencv-python`; cài thêm sẽ đổi version các gói đó), mà tạo
  venv riêng theo đúng cho phép ở trên:
  ```powershell
  & S:\M_AGENT\CV\.venv\Scripts\python.exe -m venv S:\M_AGENT\CV\.venv-datasets
  & S:\M_AGENT\CV\.venv-datasets\Scripts\python.exe -m pip install roboflow kaggle
  ```
  Kết quả: Python 3.12.14, `roboflow 1.5.1`, `kaggle 2.2.4`. Vì vậy các lệnh ở
  mục 4 và mục 5 chạy bằng python/kaggle **của `.venv-datasets`**:
  ```powershell
  cd S:\M_AGENT\CV
  & .\.venv-datasets\Scripts\python.exe download_datasets.py
  & .\.venv-datasets\Scripts\kaggle.exe datasets download -d bomaich/vnlicenseplate -p datasets/raw/raw_kaggle --unzip
  ```
- **Bổ sung cùng ngày 2026-09-26 (sau khi đã tải xong dataset):** venv
  `.venv-datasets` được cài thêm `ultralytics` (theo yêu cầu riêng của người
  dùng, để chạy thử inference — không thuộc phần tải dữ liệu của spec này):
  ```powershell
  # (1) cài ultralytics — lưu ý bước này lấy torch CPU-only từ PyPI
  & .\.venv-datasets\Scripts\python.exe -m pip install ultralytics
  # (2) thay torch/torchvision sang bản CUDA (~2.6 GB, tải từ index PyTorch)
  & .\.venv-datasets\Scripts\python.exe -m pip install --upgrade `
      --index-url https://download.pytorch.org/whl/cu126 torch torchvision
  ```
  - Kết quả: `ultralytics 8.4.163`, `torch 2.14.0+cu126`, `torchvision
    0.29.0+cu126`, `opencv-python 5.0.0.93`. `pip check` không có xung đột;
    `roboflow`/`kaggle` của bước tải dataset vẫn import bình thường.
  - **Bẫy đã gặp:** bản `torch` trên PyPI cho Windows là **CPU-only**, nên lệnh
    (1) một mình cho `torch 2.14.0+cpu` và `torch.cuda.is_available() == False`
    (YOLO sẽ chạy CPU dù máy có GPU). Phải chạy thêm lệnh (2) với
    `--index-url https://download.pytorch.org/whl/cu126` mới có bản CUDA; nếu
    thêm PyPI làm `--extra-index-url` thì pip sẽ chọn nhầm lại bản CPU (version
    PyPI cao hơn), nên chỉ dùng **một** index là index của PyTorch.
  - Máy này: **NVIDIA GeForce RTX 3050 6GB Laptop, driver 610.62** → đã xác
    nhận `torch.cuda.is_available() == True`, CUDA runtime 12.6, device
    capability `(8, 6)` (Ampere).
  - Smoke test thật (YOLO11n qua `ultralytics`, 20 ảnh
    `datasets/raw/raw_school/train/images`, `imgsz=640`): **GPU 60.8 img/s vs
    CPU 28.9 img/s**, cùng số detection (11 box) → GPU cho kết quả nhất quán
    với CPU, không sai lệch.
  - Lưu ý phạm vi: spec này **không** train model. Việc train vẫn theo kế hoạch
    ở `spec-ke-hoach-du-an-nhan-dien-bien-so-xe-v2_updated.md` (mục 6 — Google
    Colab T4); cài CUDA ở đây chỉ để chạy inference/demo nhanh trên máy local.
  - Phụ phẩm của smoke test: `yolo11n.pt` (5.35 MB, pretrained COCO) đã được
    tải về `S:\M_AGENT\CV\yolo11n.pt`. Thư mục `models/` trong cấu trúc đề xuất
    của spec v2 (mục 4) chưa tồn tại, nên file này tạm nằm ở gốc project — dời
    vào `models/` khi tạo cấu trúc đó cho Module 1.
- Có kết nối mạng ổn định (một số dataset ~1-2 GB).

## 2. Lấy credentials (làm 1 lần, trước khi chạy script)

### 2.1 Roboflow API Key
1. Vào https://app.roboflow.com/settings/api (đăng ký tài khoản free nếu
   chưa có).
2. Copy **Private API Key**.
3. **KHÔNG hardcode key vào script rồi commit lên git.** Đặt vào biến môi
   trường:
   ```powershell
   setx ROBOFLOW_API_KEY "dán_key_vào_đây"
   ```
   Sau khi `setx`, mở lại terminal mới để biến có hiệu lực.

### 2.2 Kaggle API Token
1. Vào https://www.kaggle.com/settings → mục "API" → bấm **"Create New
   Token"** → tải file `kaggle.json`.
2. Đặt file này vào:
   - Windows: `C:\Users\<tên_user>\.kaggle\kaggle.json`
   - (tạo thư mục `.kaggle` nếu chưa có)
3. **Không commit file `kaggle.json` lên git** — file này chứa credential cá
   nhân.
4. *Ghi chú cho máy này (2026-09-26):* `kaggle 2.2.4` (bản đã cài trong
   `.venv-datasets`) vẫn đọc được `kaggle.json` kiểu cũ (`AuthMethod.LEGACY_API_KEY`),
   nên cách ở trên vẫn dùng được. Ngoài ra bản này còn 2 cách mới cũng chấp
   nhận, dùng khi không muốn tạo file `kaggle.json`:
   - `& .\.venv-datasets\Scripts\kaggle.exe auth login` (đăng nhập OAuth qua
     trình duyệt, không cần copy token), hoặc
   - đặt token vào biến môi trường `KAGGLE_API_TOKEN`, hoặc lưu vào
     `C:\Users\<tên_user>\.kaggle\access_token`.

## 3. Cấu trúc thư mục đích (bắt buộc đúng như sau)

Chạy tất cả các bước tại thư mục gốc project (ví dụ `S:\M_AGENT\CV\`):

```
S:\M_AGENT\CV\
└── datasets\
    └── raw\
        ├── raw_school\        <- nguồn Roboflow "school-fuhih"
        ├── raw_cuongta\       <- nguồn Roboflow "cuong-ta-ulxex"
        ├── raw_annguyen\      <- nguồn Roboflow "annguyen" (version 8)
        └── raw_kaggle\        <- nguồn Kaggle "bomaich/vnlicenseplate"
```

Mỗi thư mục `raw_*` giữ NGUYÊN cấu trúc export gốc của nguồn đó (thường có
sẵn `train/`, `valid/`, `test/` bên trong, kèm ảnh + file nhãn) — **không**
gộp ảnh giữa các thư mục, không đổi tên file, không xoá file nào ở bước này.

> ⚠️ **Bẫy đã gặp thật (2026-09-26) — KHÔNG tạo sẵn thư mục `raw_*` rỗng trước
> khi tải.** Trong `roboflow` 1.5.1, `Version.download(..., location=dest)` mở
> đầu bằng:
> ```python
> if os.path.exists(location) and not overwrite:
>     return Dataset(...)   # coi như dataset đã tải xong -> BỎ QUA download
> ```
> nên nếu `datasets/raw/raw_school` đã tồn tại (dù rỗng), script ở mục 4 vẫn in
> `OK -> datasets/raw/raw_school` mà **không tải byte nào**, thư mục vẫn 0 ảnh
> (đúng như đã xảy ra ở lần chạy đầu). Cách chạy đúng: để chính lệnh tải tạo
> thư mục đích, **hoặc** gọi `download(..., overwrite=True)`. Riêng
> `raw_kaggle` giữ thư mục rỗng cũng không sao vì
> `kaggle datasets download -p <dir> --unzip` vẫn tải bình thường.

## 4. Tải 3 nguồn Roboflow — dùng script Python

Tạo file `download_datasets.py` tại `S:\M_AGENT\CV\` với nội dung:

```python
"""
download_datasets.py
=====================
Tải 3 dataset biển số VN từ Roboflow Universe về datasets/raw/<tên>/.
Yêu cầu: đã cài `pip install roboflow` và đã set biến môi trường
ROBOFLOW_API_KEY (xem mục 2.1 của spec).
"""

import os
from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError(
        "Chưa set biến môi trường ROBOFLOW_API_KEY. "
        "Xem mục 2.1 trong spec-tai-dataset-bien-so-vn.md"
    )

rf = Roboflow(api_key=api_key)

# (workspace_slug, project_slug, version, tên_thư_mục_đích)
SOURCES = [
    ("school-fuhih", "vietnamese-license-plate-tptd0", 1, "raw_school"),
    ("cuong-ta-ulxex", "vietnamese-car-license-plate", 1, "raw_cuongta"),
    ("annguyen", "vietnamese-license-plate-nugsi", 8, "raw_annguyen"),
]

for workspace, project_slug, version, folder_name in SOURCES:
    dest = f"datasets/raw/{folder_name}"
    print(f"--- Đang tải: {folder_name} (workspace={workspace}, "
          f"project={project_slug}, version={version}) ---")
    try:
        project = rf.workspace(workspace).project(project_slug)
        version_obj = project.version(version)
        version_obj.download("yolov8", location=dest)
        print(f"    OK -> {dest}")
    except Exception as exc:
        print(f"    LỖI khi tải {folder_name}: {exc}")
        print(
            "    -> Mở link dataset trên roboflow.com, kiểm tra đúng số "
            "version công khai (thường ghi trong URL /dataset/<số>), "
            "sửa lại số version trong SOURCES rồi chạy lại."
        )

print("\nHoàn tất. Kiểm tra thư mục datasets/raw/ để xác nhận.")
```

Chạy (dùng venv đã tạo ở mục 1):
```powershell
cd S:\M_AGENT\CV
& .\.venv-datasets\Scripts\python.exe download_datasets.py
```

> ⚠️ Nếu chạy nền / ghi log ra file (stdout bị redirect), phải set trước
> `$env:PYTHONIOENCODING='utf-8'`; nếu không Python sẽ **chết ngay ở dòng
> `print()` đầu tiên** với `UnicodeEncodeError: 'charmap' codec can't encode
> character '\u0110'` (vì stdout không phải console nên Python dùng cp1252).
> Chạy trực tiếp trong terminal thì không cần, console Windows đã là UTF-16.
> Bản đang chạy thực tế: `S:\M_AGENT\CV\download_datasets.py` (có thêm comment
> ghi rõ version đã kiểm tra và lý do bỏ nguồn `hr-gamma`).

**Xử lý lỗi thường gặp:** Nếu 1 nguồn báo lỗi "version not found", mở đúng
link dataset đó trên trình duyệt, xem số version mới nhất họ công khai
(hiện trong URL dạng `/dataset/<số>`), sửa lại số `version` tương ứng trong
`SOURCES`, chạy lại — không cần chạy lại các nguồn đã tải thành công (script
vẫn tải các nguồn còn lại nhờ khối `try/except`).

## 5. Tải nguồn Kaggle

```powershell
cd S:\M_AGENT\CV
& .\.venv-datasets\Scripts\kaggle.exe datasets download -d bomaich/vnlicenseplate -p datasets/raw/raw_kaggle --unzip
```

*Trạng thái máy này (2026-09-26): đã xác thực Kaggle thành công bằng OAuth, file
`C:\Users\MSI-VN\.kaggle\credentials.json` (không cần `kaggle.json`). Kiểm tra
nhanh bằng `& .\.venv-datasets\Scripts\kaggle.exe datasets list -m`.*

Nếu lệnh báo lỗi xác thực (`401 - Unauthorized`), kiểm tra lại vị trí file
`kaggle.json` đúng theo mục 2.2.

## 6. Xác nhận đã tải đủ (checklist — bắt buộc chạy sau khi tải xong)

Chạy lệnh sau để đếm nhanh số ảnh mỗi nguồn (PowerShell):

```powershell
Get-ChildItem -Path "S:\M_AGENT\CV\datasets\raw" -Directory | ForEach-Object {
    $count = (Get-ChildItem -Path $_.FullName -Recurse -Include *.jpg,*.jpeg,*.png).Count
    Write-Host "$($_.Name): $count ảnh"
}
```

**Kết quả kỳ vọng (số liệu tham khảo tại thời điểm viết spec — có thể lệch
nếu chủ dataset cập nhật thêm ảnh, không phải dấu hiệu lỗi):**

| Thư mục | Số ảnh kỳ vọng (khoảng) |
|---|---|
| `raw_school` | ~8,360 (v1 = 8,357) |
| `raw_cuongta` | ~8,260 (v1 = 8,255) |
| `raw_annguyen` | ~1,620 (v8 = 1,618) |
| `raw_kaggle` | ~500 (bản v3 thực tế có 498 ảnh — xem bảng dưới) |
| **Tổng** | **~18,700** (chưa lọc trùng) |

**Kết quả tải THỰC TẾ (đã chạy lệnh đếm ở trên, ngày 2026-09-26):**

| Thư mục | Ảnh thực tế | train / valid / test | File nhãn | Đối chiếu |
|---|---|---|---|---|
| `raw_school` | 8,357 | 5,845 / 1,680 / 832 | 8,357 `.txt` | ✅ khớp trang version v1 |
| `raw_cuongta` | 8,254 | 5,777 / 2,477 / — | 8,254 `.txt` | ✅ (trang ghi 8,255 → lệch đúng 1 ảnh) |
| `raw_annguyen` | 1,618 | 1,296 / 162 / 160 | 1,618 `.txt` | ✅ khớp v8 |
| `raw_kaggle` | 498 | 381 / 109 / 8 | 498 `.txt` | ⚠️ mô tả dataset ghi "1000 images" nhưng bản v3 chỉ có 498 ảnh + 498 nhãn = 996 file (đã đối chiếu bằng `kaggle datasets files` → đúng bản gốc, không phải giải nén lỗi) |
| **Tổng** | **18,727** | | 18,727 `.txt` | mỗi ảnh đều có đúng 1 nhãn đi kèm |

*(`raw_hrgamma` đã bị loại ngày 2026-09-26 — xem mục 8.1.)*

Nếu số ảnh 1 nguồn nào đó = 0 hoặc lệch quá xa (ví dụ chỉ vài chục ảnh), có
nghĩa bước tải nguồn đó thất bại âm thầm — quay lại mục 4 hoặc 5 kiểm tra
lại, đừng bỏ qua.

**Kiểm tra thêm:** mỗi thư mục `raw_*` phải có ít nhất 1 file nhãn tương ứng
với ảnh (đuôi `.txt` nếu là YOLO format, `.xml` nếu Pascal VOC, hoặc
`_annotations.coco.json` nếu COCO format) — không chỉ có ảnh trơn không
nhãn. Dataset Roboflow tải qua script ở mục 4 (định dạng `"yolov8"`) sẽ luôn
có nhãn `.txt` đi kèm; riêng `raw_kaggle` cần tự mở thư mục xem cấu trúc
thực tế vì mỗi tác giả có thể tổ chức khác nhau.

*Đã kiểm tra thực tế (2026-09-26): `raw_kaggle` cũng là YOLO format chuẩn —
`train|valid|test/{images,labels}`, 496 `.jpg` + 2 `.PNG` và 498 `.txt` nhãn
xywh; **không** có `data.yaml` (khác 3 thư mục Roboflow có `data.yaml`).*

## 7. Việc KHÔNG làm ở bước này (out of scope)

- Không xoá ảnh trùng lặp giữa các nguồn.
- Không convert định dạng nhãn (VOC/COCO → YOLO) cho đồng nhất.
- Không gộp các thư mục `raw_*` lại với nhau.
- Không chia train/val/test.
- Không chạy train YOLO.

Tất cả các việc trên sẽ được thực hiện ở bước kế tiếp (giao riêng qua
ChatDev, dùng chính các thư mục `datasets/raw/*` này làm đầu vào).

## 8. Bản ghi nguồn dữ liệu & giấy phép (điền vào cuối, dùng khi viết báo cáo)

| Thư mục | Nguồn gốc | License |
|---|---|---|
| `raw_school` | https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0 | CC BY 4.0 (đã mở trang xác nhận 2026-09-26) |
| `raw_cuongta` | https://universe.roboflow.com/cuong-ta-ulxex/vietnamese-car-license-plate | Public Domain (đã mở trang xác nhận 2026-09-26) |
| ~~`raw_hrgamma`~~ **(đã loại bỏ)** | https://universe.roboflow.com/hr-gamma/vietnamese-license-plate-2 | Không dùng — dataset nguồn đã bị xoá ("Project Not Found"), đã bỏ khỏi `SOURCES` và xoá thư mục ngày 2026-09-26 (xem mục 8.1) |
| `raw_annguyen` | https://universe.roboflow.com/annguyen/vietnamese-license-plate-nugsi | MIT (đã mở trang xác nhận 2026-09-26) |
| `raw_kaggle` | https://www.kaggle.com/datasets/bomaich/vnlicenseplate | Unknown — chủ dataset không khai báo license (kiểm tra qua Kaggle API `/api/v1/datasets/view/bomaich/vnlicenseplate` ngày 2026-09-26, trường `licenseName = "Unknown"`) |

*Ghi chú: cả 5 link đã được mở kiểm tra ngày 2026-09-26. 4/5 nguồn xác nhận
được license; riêng `raw_hrgamma` không còn tồn tại nên không có license để
ghi — nếu dùng trong báo cáo/demo công khai thì phải thay nguồn khác (xem
mục 8.1). `raw_kaggle` là "Unknown", tức là **không** có license rõ ràng: chỉ
nên dùng nội bộ/đồ án, không phát hành lại dataset.*

## 8.1 Ghi chú kiểm tra thực tế các nguồn (2026-09-26)

Số liệu dưới đây lấy trực tiếp từ trang dataset (mục Versions) và Kaggle API,
dùng để đối chiếu với bảng ở mục 6 và để sửa lại số `version` trong `SOURCES`:

| Nguồn | Version public | Ảnh trong version | Ảnh trong project | License |
|---|---|---|---|---|
| `school-fuhih/vietnamese-license-plate-tptd0` | v1 (2023-07-11) | 8,357 | 8,397 | CC BY 4.0 |
| `cuong-ta-ulxex/vietnamese-car-license-plate` | v1 (2021-08-27) | 8,255 | 8,255 | Public Domain |
| `hr-gamma/vietnamese-license-plate-2` | **không có** | — | — | dataset không còn tồn tại |
| `annguyen/vietnamese-license-plate-nugsi` | v2, v3, v5, **v8** (2025-11-28) | v8 = 1,618 | 1,815 | MIT |
| Kaggle `bomaich/vnlicenseplate` | 3 bản (bản mới nhất: v3, 2022-10-07) | 498 (đo thực tế sau khi tải) | 498 | Unknown |

Hệ quả đã xử lý (trước khi chạy mục 4–5):

1. **`raw_annguyen`: đã sửa `version` từ 1 -> 8** trong `download_datasets.py`
   (v1 không tồn tại; nếu để 1 thì script sẽ báo `version not found` và bỏ
   qua nguồn này).
2. **`raw_hrgamma`: nguồn trong spec đã chết — QUYẾT ĐỊNH ngày 2026-09-26:
   bỏ hẳn, không thay thế.** Dataset nguồn bị xoá nên không thể tải `~4,500`
   ảnh như bảng kỳ vọng cũ ở mục 6. Đã thực hiện: xoá entry khỏi `SOURCES`
   trong `download_datasets.py`, xoá thư mục rỗng `datasets/raw/raw_hrgamma`,
   sửa lại bảng kỳ vọng ở mục 6 (4 nguồn, ~18,700 ảnh thay vì ~24,000; thực tế
   tải được **18,727 ảnh**).
3. `raw_kaggle` không có version để chọn — lệnh `kaggle datasets download`
   ở mục 5 luôn lấy bản mới nhất (v3).


---

**Định nghĩa hoàn thành (Definition of Done):**
- [x] Cả 4 thư mục `datasets/raw/raw_*` tồn tại, mỗi thư mục có ảnh + nhãn.
      *(`raw_hrgamma` đã bị loại bỏ — xem mục 8.1. Tổng 18,727 ảnh, mỗi ảnh có
      đúng 1 file nhãn `.txt`.)*
- [x] Đã chạy lệnh đếm ảnh ở mục 6, số liệu không có thư mục nào = 0.
      *(raw_school 8,357 / raw_cuongta 8,254 / raw_annguyen 1,618 /
      raw_kaggle 498 — xem bảng thực tế ở mục 6.)*
- [x] Đã điền license thực tế vào bảng mục 8 (mở từng link kiểm tra).
      *(4/5 nguồn xác nhận trực tiếp; `raw_hrgamma` ghi nhận là dataset đã bị
      xoá nên không còn license — xem mục 8.1.)*
- [x] Không có thao tác nào ở mục 7 (out of scope) đã được thực hiện nhầm.
      *(chỉ tải + giải nén nguyên trạng; không dedupe/convert/merge/chia lại
      split/train.)*
