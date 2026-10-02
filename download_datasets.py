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
#
# Số version đã được kiểm tra lại trực tiếp trên trang Roboflow Universe
# (ngày 2026-09-26) theo đúng hướng dẫn "Xử lý lỗi thường gặp" ở mục 4:
#   school-fuhih/vietnamese-license-plate-tptd0  -> chỉ có v1  (8,357 ảnh, CC BY 4.0)
#   cuong-ta-ulxex/vietnamese-car-license-plate  -> chỉ có v1  (8,255 ảnh, Public Domain)
#   annguyen/vietnamese-license-plate-nugsi      -> v1 KHÔNG tồn tại, các version
#        đang public là v2/v3/v5/v8; bản mới nhất là v8 "Train6" (1,618 ảnh, MIT)
#        -> đã sửa 1 -> 8 cho khớp thực tế.
#
# ĐÃ BỎ nguồn "hr-gamma/vietnamese-license-plate-2" (thư mục raw_hrgamma) theo
# quyết định ngày 2026-09-26: dataset đó đã bị xoá ("Project Not Found"),
# workspace hr-gamma chỉ còn 1 project "vietnamese-license-plate" 454 ảnh / 0
# version -> không thể tải. Xem mục 8.1 của spec.
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
