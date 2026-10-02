"""
Gói mã nguồn cho phần mềm "Đọc biển số xe Việt Nam" (ảnh tĩnh + video).

Triển khai theo SPEC KỸ THUẬT v2:

    * Module 0 — Trích & lọc frame từ video  (``frame_extractor``)
    * Module 1 — Phát hiện phương tiện       (``detect_vehicle``)
    * Module 2 — Phát hiện biển số (tùy chọn)(``detector``)
    * Module 3 — Tracking (SORT)             (``tracker``)
    * Module 4 — Tiền xử lý ảnh biển số      (``preprocess_plate``)
    * Module 5 — OCR (PaddleOCR / EasyOCR)   (``ocr``)
    * Module 6 — Hậu xử lý luật biển VN + voting (``postprocess``)
    * Module 7 — Output & visualize          (``visualize``, ``video_pipeline``)

``video_pipeline`` là **file glue** nối Module 0 -> 1 -> 2 -> 3 -> 4/5/6 -> 7 cho
video: trích frame, dò phương tiện, tracking, dò biển số, OCR + voting theo
``track_id``.

Với **1 ảnh tĩnh**, khái niệm "voting theo ``track_id``" được thay bằng voting
trên nhiều lần đọc của cùng một ảnh (nhiều biến thể tiền xử lý, nhiều item OCR).
Logic nhóm theo ``track_id`` vẫn được cung cấp trong
``postprocess.group_readings_by_track`` và ``postprocess.vote_by_track`` để tái
sử dụng khi mở rộng sang video.
"""

__version__ = "1.1.0"

__all__ = [
    "config",
    "image_utils",
    "frame_extractor",
    "detect_vehicle",
    "detector",
    "tracker",
    "preprocess_plate",
    "ocr",
    "postprocess",
    "plate_reader",
    "video_pipeline",
    "visualize",
    "sample_data",
]
