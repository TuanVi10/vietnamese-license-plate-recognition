"""
config.py
=========
Cấu hình dùng chung cho pipeline đọc biển số xe Việt Nam từ ảnh tĩnh.

Các ngưỡng ở đây tương ứng với các tham số "khởi điểm cần tinh chỉnh" trong
spec v2 (mục 3.4, 3.5, 3.6). Chúng được gom về một chỗ để dễ hiệu chỉnh sau khi
thử trên ảnh thật, thay vì hard-code rải rác trong code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


#: Charset cho phép trên biển số VN (chỉ số và chữ cái Latin in hoa).
DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def filter_charset(text: str, charset: str = DEFAULT_CHARSET, keep_separator: bool = False) -> str:
    """Giữ lại các ký tự thuộc ``charset`` (mặc định 0-9, A-Z), bỏ phần còn lại.

    Args:
        text: Chuỗi đầu vào (có thể lẫn ký tự đặc biệt, dấu tiếng Việt...).
        charset: Tập ký tự được phép giữ lại.
        keep_separator: Nếu True, giữ thêm dấu ``-``, ``.`` và khoảng trắng.

    Returns:
        Chuỗi đã viết hoa và lọc ký tự.
    """
    if not text:
        return ""
    upper = text.upper()
    if keep_separator:
        allowed = set(charset) | {"-", ".", " "}
    else:
        allowed = set(charset)
    return "".join(ch for ch in upper if ch in allowed)


@dataclass
class PreprocessConfig:
    """Tham số cho Module 4 — Tiền xử lý ảnh biển số."""

    # Kích thước chuẩn hoá đầu vào cho OCR (giữ aspect ratio + padding).
    target_height: int = 48
    target_width: int = 320

    # Ngưỡng kích thước tối thiểu của ảnh crop biển số (spec 3.4 — tự đo thử,
    # không phải số cố định).
    min_plate_width: int = 40
    min_plate_height: int = 15
    min_ocr_width: int = 80

    # Nếu ảnh crop nhỏ hơn ``min_ocr_width`` thì phóng to (vẫn giữ aspect ratio).
    upscale_small_plates: bool = True

    # Tham số deskew (chỉnh nghiêng).
    max_skew_angle: float = 20.0
    skew_step: float = 1.0
    skew_min_angle: float = 0.5

    # Tham số CLAHE (tăng tương phản cục bộ).
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8

    # Nhận biết biển 2 dòng: nếu aspect (w/h) < ngưỡng này thì mới xét tách dòng.
    two_line_min_aspect: float = 1.8
    # Tỷ lệ bề dày vùng trống khi cắt 2 dòng (theo chiều cao ảnh).
    line_gap_ratio: float = 0.04
    # Khoảng trắng chèn giữa 2 dòng khi ghép ngang.
    merge_gap: int = 12

    # Màu nền khi padding / xoay ảnh (biển trắng -> 255).
    pad_color: int = 255

    # Nới rộng bbox biển số trước khi crop (tỷ lệ theo w/h).
    bbox_margin: float = 0.0

    # Ngưỡng sai khác để cắt viền đồng màu sau khi xoay ảnh.
    border_tolerance: int = 12


@dataclass
class OCRConfig:
    """Tham số cho Module 5 — OCR."""

    #: "auto" (thử PaddleOCR rồi fallback EasyOCR) | "paddle" | "easyocr".
    engine: str = "auto"
    lang: str = "en"
    use_gpu: bool = False
    #: BẮT BUỘC False trên CPU để tránh crash (spec 3.5).
    enable_mkldnn: bool = False
    use_angle_cls: bool = True
    charset: str = DEFAULT_CHARSET
    drop_empty: bool = True


@dataclass
class PostprocessConfig:
    """Tham số cho Module 6 — Hậu xử lý + voting."""

    charset: str = DEFAULT_CHARSET
    #: Bỏ qua các lần đọc có confidence thấp hơn ngưỡng này trước khi voting.
    min_confidence: float = 0.35
    #: Số lần đọc hợp lệ tối thiểu; ít hơn -> đánh dấu "độ tin cậy thấp".
    min_readings: int = 2
    #: Chỉ nhận các chuỗi khớp định dạng biển trắng dân sự VN.
    require_valid_format: bool = True
    #: Bật sửa ký tự dễ nhầm theo vị trí kỳ vọng (O<->0, I<->1, B<->8, S<->5...).
    apply_position_fix: bool = True
    #: Giữ dấu phân cách trong chuỗi chuẩn hoá (mặc định False).
    keep_separator: bool = False


@dataclass
class FrameExtractionConfig:
    """Tham số cho Module 0 — Trích & lọc frame từ video (spec v2, Module 0).

    Đây là các "điểm khởi điểm cần tinh chỉnh", không phải số cố định.
    """

    #: Bước nhảy frame. 1 = lấy mọi frame; 5 = ~6 frame/giây với video 30fps.
    frame_interval: int = 5
    #: Ngưỡng độ sáng trung bình tối thiểu (thang 0-255).
    brightness_threshold: float = 50.0
    #: Ngưỡng độ mờ tối thiểu (variance of Laplacian).
    blur_threshold: float = 100.0
    #: Bật/tắt bộ lọc chất lượng. False = giữ mọi frame theo ``frame_interval``.
    enable_quality_filter: bool = True
    #: Giới hạn số frame giữ lại. 0 = không giới hạn.
    max_frames: int = 0
    #: Giới hạn theo thời lượng (giây) kể từ đầu video. 0 = không giới hạn.
    max_duration: float = 0.0
    #: Bỏ qua N frame đầu video (giai đoạn camera chưa ổn định).
    skip_start_frames: int = 0
    #: Nếu > 0, resize frame đã giữ về đúng chiều cao này. 0 = giữ nguyên.
    resize_height: int = 0


@dataclass
class VehicleDetectionConfig:
    """Tham số cho Module 1 — Phát hiện phương tiện (Vehicle Detection).

    Module 1 dùng YOLOv8n (``ultralytics``) để tìm bbox phương tiện trong frame.
    Nếu ``model_path`` là ``None``, pipeline chạy ở chế độ "toàn khung" (coi cả
    frame là 1 phương tiện) để Module 3 và phần còn lại vẫn chạy được mà chưa cần
    model — rất tiện cho việc thử nghiệm/kiểm thử logic tracking.
    """

    #: Đường dẫn model YOLO (.pt). ``None`` -> chế độ "toàn khung".
    model_path: Optional[str] = None
    #: Ngưỡng confidence tối thiểu khi detect.
    conf: float = 0.25
    #: Ngưỡng IoU cho NMS của YOLO.
    iou: float = 0.45
    #: Thiết bị suy luận (``"cpu"``, ``"cuda"``, ``"0"``...). ``None`` = mặc định.
    device: Optional[str] = None
    #: Kích thước ảnh đầu vào cho YOLO.
    imgsz: int = 640
    #: Số detection tối đa mỗi frame.
    max_det: int = 100
    #: Nhóm lớp COCO coi là phương tiện (2=car, 3=motorcycle, 5=bus, 7=truck).
    classes: Tuple[int, ...] = (2, 3, 5, 7)
    #: Khi không có model, trả về 1 bbox phủ toàn khung (coi cả frame là xe).
    full_frame_fallback: bool = True


@dataclass
class TrackingConfig:
    """Tham số cho Module 3 — Tracking (SORT: Kalman + Hungarian + IoU).

    Đây là các "điểm khởi điểm cần tinh chỉnh" trên video thật, không phải số cố
    định. ``max_age`` / ``min_hits`` tính theo SỐ FRAME đã xét (sau Module 0).
    """

    #: Bật/tắt tracking. ``False`` -> mỗi detection được xử lý độc lập (không gán track_id).
    enabled: bool = True
    #: Số frame tối đa giữ 1 track không được cập nhật trước khi xoá.
    max_age: int = 30
    #: Số lần khớp liên tiếp tối thiểu để coi track là "đã xác nhận".
    min_hits: int = 3
    #: Ngưỡng IoU tối thiểu để ghép detection với track (SORT dùng ~0.3).
    iou_threshold: float = 0.3


@dataclass
class VideoPipelineConfig:
    """Tham số cho file glue — pipeline video hoàn chỉnh (Module 0 -> 7)."""

    #: Có đọc biển số (Module 4/5/6) cho từng track hay không. False = chỉ 0/1/3.
    read_plates: bool = True
    #: Lưu frame đã vẽ overlay (bbox xe + track_id + biển số) vào ``<output_dir>/tracks_frames/``.
    save_frame_overlays: bool = False
    #: Khi KHÔNG có model dò biển số (Module 2), lấy bbox phương tiện của Module 1 làm
    #: vùng biển số (cách xấp xỉ để pipeline vẫn chạy được).
    plate_fallback_to_vehicle: bool = True
    #: Tỷ lệ chiều cao vùng biển số dự phòng tính từ ĐÁY bbox phương tiện.
    #: 0.5 = chỉ lấy nửa dưới (nơi thường đặt biển số) — mặc định an toàn;
    #: 1.0 = lấy cả bbox xe (nên tránh: OCR sẽ chạy trên toàn bộ ô tô, sinh
    #: nhiều reading rác vẫn lọt vào voting).
    plate_region_ratio: float = 0.5
    #: Ngưỡng confidence tối thiểu của detection biển số (Module 2) để nhận.
    plate_min_confidence: float = 0.25
    #: Giới hạn số frame có đọc biển số (0 = không giới hạn) — hữu ích khi test nhanh.
    max_reading_frames: int = 0


@dataclass
class BestFrameConfig:
    """Tham số cho cơ chế chọn top-K crop biển số tốt nhất cho mỗi track.

    Điểm mỗi crop là tổ hợp có trọng số (mọi thành phần chuẩn hoá về ``[0, 1]``):
    độ nét (variance of Laplacian, thang log), chiều cao biển (px), độ lệch tỉ lệ
    w/h so với loại biển kỳ vọng, confidence detection và phạt vùng cháy sáng/quá tối.
    """

    #: Bật/tắt cơ chế best-frame. ``False`` = quay về OCR mọi frame (hành vi cũ).
    enabled: bool = True
    #: Số crop giữ lại tối đa cho mỗi track.
    k: int = 5
    #: Khoảng cách tối thiểu (theo số frame đã lấy mẫu) giữa các crop giữ lại.
    #: ``None`` = tự chọn ``max(2, 2 * frame_interval)`` ở runtime.
    min_frame_gap: Optional[int] = None
    #: Chiều cao cố định khi đo độ nét (resize trước khi tính Laplacian).
    eval_height: int = 48
    #: Mốc chuẩn để chuẩn hoá độ nét (thang log1p).
    sharpness_ref: float = 1000.0
    #: Mốc chuẩn chiều cao (px) để chuẩn hoá.
    height_ref: float = 60.0
    #: Dung sai (log-ratio) khi tính độ lệch aspect.
    aspect_tol: float = 1.0
    #: Ngưỡng aspect trung vị để phân biệt biển 2 dòng (dưới ngưỡng = 2 dòng).
    two_line_aspect_threshold: float = 2.5
    #: Aspect kỳ vọng của biển 2 dòng.
    two_line_aspect_ratio: float = 1.35
    #: Aspect kỳ vọng của biển 1 dòng.
    one_line_aspect_ratio: float = 4.3
    #: Ngưỡng pixel coi là quá tối (thang 0-255).
    dark_threshold: float = 30.0
    #: Ngưỡng pixel coi là cháy sáng (thang 0-255).
    bright_threshold: float = 225.0
    #: Chiều cao biển tối thiểu (px); crop thấp hơn bị loại cứng ở pha thu thập.
    min_plate_height: float = 24.0
    #: Trọng số độ nét.
    w_sharp: float = 0.4
    #: Trọng số chiều cao biển.
    w_height: float = 0.25
    #: Trọng số độ lệch aspect.
    w_aspect: float = 0.10
    #: Trọng số confidence detection.
    w_conf: float = 0.25
    #: Trọng số phạt vùng cháy sáng/quá tối (trừ điểm).
    w_exposure: float = 0.10
    #: Bật log chẩn đoán best-frame theo từng track (so sánh với legacy). Chỉ dùng debug.
    debug: bool = False


@dataclass
class FusionConfig:
    """Tham số cho fusion — gộp nhiều lần đọc 1 track theo **vị trí ký tự**.

    Khác với ``VotingAggregator`` (vote theo CẢ CHUỖI), fusion bỏ phiếu cho từng
    VỊ TRÍ ký tự nên một ký tự sai lẻ tẻ (vd: 5 đọc thành 6) không làm chia nhỏ
    phiếu. Xem :mod:`src.fusion`.
    """

    #: Bật/tắt fusion. ``False`` = dùng vote cả chuỗi cũ (để so sánh A/B).
    enabled: bool = False
    #: Bỏ qua lần đọc có confidence thấp hơn ngưỡng này (lọc mảnh vụn).
    min_confidence: float = 0.35
    #: Số mũ của confidence trong trọng số ``w = conf^a * quality^b``.
    confidence_exponent: float = 1.0
    #: Số mũ của quality_score trong trọng số ``w = conf^a * quality^b``.
    quality_exponent: float = 1.0
    #: Khoảng cách Hamming tối đa giữa 2 chuỗi CÙNG độ dài để gom chung 1 cụm
    #: (chống trộn nhiều biển trong 1 track).
    max_cluster_distance: int = 2
    #: Ngưỡng chia sẻ của cụm lớn thứ 2 (so với tổng lần đọc) để nghi "trộn nhiều biển".
    cluster_min_share: float = 0.3
    #: Mã tỉnh hợp lệ (đã nạp). Chỉ dùng làm tiêu chí phá hoà, KHÔNG loại cứng.
    province_codes: Optional[Tuple[str, ...]] = None
    #: Đường dẫn file mã tỉnh (tuỳ chọn). Nếu đặt, nạp bằng
    #: :func:`src.fusion.load_province_codes`.
    province_codes_path: Optional[str] = None


@dataclass
class PipelineConfig:
    """Gom toàn bộ cấu hình của pipeline."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    frame_extraction: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)
    vehicle_detection: VehicleDetectionConfig = field(default_factory=VehicleDetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    video: VideoPipelineConfig = field(default_factory=VideoPipelineConfig)
    best_frame: BestFrameConfig = field(default_factory=BestFrameConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    #: Đường dẫn model YOLO dò biển số (tùy chọn). None -> coi cả ảnh là biển số.
    detection_model: Optional[str] = None
    detection_conf: float = 0.25
    detection_device: Optional[str] = None
    detection_imgsz: int = 640

    output_dir: str = "results"

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển cấu hình (lồng nhau) thành dict thuần để serialize JSON."""
        return asdict(self)
