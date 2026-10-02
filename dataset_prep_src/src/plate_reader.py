"""
plate_reader.py
===============
Pipeline đọc biển số xe Việt Nam từ **1 ảnh tĩnh**.

Nối các module lại với nhau:

    Module 2 (tùy chọn)  -> dò bbox biển số bằng YOLO
    Module 4             -> tiền xử lý ảnh biển số (deskew, tách dòng, CLAHE...)
    Module 5             -> OCR (PaddleOCR / EasyOCR)
    Module 6             -> hậu xử lý luật biển VN + voting
    Module 7             -> đóng gói kết quả (JSON) + ảnh crop để đối chiếu

Vì đầu vào là ảnh tĩnh, "voting theo track_id" được thay bằng voting trên nhiều
lần đọc của cùng một ảnh (nhiều biến thể tiền xử lý + nhiều item OCR). Logic
nhóm theo ``track_id`` vẫn có sẵn trong :mod:`src.postprocess` cho video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import PipelineConfig
from .detector import PlateDetector
from .image_utils import imread_unicode
from .ocr import BaseOCREngine, create_ocr_engine
from .postprocess import PostProcessor
from .preprocess_plate import PlatePreprocessor, PreprocessResult


@dataclass
class PlateReading:
    """Kết quả đọc biển số cho 1 ảnh tĩnh."""

    #: Chuỗi biển số đã chuẩn hoá (chỉ gồm 0-9, A-Z).
    text: str
    #: Chuỗi hiển thị "best effort" (VD: ``51F-12345``).
    formatted: str
    #: Confidence trung bình của các lần đọc đã bỏ phiếu cho chuỗi thắng.
    confidence: float
    #: Chuỗi thắng khớp định dạng biển trắng dân sự VN hay không.
    valid: bool
    #: Đủ số lần đọc hợp lệ và ảnh không quá nhỏ hay không.
    reliable: bool
    #: Số phiếu của chuỗi thắng.
    vote_count: int
    #: Tổng số lần đọc được chấp nhận vào voting.
    accepted_readings: int
    #: Tổng số lần đọc (kể cả bị loại).
    total_readings: int
    #: Biển 2 dòng hay không.
    is_two_line: bool
    #: Góc nghiêng đã bù (độ).
    skew_angle: float
    #: Ảnh crop quá nhỏ -> không đủ tin cậy.
    too_small: bool
    #: Tên OCR engine đã dùng.
    engine: str
    #: Bbox biển số trong ảnh gốc (nếu có).
    bbox: Optional[List[int]] = None
    #: Cảnh báo trong quá trình xử lý.
    warnings: List[str] = field(default_factory=list)
    #: Chi tiết từng lần đọc OCR (phục vụ debug/báo cáo).
    readings: List[Dict[str, Any]] = field(default_factory=list)
    #: Kết quả voting đầy đủ.
    vote: Optional[Dict[str, Any]] = None
    #: Ảnh biển số đã tiền xử lý (không serialize khi ghi JSON).
    crop: Optional[np.ndarray] = None

    def to_dict(self, include_readings: bool = True) -> Dict[str, Any]:
        """Serialize thành dict (bỏ ảnh crop vì không JSON-serializable)."""
        data: Dict[str, Any] = {
            "plate_text": self.text,
            "plate_display": self.formatted,
            "plate_confidence": round(float(self.confidence), 4),
            "valid_format": bool(self.valid),
            "reliable": bool(self.reliable),
            "vote_count": int(self.vote_count),
            "accepted_readings": int(self.accepted_readings),
            "total_readings": int(self.total_readings),
            "is_two_line": bool(self.is_two_line),
            "skew_angle": round(float(self.skew_angle), 2),
            "too_small": bool(self.too_small),
            "engine": self.engine,
            "bbox": None if self.bbox is None else [int(v) for v in self.bbox],
            "warnings": list(self.warnings),
            "vote": self.vote,
        }
        if include_readings:
            data["readings"] = list(self.readings)
        return data


class PlateReader:
    """Ghép Module 2/4/5/6/7 thành pipeline hoàn chỉnh cho 1 ảnh tĩnh."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        ocr_engine: Optional[BaseOCREngine] = None,
        detector: Optional[PlateDetector] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.preprocessor = PlatePreprocessor(self.config.preprocess)
        self.postprocessor = PostProcessor(self.config.postprocess)
        self.ocr_engine = ocr_engine
        self.detector = detector

    # ------------------------------------------------------------------ #
    def ensure_engine(self) -> BaseOCREngine:
        """Khởi tạo OCR engine (lazy) theo cấu hình, có fallback tự động."""
        if self.ocr_engine is None:
            self.ocr_engine = create_ocr_engine(self.config.ocr)
        return self.ocr_engine

    # ------------------------------------------------------------------ #
    def read(
        self,
        image: np.ndarray,
        bbox: Optional[Sequence[float]] = None,
    ) -> PlateReading:
        """Đọc biển số từ 1 ảnh.

        Args:
            image: Ảnh BGR.
            bbox: Bbox biển số ``(x1, y1, x2, y2)``. Nếu ``None``: dùng detector
                (nếu có) để tìm biển số, ngược lại coi toàn bộ ảnh là biển số.

        Returns:
            :class:`PlateReading`.
        """
        engine = self.ensure_engine()

        chosen_bbox: Optional[List[int]] = [int(v) for v in bbox] if bbox is not None else None
        if chosen_bbox is None and self.detector is not None:
            detections = self.detector.detect(image)
            if not detections:
                return self._empty_reading(
                    engine_name=engine.name,
                    message="Không phát hiện biển số nào trong ảnh.",
                )
            chosen_bbox = [int(v) for v in detections[0].bbox]

        pre = self.preprocessor.preprocess(image, bbox=chosen_bbox)

        # --- Module 5: OCR trên từng biến thể ảnh -------------------------- #
        readings: List[Dict[str, Any]] = []
        if not pre.too_small:
            for variant_name, variant_image in self.preprocessor.iter_variants(pre):
                ocr_result = engine.recognize(variant_image)
                if ocr_result.items:
                    for item in ocr_result.items:
                        readings.append(
                            {
                                "source": variant_name,
                                "text": item.text,
                                # Giữ nguyên chuỗi thô (còn dấu phân cách) để
                                # Module 6 suy ra series 1 hay 2 chữ cái.
                                "raw_text": item.raw_text or item.text,
                                "confidence": float(item.confidence),
                                "engine": ocr_result.engine,
                            }
                        )

                    # Bổ sung chuỗi GHÉP ĐẦY ĐỦ khi OCR tách thành nhiều text-box.
                    # Đây là mấu chốt với biển 2 dòng: item trên "29B1" + item dưới
                    # "12345" -> chuỗi ghép "29B112345" mới đủ dài để qua
                    # ``validate_plate``; còn từng mảnh rời (độ dài < MIN_PLATE_LENGTH)
                    # sẽ bị Module 6 loại vì sai định dạng, khiến voting trả về RỖNG.
                    # CHỈ thêm khi có >1 item để tránh double-count trường hợp OCR
                    # trả về đúng 1 item (khi đó item.text chính là ocr_result.text).
                    if len(ocr_result.items) > 1 and ocr_result.text:
                        readings.append(
                            {
                                "source": variant_name,
                                "text": ocr_result.text,
                                # Dùng chuỗi thô ghép theo thứ tự đọc (không phải
                                # ``ocr_result.raw_text`` vốn theo thứ tự engine trả về).
                                "raw_text": self._joined_raw_text(ocr_result),
                                "confidence": float(ocr_result.confidence),
                                "engine": ocr_result.engine,
                            }
                        )
                elif ocr_result.text:
                    readings.append(
                        {
                            "source": variant_name,
                            "text": ocr_result.text,
                            "raw_text": ocr_result.raw_text or ocr_result.text,
                            "confidence": float(ocr_result.confidence),
                            "engine": ocr_result.engine,
                        }
                    )

        # --- Module 6: hậu xử lý + voting ---------------------------------- #
        vote = self.postprocessor.process(readings, group_id="image")
        formatted = (
            self.postprocessor.format(vote.text, is_two_line=pre.is_two_line) if vote.text else ""
        )

        warnings = list(pre.warnings)
        if pre.too_small:
            warnings.append("Ảnh biển số quá nhỏ — kết quả không đáng tin cậy.")
        if not vote.text:
            warnings.append("Không đọc được ký tự nào từ ảnh.")

        return PlateReading(
            text=vote.text,
            formatted=formatted,
            confidence=float(vote.mean_confidence),
            valid=bool(vote.valid),
            reliable=bool(vote.reliable and not pre.too_small),
            vote_count=int(vote.vote_count),
            accepted_readings=int(vote.accepted_readings),
            total_readings=int(vote.total_readings),
            is_two_line=bool(pre.is_two_line),
            skew_angle=float(pre.skew_angle),
            too_small=bool(pre.too_small),
            engine=engine.name,
            bbox=chosen_bbox,
            warnings=warnings,
            readings=readings,
            vote=vote.to_dict(),
            crop=pre.image,
        )

    # ------------------------------------------------------------------ #
    def read_file(
        self,
        image_path: str,
        bbox: Optional[Sequence[float]] = None,
    ) -> PlateReading:
        """Tiện dụng: đọc ảnh từ đường dẫn rồi gọi :meth:`read`."""
        image = imread_unicode(image_path)
        return self.read(image, bbox=bbox)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _joined_raw_text(ocr_result: Any) -> str:
        """Ghép ``raw_text`` của các item theo ĐÚNG thứ tự đọc của ``ocr_result.text``.

        ``OCRResult.raw_text`` được ghép theo thứ tự engine trả về, có thể khác
        thứ tự đọc đã được sắp xếp trong ``items``. Ghép lại từ ``items`` (đã
        sort theo thứ tự đọc) đảm bảo cấu trúc dấu phân cách khớp vị trí với
        ``ocr_result.text`` — cần thiết để Module 6 suy đúng series 1 hay 2 chữ.
        """
        items = getattr(ocr_result, "items", None) or []
        joined = "".join(
            (getattr(item, "raw_text", "") or getattr(item, "text", "")) for item in items
        )
        return (
            joined
            or getattr(ocr_result, "raw_text", "")
            or getattr(ocr_result, "text", "")
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _empty_reading(engine_name: str, message: str) -> PlateReading:
        """Tạo kết quả rỗng khi không có gì để đọc (VD: detector không tìm thấy)."""
        return PlateReading(
            text="",
            formatted="",
            confidence=0.0,
            valid=False,
            reliable=False,
            vote_count=0,
            accepted_readings=0,
            total_readings=0,
            is_two_line=False,
            skew_angle=0.0,
            too_small=False,
            engine=engine_name,
            bbox=None,
            warnings=[message],
            readings=[],
            vote=None,
            crop=None,
        )

    # ------------------------------------------------------------------ #
    def preprocess_only(
        self,
        image: np.ndarray,
        bbox: Optional[Sequence[float]] = None,
    ) -> PreprocessResult:
        """Chạy riêng Module 4 (hữu ích khi debug/soi ảnh tiền xử lý)."""
        return self.preprocessor.preprocess(image, bbox=bbox)
