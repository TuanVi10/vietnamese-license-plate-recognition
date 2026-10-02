"""
best_frame.py
=============
Cơ chế chọn **top-K crop biển số tốt nhất** cho mỗi ``track_id`` (bổ trợ cho video
pipeline), nhằm tránh OCR các frame mờ/nhỏ/nghiêng làm nhiễu kết quả.

Ý tưởng: trong pha thu thập, mỗi frame ta định vị bbox biển số, cắt crop, chấm điểm
chất lượng rồi đẩy vào :class:`TopKBuffer` (KHÔNG OCR). Khi track kết thúc (hoặc cuối
video), chỉ OCR ``top-K`` crop có điểm cao nhất.

Điểm của một crop là tổ hợp có trọng số của:
    * Độ nét (variance of Laplacian trên crop đã resize về chiều cao cố định, thang log).
    * Chiều cao biển (px).
    * Độ lệch tỉ lệ w/h so với loại biển kỳ vọng.
    * Confidence của detection biển số.
    * Phạt vùng cháy sáng / quá tối.
Mọi thành phần đều được chuẩn hoá về ``[0, 1]``. Trọng số nằm trong
:class:`~src.config.BestFrameConfig`.

Chỉ dùng ``cv2`` + ``numpy`` (không thêm thư viện).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .config import BestFrameConfig
from .image_utils import resize_by_height, to_gray, variance_of_laplacian


def _clip01(value: float) -> float:
    """Kẹp giá trị về ``[0, 1]``."""
    return float(max(0.0, min(1.0, value)))


@dataclass
class ScoreDetail:
    """Điểm chi tiết của 1 crop biển số (mọi thành phần đã chuẩn hoá về ``[0, 1]``)."""

    total: float
    sharpness: float
    height: float
    aspect: float
    confidence: float
    exposure: float

    def to_dict(self) -> Dict[str, float]:
        """Serialize thành dict (để ghi vào output, đối chiếu score <-> text)."""
        return {
            "score": round(float(self.total), 4),
            "score_sharpness": round(float(self.sharpness), 4),
            "score_height": round(float(self.height), 4),
            "score_aspect": round(float(self.aspect), 4),
            "score_conf": round(float(self.confidence), 4),
            "score_exposure": round(float(self.exposure), 4),
        }


@dataclass
class PlateCandidate:
    """Một crop biển số ứng viên cho 1 ``track_id`` tại 1 frame."""

    frame_index: int
    crop: np.ndarray
    plate_bbox: List[int]
    det_conf: float
    score: float
    #: Timestamp của frame (giây) — để ghi đầy đủ vào output.
    timestamp: float = 0.0
    #: Nguồn gốc bbox: ``"detection"`` (Module 2) hoặc ``"fallback"`` (xấp xỉ bbox xe).
    source: str = "detection"
    #: Từng thành phần điểm (chuẩn hoá), phục vụ phân tích tương quan.
    components: Dict[str, float] = field(default_factory=dict)


def _exposure_penalty(crop: np.ndarray, dark: float, bright: float) -> float:
    """Tỉ lệ pixel quá tối / cháy sáng (``[0, 1]``), dùng để phạt crop mất chi tiết."""
    gray = to_gray(crop)
    if gray.size == 0:
        return 0.0
    over = float(np.mean(gray >= bright))
    under = float(np.mean(gray <= dark))
    return _clip01(over + under)


def score_crop(
    crop: np.ndarray,
    plate_bbox: Sequence[float],
    det_conf: float,
    expected_aspect: float,
    config: BestFrameConfig,
) -> ScoreDetail:
    """Chấm điểm 1 crop biển số.

    Args:
        crop: Ảnh crop biển số (BGR hoặc grayscale).
        plate_bbox: Bbox ``(x1, y1, x2, y2)`` của biển trong ảnh gốc.
        det_conf: Confidence của detection biển số (``0.0`` nếu là fallback).
        expected_aspect: Tỉ lệ w/h kỳ vọng của loại biển (được ước lượng theo track).
        config: :class:`~src.config.BestFrameConfig`.

    Returns:
        :class:`ScoreDetail` gồm điểm tổng + từng thành phần (đều trong ``[0, 1]``).
    """
    if crop is None or crop.size == 0:
        return ScoreDetail(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # 1) Độ nét: variance of Laplacian trên crop đã resize về chiều cao cố định.
    #    Thang log để một vài cạnh rất mạnh không thổi phồng điểm bất thường.
    resized = resize_by_height(crop, int(config.eval_height))
    sharpness = _clip01(np.log1p(variance_of_laplacian(resized)) / np.log1p(config.sharpness_ref))

    # 2) Chiều cao biển (px) chuẩn hoá.
    x1, y1, x2, y2 = (float(v) for v in plate_bbox[:4])
    height_px = max(0.0, y2 - y1)
    height = _clip01(height_px / config.height_ref)

    # 3) Độ lệch tỉ lệ w/h so với loại biển kỳ vọng (dùng log-ratio).
    width_px = max(0.0, x2 - x1)
    if height_px <= 0 or width_px <= 0 or expected_aspect <= 0:
        aspect_score = 0.0
    else:
        deviation = abs(np.log(width_px / height_px) - np.log(expected_aspect))
        aspect_score = _clip01(1.0 - deviation / max(config.aspect_tol, 1e-6))

    # 4) Confidence detection (fallback -> 0.0).
    confidence = _clip01(det_conf)

    # 5) Phạt vùng cháy sáng / quá tối.
    exposure = _exposure_penalty(crop, config.dark_threshold, config.bright_threshold)

    total = (
        config.w_sharp * sharpness
        + config.w_height * height
        + config.w_aspect * aspect_score
        + config.w_conf * confidence
        - config.w_exposure * exposure
    )
    return ScoreDetail(
        total=_clip01(total),
        sharpness=sharpness,
        height=height,
        aspect=aspect_score,
        confidence=confidence,
        exposure=exposure,
    )


class TopKBuffer:
    """Giữ top-K :class:`PlateCandidate` cho 1 ``track_id``.

    Ràng buộc:
        * Giữ đúng tối đa ``k`` crop.
        * Khoảng cách frame giữa các crop giữ lại >= ``min_frame_gap`` (theo frame đã lấy mẫu).
        * Xếp hạng theo CẶP ``(is_detection, score)``: crop fallback KHÔNG bao giờ
          vượt crop detection thật dù điểm cao hơn.
    """

    def __init__(self, k: int = 5, min_frame_gap: int = 0) -> None:
        self.k = max(1, int(k))
        self.min_frame_gap = max(0, int(min_frame_gap))
        self._items: List[PlateCandidate] = []

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _is_detection(candidate: PlateCandidate) -> bool:
        return candidate.source != "fallback"

    @staticmethod
    def _rank_key(candidate: PlateCandidate):
        """Khoá xếp hạng: detection luôn trội fallback, cùng loại thì so điểm."""
        return (1 if candidate.source != "fallback" else 0, float(candidate.score))

    def add(self, candidate: PlateCandidate) -> bool:
        """Thêm 1 candidate; trả ``True`` nếu được giữ, ``False`` nếu bị loại."""
        is_detection = self._is_detection(candidate)

        # (a) Khi một detection thật xuất hiện, xoá hết crop fallback còn sót.
        if is_detection:
            self._items = [item for item in self._items if self._is_detection(item)]

        # (b) Fallback không bao giờ được thêm khi buffer đã có detection thật.
        if not is_detection and any(self._is_detection(item) for item in self._items):
            return False

        # (c) Xử lý xung đột ``min_frame_gap`` với nhiều crop cùng lúc.
        if self.min_frame_gap > 0:
            conflicting = [
                item
                for item in self._items
                if abs(item.frame_index - candidate.frame_index) < self.min_frame_gap
            ]
            if conflicting:
                # Chỉ thay khi trội hơn TẤT CẢ crop xung đột (theo cặp is_detection, score).
                if not all(self._rank_key(candidate) > self._rank_key(item) for item in conflicting):
                    return False
                for item in conflicting:
                    self._items.remove(item)

        # (d) Chèn theo dung lượng K.
        if len(self._items) < self.k:
            self._items.append(candidate)
            return True

        worst = min(self._items, key=self._rank_key)
        if self._rank_key(candidate) > self._rank_key(worst):
            self._items.remove(worst)
            self._items.append(candidate)
            return True
        return False

    def best(self) -> List[PlateCandidate]:
        """Trả các crop giữ lại, sắp ưu tiên giảm dần (detection trước, điểm cao trước)."""
        return sorted(self._items, key=self._rank_key, reverse=True)

    def clear(self) -> None:
        """Xoá sạch buffer (dùng khi reset pipeline)."""
        self._items = []
