"""
preprocess_plate.py
===================
Module 4 — Tiền xử lý ảnh biển số.

Thứ tự thao tác **bắt buộc** theo spec 3.4:

    1. Crop ảnh biển số theo bbox (nếu có).
    2. Deskew (chỉnh nghiêng) — LUÔN chạy TRƯỚC khi tách dòng.
    3. Tách dòng (nếu là biển 2 dòng) rồi ghép ngang.
    4. Tăng tương phản CLAHE (ảnh biển bị chói/tối một phần).
    5. Resize chuẩn hoá **giữ nguyên aspect ratio** bằng padding, không kéo dãn.
    6. Kiểm tra ngưỡng kích thước tối thiểu -> đánh dấu "không đủ tin cậy"
       và KHÔNG đưa vào OCR.

Làm ngược thứ tự deskew <-> tách dòng sẽ tách sai dòng khi ảnh còn nghiêng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import PreprocessConfig
from .image_utils import (
    binarize_otsu,
    crop_bbox,
    ensure_bgr,
    expand_bbox,
    rotate_image,
    to_gray,
    trim_uniform_borders,
)


@dataclass
class PreprocessResult:
    """Kết quả tiền xử lý 1 ảnh biển số."""

    #: Ảnh biển số đã chuẩn hoá để đưa vào OCR (BGR, đã padding).
    image: np.ndarray
    #: Danh sách các dòng sau khi tách (trên -> dưới); 1 phần tử nếu biển 1 dòng.
    lines: List[np.ndarray] = field(default_factory=list)
    #: True nếu phát hiện biển 2 dòng.
    is_two_line: bool = False
    #: Góc nghiêng đã bù (độ).
    skew_angle: float = 0.0
    #: True nếu ảnh crop nhỏ hơn ngưỡng tối thiểu -> không đủ tin cậy.
    too_small: bool = False
    #: (width, height) của ảnh crop gốc trước khi tiền xử lý.
    original_size: Tuple[int, int] = (0, 0)
    #: Các cảnh báo trong quá trình tiền xử lý.
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Hàm tiền xử lý mức thấp (có thể gọi/test độc lập)
# --------------------------------------------------------------------------- #
def estimate_skew_angle(image: np.ndarray, max_angle: float = 20.0, step: float = 1.0) -> float:
    """Ước lượng góc nghiêng cần bù bằng phương pháp projection profile.

    Ý tưởng: khi ảnh được deskew đúng, histogram chiếu theo trục dọc (tổng pixel
    theo từng hàng) có phương sai cao nhất vì các đỉnh/đáy ký tự thẳng hàng. Ta
    quét các góc trong ``[-max_angle, max_angle]``, chọn góc cho phương sai lớn
    nhất, rồi tinh chỉnh ở bước nhỏ hơn (kèm nội suy parabol).

    Args:
        image: Ảnh biển số.
        max_angle: Biên độ góc tối đa (độ).
        step: Bước quét thô (độ).

    Returns:
        Góc (độ) cần xoay để ảnh thẳng lại. 0.0 nếu không ước lượng được.
    """
    if image is None or image.size == 0:
        return 0.0

    gray = to_gray(image)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = binary.shape[:2]
    if height < 4 or width < 4:
        return 0.0

    center = (width / 2.0, height / 2.0)

    def profile_score(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        projection = rotated.sum(axis=1).astype(np.float64)
        return float(np.var(projection))

    coarse_angles = np.arange(-max_angle, max_angle + 1e-9, step)
    if coarse_angles.size == 0:
        return 0.0
    coarse_scores = [profile_score(float(a)) for a in coarse_angles]
    best_angle = float(coarse_angles[int(np.argmax(coarse_scores))])

    fine_step = max(step / 5.0, 1e-3)
    fine_angles = np.arange(best_angle - step, best_angle + step + 1e-9, fine_step)
    fine_scores = [profile_score(float(a)) for a in fine_angles]
    best_index = int(np.argmax(fine_scores))

    # Nội suy parabol để đạt độ phân giải dưới 1 bước quét.
    if 0 < best_index < len(fine_angles) - 1:
        y0 = fine_scores[best_index - 1]
        y1 = fine_scores[best_index]
        y2 = fine_scores[best_index + 1]
        denominator = y0 - 2.0 * y1 + y2
        if abs(denominator) > 1e-12:
            delta = 0.5 * (y0 - y2) / denominator
            delta = float(np.clip(delta, -1.0, 1.0))
            return float(fine_angles[best_index] + delta * fine_step)

    return float(fine_angles[best_index])


def deskew(
    image: np.ndarray,
    max_angle: float = 20.0,
    step: float = 1.0,
    min_angle: float = 0.5,
    border_value: int = 255,
) -> Tuple[np.ndarray, float]:
    """Chỉnh nghiêng ảnh biển số.

    Returns:
        (ảnh đã chỉnh nghiêng, góc đã bù). Nếu góc nhỏ hơn ``min_angle`` thì trả
        về ảnh gốc (copy) và góc 0.0 để tránh nội suy không cần thiết.
    """
    if image is None or image.size == 0:
        raise ValueError("deskew nhận ảnh rỗng")
    angle = estimate_skew_angle(image, max_angle=max_angle, step=step)
    if abs(angle) < min_angle:
        return image.copy(), 0.0
    return rotate_image(image, angle, border_value=border_value), float(angle)


def _has_ink(image: np.ndarray, min_ratio: float = 0.01) -> bool:
    """Kiểm tra một dải ảnh có chứa nét mực (ký tự) hay không."""
    if image is None or image.size == 0:
        return False
    binary = binarize_otsu(image, invert=True)
    return float(binary.mean()) > float(min_ratio) * 255.0


def split_lines(image: np.ndarray, gap_ratio: float = 0.04) -> List[np.ndarray]:
    """Tách biển 2 dòng thành danh sách các dòng theo thứ tự trên -> dưới.

    Dùng projection profile theo hàng để tìm "thung lũng" (vùng gần như trống)
    quanh khoảng giữa ảnh. Nếu không có thung lũng rõ ràng, coi như biển 1 dòng
    và trả về danh sách 1 phần tử.

    Args:
        image: Ảnh biển số (đã deskew).
        gap_ratio: Bề dày vùng cắt thêm hai bên hàng phân cách, theo tỷ lệ chiều cao.

    Returns:
        Danh sách ảnh các dòng.
    """
    if image is None or image.size == 0:
        return []

    gray = to_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height = int(binary.shape[0])
    if height < 10:
        return [image]

    projection = binary.sum(axis=1).astype(np.float64)
    low = max(1, int(height * 0.25))
    high = min(height - 1, int(height * 0.75))
    if high <= low:
        return [image]

    band = projection[low:high]
    split_row = low + int(np.argmin(band))
    mean_value = float(projection.mean())
    if mean_value <= 0.0:
        return [image]
    # Không có "thung lũng" rõ ràng -> coi như 1 dòng.
    if projection[split_row] > 0.35 * mean_value:
        return [image]

    gap = max(2, int(round(height * gap_ratio)))
    top = image[: max(1, split_row - gap), :]
    bottom = image[min(height - 1, split_row + gap):, :]
    if top.shape[0] < 3 or bottom.shape[0] < 3:
        return [image]
    if not _has_ink(top) or not _has_ink(bottom):
        return [image]
    return [top, bottom]


def merge_lines_horizontally(
    lines: Sequence[np.ndarray],
    gap: int = 12,
    background: int = 255,
) -> np.ndarray:
    """Ghép các dòng đã tách thành 1 ảnh ngang duy nhất (giữ thứ tự trên -> dưới).

    Mỗi dòng được scale về cùng chiều cao (giữ aspect ratio) rồi đặt cạnh nhau,
    chèn ``gap`` pixel nền trắng giữa các dòng.
    """
    if not lines:
        raise ValueError("merge_lines_horizontally cần ít nhất 1 dòng")
    if len(lines) == 1:
        return lines[0].copy()

    target_height = max(1, int(max(int(line.shape[0]) for line in lines)))
    resized: List[np.ndarray] = []
    for line in lines:
        height, width = line.shape[:2]
        if height <= 0 or width <= 0:
            continue
        scale = target_height / float(height)
        new_w = max(1, int(round(width * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        resized.append(cv2.resize(line, (new_w, target_height), interpolation=interpolation))
    if not resized:
        return lines[0].copy()

    total_width = sum(int(r.shape[1]) for r in resized) + int(gap) * (len(resized) - 1)
    if resized[0].ndim == 3:
        canvas = np.full(
            (target_height, total_width, resized[0].shape[2]),
            background,
            dtype=resized[0].dtype,
        )
    else:
        canvas = np.full((target_height, total_width), background, dtype=resized[0].dtype)

    cursor = 0
    for part in resized:
        canvas[:, cursor:cursor + part.shape[1]] = part
        cursor += part.shape[1] + int(gap)
    return canvas


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int] = (320, 48),
    background: int = 255,
) -> np.ndarray:
    """Resize ảnh giữ nguyên aspect ratio và padding cho đủ kích thước mục tiêu.

    Quan trọng: KHÔNG kéo dãn ảnh (sẽ làm méo ký tự, hại OCR). Ảnh được scale
    vừa khít trong khung ``target_size`` rồi canh giữa trên nền ``background``.
    """
    target_w, target_h = int(target_size[0]), int(target_size[1])
    target_w = max(1, target_w)
    target_h = max(1, target_h)

    image = ensure_bgr(image)
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return np.full((target_h, target_w, 3), background, dtype=np.uint8)

    scale = min(target_w / float(width), target_h / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((target_h, target_w, 3), background, dtype=image.dtype)
    offset_y = (target_h - new_h) // 2
    offset_x = (target_w - new_w) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Tăng tương phản cục bộ bằng CLAHE (xử lý biển bị chói/tối một phần)."""
    if image is None or image.size == 0:
        return image
    tile = (max(1, int(tile_grid)), max(1, int(tile_grid)))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile)

    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def to_ocr_binary(image: np.ndarray) -> np.ndarray:
    """Nhị phân hoá Otsu (chữ đen trên nền trắng) — một biến thể ảnh cho OCR."""
    return cv2.cvtColor(binarize_otsu(image, invert=False), cv2.COLOR_GRAY2BGR)


def upscale_if_small(image: np.ndarray, min_width: int = 80, max_scale: float = 4.0) -> np.ndarray:
    """Phóng to ảnh nếu chiều rộng nhỏ hơn ``min_width`` (giữ nguyên aspect ratio)."""
    height, width = image.shape[:2]
    if width >= int(min_width) or width <= 0:
        return image
    scale = min(float(max_scale), float(min_width) / float(width))
    if scale <= 1.0:
        return image
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- #
# Lớp chính
# --------------------------------------------------------------------------- #
class PlatePreprocessor:
    """Triển khai Module 4 — tiền xử lý ảnh biển số."""

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()

    # ------------------------------------------------------------------ #
    def preprocess(
        self,
        image: np.ndarray,
        bbox: Optional[Sequence[float]] = None,
    ) -> PreprocessResult:
        """Chạy toàn bộ pipeline tiền xử lý cho 1 ảnh biển số.

        Args:
            image: Ảnh gốc (toàn ảnh hoặc ảnh đã là biển số).
            bbox: Bbox biển số ``(x1, y1, x2, y2)`` trong hệ toạ độ ``image``.
                Nếu ``None`` thì coi toàn bộ ảnh là vùng biển số.

        Returns:
            :class:`PreprocessResult`.
        """
        cfg = self.config
        if image is None or image.size == 0:
            raise ValueError("PlatePreprocessor.preprocess nhận ảnh rỗng")

        if bbox is not None:
            height, width = image.shape[:2]
            margin = float(cfg.bbox_margin)
            if margin > 0:
                bbox = expand_bbox(bbox, margin, width, height)
            crop = crop_bbox(image, bbox)
        else:
            crop = ensure_bgr(image).copy()

        crop = ensure_bgr(crop)
        original_height, original_width = crop.shape[:2]
        result = PreprocessResult(image=crop, original_size=(original_width, original_height))

        # --- Ngưỡng kích thước tối thiểu (spec 3.4) ---------------------- #
        if original_width < cfg.min_plate_width or original_height < cfg.min_plate_height:
            result.too_small = True
            result.lines = [crop]
            result.warnings.append(
                "Ảnh crop biển số quá nhỏ "
                f"({original_width}x{original_height}) < ngưỡng tối thiểu "
                f"({cfg.min_plate_width}x{cfg.min_plate_height}) — bỏ qua OCR."
            )
            return result

        # --- 1) Deskew TRƯỚC khi tách dòng ------------------------------- #
        deskewed, angle = deskew(
            crop,
            max_angle=cfg.max_skew_angle,
            step=cfg.skew_step,
            min_angle=cfg.skew_min_angle,
            border_value=cfg.pad_color,
        )
        result.skew_angle = float(angle)
        if abs(angle) >= cfg.max_skew_angle - 1e-6:
            result.warnings.append(
                f"Góc nghiêng bù {angle:.1f}° chạm ngưỡng tối đa — có thể chưa thẳng hoàn toàn."
            )

        deskewed = trim_uniform_borders(deskewed, tolerance=cfg.border_tolerance)
        if deskewed is None or deskewed.size == 0:
            deskewed = crop

        # --- 2) Tách dòng + ghép ngang ------------------------------------ #
        if self._looks_two_line(deskewed):
            lines = split_lines(deskewed, gap_ratio=cfg.line_gap_ratio)
        else:
            lines = [deskewed]
        result.lines = lines
        result.is_two_line = len(lines) > 1

        if result.is_two_line:
            merged = merge_lines_horizontally(lines, gap=cfg.merge_gap, background=cfg.pad_color)
        else:
            merged = lines[0]

        # --- 3) CLAHE tăng tương phản ------------------------------------- #
        merged = apply_clahe(merged, clip_limit=cfg.clahe_clip_limit, tile_grid=cfg.clahe_tile_grid)

        # --- 4) Phóng to nếu ảnh quá nhỏ cho OCR -------------------------- #
        if cfg.upscale_small_plates:
            merged = upscale_if_small(merged, min_width=cfg.min_ocr_width)

        # --- 5) Resize giữ aspect ratio + padding ------------------------- #
        merged = resize_with_padding(
            merged,
            target_size=(cfg.target_width, cfg.target_height),
            background=cfg.pad_color,
        )

        result.image = merged
        return result

    # ------------------------------------------------------------------ #
    def _looks_two_line(self, image: np.ndarray) -> bool:
        """Đoán biển 2 dòng: aspect đủ nhỏ VÀ tồn tại thung lũng giữa 2 dòng."""
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return False
        aspect = width / float(height)
        if aspect >= self.config.two_line_min_aspect:
            return False
        return len(split_lines(image, gap_ratio=self.config.line_gap_ratio)) > 1

    # ------------------------------------------------------------------ #
    def iter_variants(self, result: PreprocessResult) -> Iterator[Tuple[str, np.ndarray]]:
        """Sinh nhiều biến thể ảnh từ 1 lần tiền xử lý.

        Vì chỉ có 1 ảnh tĩnh, "voting" ở Module 6 dựa trên nhiều lần đọc khác
        nhau của cùng ảnh: ảnh ghép đã chuẩn hoá, biến thể nhị phân, và từng
        dòng riêng (nếu là biển 2 dòng).

        Nếu ảnh quá nhỏ (``too_small``) thì KHÔNG sinh biến thể nào — đúng theo
        yêu cầu "không đưa vào OCR để tránh tốn compute vô ích".
        """
        if result.too_small:
            return
        cfg = self.config

        if result.image is not None and result.image.size > 0:
            yield "merged", result.image
            yield "merged_binary", to_ocr_binary(result.image)

        for index, line in enumerate(result.lines):
            if line is None or line.size == 0:
                continue
            processed = apply_clahe(line, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
            if cfg.upscale_small_plates:
                processed = upscale_if_small(processed, min_width=cfg.min_ocr_width)
            processed = resize_with_padding(
                processed,
                target_size=(cfg.target_width, cfg.target_height),
                background=cfg.pad_color,
            )
            yield f"line_{index}", processed
