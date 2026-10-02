"""
autolabel.py
============
Classical computer-vision pseudo-labeller for Vietnamese license plates.

Why a heuristic?
----------------
Vietnamese plates are extremely regular objects: they are bright, high-contrast
rectangular regions covered in characters, and - for *every* Vietnamese plate
type (long 1-row car plate 470x110 mm, 2-row car plate 280x200 mm, motorcycle
square plate 190x140 mm) - they are always **wider than they are tall**, with an
aspect ratio between roughly ``1.2`` and ``4.5``.  That prior is strong enough to
bootstrap a detector from unlabelled photos, which is exactly what YOLO
fine-tuning needs (a "warm start" label set + human review in LabelImg/CVAT).

Algorithm (per image)
---------------------
1.  bilateral filter to remove noise while keeping the plate border crisp;
2.  two independent binarisations produce candidate regions:
    *  horizontal Sobel gradient + Otsu  -> the plate *border/frame*;
    *  adaptive threshold (optional)     -> the *characters*;
3.  a wide, short morphological closing joins individual characters into one
    solid plate-shaped blob (this is the key step);
4.  every external contour is turned into a bounding box and filtered with a
    cascade of priors: size, aspect ratio, rectangularity (fill) and edge
    density (a plate is full of ink pixels, a wall is not);
5.  the survivors get a heuristic confidence score, are de-duplicated with
    class-aware NMS and the best ``max_boxes_per_image`` are returned.

The output is *pseudo-labels*: they should always be reviewed before training a
production model, but they remove most of the manual clicking work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from annotations import BBox, filter_boxes, merge_boxes
from dataset_config import AutoLabelConfig, FilterConfig
from utils import Reporter


@dataclass
class Candidate:
    """Internal container coupling a box with the features used to score it."""

    box: BBox
    aspect: float
    fill: float
    edge_density: float
    area_ratio: float
    strategy: str


class PlateAutoLabeler:
    """Detect Vietnamese-style license plates with hand-crafted image features."""

    def __init__(self, config: AutoLabelConfig, reporter: Optional[Reporter] = None) -> None:
        self.config = config
        self.reporter = reporter or Reporter(quiet=True)
        self.detections = 0
        self.processed = 0
        self.images_with_plate = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, image: Optional[np.ndarray]) -> List[BBox]:
        """Return scored pseudo-label boxes for one BGR image."""
        if image is None or image.size == 0:
            return []
        self.processed += 1
        height, width = image.shape[:2]
        if width < 16 or height < 16:
            return []

        gray = self._prepare_gray(image)
        candidates: List[Candidate] = []

        gradient_binary = self._gradient_binary(gray)
        candidates.extend(self._region_candidates(gradient_binary, width, height, "gradient"))

        if self.config.use_adaptive:
            adaptive_binary = self._adaptive_binary(gray)
            candidates.extend(self._region_candidates(adaptive_binary, width, height, "adaptive"))

        boxes = self._score_and_select(candidates, width, height)
        if boxes:
            self.images_with_plate += 1
            self.detections += len(boxes)
        return boxes

    def detect_batch(self, images: Sequence[Optional[np.ndarray]]) -> List[List[BBox]]:
        """Convenience wrapper over :meth:`detect` for a list of images."""
        return [self.detect(image) for image in images]

    def stats(self) -> dict:
        """Small summary used by the pipeline log."""
        return {
            "processed": int(self.processed),
            "images_with_plate": int(self.images_with_plate),
            "detections": int(self.detections),
            "hit_rate": round(self.images_with_plate / self.processed, 4) if self.processed else 0.0,
            "average_boxes": round(self.detections / self.processed, 4) if self.processed else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Pre-processing helpers
    # ------------------------------------------------------------------ #
    def _prepare_gray(self, image: np.ndarray) -> np.ndarray:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        diameter = int(self.config.bilateral_diameter)
        if diameter % 2 == 0:
            diameter += 1
        diameter = max(3, diameter)
        try:
            return cv2.bilateralFilter(gray, diameter, 75, 75)
        except cv2.error:
            return gray

    def _gradient_binary(self, gray: np.ndarray) -> np.ndarray:
        """Horizontal Sobel + Otsu: highlights the plate frame and characters."""
        gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient = cv2.convertScaleAbs(gradient)
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    def _adaptive_binary(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive threshold: catches characters when the contrast is low."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        block = max(15, (min(gray.shape[:2]) // 8) | 1)
        return cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            9,
        )

    # ------------------------------------------------------------------ #
    # Region proposal
    # ------------------------------------------------------------------ #
    def _region_candidates(
        self,
        binary: np.ndarray,
        width: int,
        height: int,
        strategy: str,
    ) -> List[Candidate]:
        """Close the binary mask into plate blobs and filter the contours."""
        kernel_w = max(9, int(round(width / 40.0))) | 1
        kernel_h = max(3, int(round(height / 100.0))) | 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        closed = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, close_kernel, iterations=int(self.config.close_iterations)
        )
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(closed, cv2.MORPH_DILATE, dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(width * height)
        results: List[Candidate] = []

        for contour in contours:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if box_w < 8 or box_h < 6:
                continue
            if box_w <= box_h:
                # every Vietnamese plate is wider than it is tall
                continue

            area = float(box_w * box_h)
            area_ratio = area / image_area
            if area_ratio < self.config.min_area_ratio or area_ratio > self.config.max_area_ratio:
                continue

            aspect = float(box_w) / float(box_h)
            if aspect < self.config.min_aspect_ratio or aspect > self.config.max_aspect_ratio:
                continue

            contour_area = float(cv2.contourArea(contour))
            fill = contour_area / area if area > 0 else 0.0
            if fill < self.config.min_fill_ratio:
                continue

            edge_density = self._edge_density(binary, x, y, box_w, box_h)
            if edge_density < self.config.min_edge_density or edge_density > 0.85:
                continue

            box = BBox(
                x1=float(x),
                y1=float(y),
                x2=float(x + box_w),
                y2=float(y + box_h),
                class_id=int(self.config.class_id),
                score=0.0,
                source="auto:{0}".format(strategy),
            )
            results.append(
                Candidate(
                    box=box,
                    aspect=aspect,
                    fill=fill,
                    edge_density=edge_density,
                    area_ratio=area_ratio,
                    strategy=strategy,
                )
            )
        return results

    @staticmethod
    def _edge_density(binary: np.ndarray, x: int, y: int, width: int, height: int) -> float:
        region = binary[y : y + height, x : x + width]
        if region.size == 0:
            return 0.0
        return float(np.count_nonzero(region)) / float(region.size)

    # ------------------------------------------------------------------ #
    # Scoring / selection
    # ------------------------------------------------------------------ #
    def _score(self, candidate: Candidate) -> float:
        aspect_score = self._aspect_score(candidate.aspect)
        fill_score = min(1.0, candidate.fill / 0.85)
        density_score = min(1.0, candidate.edge_density / 0.25)
        area_score = min(1.0, candidate.area_ratio / 0.03)
        strategy_bonus = 0.03 if candidate.strategy == "gradient" else 0.0
        raw = (
            0.35 * aspect_score
            + 0.25 * fill_score
            + 0.25 * density_score
            + 0.15 * area_score
            + strategy_bonus
        )
        return float(min(1.0, max(0.0, raw)))

    def _aspect_score(self, aspect: float) -> float:
        """1.0 for the typical 1.6-4.5 band, decaying towards the filter bounds."""
        ideal_low = max(self.config.min_aspect_ratio, 1.6)
        ideal_high = min(self.config.max_aspect_ratio, 4.5)
        if ideal_low <= aspect <= ideal_high:
            return 1.0
        if aspect < ideal_low:
            span = max(1e-6, ideal_low - self.config.min_aspect_ratio)
            return float(max(0.0, 1.0 - (ideal_low - aspect) / span))
        span = max(1e-6, self.config.max_aspect_ratio - ideal_high)
        return float(max(0.0, 1.0 - (aspect - ideal_high) / span))

    def _score_and_select(self, candidates: List[Candidate], width: int, height: int) -> List[BBox]:
        if not candidates:
            return []
        for candidate in candidates:
            candidate.box.score = self._score(candidate)
        scored = [candidate for candidate in candidates if candidate.box.score >= self.config.score_threshold]
        if not scored:
            return []
        scored.sort(key=lambda item: item.box.score, reverse=True)
        merged = merge_boxes([candidate.box for candidate in scored], float(self.config.nms_iou))
        merged.sort(key=lambda box: float(box.score), reverse=True)
        limit = int(self.config.max_boxes_per_image)
        selected = merged[:limit] if limit > 0 else merged
        return [box.clip(width, height) for box in selected]


# --------------------------------------------------------------------------- #
# Post-processing helpers
# --------------------------------------------------------------------------- #
def refine_auto_labels(
    boxes: Sequence[BBox],
    width: int,
    height: int,
    filter_config: FilterConfig,
    expand: float = 0.0,
) -> List[BBox]:
    """Lightly pad, clip and re-validate pseudo-labels before they are stored.

    ``expand`` grows each box by that fraction of its size on all four sides,
    which compensates for the morphological closing eating into the plate edge.
    """
    prepared: List[BBox] = []
    for box in boxes:
        if expand > 0.0:
            delta_x = box.width * expand
            delta_y = box.height * expand
            box = BBox(
                box.x1 - delta_x,
                box.y1 - delta_y,
                box.x2 + delta_x,
                box.y2 + delta_y,
                box.class_id,
                box.score,
                box.source,
            )
        prepared.append(box)
    return filter_boxes(prepared, filter_config, width, height)


def auto_label_image(
    image: np.ndarray,
    auto_config: AutoLabelConfig,
    filter_config: FilterConfig,
    expand: float = 0.0,
) -> List[BBox]:
    """One-image convenience function used by scripts and tests."""
    labeler = PlateAutoLabeler(auto_config)
    height, width = image.shape[:2]
    return refine_auto_labels(labeler.detect(image), width, height, filter_config, expand)
