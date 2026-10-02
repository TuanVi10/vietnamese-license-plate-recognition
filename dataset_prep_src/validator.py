"""
validator.py
============
Stage 3 of the pipeline: make sure the labels that are about to be written are
actually usable by YOLO, and produce the statistics a dataset card needs.

What is checked
---------------
* the image still exists on disk in ``work/images``;
* boxes are inside the image (they are clipped, and out-of-bounds boxes counted);
* boxes reference a class declared in ``config.classes`` - boxes with an
  out-of-range ``class_id`` are dropped so ``data.yaml`` and the label files
  always agree;
* boxes are not degenerate / too small / too large / absurdly shaped;
* images that end up with zero boxes are either kept as *negative samples*
  (``filter.keep_empty_images = True``) or dropped from the dataset.

The stage returns both the cleaned annotation map and a JSON-friendly report.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from annotations import (
    BBox,
    ImageAnnotation,
    box_issues,
    count_classes,
    filter_boxes,
)
from dataset_config import DatasetConfig
from preprocess import ImageRecord
from utils import Reporter, percentile


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def validate_annotations(
    records: List[ImageRecord],
    annotations: Dict[str, ImageAnnotation],
    config: DatasetConfig,
    reporter: Optional[Reporter] = None,
) -> Tuple[Dict[str, ImageAnnotation], Dict[str, Any]]:
    """Validate, clean and summarise ``annotations``. Mutates ``records`` drop flags."""
    reporter = reporter or Reporter(quiet=True)
    filter_config = config.filter

    issue_counter: Counter = Counter()
    cleaned: Dict[str, ImageAnnotation] = {}

    boxes_in = 0
    boxes_out = 0
    boxes_clipped = 0
    images_with_boxes = 0
    empty_images: List[str] = []
    dropped_empty_images: List[str] = []

    for record in records:
        if record.dropped:
            continue
        if not record.name:
            record.issues.append("no_output_name")
            continue

        annotation = annotations.get(record.name)
        width = float(record.out_width or (annotation.width if annotation else 0))
        height = float(record.out_height or (annotation.height if annotation else 0))
        source_boxes: List[BBox] = list(annotation.boxes) if annotation else []

        boxes_in += len(source_boxes)
        # Class ids must reference a class declared in ``config.classes`` (and
        # therefore in ``data.yaml``): anything outside ``[0, len(classes))`` is
        # dropped here so the written labels can never disagree with ``nc``.
        num_classes = len(config.classes)
        for box in source_boxes:
            if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
                boxes_clipped += 1
            for issue in box_issues(box, width, height, filter_config, num_classes=num_classes):
                if issue != "out_of_bounds":
                    issue_counter[issue] += 1

        kept = filter_boxes(source_boxes, filter_config, width, height, num_classes=num_classes)
        boxes_out += len(kept)

        if kept:
            images_with_boxes += 1
        else:
            empty_images.append(record.name)
            if not filter_config.keep_empty_images:
                record.dropped = True
                record.issues.append("no_boxes")
                dropped_empty_images.append(record.name)
                continue
            if source_boxes:
                # every box was rejected -> the image becomes an (unintended)
                # negative sample, flag it so the user can review it.
                record.issues.append("all_boxes_rejected")

        cleaned[record.name] = ImageAnnotation(
            image=record.name,
            width=int(width),
            height=int(height),
            boxes=kept,
        )

    statistics = compute_statistics(cleaned, config.classes)

    report: Dict[str, Any] = {
        "images_considered": int(sum(1 for record in records if not record.dropped)),
        "images_with_boxes": int(images_with_boxes),
        "images_without_boxes": int(len(empty_images)),
        "images_dropped_without_boxes": int(len(dropped_empty_images)),
        "boxes_in": int(boxes_in),
        "boxes_out": int(boxes_out),
        "boxes_dropped": int(boxes_in - boxes_out),
        "boxes_clipped": int(boxes_clipped),
        "issues": {key: int(value) for key, value in sorted(issue_counter.items())},
        "dropped_without_boxes": dropped_empty_images[:200],
        "statistics": statistics,
    }

    reporter.info(
        "validate: {0} image(s), {1} box(es) kept of {2} ({3} dropped, {4} clipped)".format(
            report["images_considered"],
            report["boxes_out"],
            report["boxes_in"],
            report["boxes_dropped"],
            report["boxes_clipped"],
        )
    )
    if report["images_without_boxes"]:
        reporter.info(
            "validate: {0} image(s) have no boxes (kept as negatives: {1})".format(
                report["images_without_boxes"], filter_config.keep_empty_images
            )
        )
    if issue_counter:
        reporter.warn(
            "validate: issue summary -> {0}".format(
                ", ".join("{0}={1}".format(k, v) for k, v in sorted(issue_counter.items()))
            )
        )
    return cleaned, report


def compute_statistics(
    annotations: Dict[str, ImageAnnotation],
    classes: List[str],
) -> Dict[str, Any]:
    """Collect the numbers printed in the dataset card / ``dataset_stats.json``."""
    widths: List[float] = []
    heights: List[float] = []
    areas: List[float] = []
    normalized_areas: List[float] = []
    aspects: List[float] = []
    boxes_per_image: List[int] = []
    per_class = Counter()

    for annotation in annotations.values():
        boxes_per_image.append(len(annotation.boxes))
        image_area = float(max(1, annotation.width * annotation.height))
        for box in annotation.boxes:
            widths.append(box.width)
            heights.append(box.height)
            areas.append(box.area)
            normalized_areas.append(box.area / image_area)
            aspects.append(box.aspect_ratio)
            per_class[int(box.class_id)] += 1

    named_classes = {}
    for index, name in enumerate(classes):
        named_classes[str(index)] = {
            "name": str(name),
            "instances": int(per_class.get(index, 0)),
        }

    return {
        "images": int(len(annotations)),
        "images_with_boxes": int(sum(1 for value in boxes_per_image if value > 0)),
        "boxes_total": int(len(widths)),
        "boxes_per_image_mean": round(float(np.mean(boxes_per_image)) if boxes_per_image else 0.0, 4),
        "boxes_per_image_max": int(max(boxes_per_image)) if boxes_per_image else 0,
        "classes": named_classes,
        "unknown_class_ids": {
            str(key): int(value)
            for key, value in sorted(per_class.items())
            if key < 0 or key >= len(classes)
        },
        "width_px": {
            "min": round(min(widths), 2) if widths else 0.0,
            "p50": round(percentile(widths, 50), 2),
            "p95": round(percentile(widths, 95), 2),
            "max": round(max(widths), 2) if widths else 0.0,
        },
        "height_px": {
            "min": round(min(heights), 2) if heights else 0.0,
            "p50": round(percentile(heights, 50), 2),
            "p95": round(percentile(heights, 95), 2),
            "max": round(max(heights), 2) if heights else 0.0,
        },
        "area_px": {
            "min": round(min(areas), 2) if areas else 0.0,
            "p50": round(percentile(areas, 50), 2),
            "p95": round(percentile(areas, 95), 2),
            "max": round(max(areas), 2) if areas else 0.0,
        },
        "area_ratio": {
            "min": round(min(normalized_areas), 6) if normalized_areas else 0.0,
            "p50": round(percentile(normalized_areas, 50), 6),
            "p95": round(percentile(normalized_areas, 95), 6),
            "max": round(max(normalized_areas), 6) if normalized_areas else 0.0,
        },
        "aspect_ratio": {
            "min": round(min(aspects), 4) if aspects else 0.0,
            "p50": round(percentile(aspects, 50), 4),
            "p95": round(percentile(aspects, 95), 4),
            "max": round(max(aspects), 4) if aspects else 0.0,
        },
    }


def cross_check_records(
    records: List[ImageRecord],
    annotations: Dict[str, ImageAnnotation],
    classes: List[str],
) -> Dict[str, Any]:
    """Quick consistency check between the manifest and the annotation map."""
    kept = [record for record in records if not record.dropped]
    missing = [record.name for record in kept if record.name not in annotations]
    orphans = [name for name in annotations if name not in {record.name for record in kept}]
    per_class = count_classes(annotations.values())
    unknown = sorted(key for key in per_class if key < 0 or key >= len(classes))
    return {
        "kept_images": len(kept),
        "annotated_images": len(annotations),
        "missing_annotations": missing[:200],
        "orphan_annotations": orphans[:200],
        "class_counts": {str(key): int(value) for key, value in per_class.items()},
        "unknown_class_ids": unknown,
        "ok": not missing and bool(kept),
    }
