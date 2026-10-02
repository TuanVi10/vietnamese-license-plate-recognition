"""
pipeline.py
===========
The orchestrator that chains every stage of the dataset preparation:

``preprocess`` -> ``labels`` -> ``validate`` -> ``split`` -> ``write`` -> ``preview``

Each stage is independently runnable and persists its artifact as JSON inside
``work_dir``, so a long job can be resumed without recomputing everything::

    python main.py --stages preprocess
    python main.py --stages labels validate
    python main.py --stages split write preview

Artifacts
---------
* ``work/manifest.json``     – :mod:`preprocess` output (one record per image).
* ``work/annotations.json``  – :mod:`annotations` map, final coordinate frame.
* ``work/report.json``       – :mod:`validator` output.
* ``work/splits.json``       – :mod:`splitter` output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from annotations import (
    ImageAnnotation,
    load_annotations_file,
    merge_boxes,
    save_annotations,
)
from autolabel import PlateAutoLabeler
from converters import build_images_index, load_annotations as load_imported_labels
from dataset_config import DatasetConfig
from preprocess import ImagePreprocessor, ImageRecord
from splitter import load_splits, save_splits, split_dataset, split_summary
from utils import Reporter, ensure_dir, imread_unicode, read_json, write_json
from validator import validate_annotations
from visualize import save_qc_samples, save_split_contact_sheets
from writer import DatasetWriter

STAGE_ORDER: Sequence[str] = ("preprocess", "labels", "validate", "split", "write", "preview")
_CORE_STAGES: Sequence[str] = ("preprocess", "labels", "validate", "split", "write")

# How much a pseudo-label is grown before being stored (compensates the
# morphological closing that shaves the plate border).
AUTO_LABEL_EXPAND = 0.02


class DatasetPipeline:
    """Run the whole (or part of the) preparation workflow for one config."""

    def __init__(self, config: DatasetConfig, reporter: Optional[Reporter] = None) -> None:
        config.validate()
        self.config = config
        self.reporter = reporter or Reporter()
        self.work_dir = ensure_dir(config.work_dir)
        self.images_dir = Path(config.work_dir) / "images"
        self.manifest_path = Path(config.work_dir) / "manifest.json"
        self.annotations_path = Path(config.work_dir) / "annotations.json"
        self.report_path = Path(config.work_dir) / "report.json"
        self.splits_path = Path(config.work_dir) / "splits.json"

        self.records: List[ImageRecord] = []
        self.annotations: Dict[str, ImageAnnotation] = {}
        self.report: Dict[str, Any] = {}
        self.splits: Dict[str, List[str]] = {}
        self.written: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _kept_records(self) -> List[ImageRecord]:
        return [record for record in self.records if not record.dropped]

    def _require_records(self) -> List[ImageRecord]:
        if not self.records:
            self.records, _ = ImagePreprocessor.load_manifest(self.manifest_path)
        return self.records

    def _require_annotations(self) -> Dict[str, ImageAnnotation]:
        if not self.annotations:
            self.annotations = load_annotations_file(self.annotations_path)
        return self.annotations

    def _require_splits(self) -> Dict[str, List[str]]:
        if not self.splits:
            self.splits = load_splits(self.splits_path)
        return self.splits

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #
    def stage_preprocess(self) -> List[ImageRecord]:
        """Decode, clean, de-duplicate, rename and (optionally) resize the raw images."""
        self.reporter.stage("preprocess: scanning {0}".format(self.config.raw_dir))
        self.records = ImagePreprocessor(self.config, self.reporter).run()
        return self.records

    def stage_labels(self) -> Dict[str, ImageAnnotation]:
        """Build the annotation map: imported labels and/or pseudo-labels."""
        self.reporter.stage("labels: mode={0}".format(self.config.label_mode))
        self._require_records()
        kept = self._kept_records()

        imported = None
        if self.config.label_mode in ("import", "merge"):
            imported = load_imported_labels(
                self.config.labels_format,
                self.config.labels_path,
                self.config.classes,
                build_images_index(self.records),
                self.reporter,
            )
            self.reporter.info(
                "labels: imported {0} annotated image(s) from {1}".format(
                    len(imported), self.config.labels_format
                )
            )
            if imported.skipped_classes:
                self.reporter.warn(
                    "labels: ignored class names not present in config: {0}".format(
                        ", ".join(sorted(set(imported.skipped_classes))[:10])
                    )
                )

        labeler = None
        if self.config.label_mode in ("auto", "merge") and self.config.autolabel.enabled:
            labeler = PlateAutoLabeler(self.config.autolabel, self.reporter)

        annotations: Dict[str, ImageAnnotation] = {}
        auto_hits = 0
        for record in kept:
            boxes = []
            if imported is not None:
                boxes.extend(self._imported_boxes(record, imported))
            if labeler is not None:
                image = imread_unicode(self.images_dir / record.name)
                if image is None:
                    self.reporter.warn("labels: cannot re-read {0}, skipping auto-label".format(record.name))
                else:
                    detected = labeler.detect(image)
                    if detected:
                        auto_hits += 1
                    boxes.extend(detected)
            if self.config.label_mode == "merge":
                boxes = merge_boxes(boxes, float(self.config.autolabel.nms_iou))

            annotations[record.name] = ImageAnnotation(
                image=record.name,
                width=int(record.out_width),
                height=int(record.out_height),
                boxes=boxes,
            )

        self.annotations = annotations
        save_annotations(self.annotations_path, annotations, self.config.classes, self.config.label_mode)

        total_boxes = sum(len(annotation.boxes) for annotation in annotations.values())
        self.reporter.info(
            "labels: {0} image(s), {1} box(es), {2} image(s) with plate".format(
                len(annotations), total_boxes, auto_hits if labeler else _count_with_boxes(annotations)
            )
        )
        if labeler is not None:
            self.reporter.info("labels: auto-labeller stats {0}".format(labeler.stats()))
        return annotations

    def _imported_boxes(self, record: ImageRecord, imported: Any) -> List:
        """Attach imported boxes to a record, fixing size and resize mismatches."""
        entry = imported.find(record.source_name, record.source_stem)
        if entry is None:
            return []
        boxes, declared = entry
        prepared = []
        for box in boxes:
            candidate = box.clone()
            if declared and declared[0] > 0 and declared[1] > 0:
                factor_x = float(record.width) / float(declared[0])
                factor_y = float(record.height) / float(declared[1])
                if abs(factor_x - 1.0) > 1e-3 or abs(factor_y - 1.0) > 1e-3:
                    candidate = candidate.scaled(factor_x, factor_y)
            candidate = candidate.transformed(record.transform)
            candidate.source = imported.source
            prepared.append(candidate)
        return prepared

    def stage_validate(self) -> Dict[str, Any]:
        """Clean the labels and compute the dataset statistics."""
        self.reporter.stage("validate: checking geometry and coverage")
        self._require_records()
        self._require_annotations()
        self.annotations, self.report = validate_annotations(
            self.records, self.annotations, self.config, self.reporter
        )
        # Persist the cleaned annotations so later stages (and later processes
        # resumed from the 'split'/'write' stages) reuse the filtered/clipped
        # boxes instead of silently reloading the un-validated labels.
        save_annotations(
            self.annotations_path,
            self.annotations,
            self.config.classes,
            self.config.label_mode,
        )
        # Persist the updated record drop-flags so empty images that were removed
        # here stay removed when the user resumes from a later stage.
        ImagePreprocessor(self.config, self.reporter).save_manifest(self.records)
        write_json(self.report_path, self.report)
        return self.report

    def stage_split(self) -> Dict[str, List[str]]:
        """Deterministically partition the images into train/val/test."""
        self.reporter.stage("split: val={0:.2f} test={1:.2f}".format(
            self.config.split.val_ratio, self.config.split.test_ratio
        ))
        self._require_records()
        self._require_annotations()
        self.splits = split_dataset(self._kept_records(), self.annotations, self.config, self.reporter)
        save_splits(self.splits_path, self.splits)
        return self.splits

    def stage_write(self) -> Dict[str, str]:
        """Copy images + labels into the final YOLO tree and emit ``data.yaml``."""
        self.reporter.stage("write: materialising {0}".format(self.config.output_dir))
        self._require_records()
        self._require_annotations()
        self._require_splits()
        writer = DatasetWriter(self.config, self.reporter)
        self.written = writer.write(self.records, self.annotations, self.splits, self.report)
        return self.written

    def stage_preview(self) -> Dict[str, str]:
        """Render the QC contact sheets."""
        self.reporter.stage("preview: rendering QC sheets")
        self._require_records()
        self._require_annotations()
        produced: Dict[str, str] = {}
        if not self.config.make_preview:
            self.reporter.info("preview: disabled by configuration")
            return produced
        sheet = save_qc_samples(self.config, self.records, self.annotations, self.reporter)
        if sheet is not None:
            produced["all"] = str(sheet)
        if self.splits:
            produced.update(
                save_split_contact_sheets(
                    self.config,
                    self.records,
                    self.annotations,
                    self.splits,
                    limit_per_split=max(4, int(self.config.preview_limit) // 2),
                    columns=int(self.config.preview_columns),
                    reporter=self.reporter,
                )
            )
        return produced

    # ------------------------------------------------------------------ #
    # Runner
    # ------------------------------------------------------------------ #
    def run(self, stages: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Execute ``stages`` (defaults to the full core pipeline) and summarise."""
        selected = list(stages) if stages else list(_CORE_STAGES)
        unknown = [name for name in selected if name not in STAGE_ORDER]
        if unknown:
            raise ValueError("unknown stage(s): {0}".format(", ".join(unknown)))
        # Execute in canonical order regardless of the order given by the user.
        ordered = [name for name in STAGE_ORDER if name in selected]

        self.reporter.info("pipeline: stages -> {0}".format(", ".join(ordered)))
        for name in ordered:
            handler = getattr(self, "stage_{0}".format(name))
            handler()
        return self.summary()

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        """Return the final report dictionary (also used as the CLI output)."""
        kept = self._kept_records()
        summary: Dict[str, Any] = {
            "dataset_name": self.config.dataset_name,
            "raw_dir": str(self.config.raw_dir),
            "work_dir": str(self.config.work_dir),
            "output_dir": str(self.config.output_dir),
            "images_total": len(self.records),
            "images_kept": len(kept),
            "images_dropped": len(self.records) - len(kept),
            "boxes_total": int(sum(len(annotation.boxes) for annotation in self.annotations.values())),
            "splits": split_summary(self.splits, self.annotations) if self.splits else {},
            "written": self.written,
            "artifacts": {
                "manifest": str(self.manifest_path) if self.manifest_path.exists() else "",
                "annotations": str(self.annotations_path) if self.annotations_path.exists() else "",
                "report": str(self.report_path) if self.report_path.exists() else "",
                "splits": str(self.splits_path) if self.splits_path.exists() else "",
            },
        }
        return summary


def _count_with_boxes(annotations: Dict[str, ImageAnnotation]) -> int:
    return sum(1 for annotation in annotations.values() if annotation.boxes)


def load_manifest(path: Any) -> List[ImageRecord]:
    """Public re-export so callers do not need to import :mod:`preprocess`."""
    records, _ = ImagePreprocessor.load_manifest(path)
    return records


def load_manifest_payload(path: Any) -> Dict[str, Any]:
    """Load the raw manifest JSON (images dir, resize mode, counts...)."""
    return read_json(path, default={}) or {}


def run_pipeline(
    config: DatasetConfig,
    stages: Optional[Sequence[str]] = None,
    reporter: Optional[Reporter] = None,
) -> Dict[str, Any]:
    """Module level convenience entry point."""
    return DatasetPipeline(config, reporter).run(stages)
