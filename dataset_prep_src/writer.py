"""
writer.py
=========
Stage 5 of the pipeline: materialise the final, ready-to-train YOLO dataset.

Produced layout
---------------
::

    <output_dir>/
    ├── data.yaml                 # ultralytics entry point
    ├── dataset_stats.json        # raw numbers (machine readable)
    ├── DATASET_CARD.md           # human readable summary
    ├── images/{train,val,test}/  # the images, exactly as labelled
    └── labels/{train,val,test}/  # one .txt per image (empty file = negative)

``data.yaml`` uses the modern Ultralytics schema::

    path: /abs/path/to/output_dir
    train: images/train
    val: images/val
    test: images/test
    nc: 1
    names:
      0: license_plate

The writer never *moves* the files produced by :mod:`preprocess`; it copies them
so the intermediate ``work`` folder can still be inspected/re-generated.

On every run the pipeline-owned ``images``/``labels``/``qc`` folders inside
``output_dir`` are wiped *before* the fresh files are copied in, so a re-run can
never keep stale files from a previous build (Ultralytics globs the whole folder
tree and would otherwise silently train on images that are absent from the
current ``data.yaml``).
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from annotations import ImageAnnotation, save_yolo_labels
from dataset_config import DatasetConfig
from preprocess import ImageRecord
from splitter import SPLIT_NAMES, split_summary
from utils import Reporter, clean_dir, ensure_dir, human_bytes, paths_overlap, write_json


class DatasetWriter:
    """Copy images + labels into the final dataset tree and emit the metadata."""

    def __init__(self, config: DatasetConfig, reporter: Optional[Reporter] = None) -> None:
        self.config = config
        self.reporter = reporter or Reporter(quiet=True)
        self.root = Path(config.output_dir)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def write(
        self,
        records: Sequence[ImageRecord],
        annotations: Dict[str, ImageAnnotation],
        splits: Dict[str, List[str]],
        report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Materialise the dataset and return a mapping of the produced paths."""
        ensure_dir(self.root)

        # A previous run may have left files behind. Ultralytics globs the whole
        # ``images``/``labels`` tree, so any stale image/label would be trained on
        # (via the folder scan) even though it is absent from the freshly written
        # ``data.yaml``. Wipe the pipeline-owned folders before copying the new
        # run so the produced dataset only ever reflects the current split.
        # ``images`` and ``labels`` are mandatory; ``qc`` is wiped too so preview
        # sheets cannot be mistaken for current output. ``clean_dir`` keeps the
        # folders themselves, so the loops below can still rely on them existing.
        source_dir = Path(self.config.work_dir) / "images"
        for subdir in ("images", "labels", "qc"):
            target = self.root / subdir
            if paths_overlap(target, source_dir):
                raise ValueError(
                    "output_dir '{0}' overlaps the preprocessed source images "
                    "'{1}'; refusing to wipe them before writing. Configure "
                    "output_dir so it does not contain (or live inside) "
                    "work_dir/images.".format(self.root, source_dir)
                )
            clean_dir(target)

        records_by_name = {record.name: record for record in records if not record.dropped}

        written_images = 0
        written_labels = 0
        written_boxes = 0
        copied_bytes = 0
        skipped: List[str] = []

        for split_name in SPLIT_NAMES:
            names = splits.get(split_name, [])
            if not names:
                continue
            image_dir = ensure_dir(self.root / "images" / split_name)
            label_dir = ensure_dir(self.root / "labels" / split_name)

            for name in names:
                record = records_by_name.get(name)
                if record is None:
                    skipped.append(name)
                    continue

                source_image = source_dir / name
                if not source_image.exists():
                    skipped.append(name)
                    continue
                destination_image = image_dir / name
                shutil.copy2(str(source_image), str(destination_image))
                written_images += 1
                copied_bytes += destination_image.stat().st_size

                annotation = annotations.get(name)
                boxes = annotation.boxes if annotation is not None else []
                width = record.out_width or (annotation.width if annotation else 0)
                height = record.out_height or (annotation.height if annotation else 0)
                save_yolo_labels(
                    label_dir / "{0}.txt".format(Path(name).stem),
                    boxes,
                    width,
                    height,
                    num_classes=len(self.config.classes),
                )
                written_labels += 1
                written_boxes += len(boxes)

        if skipped:
            self.reporter.warn(
                "write: {0} image(s) referenced by the split were missing and skipped".format(len(skipped))
            )

        produced: Dict[str, str] = {}
        produced["data_yaml"] = str(self.write_data_yaml(splits))
        produced["stats"] = str(self.write_stats(records, annotations, splits, report, written_images, written_boxes))
        produced["card"] = str(self.write_dataset_card(splits, annotations, records, report))

        self.reporter.info(
            "write: {0} image(s), {1} label file(s), {2} box(es), {3} -> {4}".format(
                written_images,
                written_labels,
                written_boxes,
                human_bytes(copied_bytes),
                self.root,
            )
        )
        return produced

    # ------------------------------------------------------------------ #
    # data.yaml
    # ------------------------------------------------------------------ #
    def write_data_yaml(self, splits: Dict[str, List[str]]) -> Path:
        """Write the Ultralytics ``data.yaml`` pointing at the produced folders."""
        path = self.root / "data.yaml"
        root = self.root.resolve().as_posix()
        lines = [
            "# Auto-generated by prepare_vn_plate_dataset (dataset: {0})".format(self.config.dataset_name),
            "# Classes: {0}".format(", ".join(self.config.classes)),
            "path: {0}".format(root),
            "train: images/train",
        ]
        lines.append("val: images/val" if splits.get("val") else "val: images/train")
        if splits.get("test"):
            lines.append("test: images/test")
        lines.append("nc: {0}".format(len(self.config.classes)))
        lines.append("names:")
        for index, name in enumerate(self.config.classes):
            lines.append("  {0}: {1}".format(index, name))
        lines.append("")
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return path

    # ------------------------------------------------------------------ #
    # stats / card
    # ------------------------------------------------------------------ #
    def write_stats(
        self,
        records: Sequence[ImageRecord],
        annotations: Dict[str, ImageAnnotation],
        splits: Dict[str, List[str]],
        report: Optional[Dict[str, Any]],
        written_images: int,
        written_boxes: int,
    ) -> Path:
        """Persist ``dataset_stats.json``."""
        kept = [record for record in records if not record.dropped]
        payload: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_name": self.config.dataset_name,
            "classes": list(self.config.classes),
            "label_mode": self.config.label_mode,
            "labels_format": self.config.labels_format,
            "preprocess": self.config.preprocess.to_dict(),
            "filter": self.config.filter.to_dict(),
            "split_config": self.config.split.to_dict(),
            "records": {
                "total": len(records),
                "kept": len(kept),
                "dropped": len(records) - len(kept),
                "drop_reasons": _drop_reasons(records),
            },
            "splits": split_summary(splits, annotations),
            "written": {"images": written_images, "boxes": written_boxes},
            "validation": report or {},
            "annotation_source_counts": _source_counts(annotations),
        }
        return write_json(self.root / "dataset_stats.json", payload)

    def write_dataset_card(
        self,
        splits: Dict[str, List[str]],
        annotations: Dict[str, ImageAnnotation],
        records: Sequence[ImageRecord],
        report: Optional[Dict[str, Any]],
    ) -> Path:
        """Write a short Markdown summary next to ``data.yaml``."""
        summary = split_summary(splits, annotations)
        stats = (report or {}).get("statistics", {})
        dropped = [record for record in records if record.dropped]

        lines: List[str] = []
        lines.append("# {0}".format(self.config.dataset_name))
        lines.append("")
        lines.append("YOLO-format dataset of Vietnamese license plates, generated on {0}.".format(
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
        lines.append("")
        lines.append("## Classes")
        lines.append("")
        for index, name in enumerate(self.config.classes):
            lines.append("- `{0}`: {1}".format(index, name))
        lines.append("")
        lines.append("## Splits")
        lines.append("")
        lines.append("| split | images | boxes | images with plate | images without plate |")
        lines.append("|-------|--------|-------|-------------------|----------------------|")
        for split_name in SPLIT_NAMES:
            if split_name not in summary:
                continue
            item = summary[split_name]
            lines.append(
                "| {0} | {1} | {2} | {3} | {4} |".format(
                    split_name,
                    item["images"],
                    item["boxes"],
                    item["images_with_boxes"],
                    item["images_without_boxes"],
                )
            )
        lines.append("")
        lines.append("## Box geometry (pixels, dataset wide)")
        lines.append("")
        if stats:
            lines.append("| metric | min | p50 | p95 | max |")
            lines.append("|--------|-----|-----|-----|-----|")
            for key, title in (
                ("width_px", "width"),
                ("height_px", "height"),
                ("area_px", "area"),
            ):
                block = stats.get(key, {})
                lines.append(
                    "| {0} | {1} | {2} | {3} | {4} |".format(
                        title,
                        block.get("min", 0),
                        block.get("p50", 0),
                        block.get("p95", 0),
                        block.get("max", 0),
                    )
                )
            aspect = stats.get("aspect_ratio", {})
            lines.append(
                "| aspect ratio | {0} | {1} | {2} | {3} |".format(
                    aspect.get("min", 0),
                    aspect.get("p50", 0),
                    aspect.get("p95", 0),
                    aspect.get("max", 0),
                )
            )
        else:
            lines.append("_no statistics available_")
        lines.append("")
        lines.append("## Pipeline settings")
        lines.append("")
        lines.append("- label mode: `{0}` (format `{1}`)".format(self.config.label_mode, self.config.labels_format))
        lines.append("- resize: `{0}` @ {1}px".format(
            self.config.preprocess.resize_mode, self.config.preprocess.image_size
        ))
        lines.append("- seed: `{0}`, group aware: `{1}`, stratified: `{2}`".format(
            self.config.split.seed, self.config.split.group_aware, self.config.split.stratify
        ))
        lines.append("- dropped images: `{0}`".format(len(dropped)))
        if dropped:
            reasons = _drop_reasons(records)
            lines.append("  - " + ", ".join("{0}={1}".format(k, v) for k, v in sorted(reasons.items())))
        lines.append("")
        lines.append("## How to train")
        lines.append("")
        lines.append("```bash")
        lines.append("yolo detect train model=yolo11n.pt data={0} imgsz=640 epochs=100".format(
            (self.root / "data.yaml").as_posix()
        ))
        lines.append("```")
        lines.append("")
        if self.config.label_mode in ("auto", "merge"):
            lines.append(
                "> NOTE: labels in this build come (partly) from automatic pseudo-labelling. "
                "Review them in LabelImg / CVAT before trusting production metrics."
            )
            lines.append("")
        path = self.root / "DATASET_CARD.md"
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _drop_reasons(records: Sequence[ImageRecord]) -> Dict[str, int]:
    reasons: Dict[str, int] = {}
    for record in records:
        if not record.dropped:
            continue
        if not record.issues:
            reasons["unknown"] = reasons.get("unknown", 0) + 1
            continue
        for issue in record.issues:
            key = issue.split(":", 1)[0]
            reasons[key] = reasons.get(key, 0) + 1
    return reasons


def _source_counts(annotations: Dict[str, ImageAnnotation]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for annotation in annotations.values():
        for box in annotation.boxes:
            key = (box.source or "unknown").split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
    return counts


def write_dataset(
    config: DatasetConfig,
    records: Sequence[ImageRecord],
    annotations: Dict[str, ImageAnnotation],
    splits: Dict[str, List[str]],
    report: Optional[Dict[str, Any]] = None,
    reporter: Optional[Reporter] = None,
) -> Dict[str, str]:
    """Convenience wrapper around :class:`DatasetWriter`."""
    return DatasetWriter(config, reporter).write(records, annotations, splits, report)
