"""
dataset_config.py
=================
Typed configuration objects for the Vietnamese license-plate dataset builder.

Everything the pipeline needs is expressed here so the behaviour of each stage
can be reproduced from a single JSON file (see ``config.default.json``).

Every dataclass exposes ``to_dict()`` / ``from_dict()`` and the top level
:class:`DatasetConfig` is the object handed to :class:`pipeline.DatasetPipeline`.

Typical usage
-------------
>>> from dataset_config import DatasetConfig
>>> cfg = DatasetConfig.load("config.default.json")
>>> cfg.raw_dir = "data/raw"
>>> cfg.validate()
True
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_type_hints

from utils import ensure_dir, read_json, write_json

# --------------------------------------------------------------------------- #
# Helpers to make nested (de)serialisation painless
# --------------------------------------------------------------------------- #
_RESIZE_MODES = ("none", "letterbox", "stretch")
_LABEL_FORMATS = ("none", "voc", "coco", "yolo")
_LABEL_MODES = ("import", "auto", "merge")
_TUPLE_FIELDS = {"letterbox_color"}
_LIST_FIELDS = {"classes"}


def _coerce(field_name: str, value: Any) -> Any:
    """Normalise JSON lists back into the tuples/lists the dataclasses expect."""
    if field_name in _TUPLE_FIELDS and isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    if field_name in _LIST_FIELDS and isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return value


def apply_dict(target: Any, data: Optional[Dict[str, Any]]) -> Any:
    """Recursively copy ``data`` into the dataclass instance ``target``."""
    if not data:
        return target
    hints = get_type_hints(type(target))
    for dataclass_field in dataclasses.fields(target):
        name = dataclass_field.name
        if name not in data or data[name] is None:
            continue
        value = data[name]
        field_type = hints.get(name, dataclass_field.type)
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            apply_dict(getattr(target, name), value)
        else:
            setattr(target, name, _coerce(name, value))
    return target


class _Mixin:
    """Shared ``to_dict`` / ``from_dict`` behaviour for every config dataclass."""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        instance = cls()
        apply_dict(instance, data)
        return instance


# --------------------------------------------------------------------------- #
# Stage configurations
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessConfig(_Mixin):
    """Controls reading, cleaning, de-duplicating and re-sizing of raw images."""

    resize_mode: str = "none"          # none | letterbox | stretch
    image_size: int = 640              # target size (square) for letterbox/stretch
    letterbox_color: Tuple[int, int, int] = (114, 114, 114)
    scale_up: bool = False             # allow up-scaling small images
    jpeg_quality: int = 95             # quality when re-encoding JPEG/WebP
    keep_unicode_names: bool = False   # keep Vietnamese file names (else ASCII slug)
    min_width: int = 64                # images narrower than this are dropped
    min_height: int = 48               # images shorter than this are dropped
    dedupe_exact: bool = True          # MD5 duplicate removal
    dedupe_near: bool = False          # perceptual-hash duplicate removal (slow)
    phash_hamming_threshold: int = 4   # <= this distance means "same picture"
    max_compare: int = 20000           # disable near-dupe search above this count
    limit: int = 0                     # 0 = use every image (useful for smoke tests)


@dataclass
class AutoLabelConfig(_Mixin):
    """Heuristic settings of the classical-CV Vietnamese plate pseudo-labeller."""

    enabled: bool = True
    class_id: int = 0
    min_area_ratio: float = 0.0025     # plate area / image area lower bound
    max_area_ratio: float = 0.35       # plate area / image area upper bound
    min_aspect_ratio: float = 1.15     # Vietnamese plates are wider than tall
    max_aspect_ratio: float = 6.0
    min_fill_ratio: float = 0.52       # contour area / bounding box area
    min_edge_density: float = 0.07     # ink/edge pixel ratio inside the box
    score_threshold: float = 0.45      # discard weak candidates
    nms_iou: float = 0.40              # merge overlapping candidates
    max_boxes_per_image: int = 4
    bilateral_diameter: int = 9
    close_iterations: int = 2
    use_adaptive: bool = True          # second candidate source (adaptive threshold)


@dataclass
class FilterConfig(_Mixin):
    """Sanity rules applied to every box before it is written to disk."""

    min_width_px: float = 12.0
    min_height_px: float = 10.0
    min_area_ratio: float = 1e-4       # of the whole image
    max_area_ratio: float = 0.90
    min_aspect_ratio: float = 1.0
    max_aspect_ratio: float = 7.0
    keep_empty_images: bool = True     # keep images with zero boxes as negatives


@dataclass
class SplitConfig(_Mixin):
    """Train / validation / test partitioning."""

    val_ratio: float = 0.15
    test_ratio: float = 0.10
    seed: int = 42
    stratify: bool = True              # keep the positive/negative ratio balanced
    group_aware: bool = True           # never split one source folder across splits


# --------------------------------------------------------------------------- #
# Top level configuration
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig(_Mixin):
    """Complete description of one dataset build."""

    raw_dir: str = "raw"
    work_dir: str = "work"
    output_dir: str = "dataset"
    dataset_name: str = "vn_license_plate"
    labels_format: str = "none"        # none | voc | coco | yolo
    labels_path: str = ""              # folder (voc/yolo) or json file (coco)
    label_mode: str = "auto"           # import | auto | merge
    classes: List[str] = field(default_factory=lambda: ["license_plate"])

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    autolabel: AutoLabelConfig = field(default_factory=AutoLabelConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    make_preview: bool = True
    preview_limit: int = 24
    preview_columns: int = 4

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: Any) -> "DatasetConfig":
        """Load a configuration from a JSON file (missing keys keep defaults)."""
        data = read_json(path, default=None)
        if data is None:
            return cls()
        return cls.from_dict(data)

    def save(self, path: Any) -> Path:
        """Persist the configuration as UTF-8 JSON."""
        return write_json(path, self.to_dict())

    # ------------------------------------------------------------------ #
    # Validation / path helpers
    # ------------------------------------------------------------------ #
    def validate(self) -> bool:
        """Raise ``ValueError`` when the configuration is inconsistent."""
        errors: List[str] = []

        if self.preprocess.resize_mode not in _RESIZE_MODES:
            errors.append(
                "preprocess.resize_mode must be one of {0} (got {1!r})".format(
                    _RESIZE_MODES, self.preprocess.resize_mode
                )
            )
        if int(self.preprocess.image_size) < 32:
            errors.append("preprocess.image_size must be >= 32")

        if self.labels_format not in _LABEL_FORMATS:
            errors.append(
                "labels_format must be one of {0} (got {1!r})".format(
                    _LABEL_FORMATS, self.labels_format
                )
            )
        if self.label_mode not in _LABEL_MODES:
            errors.append(
                "label_mode must be one of {0} (got {1!r})".format(
                    _LABEL_MODES, self.label_mode
                )
            )
        if self.label_mode in ("import", "merge") and self.labels_format == "none":
            errors.append("label_mode {0!r} needs a labels_format".format(self.label_mode))
        if self.labels_format != "none" and not self.labels_path:
            errors.append("labels_path is required when labels_format is set")

        if not self.classes:
            errors.append("classes must contain at least one class name")

        split = self.split
        if split.val_ratio < 0.0 or split.test_ratio < 0.0:
            errors.append("split ratios must be >= 0")
        if split.val_ratio + split.test_ratio >= 1.0:
            errors.append("split.val_ratio + split.test_ratio must be < 1.0")

        if self.filter.min_aspect_ratio <= 0 or self.filter.max_aspect_ratio <= 0:
            errors.append("filter aspect ratios must be > 0")
        if self.filter.min_aspect_ratio > self.filter.max_aspect_ratio:
            errors.append("filter.min_aspect_ratio must be <= max_aspect_ratio")

        auto = self.autolabel
        if auto.min_area_ratio >= auto.max_area_ratio:
            errors.append("autolabel.min_area_ratio must be < max_area_ratio")
        if auto.min_aspect_ratio >= auto.max_aspect_ratio:
            errors.append("autolabel.min_aspect_ratio must be < max_aspect_ratio")
        if auto.max_boxes_per_image < 1:
            errors.append("autolabel.max_boxes_per_image must be >= 1")

        if errors:
            raise ValueError("invalid configuration:\n - " + "\n - ".join(errors))
        return True

    # ------------------------------------------------------------------ #
    def absolute_paths(self) -> Dict[str, str]:
        """Return the three main folders as absolute, resolved paths."""
        return {
            "raw_dir": str(Path(self.raw_dir).expanduser().resolve()),
            "work_dir": str(Path(self.work_dir).expanduser().resolve()),
            "output_dir": str(Path(self.output_dir).expanduser().resolve()),
        }

    def prepare_directories(self) -> Dict[str, Path]:
        """Create ``work_dir`` and ``output_dir`` and return them as ``Path``s."""
        return {
            "work_dir": ensure_dir(self.work_dir),
            "output_dir": ensure_dir(self.output_dir),
        }

    def describe(self) -> str:
        """Human readable one-block summary used by the CLI."""
        return (
            "dataset     : {0}\n"
            "raw_dir     : {1}\n"
            "work_dir    : {2}\n"
            "output_dir  : {3}\n"
            "classes     : {4}\n"
            "labels      : format={5!r} mode={6!r} path={7!r}\n"
            "resize      : mode={8!r} size={9}\n"
            "split       : val={10:.2f} test={11:.2f} seed={12} stratify={13}"
        ).format(
            self.dataset_name,
            self.raw_dir,
            self.work_dir,
            self.output_dir,
            ", ".join(self.classes),
            self.labels_format,
            self.label_mode,
            self.labels_path,
            self.preprocess.resize_mode,
            self.preprocess.image_size,
            self.split.val_ratio,
            self.split.test_ratio,
            self.split.seed,
            self.split.stratify,
        )
