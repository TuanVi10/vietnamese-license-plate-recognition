"""
splitter.py
===========
Stage 4 of the pipeline: partition the cleaned dataset into ``train`` / ``val``
/ ``test``.

Two properties matter for a *license plate* dataset:

* **group awareness** – consecutive video frames (or bursts) end up in the same
  folder; if two near-identical frames landed in different splits the reported
  validation accuracy would be optimistic.  Images are therefore grouped (by the
  relative folder they came from, see :func:`utils.guess_group`) and whole
  groups are assigned to a single split.
* **stratification** – the ratio between images *with* a plate and images
  *without* one is preserved in every split, which keeps the false-positive rate
  measurable.

The function is fully deterministic for a given ``seed``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from annotations import ImageAnnotation
from dataset_config import DatasetConfig
from preprocess import ImageRecord
from utils import Reporter

SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test")


def group_records(records: Sequence[ImageRecord], group_aware: bool = True) -> Dict[str, List[str]]:
    """Bucket image names by their source group (or by their own name)."""
    groups: Dict[str, List[str]] = {}
    for record in records:
        if record.dropped or not record.name:
            continue
        key = record.group if (group_aware and record.group) else record.name
        groups.setdefault(key, []).append(record.name)
    for key in groups:
        groups[key] = sorted(groups[key])
    return groups


def split_dataset(
    records: Sequence[ImageRecord],
    annotations: Dict[str, ImageAnnotation],
    config: DatasetConfig,
    reporter: Optional[Reporter] = None,
) -> Dict[str, List[str]]:
    """Assign every kept image to exactly one split and return ``{split: names}``."""
    reporter = reporter or Reporter(quiet=True)
    split_config = config.split

    groups = group_records(records, group_aware=split_config.group_aware)
    if not groups:
        raise ValueError("split_dataset() received no usable image records")

    def is_positive(group_name: str) -> bool:
        for image_name in groups[group_name]:
            annotation = annotations.get(image_name)
            if annotation is not None and annotation.boxes:
                return True
        return False

    positive_groups = sorted(name for name in groups if is_positive(name))
    negative_groups = sorted(name for name in groups if not is_positive(name))

    rng = random.Random(int(split_config.seed))
    assignment: Dict[str, str] = {}

    buckets: List[List[str]]
    if split_config.stratify:
        buckets = [positive_groups, negative_groups]
    else:
        buckets = [positive_groups + negative_groups]

    for bucket in buckets:
        for group_name, split_name in _allocate(bucket, split_config.val_ratio, split_config.test_ratio, rng).items():
            assignment[group_name] = split_name

    splits: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}
    for group_name, split_name in assignment.items():
        splits[split_name].extend(groups[group_name])
    for split_name in splits:
        splits[split_name].sort()

    splits = {name: names for name, names in splits.items() if names}

    # A missing ``val`` split is the dangerous case: ``writer.write_data_yaml``
    # then emits ``val: images/train`` and the model would be evaluated on the
    # very data it is trained on.  Surface it loudly instead of failing silently.
    has_val = bool(splits.get("val"))
    has_test = bool(splits.get("test"))
    if split_config.val_ratio > 0.0 and not has_val:
        reporter.warn(
            "split: no validation images could be created (only {0} group(s) available); "
            "data.yaml will fall back to 'val: images/train'. Add more source folders "
            "or set split.val_ratio=0 to silence this warning.".format(len(assignment))
        )
    elif split_config.test_ratio > 0.0 and not has_test:
        reporter.warn(
            "split: no test images could be created; only a train/val split was produced "
            "(lower split.test_ratio or add more source folders if a test split is required)."
        )

    reporter.info(
        "split: {0} -> {1}".format(
            ", ".join("{0}={1}".format(name, len(names)) for name, names in sorted(splits.items())),
            "group-aware" if split_config.group_aware else "image-level",
        )
    )
    reporter.info(
        "split: positive groups={0}, negative groups={1}".format(
            len(positive_groups), len(negative_groups)
        )
    )
    return splits


def _allocate(
    group_names: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, str]:
    """Randomly assign ``group_names`` to the three splits using the given ratios.

    Whole groups are assigned atomically so a single source folder never leaks
    across splits.  A held-out split is created whenever it is *possible*:
    even with only two groups (and a non-zero evaluation ratio) at least one
    group is kept out of ``train`` – otherwise validation would silently run on
    the training data.
    """
    items = list(group_names)
    rng.shuffle(items)
    total = len(items)
    assignment: Dict[str, str] = {}

    if total == 0:
        return assignment

    eval_ratio = float(val_ratio) + float(test_ratio)

    # Only fall back to "everything is train" when a split is genuinely
    # impossible: a single group, or the user asked for no evaluation split.
    if total == 1 or eval_ratio <= 0.0:
        for name in items:
            assignment[name] = "train"
        return assignment

    eval_count = max(1, min(total - 1, int(round(total * eval_ratio))))

    # When only one evaluation slot exists but *both* val and test are wanted,
    # keep the split with the larger ratio (the caller logs a warning).
    if eval_count == 1 and float(val_ratio) > 0.0 and float(test_ratio) > 0.0:
        val_count = 1 if float(val_ratio) >= float(test_ratio) else 0
        test_count = 1 - val_count
    else:
        ratio_sum = float(val_ratio) + float(test_ratio)
        val_count = int(round(eval_count * float(val_ratio) / ratio_sum)) if ratio_sum > 0 else 0
        test_count = eval_count - val_count

        # Never let rounding starve a requested split when there is room.
        if float(val_ratio) > 0.0 and eval_count >= 2 and val_count == 0:
            val_count, test_count = 1, eval_count - 1
        if float(test_ratio) > 0.0 and eval_count >= 2 and test_count == 0:
            test_count, val_count = 1, eval_count - 1

    for index, name in enumerate(items):
        if index < test_count:
            assignment[name] = "test"
        elif index < test_count + val_count:
            assignment[name] = "val"
        else:
            assignment[name] = "train"
    return assignment


def split_summary(
    splits: Dict[str, List[str]],
    annotations: Dict[str, ImageAnnotation],
) -> Dict[str, Dict[str, int]]:
    """Per-split counts of images / boxes / positive images (for the report)."""
    summary: Dict[str, Dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        names = splits.get(split_name, [])
        boxes = 0
        positives = 0
        for image_name in names:
            annotation = annotations.get(image_name)
            count = len(annotation.boxes) if annotation else 0
            boxes += count
            if count > 0:
                positives += 1
        summary[split_name] = {
            "images": len(names),
            "boxes": boxes,
            "images_with_boxes": positives,
            "images_without_boxes": len(names) - positives,
        }
    return summary


def save_splits(path: Any, splits: Dict[str, List[str]]) -> Path:
    """Persist the split assignment as JSON."""
    from utils import write_json

    return write_json(path, {name: sorted(names) for name, names in splits.items()})


def load_splits(path: Any) -> Dict[str, List[str]]:
    """Load a split assignment previously written by :func:`save_splits`."""
    from utils import read_json

    payload = read_json(path, default=None)
    if not payload:
        raise FileNotFoundError(
            "splits not found at {0}; run the 'split' stage first".format(path)
        )
    return {name: sorted(names) for name, names in payload.items()}


def dataset_paths(output_dir: Any, split_name: str) -> Dict[str, Path]:
    """Return the conventional YOLO folders of one split."""
    root = Path(output_dir)
    return {
        "images": root / "images" / split_name,
        "labels": root / "labels" / split_name,
    }
