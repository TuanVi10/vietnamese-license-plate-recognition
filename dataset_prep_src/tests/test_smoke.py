"""
tests/test_smoke.py
===================
End-to-end smoke test that needs no external data: it synthesises a handful of
synthetic "license plates" (a bright plate with dark characters on a street-like
background), runs the full pipeline and asserts that a valid YOLO dataset is
produced.

Run it with::

    python tests/test_smoke.py

It is intentionally dependency-free (only numpy + opencv-python) so it can be
used as the project's pre-flight check.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotations import BBox, ImageAnnotation, filter_boxes  # noqa: E402
from converters import load_yolo_dir_annotations  # noqa: E402
from dataset_config import DatasetConfig, FilterConfig  # noqa: E402
from pipeline import DatasetPipeline  # noqa: E402
from preprocess import ImageRecord  # noqa: E402
from splitter import _allocate, split_dataset  # noqa: E402
from utils import Reporter, ensure_dir, imwrite_unicode, read_json  # noqa: E402


def make_synthetic_image(index: int, width: int = 640, height: int = 480) -> np.ndarray:
    """Create a road-like scene with one plate-shaped bright rectangle."""
    rng = np.random.RandomState(1000 + index)
    image = np.full((height, width, 3), 70, dtype=np.uint8)
    image[:, :, 1] = 78
    image = cv2.add(image, (rng.rand(height, width, 3) * 25).astype(np.uint8))

    # A slightly rotated plate: Vietnamese plates are wider than tall.
    plate_w, plate_h = rng.randint(120, 220), rng.randint(45, 75)
    x = rng.randint(30, max(31, width - plate_w - 30))
    y = rng.randint(40, max(41, height - plate_h - 40))

    plate = np.full((plate_h, plate_w, 3), 245, dtype=np.uint8)
    cv2.rectangle(plate, (0, 0), (plate_w - 1, plate_h - 1), (255, 255, 255), 3)
    # Characters: a few dark vertical bars + a dot.
    count = rng.randint(5, 8)
    step = max(6, plate_w // (count + 1))
    for position in range(count):
        bar_x = step // 2 + position * step
        bar_w = max(3, plate_w // 40)
        cv2.rectangle(plate, (bar_x, plate_h // 5), (bar_x + bar_w, plate_h - plate_h // 5), (10, 10, 10), -1)
    cv2.circle(plate, (plate_w - 12, plate_h // 2), max(2, plate_h // 10), (10, 10, 10), -1)

    angle = float(rng.randint(-8, 9))
    center = (plate_w / 2.0, plate_h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(plate_h * sine + plate_w * cosine)
    new_h = int(plate_h * cosine + plate_w * sine)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    rotated = cv2.warpAffine(plate, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    paste_x = min(x, width - new_w - 1)
    paste_y = min(y, height - new_h - 1)
    image[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = rotated
    return image


def build_raw_folder(root: Path, count: int = 12) -> Path:
    """Write ``count`` synthetic images, duplicated once to exercise de-duplication."""
    raw = ensure_dir(root / "raw")
    # Two sub folders so the group-aware splitter has something to group.
    for index in range(count):
        folder = ensure_dir(raw / ("clip_a" if index % 2 == 0 else "clip_b"))
        image = make_synthetic_image(index)
        assert imwrite_unicode(folder / ("frame_{0:03d}.jpg".format(index)), image)
    # Exact duplicate of the very first frame.
    shutil.copy2(str(raw / "clip_a" / "frame_000.jpg"), str(raw / "clip_a" / "frame_000_copy.jpg"))
    return raw


def test_class_id_filtering() -> None:
    """Regression: labels must never reference a class outside the config.

    A YOLO label file may carry a ``class_id`` larger than the number of classes
    declared in ``data.yaml``. Such boxes must be dropped both on import and by
    the box filter so the produced dataset stays valid for Ultralytics (i.e. the
    ``nc`` in ``data.yaml`` matches the ids actually written to the ``.txt``).
    """
    workspace = Path(tempfile.mkdtemp(prefix="vn_plate_class_"))
    try:
        labels_dir = ensure_dir(workspace / "labels")
        # One image (100x50) with three boxes: valid id, out-of-range id, huge id.
        (labels_dir / "plate_000.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.4 0.4\n7 0.5 0.5 0.4 0.4\n",
            encoding="utf-8",
        )
        images_index = {"plate_000": ("plate_000.jpg", 100, 50)}
        imported = load_yolo_dir_annotations(
            labels_dir, images_index, ["license_plate"], Reporter(quiet=True)
        )
        entry = imported.find("plate_000", "plate_000")
        assert entry is not None, "yolo importer lost the label file"
        boxes, _ = entry
        kept_ids = [int(box.class_id) for box in boxes]
        assert kept_ids == [0], (
            "out-of-range class ids must be dropped on import, got {0}".format(kept_ids)
        )

        # ``filter_boxes`` must drop out-of-range ids when ``num_classes`` is set.
        candidates = [
            BBox(10, 10, 60, 35, 0),
            BBox(10, 10, 60, 35, 4),
        ]
        kept = filter_boxes(candidates, FilterConfig(), 100, 50, num_classes=1)
        assert [int(box.class_id) for box in kept] == [0], (
            "filter_boxes(num_classes=1) must keep only class 0"
        )

        # Without ``num_classes`` the historic geometry-only behaviour is kept.
        assert len(filter_boxes(candidates, FilterConfig(), 100, 50)) == 2, (
            "without num_classes no class filtering must happen"
        )
        print("CLASS-ID FILTERING TEST PASSED")
    finally:
        shutil.rmtree(str(workspace), ignore_errors=True)


def test_split_holds_out_eval_groups() -> None:
    """Regression: small datasets must still produce a held-out split.

    Before the fix ``_allocate`` sent *every* group to ``train`` whenever a
    bucket held fewer than three groups.  Combined with the default
    ``group_aware=True`` + ``stratify=True`` that meant a dataset built from
    only one or two source folders produced an empty ``val``/``test``; the
    writer then emitted ``val: images/train`` and the model was validated on the
    very frames it was trained on, making fine-tuning metrics meaningless.
    """
    import random as _random

    # Two groups, both ratios requested -> at least one group must leave train.
    assignment = _allocate(["group_a", "group_b"], 0.2, 0.2, _random.Random(0))
    assert assignment, "allocation must not be empty"
    assert assignment["group_a"] != "train" or assignment["group_b"] != "train", (
        "two groups with a val/test ratio must keep at least one group out of train"
    )
    # A single held-out slot with equal ratios goes to val (the split that
    # actually gates fine-tuning quality) rather than being wasted on test.
    assert list(assignment.values()).count("val") >= 1, (
        "the single held-out slot must prefer validation over test"
    )

    # One group is genuinely unsplittable.
    assert _allocate(["only_group"], 0.2, 0.2, _random.Random(0)) == {"only_group": "train"}

    # No evaluation ratio requested -> everything legitimately stays in train.
    none = _allocate(["g1", "g2", "g3"], 0.0, 0.0, _random.Random(0))
    assert set(none.values()) == {"train"}, "zero eval ratio must keep every group in train"
    print("SPLIT HOLDOUT TEST PASSED")


def test_split_stratified_two_plus_two() -> None:
    """2 positive + 2 negative groups must not collapse entirely into train."""
    records: list = []
    annotations: dict = {}

    def add(name: str, group: str, positive: bool) -> None:
        records.append(ImageRecord(name=name, group=group))
        boxes = [BBox(10.0, 10.0, 60.0, 35.0, 0)] if positive else []
        annotations[name] = ImageAnnotation(image=name, width=100, height=50, boxes=boxes)

    for index in range(2):
        add("pos_a_{0}".format(index), "pos_a", True)
        add("pos_b_{0}".format(index), "pos_b", True)
        add("neg_a_{0}".format(index), "neg_a", False)
        add("neg_b_{0}".format(index), "neg_b", False)

    config = DatasetConfig()
    config.split.stratify = True
    config.split.group_aware = True
    config.split.val_ratio = 0.2
    config.split.test_ratio = 0.2

    splits = split_dataset(records, annotations, config, Reporter(quiet=True))
    total = sum(len(names) for names in splits.values())
    assert total == len(records), "every record must land in exactly one split"
    assert splits.get("val"), "2 positive + 2 negative groups must still yield a val split"
    assert len(splits.get("train", [])) < total, (
        "the stratified split collapsed every image into train; val/test are empty"
    )
    # Stratification keeps both polarities represented in the held-out split.
    val_names = splits.get("val", [])
    assert any(name.startswith("pos_") for name in val_names), "no positive group in val"
    assert any(name.startswith("neg_") for name in val_names), "no negative group in val"
    print("STRATIFIED SMALL-SPLIT TEST PASSED")


def test_writer_wipes_stale_output() -> None:
    """Regression: a re-run must not keep stale dataset files behind.

    ``DatasetWriter.write`` used to only *add* files to ``output_dir``. A second
    run with fewer images (or a different split) therefore left the previous
    images/labels in place; Ultralytics globs the whole ``images``/``labels``
    tree, so those leftovers were silently trained on even though they were not
    listed in the current ``data.yaml``. The writer must now wipe every
    pipeline-owned folder (``images``/``labels``/``qc``) before copying.

    It must also refuse to run when ``output_dir`` overlaps the preprocessed
    source images, otherwise the wipe would delete the data it is about to copy.
    """
    from dataset_config import DatasetConfig
    from writer import DatasetWriter

    workspace = Path(tempfile.mkdtemp(prefix="vn_plate_write_"))
    try:
        work_images = ensure_dir(workspace / "work" / "images")
        keep_name = "keep_plate.jpg"
        assert imwrite_unicode(work_images / keep_name, make_synthetic_image(1, 320, 240))

        output = ensure_dir(workspace / "dataset")
        # Artefacts a hypothetical previous run would have left behind.
        stale_image = ensure_dir(output / "images" / "train") / "stale_plate.jpg"
        stale_image.write_bytes(b"old")
        stale_label = ensure_dir(output / "labels" / "train") / "stale_plate.txt"
        stale_label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        stale_qc = ensure_dir(output / "qc") / "preview.jpg"
        stale_qc.write_bytes(b"old")
        # A whole stale split the new run will not recreate.
        ensure_dir(output / "images" / "test")

        config = DatasetConfig()
        config.work_dir = str(workspace / "work")
        config.output_dir = str(output)
        config.classes = ["license_plate"]

        record = ImageRecord(
            source_path=str(work_images / keep_name),
            source_rel=keep_name,
            source_name=keep_name,
            source_stem="keep_plate",
            name=keep_name,
            width=320,
            height=240,
            out_width=320,
            out_height=240,
            group="root",
        )
        annotation = ImageAnnotation(
            image=keep_name,
            width=320,
            height=240,
            boxes=[BBox(10.0, 10.0, 100.0, 60.0, 0)],
        )
        splits = {"train": [keep_name], "val": [], "test": []}

        DatasetWriter(config, Reporter(quiet=True)).write(
            [record], {keep_name: annotation}, splits
        )

        assert (output / "images" / "train" / keep_name).exists(), "fresh image not written"
        assert not stale_image.exists(), "stale image survived a re-run"
        assert not stale_label.exists(), "stale label survived a re-run"
        assert not stale_qc.exists(), "stale QC preview survived a re-run"
        assert not (output / "images" / "test").exists(), (
            "a dropped split must not leave its folder (or files) behind"
        )
        assert (output / "data.yaml").exists(), "data.yaml missing"

        # Safety net: output_dir == work_dir would wipe the source images.
        overlapping = DatasetConfig()
        overlapping.work_dir = str(workspace / "work")
        overlapping.output_dir = str(workspace / "work")
        try:
            DatasetWriter(overlapping, Reporter(quiet=True)).write(
                [record], {keep_name: annotation}, splits
            )
        except ValueError:
            pass
        else:
            raise AssertionError("output_dir overlapping work_dir must be rejected")
        assert (work_images / keep_name).exists(), "source image must never be deleted"
        print("WRITER STALE-OUTPUT TEST PASSED")
    finally:
        shutil.rmtree(str(workspace), ignore_errors=True)


def test_preprocess_refuses_raw_work_overlap() -> None:
    """Regression: preprocessing must never wipe the user's original images.

    ``ImagePreprocessor.run`` wipes ``work_dir/images`` *before* it scans
    ``raw_dir``. When ``work_dir/images`` equals (or contains, or lives inside)
    ``raw_dir`` — e.g. ``raw_dir=data/raw/images`` with ``work_dir=data/raw`` —
    that wipe would delete the original images before ``list_images`` ever sees
    them, which is unrecoverable data loss. ``run`` must refuse to run and leave
    the files untouched, mirroring the guard already present in
    ``DatasetWriter.write``.
    """
    from preprocess import ImagePreprocessor

    workspace = Path(tempfile.mkdtemp(prefix="vn_plate_pre_overlap_"))
    try:
        raw_images = ensure_dir(workspace / "images")
        raw_image = raw_images / "plate_000.jpg"
        assert imwrite_unicode(raw_image, make_synthetic_image(0, 320, 240))

        # Case A: work_dir/images *is* raw_dir (raw_dir == workspace/images).
        config = DatasetConfig()
        config.raw_dir = str(raw_images)
        config.work_dir = str(workspace)  # -> work_dir/images == workspace/images
        try:
            ImagePreprocessor(config, Reporter(quiet=True)).run()
        except ValueError:
            pass
        else:
            raise AssertionError("work_dir/images == raw_dir must be rejected")
        assert raw_image.exists(), "the original raw image must never be deleted"

        # Case B: work_dir/images is nested *inside* raw_dir.
        nested_work = ensure_dir(workspace / "work")
        config_b = DatasetConfig()
        config_b.raw_dir = str(workspace)          # contains work/images
        config_b.work_dir = str(nested_work)       # -> workspace/work/images
        try:
            ImagePreprocessor(config_b, Reporter(quiet=True)).run()
        except ValueError:
            pass
        else:
            raise AssertionError("work_dir/images nested inside raw_dir must be rejected")
        assert raw_image.exists(), "the original raw image must never be deleted"

        # A genuinely separate work_dir must still be accepted.
        safe_work = ensure_dir(workspace / "safe")
        config_ok = DatasetConfig()
        config_ok.raw_dir = str(raw_images)
        config_ok.work_dir = str(safe_work)
        records = ImagePreprocessor(config_ok, Reporter(quiet=True)).run()
        assert records, "a non-overlapping work_dir must still preprocess normally"
        print("PREPROCESS OVERLAP TEST PASSED")
    finally:
        shutil.rmtree(str(workspace), ignore_errors=True)


def test_read_json_with_bom(tmp_path: "Path | None" = None) -> None:
    """Regression: JSON configs may be stored as UTF-8 *with* a BOM.

    ``utils.read_json`` used to open files with plain ``encoding="utf-8"`` while
    the shipped ``config.default.json`` (and configs re-saved by Windows
    tooling) start with a UTF-8 BOM. ``json.load`` then raised
    ``JSONDecodeError: Unexpected UTF-8 BOM`` and the CLI died while loading its
    default config, before the pipeline could run. Reading with ``utf-8-sig``
    must accept both BOM-prefixed and BOM-less UTF-8 files. Works as a pytest
    test (via the ``tmp_path`` fixture) and inside the standalone smoke runner.
    """
    owns_dir = tmp_path is None
    base = Path(tempfile.mkdtemp(prefix="vn_plate_jsonbom_")) if owns_dir else Path(tmp_path)
    try:
        with_bom = base / "config.json"
        with_bom.write_bytes(b'\xef\xbb\xbf{"raw_dir": "raw"}')
        assert read_json(with_bom)["raw_dir"] == "raw", "UTF-8 BOM was not stripped"

        without_bom = base / "plain.json"
        without_bom.write_bytes('{"raw_dir": "raw"}'.encode("utf-8"))
        assert read_json(without_bom)["raw_dir"] == "raw", "plain UTF-8 must keep working"

        # The real, shipped default config must also load despite its BOM.
        default_cfg = Path(__file__).resolve().parent.parent / "config.default.json"
        if default_cfg.exists():
            assert DatasetConfig.load(str(default_cfg)) is not None

        print("READ_JSON BOM TEST PASSED")
    finally:
        if owns_dir:
            shutil.rmtree(str(base), ignore_errors=True)


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="vn_plate_smoke_"))
    try:
        raw_dir = build_raw_folder(workspace)
        config = DatasetConfig()
        config.raw_dir = str(raw_dir)
        config.work_dir = str(workspace / "work")
        config.output_dir = str(workspace / "dataset")
        config.label_mode = "auto"
        config.preprocess.resize_mode = "letterbox"
        config.preprocess.image_size = 640
        config.split.val_ratio = 0.2
        config.split.test_ratio = 0.2
        config.preview_limit = 8
        config.validate()

        reporter = Reporter(quiet=True)
        summary = DatasetPipeline(config, reporter).run()

        output = Path(config.output_dir)
        assert (output / "data.yaml").exists(), "data.yaml missing"
        assert (output / "dataset_stats.json").exists(), "dataset_stats.json missing"
        assert (output / "DATASET_CARD.md").exists(), "DATASET_CARD.md missing"
        assert (output / "images" / "train").is_dir(), "train images folder missing"
        assert (output / "labels" / "train").is_dir(), "train labels folder missing"

        label_files = list((output / "labels").rglob("*.txt"))
        image_files = list((output / "images").rglob("*.jpg")) + list((output / "images").rglob("*.png"))
        assert label_files, "no label files produced"
        assert len(label_files) == len(image_files), "every image must have a label file"

        boxes = sum(
            1
            for label_file in label_files
            for line in label_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        assert boxes > 0, "the pseudo-labeller found no plate at all"

        # Every label line must be a valid YOLO row.
        for label_file in label_files:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                assert len(parts) >= 5, "malformed label line: {0}".format(line)
                values = [float(value) for value in parts[1:5]]
                assert all(0.0 <= value <= 1.0 for value in values), "unnormalised label: {0}".format(line)

        data_yaml = (output / "data.yaml").read_text(encoding="utf-8")
        assert "nc: 1" in data_yaml, "nc missing from data.yaml"
        assert "license_plate" in data_yaml, "class name missing from data.yaml"

        # The output labels must only ever reference a class declared in data.yaml.
        for label_file in label_files:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id = int(float(line.split()[0]))
                assert 0 <= class_id < 1, "label references undeclared class: {0}".format(line)

        test_class_id_filtering()
        test_split_holds_out_eval_groups()
        test_split_stratified_two_plus_two()
        test_writer_wipes_stale_output()
        test_preprocess_refuses_raw_work_overlap()
        test_read_json_with_bom()

        print("SMOKE TEST PASSED")
        print("  images  : {0}".format(len(image_files)))
        print("  boxes   : {0}".format(boxes))
        print("  splits  : {0}".format(summary.get("splits")))
        print("  dataset : {0}".format(output))
        return 0
    finally:
        shutil.rmtree(str(workspace), ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
