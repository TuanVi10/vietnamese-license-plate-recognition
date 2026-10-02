"""
main.py
=======
Command line entry point of the Vietnamese license-plate dataset builder.

Examples
--------
Full run with automatic pseudo-labelling::

    python main.py --raw-dir data/raw --output-dir dataset --label-mode auto

Import existing Pascal VOC / COCO / YOLO annotations::

    python main.py --raw-dir data/raw --labels-format voc --labels-path data/labels_voc

Merge both sources and review everything afterwards::

    python main.py --raw-dir data/raw --labels-format coco --labels-path data/annotations.json \
        --label-mode merge --stages labels write

Resume a long job after a crash (stages are independent and cached)::

    python main.py --config my_config.json --stages split write preview

Persist the effective configuration for full reproducibility::

    python main.py --raw-dir data/raw --save-config runs/run01.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dataset_config import DatasetConfig
from pipeline import STAGE_ORDER, DatasetPipeline
from utils import Reporter, write_json

_DEFAULT_CONFIG = Path(__file__).with_name("config.default.json")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="prepare_vn_plate_dataset",
        description=(
            "Prepare a YOLO-format dataset for Vietnamese license-plate detection:\n"
            "clean images, import or bootstrap labels, validate, split and export."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="", help="JSON config file to start from")
    parser.add_argument("--raw-dir", type=str, default="", help="folder with the raw images")
    parser.add_argument("--work-dir", type=str, default="", help="folder for intermediate artifacts")
    parser.add_argument("--output-dir", type=str, default="", help="folder of the final YOLO dataset")
    parser.add_argument("--dataset-name", type=str, default="", help="name used in the dataset card")

    parser.add_argument(
        "--labels-format",
        type=str,
        default="",
        choices=["", "none", "voc", "coco", "yolo"],
        help="format of the existing annotations",
    )
    parser.add_argument("--labels-path", type=str, default="", help="annotation folder (voc/yolo) or json (coco)")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="",
        choices=["", "import", "auto", "merge"],
        help="where the labels come from",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="comma separated class names, e.g. 'license_plate'",
    )

    parser.add_argument(
        "--resize-mode",
        type=str,
        default="",
        choices=["", "none", "letterbox", "stretch"],
        help="resize images before labelling (default: none)",
    )
    parser.add_argument("--image-size", type=int, default=0, help="target size for letterbox/stretch")
    parser.add_argument("--min-width", type=int, default=0, help="drop images narrower than this")
    parser.add_argument("--min-height", type=int, default=0, help="drop images shorter than this")
    parser.add_argument("--jpeg-quality", type=int, default=0, help="re-encode quality (1-100)")
    parser.add_argument("--no-dedupe", action="store_true", help="disable exact duplicate removal")
    parser.add_argument("--near-dedupe", action="store_true", help="also remove perceptual duplicates")
    parser.add_argument("--keep-unicode-names", action="store_true", help="keep Vietnamese file names")
    parser.add_argument("--limit", type=int, default=0, help="only use the first N raw images (0 = all)")

    parser.add_argument("--val-ratio", type=float, default=-1.0, help="validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=-1.0, help="test split ratio")
    parser.add_argument("--seed", type=int, default=-1, help="split seed")
    parser.add_argument("--no-stratify", action="store_true", help="disable stratified splitting")
    parser.add_argument("--no-group-aware", action="store_true", help="split per image instead of per folder")

    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep images without boxes as negative samples (default from config)",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="drop images that end up without any box",
    )

    parser.add_argument(
        "--stages",
        type=str,
        default="",
        help="space/comma separated subset of: {0}".format(", ".join(STAGE_ORDER)),
    )
    parser.add_argument("--save-config", type=str, default="", help="write the effective config to this path")
    parser.add_argument("--print-config", action="store_true", help="print the effective config and exit")
    parser.add_argument("--json-report", type=str, default="", help="write the final summary to this JSON file")
    parser.add_argument("--quiet", action="store_true", help="only print the final summary")
    return parser


def parse_stages(value: str) -> Optional[List[str]]:
    """Turn ``"labels, write"`` into ``["labels", "write"]`` (``None`` when empty)."""
    if not value:
        return None
    tokens = [token for token in value.replace(",", " ").split() if token]
    if tokens == ["all"]:
        return list(STAGE_ORDER)
    return tokens


def build_config(args: argparse.Namespace) -> DatasetConfig:
    """Merge the JSON config file, the CLI flags and the built-in defaults."""
    config_path = args.config or (str(_DEFAULT_CONFIG) if _DEFAULT_CONFIG.exists() else "")
    if config_path and Path(config_path).exists():
        config = DatasetConfig.load(config_path)
    else:
        config = DatasetConfig()

    if args.raw_dir:
        config.raw_dir = args.raw_dir
    if args.work_dir:
        config.work_dir = args.work_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.dataset_name:
        config.dataset_name = args.dataset_name

    if args.labels_format:
        config.labels_format = args.labels_format
    if args.labels_path:
        config.labels_path = args.labels_path
    if args.label_mode:
        config.label_mode = args.label_mode
    if args.classes:
        config.classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    # Sensible defaults: imported labels imply "import", everything else "auto".
    if not args.label_mode:
        if config.labels_format != "none" and config.labels_path:
            config.label_mode = "import"
        elif config.label_mode == "import":
            config.label_mode = "auto"

    if args.resize_mode:
        config.preprocess.resize_mode = args.resize_mode
    if args.image_size > 0:
        config.preprocess.image_size = args.image_size
    if args.min_width > 0:
        config.preprocess.min_width = args.min_width
    if args.min_height > 0:
        config.preprocess.min_height = args.min_height
    if args.jpeg_quality > 0:
        config.preprocess.jpeg_quality = max(1, min(100, args.jpeg_quality))
    if args.no_dedupe:
        config.preprocess.dedupe_exact = False
    if args.near_dedupe:
        config.preprocess.dedupe_near = True
    if args.keep_unicode_names:
        config.preprocess.keep_unicode_names = True
    if args.limit:
        config.preprocess.limit = args.limit

    if args.val_ratio >= 0.0:
        config.split.val_ratio = args.val_ratio
    if args.test_ratio >= 0.0:
        config.split.test_ratio = args.test_ratio
    if args.seed >= 0:
        config.split.seed = args.seed
    if args.no_stratify:
        config.split.stratify = False
    if args.no_group_aware:
        config.split.group_aware = False

    if args.keep_empty:
        config.filter.keep_empty_images = True
    if args.drop_empty:
        config.filter.keep_empty_images = False

    config.validate()
    return config


def format_summary(summary: Dict[str, Any]) -> str:
    """Render the pipeline summary as readable text."""
    lines: List[str] = []
    lines.append("=" * 66)
    lines.append("DATASET SUMMARY - {0}".format(summary.get("dataset_name", "")))
    lines.append("=" * 66)
    lines.append("raw_dir    : {0}".format(summary.get("raw_dir", "")))
    lines.append("work_dir   : {0}".format(summary.get("work_dir", "")))
    lines.append("output_dir : {0}".format(summary.get("output_dir", "")))
    lines.append(
        "images     : {0} kept / {1} total ({2} dropped)".format(
            summary.get("images_kept", 0),
            summary.get("images_total", 0),
            summary.get("images_dropped", 0),
        )
    )
    lines.append("boxes      : {0}".format(summary.get("boxes_total", 0)))
    splits = summary.get("splits") or {}
    if splits:
        lines.append("splits:")
        for name in ("train", "val", "test"):
            if name not in splits:
                continue
            item = splits[name]
            lines.append(
                "  - {0:<5s}: {1:>6d} images | {2:>7d} boxes | {3:>6d} with plate".format(
                    name,
                    int(item.get("images", 0)),
                    int(item.get("boxes", 0)),
                    int(item.get("images_with_boxes", 0)),
                )
            )
    written = summary.get("written") or {}
    if written:
        lines.append("written:")
        for key, value in sorted(written.items()):
            lines.append("  - {0}: {1}".format(key, value))
    lines.append("=" * 66)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse the CLI, run the pipeline and print/write the summary."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = build_config(args)
    except ValueError as error:
        print("[ERROR] {0}".format(error), file=sys.stderr)
        return 2

    if args.print_config:
        print(config.describe())
        if args.save_config:
            print("saved -> {0}".format(config.save(args.save_config)))
        return 0

    if args.save_config:
        config.save(args.save_config)

    reporter = Reporter(quiet=args.quiet)
    reporter.info("configuration:\n{0}".format(config.describe()))

    stages = parse_stages(args.stages)
    try:
        pipeline = DatasetPipeline(config, reporter)
        summary = pipeline.run(stages)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print("[ERROR] {0}".format(error), file=sys.stderr)
        return 1

    print(format_summary(summary))
    if args.json_report:
        write_json(args.json_report, {"summary": summary, "logs": reporter.as_dict()})
        print("json report -> {0}".format(args.json_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
