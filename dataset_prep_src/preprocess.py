"""
preprocess.py
=============
Stage 1 of the pipeline: turn a pile of raw images into a clean, deterministic,
uniform set of files that every later stage can rely on.

Responsibilities
----------------
* walk the raw folder (any depth) and decode every supported image;
* drop unreadable files, images that are too small and (optionally) duplicates
  – both exact (MD5) and perceptual (pHASH);
* give every surviving image a stable, ASCII-safe, collision-free name;
* optionally re-size (letterbox / stretch) to a fixed network input size while
  recording the exact geometric transform, so labels can be re-mapped later;
* persist a JSON manifest describing every decision that was made.

The manifest is the contract between this stage and the rest of the pipeline
(:mod:`converters`, :mod:`autolabel`, :mod:`validator`, :mod:`splitter`,
:mod:`writer`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from dataset_config import DatasetConfig
from utils import (
    GeoTransform,
    Reporter,
    ascii_slug,
    clean_dir,
    ensure_dir,
    guess_group,
    hamming_distance,
    human_bytes,
    imread_unicode,
    imwrite_unicode,
    list_images,
    md5_of_file,
    paths_overlap,
    phash,
    read_json,
    resize_image,
    sanitize_filename,
    unique_stem,
    write_json,
)

MANIFEST_VERSION = 1
_NORMALISED_SUFFIXES = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png", ".bmp": ".png", ".webp": ".webp", ".tif": ".png", ".tiff": ".png", ".ppm": ".png"}


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
@dataclass
class ImageRecord:
    """Everything the pipeline knows about one raw image."""

    source_path: str = ""
    source_rel: str = ""
    source_name: str = ""
    source_stem: str = ""
    name: str = ""
    width: int = 0
    height: int = 0
    out_width: int = 0
    out_height: int = 0
    transform: GeoTransform = field(default_factory=GeoTransform)
    group: str = "root"
    md5: str = ""
    phash: int = 0
    dropped: bool = False
    duplicate_of: str = ""
    issues: List[str] = field(default_factory=list)

    @property
    def relative_path(self) -> Path:
        return Path(self.source_rel)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_rel": self.source_rel,
            "source_name": self.source_name,
            "source_stem": self.source_stem,
            "name": self.name,
            "width": int(self.width),
            "height": int(self.height),
            "out_width": int(self.out_width),
            "out_height": int(self.out_height),
            "transform": self.transform.to_dict(),
            "group": self.group,
            "md5": self.md5,
            "phash": int(self.phash),
            "dropped": bool(self.dropped),
            "duplicate_of": self.duplicate_of,
            "issues": list(self.issues),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageRecord":
        return cls(
            source_path=str(data.get("source_path", "")),
            source_rel=str(data.get("source_rel", "")),
            source_name=str(data.get("source_name", "")),
            source_stem=str(data.get("source_stem", "")),
            name=str(data.get("name", "")),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            out_width=int(data.get("out_width", 0)),
            out_height=int(data.get("out_height", 0)),
            transform=GeoTransform.from_dict(data.get("transform")),
            group=str(data.get("group", "root")),
            md5=str(data.get("md5", "")),
            phash=int(data.get("phash", 0)),
            dropped=bool(data.get("dropped", False)),
            duplicate_of=str(data.get("duplicate_of", "")),
            issues=list(data.get("issues", [])),
        )


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
class ImagePreprocessor:
    """Clean / normalise the raw image folder and write ``work/images``."""

    def __init__(self, config: DatasetConfig, reporter: Optional[Reporter] = None) -> None:
        self.config = config
        self.cfg = config.preprocess
        self.reporter = reporter or Reporter(quiet=True)
        self.images_dir = Path(config.work_dir) / "images"
        self.manifest_path = Path(config.work_dir) / "manifest.json"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> List[ImageRecord]:
        """Execute the whole stage and return every record (kept + dropped)."""
        self.config.validate()

        # ``clean_dir`` wipes ``work_dir/images`` *before* the raw folder is
        # read. If that folder is the raw folder itself (or contains/lives
        # inside it) the wipe would delete the user's original images before
        # ``list_images`` ever sees them — unrecoverable data loss. Refuse to
        # run in that case, mirroring the guard in ``DatasetWriter.write``.
        raw_root = Path(self.config.raw_dir)
        if paths_overlap(raw_root, self.images_dir):
            raise ValueError(
                "work_dir/images '{0}' overlaps raw_dir '{1}'; refusing to "
                "clean it before preprocessing, as that would delete the "
                "original images. Configure work_dir so it lives outside "
                "raw_dir.".format(self.images_dir.resolve(), raw_root.resolve())
            )

        ensure_dir(self.images_dir)
        clean_dir(self.images_dir)

        sources = list_images(self.config.raw_dir, recursive=True)
        if self.cfg.limit and int(self.cfg.limit) > 0:
            sources = sources[: int(self.cfg.limit)]
        if not sources:
            raise FileNotFoundError(
                "no images found in raw_dir={0!r}".format(self.config.raw_dir)
            )
        self.reporter.info(
            "preprocess: {0} candidate image(s) found in {1}".format(len(sources), self.config.raw_dir)
        )

        records: List[ImageRecord] = []
        used_names: set = set()
        seen_md5: Dict[str, str] = {}
        kept_hashes: List[Tuple[int, str]] = []

        for index, path in enumerate(sources, start=1):
            record = self._process_one(path, index, used_names, seen_md5, kept_hashes)
            records.append(record)

        self.save_manifest(records)
        kept = [record for record in records if not record.dropped]
        dropped = len(records) - len(kept)
        self.reporter.info(
            "preprocess: kept {0} image(s), dropped {1} (duplicates/invalid)".format(len(kept), dropped)
        )
        return records

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _relative(self, path: Path) -> Path:
        try:
            return path.relative_to(Path(self.config.raw_dir))
        except ValueError:
            return Path(path.name)

    def _process_one(
        self,
        path: Path,
        index: int,
        used_names: set,
        seen_md5: Dict[str, str],
        kept_hashes: List[Tuple[int, str]],
    ) -> ImageRecord:
        cfg = self.cfg
        relative = self._relative(path)
        record = ImageRecord(
            source_path=str(path),
            source_rel=relative.as_posix(),
            source_name=path.name,
            source_stem=path.stem,
            group=guess_group(relative),
        )

        # --- exact duplicates ----------------------------------------- #
        if cfg.dedupe_exact:
            try:
                digest = md5_of_file(path)
            except OSError:
                record.dropped = True
                record.issues.append("unreadable")
                return record
            record.md5 = digest
            known = seen_md5.get(digest)
            if known is not None:
                record.dropped = True
                record.duplicate_of = known
                record.issues.append("exact_duplicate")
                return record

        # --- decode ---------------------------------------------------- #
        image = imread_unicode(path, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            record.dropped = True
            record.issues.append("decode_failed")
            return record

        height, width = image.shape[:2]
        record.width = int(width)
        record.height = int(height)

        if width < int(cfg.min_width) or height < int(cfg.min_height):
            record.dropped = True
            record.issues.append("too_small:{0}x{1}".format(width, height))
            return record

        if cfg.dedupe_exact:
            seen_md5.setdefault(record.md5, relative.as_posix())

        # --- near duplicates ------------------------------------------- #
        if cfg.dedupe_near:
            digest = phash(image)
            record.phash = int(digest)
            duplicate = self._find_near_duplicate(digest, kept_hashes)
            if duplicate is not None:
                record.dropped = True
                record.duplicate_of = duplicate
                record.issues.append("near_duplicate")
                return record
            kept_hashes.append((int(digest), relative.as_posix()))

        # --- naming ---------------------------------------------------- #
        suffix = _NORMALISED_SUFFIXES.get(path.suffix.lower(), ".jpg")
        if cfg.keep_unicode_names:
            base = Path(sanitize_filename(path.name, keep_unicode=True)).stem
        else:
            base = ascii_slug(relative.stem, max_length=100)
        stem = unique_stem(base or "img_{0:06d}".format(index), used_names)
        name = "{0}{1}".format(stem, suffix)

        # --- resize + write -------------------------------------------- #
        try:
            output, transform = resize_image(
                image,
                cfg.image_size,
                mode=cfg.resize_mode,
                color=cfg.letterbox_color,
                scale_up=cfg.scale_up,
            )
        except (ValueError, cv2.error) as exc:
            record.dropped = True
            record.issues.append("resize_failed:{0}".format(exc))
            return record

        destination = self.images_dir / name
        if not imwrite_unicode(destination, output, quality=int(cfg.jpeg_quality)):
            record.dropped = True
            record.issues.append("write_failed")
            return record

        record.name = name
        record.out_width = int(output.shape[1])
        record.out_height = int(output.shape[0])
        record.transform = transform
        return record

    def _find_near_duplicate(
        self, digest: int, kept_hashes: List[Tuple[int, str]]
    ) -> Optional[str]:
        if not kept_hashes:
            return None
        if len(kept_hashes) > int(self.cfg.max_compare):
            return None
        threshold = int(self.cfg.phash_hamming_threshold)
        for value, label in kept_hashes:
            if hamming_distance(digest, value) <= threshold:
                return label
        return None

    # ------------------------------------------------------------------ #
    # Manifest
    # ------------------------------------------------------------------ #
    def manifest_payload(self, records: List[ImageRecord]) -> Dict[str, Any]:
        kept = [record for record in records if not record.dropped]
        total_bytes = 0
        for record in kept:
            file_path = self.images_dir / record.name
            if file_path.exists():
                total_bytes += file_path.stat().st_size
        return {
            "version": MANIFEST_VERSION,
            "raw_dir": str(self.config.raw_dir),
            "images_dir": str(self.images_dir),
            "resize_mode": self.cfg.resize_mode,
            "image_size": int(self.cfg.image_size),
            "source_count": len(records),
            "kept_count": len(kept),
            "dropped_count": len(records) - len(kept),
            "output_bytes": total_bytes,
            "output_bytes_human": human_bytes(total_bytes),
            "records": [record.to_dict() for record in records],
        }

    def save_manifest(self, records: List[ImageRecord]) -> Path:
        payload = self.manifest_payload(records)
        write_json(self.manifest_path, payload)
        self.reporter.info(
            "preprocess: manifest written to {0} ({1})".format(
                self.manifest_path, payload["output_bytes_human"]
            )
        )
        return self.manifest_path

    @staticmethod
    def load_manifest(path: Any) -> Tuple[List[ImageRecord], Dict[str, Any]]:
        """Load the manifest produced by :meth:`save_manifest`."""
        payload = read_json(path, default=None)
        if not payload:
            raise FileNotFoundError(
                "manifest not found at {0}; run the 'preprocess' stage first".format(path)
            )
        records = [ImageRecord.from_dict(item) for item in payload.get("records", [])]
        return records, payload


def preprocess_dataset(config: DatasetConfig, reporter: Optional[Reporter] = None) -> List[ImageRecord]:
    """Convenience wrapper around :class:`ImagePreprocessor` for scripting."""
    return ImagePreprocessor(config, reporter).run()
