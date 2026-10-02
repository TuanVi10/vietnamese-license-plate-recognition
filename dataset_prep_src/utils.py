"""
utils.py
========
Reusable, dependency-light helpers shared by every stage of the Vietnamese
license-plate dataset preparation pipeline.

The module only relies on the Python standard library, ``numpy`` and
``opencv-python`` because that is the environment this project targets
(no PIL, no ultralytics, no PyYAML).

Sections
--------
1. Reporting                : :class:`Reporter`
2. Filesystem helpers       : ensure_dir, list_images, read_json, write_json
3. Unicode-safe image IO    : imread_unicode, imwrite_unicode
4. Image resizing           : letterbox_resize, stretch_resize, resize_image
5. Geometry                 : clip_xyxy, box_*, iou_xyxy, nms_xyxy, GeoTransform
6. Hashing / de-duplication : md5_of_file, phash, hamming_distance
7. Naming helpers           : ascii_slug, sanitize_filename, unique_stem
8. Misc                     : human_bytes, guess_group
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #
Box = Tuple[float, float, float, float]

IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".ppm",
)


# --------------------------------------------------------------------------- #
# 1. Reporting
# --------------------------------------------------------------------------- #
class Reporter:
    """Tiny stdout reporter so every pipeline stage logs in a consistent way."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = bool(quiet)
        self.events: List[Tuple[str, str]] = []

    def _emit(self, level: str, message: str) -> None:
        self.events.append((level, message))
        if not self.quiet:
            print("[{0:5s}] {1}".format(level, message), flush=True)

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warn(self, message: str) -> None:
        self._emit("WARN", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    def stage(self, message: str) -> None:
        self._emit("STAGE", message)

    def as_dict(self) -> Dict[str, List[str]]:
        """Return the accumulated log grouped by level."""
        grouped: Dict[str, List[str]] = {}
        for level, message in self.events:
            grouped.setdefault(level, []).append(message)
        return grouped


# --------------------------------------------------------------------------- #
# 2. Filesystem helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: Any) -> Path:
    """Create ``path`` (and its parents) if needed and return it as a ``Path``."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def list_images(root: Any, recursive: bool = True) -> List[Path]:
    """Return every image file below ``root``, sorted case-insensitively."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    iterator = root_path.rglob("*") if recursive else root_path.glob("*")
    files = [
        item
        for item in iterator
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda item: str(item).lower())
    return files


def paths_overlap(first: Any, second: Any) -> bool:
    """Return ``True`` when two paths are equal or one contains the other.

    Safety net shared by every stage that calls :func:`clean_dir` on a
    pipeline-owned folder. Both paths are ``~``-expanded and resolved so that
    relative paths, ``..`` segments and symlinks cannot hide an overlap: a
    misconfigured ``work_dir`` must never be allowed to delete the raw images
    it is about to read (see :class:`preprocess.ImagePreprocessor` and
    :class:`writer.DatasetWriter`).
    """
    first_parts = Path(first).expanduser().resolve().parts
    second_parts = Path(second).expanduser().resolve().parts
    shortest = min(len(first_parts), len(second_parts))
    return first_parts[:shortest] == second_parts[:shortest]


def read_json(path: Any, default: Any = None) -> Any:
    """Read a JSON file; return ``default`` when the file does not exist."""
    file_path = Path(path)
    if not file_path.exists():
        return default
    # ``utf-8-sig`` transparently accepts BOM-less UTF-8 *and* strips the UTF-8
    # BOM that Windows tooling / editors prepend to JSON files (e.g. the shipped
    # ``config.default.json``). Plain ``"utf-8"`` would make ``json.load`` raise
    # ``JSONDecodeError: Unexpected UTF-8 BOM`` and abort the CLI at start-up.
    with file_path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Any, data: Any, indent: int = 2) -> Path:
    """Write ``data`` as UTF-8 JSON (``ensure_ascii=False`` keeps Vietnamese text)."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False, default=_json_default)
        handle.write("\n")
    return file_path


def _json_default(value: Any) -> Any:
    """Fallback encoder: numpy scalars/arrays and Paths become plain Python types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def clean_dir(path: Any) -> Path:
    """Remove every file (recursively) inside ``path`` but keep the folder itself."""
    folder = ensure_dir(path)
    for entry in sorted(folder.rglob("*"), reverse=True):
        try:
            if entry.is_file() or entry.is_symlink():
                entry.unlink()
            elif entry.is_dir():
                entry.rmdir()
        except OSError:
            # A locked/protected entry is simply left in place; the next stage
            # never relies on the folder being 100% empty.
            continue
    return folder


# --------------------------------------------------------------------------- #
# 3. Unicode-safe image IO
#    cv2.imread / cv2.imwrite fail on non-ASCII paths under Windows; the
#    imdecode/imencode + numpy tofile/fromfile route always works.
# --------------------------------------------------------------------------- #
def imread_unicode(path: Any, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image from a possibly non-ASCII path. Returns ``None`` on failure."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buffer.size == 0:
        return None
    try:
        image = cv2.imdecode(buffer, flags)
    except cv2.error:
        return None
    if image is None:
        return None
    if flags == cv2.IMREAD_COLOR and image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def imwrite_unicode(path: Any, image: np.ndarray, quality: int = 95) -> bool:
    """Write ``image`` to a possibly non-ASCII path. Returns ``True`` on success."""
    file_path = Path(path)
    extension = file_path.suffix.lower() or ".jpg"
    if extension in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif extension == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    elif extension == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
    else:
        params = []
    try:
        ok, buffer = cv2.imencode(extension, image, params)
    except cv2.error:
        return False
    if not ok:
        return False
    ensure_dir(file_path.parent)
    try:
        buffer.tofile(str(file_path))
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------- #
# 4. Image resizing
# --------------------------------------------------------------------------- #
@dataclass
class GeoTransform:
    """Per-axis affine map (scale + translation) between two coordinate frames.

    ``forward`` maps *source* image coordinates to *destination* image
    coordinates, exactly like the letterbox used by YOLO at training time.
    """

    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0

    def forward_point(self, x: float, y: float) -> Tuple[float, float]:
        return (float(x) * self.scale_x + self.pad_x, float(y) * self.scale_y + self.pad_y)

    def inverse_point(self, x: float, y: float) -> Tuple[float, float]:
        scale_x = self.scale_x if abs(self.scale_x) > 1e-12 else 1.0
        scale_y = self.scale_y if abs(self.scale_y) > 1e-12 else 1.0
        return ((float(x) - self.pad_x) / scale_x, (float(y) - self.pad_y) / scale_y)

    def forward_box(self, box: Box) -> Box:
        """Map a source ``(x1, y1, x2, y2)`` box into destination coordinates."""
        x1, y1 = self.forward_point(box[0], box[1])
        x2, y2 = self.forward_point(box[2], box[3])
        return (x1, y1, x2, y2)

    def inverse_box(self, box: Box) -> Box:
        """Map a destination ``(x1, y1, x2, y2)`` box back into source coordinates."""
        x1, y1 = self.inverse_point(box[0], box[1])
        x2, y2 = self.inverse_point(box[2], box[3])
        return (x1, y1, x2, y2)

    def is_identity(self) -> bool:
        return (
            abs(self.scale_x - 1.0) < 1e-9
            and abs(self.scale_y - 1.0) < 1e-9
            and abs(self.pad_x) < 1e-9
            and abs(self.pad_y) < 1e-9
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "scale_x": float(self.scale_x),
            "scale_y": float(self.scale_y),
            "pad_x": float(self.pad_x),
            "pad_y": float(self.pad_y),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GeoTransform":
        data = data or {}
        return cls(
            scale_x=float(data.get("scale_x", 1.0)),
            scale_y=float(data.get("scale_y", 1.0)),
            pad_x=float(data.get("pad_x", 0.0)),
            pad_y=float(data.get("pad_y", 0.0)),
        )


def normalise_size(target_size: Any) -> Tuple[int, int]:
    """Normalise ``target_size`` to an explicit ``(width, height)`` tuple.

    Accepts an ``int`` (square output) or a 2-item sequence interpreted as
    ``(width, height)``.
    """
    if isinstance(target_size, (int, np.integer)):
        side = int(target_size)
        return side, side
    values = list(target_size)
    if len(values) != 2:
        raise ValueError("target_size must be an int or a (width, height) pair")
    return int(values[0]), int(values[1])


def letterbox_resize(
    image: np.ndarray,
    target_size: Any,
    color: Sequence[int] = (114, 114, 114),
    scale_up: bool = False,
) -> Tuple[np.ndarray, GeoTransform]:
    """Resize keeping the aspect ratio and pad with ``color`` to ``target_size``.

    Returns the padded image together with the :class:`GeoTransform` that maps
    original coordinates onto the padded (output) coordinates.
    """
    if image is None:
        raise ValueError("letterbox_resize() received None")
    new_w, new_h = normalise_size(target_size)
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise ValueError("letterbox_resize() received a degenerate image")

    ratio = min(new_w / float(src_w), new_h / float(src_h))
    if not scale_up:
        ratio = min(ratio, 1.0)

    inner_w = max(1, int(round(src_w * ratio)))
    inner_h = max(1, int(round(src_h * ratio)))
    interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (inner_w, inner_h), interpolation=interpolation)

    pad_w = new_w - inner_w
    pad_h = new_h - inner_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    if pad_w > 0 or pad_h > 0:
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=tuple(int(channel) for channel in color),
        )
    else:
        padded = resized

    transform = GeoTransform(
        scale_x=float(inner_w) / float(src_w),
        scale_y=float(inner_h) / float(src_h),
        pad_x=float(left),
        pad_y=float(top),
    )
    return padded, transform


def stretch_resize(image: np.ndarray, target_size: Any) -> Tuple[np.ndarray, GeoTransform]:
    """Resize ignoring the aspect ratio (fast but distorts the plate shape)."""
    if image is None:
        raise ValueError("stretch_resize() received None")
    new_w, new_h = normalise_size(target_size)
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise ValueError("stretch_resize() received a degenerate image")
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    transform = GeoTransform(
        scale_x=float(new_w) / float(src_w),
        scale_y=float(new_h) / float(src_h),
    )
    return resized, transform


def resize_image(
    image: np.ndarray,
    target_size: Any,
    mode: str = "none",
    color: Sequence[int] = (114, 114, 114),
    scale_up: bool = False,
) -> Tuple[np.ndarray, GeoTransform]:
    """Resize ``image`` according to ``mode`` and return ``(image, transform)``.

    Supported modes: ``"none"`` (identity), ``"letterbox"``, ``"stretch"``.
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return image, GeoTransform()
    if mode == "letterbox":
        return letterbox_resize(image, target_size, color=color, scale_up=scale_up)
    if mode == "stretch":
        return stretch_resize(image, target_size)
    raise ValueError("unknown resize mode: {0!r}".format(mode))


# --------------------------------------------------------------------------- #
# 5. Geometry
# --------------------------------------------------------------------------- #
def clip_xyxy(box: Sequence[float], width: float, height: float) -> Box:
    """Clamp ``box`` to the ``[0, width] x [0, height]`` image rectangle."""
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return (x1, y1, x2, y2)


def box_width(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0]))


def box_height(box: Sequence[float]) -> float:
    return max(0.0, float(box[3]) - float(box[1]))


def box_area(box: Sequence[float]) -> float:
    return box_width(box) * box_height(box)


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def iou_xyxy(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    inter_x1 = max(float(first[0]), float(second[0]))
    inter_y1 = max(float(first[1]), float(second[1]))
    inter_x2 = min(float(first[2]), float(second[2]))
    inter_y2 = min(float(first[3]), float(second[3]))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = box_area(first) + box_area(second) - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def nms_xyxy(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float = 0.5,
) -> List[int]:
    """Greedy non-maximum suppression; returns the indices of the kept boxes."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda index: float(scores[index]), reverse=True)
    keep: List[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        remaining: List[int] = []
        for index in order:
            if iou_xyxy(boxes[current], boxes[index]) <= iou_threshold:
                remaining.append(index)
        order = remaining
    return keep


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile (``q`` in ``[0, 100]``) that is safe on empty input."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.percentile(array, q))


# --------------------------------------------------------------------------- #
# 6. Hashing / de-duplication
# --------------------------------------------------------------------------- #
def md5_of_file(path: Any, chunk_size: int = 1 << 20) -> str:
    """MD5 of the raw file bytes (used for exact duplicate detection)."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """DCT-based perceptual hash packed into a Python ``int``.

    Images that differ only by compression, light blur or a small crop produce
    hashes with a small Hamming distance, which makes them bad training samples
    when they are near-duplicates.
    """
    if image is None or image.size == 0:
        return 0
    size = int(hash_size) * int(highfreq_factor)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray = np.float32(gray)
    try:
        dct = cv2.dct(gray)
    except cv2.error:
        return 0
    low = dct[:hash_size, :hash_size]
    median = float(np.median(low))
    value = 0
    for bit in (low > median).flatten().tolist():
        value = (value << 1) | (1 if bit else 0)
    return value


def hamming_distance(first: int, second: int) -> int:
    """Number of differing bits between two perceptual hashes."""
    return int(bin(int(first) ^ int(second)).count("1"))


# --------------------------------------------------------------------------- #
# 7. Naming helpers
# --------------------------------------------------------------------------- #
def ascii_slug(text: Any, max_length: int = 100) -> str:
    """Fold Vietnamese diacritics away and produce an ASCII ``snake_case`` slug."""
    folded = unicodedata.normalize("NFKD", str(text))
    folded = folded.encode("ascii", "ignore").decode("ascii")
    folded = re.sub(r"[^A-Za-z0-9]+", "_", folded).strip("_")
    if not folded:
        folded = "item"
    return folded[:max_length]


def sanitize_filename(name: Any, keep_unicode: bool = True, max_length: int = 120) -> str:
    """Strip characters that are illegal in file names while keeping the extension."""
    path = Path(str(name))
    stem = path.stem
    suffix = path.suffix.lower()
    if keep_unicode:
        cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", stem)
    else:
        cleaned = ascii_slug(stem, max_length)
    cleaned = cleaned.strip("._") or "image"
    return "{0}{1}".format(cleaned[:max_length], suffix)


def unique_stem(stem: str, used: set) -> str:
    """Return ``stem`` or ``stem_<n>`` so that no two images share a name."""
    if stem not in used:
        used.add(stem)
        return stem
    index = 1
    while "{0}_{1}".format(stem, index) in used:
        index += 1
    candidate = "{0}_{1}".format(stem, index)
    used.add(candidate)
    return candidate


# --------------------------------------------------------------------------- #
# 8. Misc
# --------------------------------------------------------------------------- #
def human_bytes(size: float) -> str:
    """Format a byte count in a compact human readable way."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return "{0:.1f}{1}".format(value, unit)
        value /= 1024.0
    return "{0:.1f}TB".format(value)


def guess_group(relative_path: Path) -> str:
    """Group images that most likely come from the same source (folder / video)."""
    parent = relative_path.parent.as_posix()
    if parent in (".", ""):
        return "root"
    return parent
