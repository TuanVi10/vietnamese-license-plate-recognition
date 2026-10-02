"""
converters.py
=============
Importers that turn already-existing annotations into the single internal
representation used by the pipeline (``list[BBox]`` in *source image* pixels).

Supported formats
-----------------
* ``voc``  - one Pascal VOC ``*.xml`` per image (LabelImg / CVAT export).
* ``coco`` - a single COCO ``instances_*.json`` file.
* ``yolo`` - one ``*.txt`` per image, ``class cx cy w h`` normalised.

All importers return an :class:`ImportedLabels` object keyed by file name, so a
later stage can attach the boxes to the matching image record and re-map them
onto the re-sized output image.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from annotations import BBox, load_yolo_labels
from utils import Reporter, read_json


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class ImportedLabels:
    """Boxes imported from an external annotation format.

    ``boxes_by_name`` is keyed by *file name*; the stem is registered as well so
    a lookup can try both spellings (annotation tools are inconsistent).
    """

    source: str = "unknown"
    boxes_by_name: Dict[str, List[BBox]] = field(default_factory=dict)
    sizes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    class_names: List[str] = field(default_factory=list)
    skipped_classes: List[str] = field(default_factory=list)
    missing_size: List[str] = field(default_factory=list)

    # -- registration / lookup ------------------------------------------ #
    def add(self, key: str, boxes: List[BBox], size: Optional[Tuple[int, int]] = None) -> None:
        """Register boxes (and the declared image size) under a file name + stem."""
        self.boxes_by_name[key] = boxes
        self.boxes_by_name.setdefault(Path(key).stem, boxes)
        if size is not None:
            self.sizes[key] = size
            self.sizes.setdefault(Path(key).stem, size)

    def find(self, name: str, stem: str = "") -> Optional[Tuple[List[BBox], Optional[Tuple[int, int]]]]:
        """Look up boxes by file name, then by stem. Returns ``None`` if absent."""
        for key in (name, stem):
            if key and key in self.boxes_by_name:
                return self.boxes_by_name[key], self.sizes.get(key)
        return None

    def __len__(self) -> int:
        return len(self.boxes_by_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "class_names": list(self.class_names),
            "images": {
                key: [box.to_dict() for box in boxes]
                for key, boxes in self.boxes_by_name.items()
            },
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_images_index(records: Iterable[Any]) -> Dict[str, Tuple[str, int, int]]:
    """Build ``{key: (source_name, width, height)}`` from the preprocess records.

    Both the full file name and the stem are registered so an annotation key
    matches regardless of the extension used by the annotation tool.
    """
    index: Dict[str, Tuple[str, int, int]] = {}
    for record in records:
        source_name = getattr(record, "source_name", "") or Path(getattr(record, "source_path", "")).name
        stem = getattr(record, "source_stem", "") or Path(source_name).stem
        width = int(getattr(record, "width", 0) or 0)
        height = int(getattr(record, "height", 0) or 0)
        if source_name:
            index[source_name] = (source_name, width, height)
        if stem:
            index.setdefault(stem, (source_name, width, height))
    return index


def _class_id(class_names: Sequence[str], name: str, fallback: int = 0) -> Optional[int]:
    """Map a class name to its index, or ``None`` when the name is unknown."""
    cleaned = str(name).strip()
    for index, candidate in enumerate(class_names):
        if candidate.strip().lower() == cleaned.lower():
            return index
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _xml_text(node: Optional[ET.Element], default: str = "") -> str:
    if node is None or node.text is None:
        return default
    return node.text.strip()


# --------------------------------------------------------------------------- #
# Pascal VOC
# --------------------------------------------------------------------------- #
def load_voc_annotations(
    xml_dir: Any,
    class_names: Sequence[str],
    reporter: Optional[Reporter] = None,
) -> ImportedLabels:
    """Read every ``*.xml`` below ``xml_dir`` (recursively)."""
    reporter = reporter or Reporter(quiet=True)
    imported = ImportedLabels(source="voc", class_names=list(class_names))
    root = Path(xml_dir)
    xml_files = sorted(root.rglob("*.xml")) if root.exists() else []

    for xml_path in xml_files:
        try:
            tree = ET.parse(str(xml_path))
        except ET.ParseError:
            reporter.warn("voc: cannot parse {0}".format(xml_path))
            continue
        root_node = tree.getroot()

        filename = _xml_text(root_node.find("filename"), xml_path.stem)
        if not filename:
            filename = xml_path.stem

        size_node = root_node.find("size")
        width = int(_to_float(_xml_text(size_node.find("width")), 0)) if size_node is not None else 0
        height = int(_to_float(_xml_text(size_node.find("height")), 0)) if size_node is not None else 0

        boxes: List[BBox] = []
        for object_node in root_node.findall("object"):
            raw_name = _xml_text(object_node.find("name"))
            class_id = _class_id(class_names, raw_name)
            if class_id is None:
                if raw_name and raw_name not in imported.skipped_classes:
                    imported.skipped_classes.append(raw_name)
                continue
            bndbox = object_node.find("bndbox")
            if bndbox is None:
                continue
            x1 = _to_float(_xml_text(bndbox.find("xmin")))
            y1 = _to_float(_xml_text(bndbox.find("ymin")))
            x2 = _to_float(_xml_text(bndbox.find("xmax")))
            y2 = _to_float(_xml_text(bndbox.find("ymax")))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(BBox(x1, y1, x2, y2, class_id, 1.0, "voc"))

        imported.add(filename, boxes, (width, height) if width and height else None)

    reporter.info("voc: parsed {0} annotation file(s)".format(len(xml_files)))
    return imported


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #
def load_coco_annotations(
    json_path: Any,
    class_names: Sequence[str],
    reporter: Optional[Reporter] = None,
) -> ImportedLabels:
    """Read a single COCO ``instances_*.json`` file."""
    reporter = reporter or Reporter(quiet=True)
    imported = ImportedLabels(source="coco", class_names=list(class_names))
    payload = read_json(json_path, default=None)
    if not payload:
        reporter.warn("coco: cannot read {0}".format(json_path))
        return imported

    category_by_id: Dict[int, str] = {}
    for category in payload.get("categories", []):
        category_by_id[int(category.get("id", -1))] = str(category.get("name", ""))

    image_by_id: Dict[int, Dict[str, Any]] = {}
    for image in payload.get("images", []):
        image_id = int(image.get("id", -1))
        image_by_id[image_id] = image
        file_name = str(image.get("file_name") or image.get("name") or "")
        width = int(_to_float(image.get("width"), 0))
        height = int(_to_float(image.get("height"), 0))
        if not file_name:
            continue
        imported.add(file_name, [], (width, height) if width and height else None)

    for annotation in payload.get("annotations", []):
        image_id = int(annotation.get("image_id", -1))
        image = image_by_id.get(image_id)
        if image is None:
            continue
        file_name = str(image.get("file_name") or image.get("name") or "")
        if not file_name:
            continue
        category_name = category_by_id.get(int(annotation.get("category_id", -1)), "")
        class_id = _class_id(class_names, category_name)
        if class_id is None:
            if category_name and category_name not in imported.skipped_classes:
                imported.skipped_classes.append(category_name)
            continue
        bbox = annotation.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x, y, box_w, box_h = (_to_float(value) for value in bbox[:4])
        if box_w <= 1 or box_h <= 1:
            continue
        score = _to_float(annotation.get("score", 1.0), 1.0)
        imported.boxes_by_name.setdefault(file_name, [])
        imported.boxes_by_name[file_name].append(
            BBox(x, y, x + box_w, y + box_h, class_id, score, "coco")
        )
        imported.boxes_by_name.setdefault(Path(file_name).stem, imported.boxes_by_name[file_name])

    reporter.info("coco: imported {0} image(s) from {1}".format(len(imported), json_path))
    return imported


# --------------------------------------------------------------------------- #
# YOLO
# --------------------------------------------------------------------------- #
def load_yolo_dir_annotations(
    labels_dir: Any,
    images_index: Dict[str, Tuple[str, int, int]],
    class_names: Sequence[str],
    reporter: Optional[Reporter] = None,
) -> ImportedLabels:
    """Read a directory of YOLO ``*.txt`` files, using the source image size.

    ``images_index`` comes from :func:`build_images_index`; it is required
    because the YOLO format only stores normalised coordinates.
    """
    reporter = reporter or Reporter(quiet=True)
    imported = ImportedLabels(source="yolo", class_names=list(class_names))
    root = Path(labels_dir)
    txt_files = sorted(root.rglob("*.txt")) if root.exists() else []

    unmatched = 0
    for txt_path in txt_files:
        key = txt_path.stem
        entry = images_index.get(key) or images_index.get(txt_path.name)
        if entry is None:
            unmatched += 1
            continue
        source_name, width, height = entry
        if width <= 0 or height <= 0:
            imported.missing_size.append(source_name)
            continue
        boxes = load_yolo_labels(txt_path, width, height, default_class=0)
        num_classes = len(class_names)
        valid_boxes: List[BBox] = []
        for box in boxes:
            # Guard against label files that carry a class index the config does
            # not declare: keeping such a box would make ``data.yaml`` (``nc``)
            # and the written ``.txt`` disagree and break Ultralytics training.
            if num_classes > 0 and not (0 <= int(box.class_id) < num_classes):
                reporter.warn(
                    "yolo: drop box with class_id={0} in {1}".format(box.class_id, txt_path)
                )
                continue
            box.source = "yolo"
            valid_boxes.append(box)
        imported.add(source_name, valid_boxes, (width, height))

    if unmatched:
        reporter.warn("yolo: {0} label file(s) had no matching source image".format(unmatched))
    reporter.info("yolo: parsed {0} label file(s)".format(len(txt_files)))
    return imported


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def load_annotations(
    labels_format: str,
    labels_path: Any,
    class_names: Sequence[str],
    images_index: Optional[Dict[str, Tuple[str, int, int]]] = None,
    reporter: Optional[Reporter] = None,
) -> ImportedLabels:
    """Load annotations in ``labels_format`` (``voc`` / ``coco`` / ``yolo``)."""
    labels_format = (labels_format or "none").lower()
    if labels_format == "voc":
        return load_voc_annotations(labels_path, class_names, reporter)
    if labels_format == "coco":
        return load_coco_annotations(labels_path, class_names, reporter)
    if labels_format == "yolo":
        return load_yolo_dir_annotations(labels_path, images_index or {}, class_names, reporter)
    raise ValueError("unsupported labels_format: {0!r}".format(labels_format))
