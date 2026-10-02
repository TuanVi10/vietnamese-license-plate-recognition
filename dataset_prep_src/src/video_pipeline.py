"""
video_pipeline.py
=================
**File glue** — nối các module rời rạc thành pipeline video hoàn chỉnh:

    Module 0 (Frame Extraction & Filtering)
        -> Module 1 (Vehicle Detection)          src/detect_vehicle.py
        -> Module 2 (Plate Detection, tùy chọn)   src/detector.py
        -> Module 3 (Tracking - SORT)             src/tracker.py
        -> Module 4/5/6 (Plate OCR + voting)      src/plate_reader.py
        -> Module 7 (Output & visualize)          JSON + frame overlay

Ý tưởng chính:

* Module 0 trích & lọc frame, giữ lại **chỉ số frame gốc** + ``timestamp``.
* Module 1 dò **bbox phương tiện**; Module 3 gán ``track_id`` ổn định qua frame.
* Với mỗi track, glue xác định **vùng biển số** (Module 2 nếu có model; nếu
  không thì xấp xỉ bằng bbox phương tiện) rồi gọi Module 4/5/6 để đọc biển số.
* Mọi lần đọc được gom theo ``track_id``; cuối video glue gọi
  :meth:`src.postprocess.PostProcessor.process` cho TỪNG ``track_id`` để ra
  **1 kết quả biển số cho mỗi xe** rồi :meth:`PostProcessor.format` để hiển thị
  (đúng tinh thần "voting theo ``track_id``" của spec 3.6;
  :func:`src.postprocess.vote_by_track` là API tiện dụng tương đương).

Pipeline vẫn chạy được khi **thiếu OCR engine** (chỉ ra danh sách track, không có
text biển số) và khi **thiếu model YOLO** (Module 1 rơi về chế độ "toàn khung").
Ngoài ra, nếu model YOLOv8n **lỗi lúc suy luận** (thiếu ``ultralytics``, sai đường
dẫn, model hỏng...) thì Module 1 cũng được bọc lỗi: ghi cảnh báo rồi fallback về
"toàn khung" thay vì làm sập cả pipeline video, nên vẫn có thể thử nghiệm/kiểm thử
logic ngay cả trong môi trường tối thiểu.

Ví dụ::

    from src.config import PipelineConfig
    from src.video_pipeline import VideoPipeline

    pipeline = VideoPipeline(PipelineConfig())
    result = pipeline.run("data/samples/test_video.mp4")
    for track in result.tracks:
        print(track.track_id, track.formatted)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .best_frame import PlateCandidate, TopKBuffer, score_crop
from .config import PipelineConfig
from .detect_vehicle import (
    VehicleDetection,
    VehicleDetector,
    build_vehicle_detector,
    clip_bbox,
)
from .frame_extractor import FrameExtractor, FrameRecord
from .fusion import FusionResult, fuse_readings
from .image_utils import crop_bbox, expand_bbox, imread_unicode, imwrite_unicode
from .postprocess import PostProcessor
from .plate_reader import PlateReader
from .tracker import SortTracker, TrackedObject, box_iou
from .visualize import draw_bbox


#: Tên module (ghi vào manifest kết quả).
MODULE_NAME = "Video Pipeline (Module 0 -> 1 -> 2 -> 3 -> 4/5/6 -> 7)"


@dataclass
class TrackReading:
    """Một lần đọc biển số gắn với 1 ``track_id`` tại 1 frame."""

    track_id: int
    frame_index: int
    timestamp: float
    text: str
    raw_text: str
    confidence: float
    engine: str
    vehicle_bbox: List[int]
    plate_bbox: Optional[List[int]] = None
    valid: bool = False
    is_two_line: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thành dict (dùng cho JSON output)."""
        return {
            "track_id": int(self.track_id),
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 4),
            "text": self.text,
            "raw_text": self.raw_text,
            "confidence": round(float(self.confidence), 4),
            "engine": self.engine,
            "vehicle_bbox": [int(v) for v in self.vehicle_bbox],
            "plate_bbox": None if self.plate_bbox is None else [int(v) for v in self.plate_bbox],
            "valid": bool(self.valid),
            "is_two_line": bool(self.is_two_line),
        }


@dataclass
class TrackResult:
    """Kết quả cuối cùng cho 1 ``track_id`` sau khi voting qua nhiều frame."""

    track_id: int
    text: str = ""
    formatted: str = ""
    confidence: float = 0.0
    valid: bool = False
    reliable: bool = False
    is_two_line: bool = False
    vote_count: int = 0
    total_readings: int = 0
    accepted_readings: int = 0
    hits: int = 0
    class_id: int = -1
    label: str = "vehicle"
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None
    vehicle_bbox: Optional[List[int]] = None
    readings: List[Dict[str, Any]] = field(default_factory=list)
    #: (Fusion) confidence từng vị trí ký tự; ``None`` nếu không dùng fusion.
    per_char_confidence: Optional[List[float]] = None
    #: (Fusion) số crop (frame_index) ủng hộ chuỗi cuối.
    supporting_frames: int = 0
    #: (Fusion) ghi chú (nghi trộn biển, lùi vote cũ...).
    fusion_notes: List[str] = field(default_factory=list)
    #: Có dùng fusion cho track này hay không.
    fusion_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thành dict (dùng cho JSON output)."""
        return {
            "track_id": int(self.track_id),
            "plate_text": self.text,
            "plate_display": self.formatted,
            "confidence": round(float(self.confidence), 4),
            "valid_format": bool(self.valid),
            "reliable": bool(self.reliable),
            "is_two_line": bool(self.is_two_line),
            "vote_count": int(self.vote_count),
            "total_readings": int(self.total_readings),
            "accepted_readings": int(self.accepted_readings),
            "hits": int(self.hits),
            "class_id": int(self.class_id),
            "label": self.label,
            "first_frame": None if self.first_frame is None else int(self.first_frame),
            "last_frame": None if self.last_frame is None else int(self.last_frame),
            "first_timestamp": None if self.first_timestamp is None else round(float(self.first_timestamp), 4),
            "last_timestamp": None if self.last_timestamp is None else round(float(self.last_timestamp), 4),
            "vehicle_bbox": self.vehicle_bbox,
            "readings": list(self.readings),
            "per_char_confidence": (
                None
                if self.per_char_confidence is None
                else [round(float(c), 4) for c in self.per_char_confidence]
            ),
            "supporting_frames": int(self.supporting_frames),
            "fusion_notes": list(self.fusion_notes),
            "fusion_used": bool(self.fusion_used),
        }


@dataclass
class FramePipelineResult:
    """Kết quả trung gian cho 1 frame (metadata + tracks + readings)."""

    frame_index: int
    timestamp: float
    tracks: List[TrackedObject] = field(default_factory=list)
    readings: List[TrackReading] = field(default_factory=list)
    #: Ảnh frame (không serialize — chỉ dùng nội bộ khi lưu overlay). Mặc định
    #: ``None`` sau bước :meth:`VideoPipeline.run` để không giữ cả video trong RAM.
    frame: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata của frame (KHÔNG kèm ảnh)."""
        return {
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 4),
            "num_tracks": len(self.tracks),
            "tracks": [track.to_dict() for track in self.tracks],
            "readings": [reading.to_dict() for reading in self.readings],
        }


@dataclass
class VideoPipelineResult:
    """Kết quả đầy đủ của pipeline video cho 1 video."""

    video_path: str
    module: str = MODULE_NAME
    frames: List[FramePipelineResult] = field(default_factory=list)
    tracks: List[TrackResult] = field(default_factory=list)
    readings: List[TrackReading] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self, include_frames: bool = True) -> Dict[str, Any]:
        """Serialize thành dict thuần để ghi JSON."""
        data: Dict[str, Any] = {
            "module": self.module,
            "video": self.video_path,
            "stats": dict(self.stats),
            "tracks": [track.to_dict() for track in self.tracks],
            "readings": [reading.to_dict() for reading in self.readings],
            "warnings": list(self.warnings),
        }
        if include_frames:
            data["frames"] = [frame.to_dict() for frame in self.frames]
        return data


class VideoPipeline:
    """Ghép Module 0 -> 1 -> 2 -> 3 -> 4/5/6 -> 7 cho 1 video."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        vehicle_detector: Optional[VehicleDetector] = None,
        plate_detector: Optional[Any] = None,
        tracker: Optional[SortTracker] = None,
        reader: Optional[PlateReader] = None,
        extractor: Optional[FrameExtractor] = None,
    ) -> None:
        """Khởi tạo pipeline với các module có thể inject (tiện test/thay thế).

        Args:
            config: Cấu hình tổng.
            vehicle_detector: Module 1. ``None`` -> dựng từ ``config.vehicle_detection``.
            plate_detector: Module 2 (``PlateDetector``). ``None`` -> dựng từ
                ``config.detection_model`` nếu được cấu hình; nếu không có model thì
                xấp xỉ vùng biển số bằng bbox phương tiện.
            tracker: Module 3. ``None`` -> :class:`~src.tracker.SortTracker`.
            reader: Module 4/5/6. ``None`` -> khởi tạo lazy (cần OCR engine).
            extractor: Module 0. ``None`` -> :class:`~src.frame_extractor.FrameExtractor`.
        """
        self.config = config or PipelineConfig()
        self.vehicle_detector = vehicle_detector or build_vehicle_detector(self.config.vehicle_detection)

        # Trạng thái nội bộ — PHẢI tạo TRƯỚC khi dựng Module 2 vì
        # ``_build_plate_detector`` cần ``_warnings`` để ghi cảnh báo.
        self._readings_by_track: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._track_meta: Dict[int, Dict[str, Any]] = {}
        #: Frame/timestamp ĐẦU TIÊN mỗi ``track_id`` xuất hiện — kể cả khi track
        #: CHƯA đủ ``min_hits`` để được tracker trả về. Dùng để ``first_frame`` /
        #: ``first_timestamp`` phản ánh đúng lúc phương tiện xuất hiện.
        self._track_first_seen: Dict[int, Tuple[int, float]] = {}
        self._warnings: List[str] = []
        #: Cảnh báo phát sinh lúc khởi tạo — sống sót qua :meth:`reset`.
        self._setup_warnings: List[str] = []
        self._reader_unavailable = False
        self._reading_frames = 0
        #: Buffer top-K crop cho mỗi track (best-frame mode).
        self._best_buffers: Dict[int, TopKBuffer] = {}
        #: Lịch sử aspect (w/h) của detection THẬT theo track (để ước lượng loại biển).
        self._track_aspects: Dict[int, List[float]] = {}
        #: Các track đã từng có detection biển thật (để chặn fallback).
        self._track_seen_detection: Set[int] = set()
        #: Log overlay theo frame_index -> [(track_id, vehicle_bbox, label)] (post-pass).
        self._overlay_log: Dict[int, List[Tuple[int, List[int], str]]] = {}
        #: Corner regressor (4 góc biển) tuỳ chọn + ngưỡng góc để warp (cách B).
        self._corner_regressor = None
        self._corner_angle_threshold = 15.0
        #: Thống kê thu thập best-frame theo track (debug): collected/rejected_height/detection/fallback.
        self._best_stats: Dict[int, Dict[str, int]] = {}
        #: Legacy readings theo track (chỉ khi debug, để đối chiếu với best-frame).
        self._legacy_readings_by_track: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
        #: Manifest các crop đã lưu (plate_crops/manifest.json).
        self._crop_manifest: List[Dict[str, Any]] = []

        # Module 2: ưu tiên module được inject, nếu không thì tự dựng từ config.
        # CHỈ dựng/giữ plate detector khi BẬT đọc biển số (``read_plates``): khi
        # người dùng tắt đọc biển số (``--no-plate-reading``) thì KHÔNG tải/chạy
        # Module 2, đúng tài liệu (``False`` = chỉ chạy Module 0/1/3) và tránh
        # tốn tài nguyên / lỗi model không cần thiết.
        if self.config.video.read_plates:
            self.plate_detector = self._build_plate_detector(plate_detector)
        else:
            self.plate_detector = None
        self.tracker = tracker or SortTracker(self.config.tracking)
        self.reader = reader
        self.extractor = extractor or FrameExtractor(self.config.frame_extraction)
        self.postprocessor = PostProcessor(self.config.postprocess)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset tracker + mọi bộ đếm để chạy lại từ đầu (dùng lại cho video khác)."""
        self.tracker.reset()
        self._readings_by_track = defaultdict(list)
        self._track_meta = {}
        self._track_first_seen = {}
        self._warnings = []
        self._reader_unavailable = False
        self._reading_frames = 0
        self._best_buffers = {}
        self._track_aspects = {}
        self._track_seen_detection = set()
        self._overlay_log = {}
        self._best_stats = {}
        self._legacy_readings_by_track = defaultdict(list)
        self._crop_manifest = []

    def _warn(self, message: str) -> None:
        """Thêm cảnh báo (không trùng lặp)."""
        if message not in self._warnings:
            self._warnings.append(message)

    def _warn_setup(self, message: str) -> None:
        """Cảnh báo phát sinh lúc khởi tạo — được giữ lại qua các lần :meth:`reset`."""
        self._warn(message)
        if message not in self._setup_warnings:
            self._setup_warnings.append(message)

    # ------------------------------------------------------------------ #
    def _build_plate_detector(self, plate_detector: Optional[Any]) -> Optional[Any]:
        """Dựng Module 2 (plate detector) từ ``config`` nếu người dùng không inject sẵn.

        Thứ tự ưu tiên:

        1. ``plate_detector`` được inject -> dùng luôn (giữ nguyên hành vi cũ).
        2. ``config.detection_model`` có giá trị -> tự dựng
           :class:`~src.detector.PlateDetector` từ các trường
           ``detection_conf`` / ``detection_device`` / ``detection_imgsz``.
        3. Không có gì -> ``None``; glue rơi về xấp xỉ vùng biển số bằng bbox xe
           (xem :meth:`_fallback_plate_bbox`).
        """
        if plate_detector is not None:
            return plate_detector

        if not self.config.detection_model:
            return None

        try:
            from .detector import PlateDetector

            detector = PlateDetector(
                self.config.detection_model,
                conf=self.config.detection_conf,
                device=self.config.detection_device,
                imgsz=self.config.detection_imgsz,
            )
        except Exception as exc:  # noqa: BLE001 - thiếu ultralytics / model lỗi -> vẫn fallback
            self._warn_setup(
                f"Không khởi tạo được Module 2 (plate detector): {exc}. "
                "Sẽ xấp xỉ vùng biển số bằng bbox phương tiện."
            )
            return None

        return detector

    # ------------------------------------------------------------------ #
    def _ensure_reader(self) -> Optional[PlateReader]:
        """Khởi tạo Module 4/5/6 (lazy). Trả ``None`` nếu thiếu OCR engine."""
        if self.reader is not None:
            return self.reader
        if not self.config.video.read_plates:
            return None
        if self._reader_unavailable:
            return None
        try:
            from .ocr import create_ocr_engine

            engine = create_ocr_engine(self.config.ocr)
            self.reader = PlateReader(config=self.config, ocr_engine=engine, detector=None)
        except Exception as exc:  # noqa: BLE001 - thiếu OCR engine -> vẫn tracking được
            self._reader_unavailable = True
            self._warn(
                f"Không khởi tạo được OCR (Module 5): {exc}. "
                "Chỉ chạy tracking, không đọc biển số."
            )
            return None
        return self.reader

    # ------------------------------------------------------------------ #
    def _update_track_meta(self, track: TrackedObject, frame_index: int, timestamp: float) -> Dict[str, Any]:
        """Cập nhật metadata (vòng đời + bbox) của 1 track."""
        meta = self._track_meta.get(track.track_id)
        if meta is None:
            # ``first_frame`` / ``first_timestamp`` phải là frame ĐẦU TIÊN phương
            # tiện xuất hiện — KHÔNG phải frame mà track đủ ``min_hits`` mới được
            # tracker xác nhận và trả về (xem ``_track_first_seen``).
            first_frame, first_timestamp = self._track_first_seen.get(
                track.track_id, (frame_index, timestamp)
            )
            meta = {
                "track_id": int(track.track_id),
                "first_frame": int(first_frame),
                "first_timestamp": float(first_timestamp),
                "hits": 0,
                "class_id": int(track.class_id),
                "label": str(track.label),
                "vehicle_bbox": [int(v) for v in track.bbox],
                "readings": [],
            }
            self._track_meta[track.track_id] = meta
        meta["last_frame"] = int(frame_index)
        meta["last_timestamp"] = float(timestamp)
        meta["hits"] = max(int(meta.get("hits", 0)), int(track.hits))
        meta["vehicle_bbox"] = [int(v) for v in track.bbox]
        meta["class_id"] = int(track.class_id)
        meta["label"] = str(track.label)
        return meta

    # ------------------------------------------------------------------ #
    @staticmethod
    def _containment_score(vehicle_bbox: Sequence[float], plate_bbox: Sequence[float]) -> float:
        """Điểm khớp giữa bbox biển số và bbox phương tiện.

        Biển nằm TRONG xe được ưu tiên (cộng thêm 1.0) so với biển chỉ chồng lấn.
        """
        vx1, vy1, vx2, vy2 = (float(v) for v in vehicle_bbox[:4])
        px1, py1, px2, py2 = (float(v) for v in plate_bbox[:4])
        center_x = (px1 + px2) / 2.0
        center_y = (py1 + py2) / 2.0
        inside = vx1 <= center_x <= vx2 and vy1 <= center_y <= vy2
        overlap = box_iou(vehicle_bbox, plate_bbox)
        if inside:
            return 1.0 + overlap
        return overlap

    def _best_plate_score(self, track: TrackedObject, plate_detections: Sequence[Any]) -> float:
        """Điểm khớp cao nhất giữa ``track`` và các plate detection hiện có.

        Dùng để sắp thứ tự gán biển -> track một cách xác định (không phụ thuộc
        thứ tự tracker trả về). Bỏ qua bbox méo/không hợp lệ thay vì ném lỗi.
        """
        best = 0.0
        for detection in plate_detections:
            candidate = getattr(detection, "bbox", None)
            if candidate is None:
                continue
            try:
                best = max(best, self._containment_score(track.bbox, candidate))
            except (TypeError, ValueError):
                continue
        return best

    def _fallback_plate_bbox(
        self,
        vehicle_bbox: Sequence[float],
        frame_shape: Sequence[int],
    ) -> Optional[List[int]]:
        """Vùng biển số xấp xỉ khi KHÔNG có model dò biển số (Module 2)."""
        if not self.config.video.plate_fallback_to_vehicle:
            return None
        height, width = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = clip_bbox(vehicle_bbox, width, height)
        ratio = float(self.config.video.plate_region_ratio)
        ratio = max(0.0, min(1.0, ratio))
        if ratio >= 1.0:
            return [x1, y1, x2, y2]
        region_h = max(1, int(round((y2 - y1) * ratio)))
        top = max(y1, y2 - region_h)
        return [x1, top, x2, y2]

    def _locate_plate_candidate(
        self,
        frame: np.ndarray,
        vehicle_bbox: Sequence[float],
        plate_detections: Optional[List[Any]],
        used_plate_indices: Optional[Set[int]] = None,
    ) -> Tuple[Optional[List[int]], float, str]:
        """Xác định ``(plate_bbox, det_conf, source)`` cho 1 phương tiện.

        Ưu tiên detection của Module 2 nằm trong/chồng lấn bbox xe (``source =
        "detection"``, ``det_conf`` = confidence của detection); nếu không có thì
        xấp xỉ bằng (phần dưới) bbox xe (``source = "fallback"``, ``det_conf = 0.0``).

        ``used_plate_indices`` là tập chỉ số các plate detection **đã được gán**
        cho track khác trong cùng frame, để mỗi detection biển số chỉ thuộc về
        ĐÚNG MỘT track. Khi match thành công, chỉ số vừa dùng được thêm vào tập
        này để track sau bỏ qua.
        """
        used = used_plate_indices if used_plate_indices is not None else set()

        if plate_detections:
            min_confidence = float(self.config.video.plate_min_confidence)
            best_bbox: Optional[Sequence[float]] = None
            best_conf = 0.0
            best_score = 0.0
            best_index = -1
            for index, detection in enumerate(plate_detections):
                if index in used:
                    continue
                confidence = float(getattr(detection, "confidence", 1.0))
                if confidence < min_confidence:
                    continue
                candidate = getattr(detection, "bbox", None)
                if candidate is None:
                    continue
                score = self._containment_score(vehicle_bbox, candidate)
                if score > best_score:
                    best_score = score
                    best_bbox = candidate
                    best_conf = confidence
                    best_index = index
            if best_bbox is not None and best_score > 0.0:
                used.add(best_index)
                height, width = frame.shape[:2]
                return list(clip_bbox(best_bbox, width, height)), best_conf, "detection"

        fallback = self._fallback_plate_bbox(vehicle_bbox, frame.shape)
        return fallback, 0.0, "fallback"

    def _locate_plate_bbox(
        self,
        frame: np.ndarray,
        vehicle_bbox: Sequence[float],
        plate_detections: Optional[List[Any]],
        used_plate_indices: Optional[Set[int]] = None,
    ) -> Optional[List[int]]:
        """Xác định bbox biển số cho 1 phương tiện (giữ chữ ký cũ, chỉ trả bbox)."""
        bbox, _, _ = self._locate_plate_candidate(
            frame, vehicle_bbox, plate_detections, used_plate_indices
        )
        return bbox

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reading_candidates(reading: Any) -> List[Dict[str, Any]]:
        """Bóc danh sách ứng viên (text/raw_text/confidence) từ 1 ``PlateReading``."""
        candidates: List[Dict[str, Any]] = []
        items = getattr(reading, "readings", None) or []
        for item in items:
            if isinstance(item, dict):
                text = item.get("text")
                raw_text = item.get("raw_text")
                confidence = item.get("confidence", 0.0)
                engine = item.get("engine", getattr(reading, "engine", ""))
                source = item.get("source", "")
            else:
                text = getattr(item, "text", None)
                raw_text = getattr(item, "raw_text", None)
                confidence = getattr(item, "confidence", 0.0)
                engine = getattr(item, "engine", getattr(reading, "engine", ""))
                source = getattr(item, "source", "")
            if not text:
                continue
            candidates.append(
                {
                    "text": str(text),
                    "raw_text": str(raw_text or text),
                    "confidence": float(confidence),
                    "engine": str(engine or ""),
                    "variant": str(source or ""),
                }
            )

        # Không có ứng viên rời nhưng vẫn có text -> dùng chính text của frame.
        if not candidates:
            text = getattr(reading, "text", "")
            if text:
                candidates.append(
                    {
                        "text": str(text),
                        "raw_text": str(getattr(reading, "raw_text", "") or text),
                        "confidence": float(getattr(reading, "confidence", 0.0)),
                        "engine": str(getattr(reading, "engine", "") or ""),
                        "variant": "frame",
                    }
                )
        return candidates

    def _read_plate(
        self,
        frame: np.ndarray,
        track: TrackedObject,
        plate_detections: Optional[List[Any]],
        used_plate_indices: Optional[Set[int]],
        frame_index: int,
        timestamp: float,
    ) -> Optional[Tuple[TrackReading, List[Dict[str, Any]]]]:
        """Đọc biển số cho 1 track; trả ``(record, candidates)`` hoặc ``None``.

        ``used_plate_indices`` được truyền tiếp cho :meth:`_locate_plate_bbox` để
        các plate detection **đã gán cho track trước đó** trong cùng frame không
        bị dùng lại cho track này.
        """
        reader = self.reader
        if reader is None:
            return None

        plate_bbox = self._locate_plate_bbox(
            frame,
            track.bbox,
            plate_detections,
            used_plate_indices,
        )
        if plate_bbox is None:
            return None

        try:
            reading = reader.read(frame, bbox=plate_bbox)
        except Exception as exc:  # noqa: BLE001 - 1 frame lỗi không làm hỏng cả video
            self._warn(f"Lỗi đọc biển số cho track {track.track_id}: {exc}")
            return None

        candidates = self._reading_candidates(reading)
        if not getattr(reading, "text", "") and not candidates:
            return None

        # ``raw_text`` phải là chuỗi OCR thô (còn dấu phân cách, ví dụ
        # ``51F-12345``), KHÔNG phải ``reading.text`` đã được chuẩn hóa. Ưu tiên
        # lấy từ các ứng viên rời (``reading.readings``) để giữ nguyên thông tin
        # gốc phục vụ debug/đối chiếu; chỉ fallback về ``reading.text`` khi không
        # có ứng viên nào.
        raw_texts = [
            str(candidate.get("raw_text") or candidate.get("text") or "")
            for candidate in candidates
        ]
        raw_text = " ".join(dict.fromkeys(raw_texts)) or str(
            getattr(reading, "text", "") or ""
        )

        record = TrackReading(
            track_id=track.track_id,
            frame_index=frame_index,
            timestamp=timestamp,
            text=str(getattr(reading, "text", "") or ""),
            raw_text=raw_text,
            confidence=float(getattr(reading, "confidence", 0.0) or 0.0),
            engine=str(getattr(reading, "engine", "") or ""),
            vehicle_bbox=[int(v) for v in track.bbox],
            plate_bbox=[int(v) for v in plate_bbox],
            valid=bool(getattr(reading, "valid", False)),
            is_two_line=bool(getattr(reading, "is_two_line", False)),
        )
        return record, candidates

    # ------------------------------------------------------------------ #
    def _annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
        readings: Sequence[TrackReading],
    ) -> np.ndarray:
        """Vẽ bbox xe + track_id + biển số đọc được lên bản copy của frame."""
        by_track = {reading.track_id: reading for reading in readings}
        canvas = frame.copy()
        for track in tracks:
            label = f"#{track.track_id} {track.label}"
            reading = by_track.get(track.track_id)
            if reading is not None and reading.text:
                label = f"{label} | {reading.text} {reading.confidence:.2f}"
            canvas = draw_bbox(canvas, track.bbox, label=label)
        return canvas

    def _save_overlay(self, frame_index: int, annotated: np.ndarray) -> None:
        """Ghi 1 frame overlay vào ``<output_dir>/tracks_frames/``."""
        output_dir = os.path.join(self.config.output_dir, "tracks_frames")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"frame_{int(frame_index):06d}.png")
        imwrite_unicode(path, annotated)

    # ------------------------------------------------------------------ #
    # Best-frame: chọn top-K crop tốt nhất cho mỗi track
    # ------------------------------------------------------------------ #
    def _effective_min_frame_gap(self) -> int:
        """Khoảng cách frame tối thiểu giữa các crop giữ lại (auto = 2*frame_interval)."""
        gap = self.config.best_frame.min_frame_gap
        if gap is not None:
            return max(0, int(gap))
        interval = max(1, int(self.config.frame_extraction.frame_interval))
        return max(2, 2 * interval)

    def _expected_aspect(self, track_id: int) -> float:
        """Ước lượng aspect kỳ vọng theo track từ median aspect của detection thật.

        Ghi chú: ``expected_aspect`` được ước lượng TRỰC TUYẾN (dựa trên các bbox
        đã thấy đến thời điểm hiện tại) nên chỉ là XẤP XỈ, có thể lệch ở những
        frame đầu khi track chưa có đủ detection.
        """
        aspects = self._track_aspects.get(track_id, [])
        if not aspects:
            return float(self.config.best_frame.one_line_aspect_ratio)
        median = float(np.median(aspects))
        if median < float(self.config.best_frame.two_line_aspect_threshold):
            return float(self.config.best_frame.two_line_aspect_ratio)
        return float(self.config.best_frame.one_line_aspect_ratio)

    def _crop_plate(self, frame: np.ndarray, plate_bbox: Sequence[float]) -> np.ndarray:
        """Cắt crop biển số từ frame (nới ``bbox_margin`` nếu cấu hình > 0)."""
        height, width = frame.shape[:2]
        margin = float(self.config.preprocess.bbox_margin)
        if margin > 0:
            plate_bbox = expand_bbox(plate_bbox, margin, width, height)
        return crop_bbox(frame, plate_bbox)

    def _collect_best_candidate(
        self,
        frame: np.ndarray,
        track: TrackedObject,
        plate_detections: Optional[List[Any]],
        used_plate_indices: Optional[Set[int]],
        frame_index: int,
        timestamp: float,
    ) -> Optional[PlateCandidate]:
        """Pha THU THẬP: định vị bbox, cắt crop, chấm điểm, đẩy vào TopKBuffer (KHÔNG OCR)."""
        bbox, det_conf, source = self._locate_plate_candidate(
            frame, track.bbox, plate_detections, used_plate_indices
        )
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        plate_height = float(y2 - y1)
        plate_width = float(x2 - x1)

        stats = self._best_stats.setdefault(
            int(track.track_id), {"collected": 0, "rejected_height": 0, "detection": 0, "fallback": 0}
        )

        if source == "detection":
            stats["detection"] += 1
            # Track đã có detection thật -> chặn fallback về sau; ghi aspect (bbox gốc detector).
            self._track_seen_detection.add(int(track.track_id))
            if plate_height > 0:
                self._track_aspects.setdefault(int(track.track_id), []).append(plate_width / plate_height)
        else:
            stats["fallback"] += 1
            if int(track.track_id) in self._track_seen_detection:
                # Fallback không được dùng khi track đã từng có detection thật.
                return None

        # Loại cứng crop quá nhỏ.
        if plate_height < float(self.config.best_frame.min_plate_height):
            stats["rejected_height"] += 1
            return None

        crop = self._crop_plate(frame, bbox)
        if crop is None or crop.size == 0:
            return None

        expected_aspect = self._expected_aspect(int(track.track_id))
        detail = score_crop(crop, bbox, det_conf, expected_aspect, self.config.best_frame)

        candidate = PlateCandidate(
            frame_index=int(frame_index),
            timestamp=float(timestamp),
            crop=crop,
            plate_bbox=list(bbox),
            det_conf=float(det_conf),
            score=float(detail.total),
            source=source,
            components=detail.to_dict(),
        )
        buffer = self._best_buffers.setdefault(
            int(track.track_id),
            TopKBuffer(k=int(self.config.best_frame.k), min_frame_gap=self._effective_min_frame_gap()),
        )
        buffer.add(candidate)
        stats["collected"] += 1
        return candidate

    def set_corner_regressor(self, regressor, angle_threshold: float = 15.0) -> None:
        """Gắn corner regressor (tuỳ chọn) + ngưỡng góc để quyết định warp."""
        self._corner_regressor = regressor
        self._corner_angle_threshold = float(angle_threshold)

    def _apply_corner(self, crop: np.ndarray) -> np.ndarray:
        """Nếu có corner regressor và |góc| > ngưỡng -> warpPerspective(4 góc), ngược lại giữ crop."""
        if self._corner_regressor is None:
            return crop
        res = self._corner_regressor.predict(crop)
        if res is None or abs(res.angle) <= self._corner_angle_threshold:
            return crop
        import cv2

        tl, tr, br, bl = res.corners
        w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
        h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
        w, h = max(8, w), max(8, h)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        m = cv2.getPerspectiveTransform(res.corners.astype(np.float32), dst)
        return cv2.warpPerspective(crop, m, (w, h), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

    def _flush_track_best(self, track_id: int) -> None:
        """Pha ĐỌC: OCR top-K crop của 1 track, lưu crop + manifest, đẩy vào ``_readings_by_track``."""
        buffer = self._best_buffers.get(track_id)
        if not buffer:
            return
        reader = self.reader
        if reader is None:
            return
        max_reading_frames = int(self.config.video.max_reading_frames)
        meta = self._track_meta.get(track_id)
        for rank, candidate in enumerate(buffer.best()):
            if max_reading_frames > 0 and self._reading_frames >= max_reading_frames:
                break
            try:
                reading = reader.read(self._apply_corner(candidate.crop))
            except Exception as exc:  # noqa: BLE001 - 1 crop lỗi không làm hỏng video
                self._warn(f"Lỗi đọc crop biển số cho track {track_id}: {exc}")
                continue
            candidates = self._reading_candidates(reading)
            if not getattr(reading, "text", "") and not candidates:
                continue
            text = str(getattr(reading, "text", "") or "")
            self._reading_frames += 1
            if getattr(reading, "is_two_line", False) and meta is not None:
                meta["is_two_line"] = True
            # Lưu toàn bộ top-K crop vào plate_crops/ + ghi manifest.
            self._save_best_crop(track_id, rank, candidate, text)
            for cand in candidates:
                enriched = dict(cand)
                enriched["track_id"] = int(track_id)
                enriched["frame_index"] = int(candidate.frame_index)
                enriched["timestamp"] = float(candidate.timestamp)
                enriched["score"] = float(candidate.score)
                enriched["crop_source"] = str(candidate.source)
                enriched.update(candidate.components)
                self._readings_by_track[track_id].append(enriched)
                if meta is not None:
                    meta["readings"].append(enriched)

    def _save_best_crop(self, track_id: int, rank: int, candidate: PlateCandidate, text: str) -> None:
        """Lưu 1 top-K crop vào ``plate_crops/`` và thêm entry vào ``_crop_manifest``."""
        output_dir = os.path.join(self.config.output_dir, "plate_crops")
        os.makedirs(output_dir, exist_ok=True)
        name = (
            f"track{int(track_id)}_rank{int(rank)}_f{int(candidate.frame_index)}"
            f"_s{float(candidate.score):.3f}_{candidate.source}.jpg"
        )
        imwrite_unicode(os.path.join(output_dir, name), candidate.crop)

        entry: Dict[str, Any] = {
            "track_id": int(track_id),
            "rank": int(rank),
            "frame_index": int(candidate.frame_index),
            "score": round(float(candidate.score), 4),
            "source": str(candidate.source),
            "text": text,
        }
        entry.update(candidate.components)
        self._crop_manifest.append(entry)

    def _write_crop_manifest(self) -> None:
        """Ghi ``plate_crops/manifest.json`` (danh sách crop đã lưu)."""
        output_dir = os.path.join(self.config.output_dir, "plate_crops")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._crop_manifest, handle, ensure_ascii=False, indent=2)

    def _legacy_vote(self, track_id: int) -> str:
        """Vote legacy readings của 1 track (chỉ dùng khi debug, để đối chiếu)."""
        items = self._legacy_readings_by_track.get(track_id, [])
        if not items:
            return ""
        vote = self.postprocessor.process(items, group_id=f"legacy-{track_id}")
        return str(getattr(vote, "text", "") or "")

    def _report_best_frame_debug(self, tracks: Sequence[Any]) -> None:
        """In chẩn đoán best-frame theo từng track (so sánh với legacy)."""
        if not self.config.best_frame.debug:
            return
        print("\n[debug-best-frame] ==== CHẨN ĐOÁN PER-TRACK ====")
        for track in tracks:
            stats = self._best_stats.get(int(track.track_id), {})
            topk = self._best_buffers.get(int(track.track_id))
            topk_desc: List[str] = []
            if topk:
                for rank, candidate in enumerate(topk.best()):
                    topk_desc.append(
                        f"rank{rank}({candidate.source},s={candidate.score:.3f},f={candidate.frame_index})"
                    )
            legacy_text = self._legacy_vote(int(track.track_id))
            print(
                f"[debug-best-frame] track {track.track_id} ({track.label}) "
                f"bbox={track.vehicle_bbox} first_frame={track.first_frame} "
                f"| collected={stats.get('collected', 0)} rejected_h={stats.get('rejected_height', 0)} "
                f"det={stats.get('detection', 0)} fallback={stats.get('fallback', 0)}"
            )
            print(
                f"[debug-best-frame]   best='{track.text}' legacy='{legacy_text}' "
                f"| topK: {', '.join(topk_desc) or '-'}"
            )
        print("[debug-best-frame] ================================\n")

    def _flush_all_tracks(self) -> None:
        """Flush top-K của MỌI track (gọi 1 lần ở cuối video)."""
        if not self.config.best_frame.enabled:
            return
        if self._ensure_reader() is None:
            return
        for track_id in list(self._best_buffers.keys()):
            self._flush_track_best(track_id)
        if self._crop_manifest:
            self._write_crop_manifest()

    def _annotate_overlays_post(self, tracks: Sequence[Any]) -> None:
        """Sau khi voting, vẽ lại text biển số cuối lên overlay đã lưu (best-frame)."""
        by_track = {int(track.track_id): track for track in tracks}
        output_dir = os.path.join(self.config.output_dir, "tracks_frames")
        for frame_index, entries in self._overlay_log.items():
            path = os.path.join(output_dir, f"frame_{int(frame_index):06d}.png")
            if not os.path.isfile(path):
                continue
            image = imread_unicode(path)
            for track_id, vehicle_bbox, label in entries:
                result = by_track.get(int(track_id))
                text = ""
                if result is not None:
                    label = str(result.label)
                    if result.text:
                        text = str(result.text)
                display = f"#{track_id} {label}"
                if text:
                    display = f"{display} | {text} {result.confidence:.2f}"
                image = draw_bbox(image, vehicle_bbox, label=display)
            imwrite_unicode(path, image)

    # ------------------------------------------------------------------ #
    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> FramePipelineResult:
        """Chạy Module 1 + 3 (và 2/4/5/6 nếu bật) trên 1 frame.

        State (tracker + bộ gom reading) được giữ trên instance, nên hàm này phải
        được gọi tuần tự theo thứ tự frame (thường thông qua :meth:`run`).
        """
        # --- Module 1: dò phương tiện -------------------------------------- #
        # Bọc ``try/except`` để lỗi model (thiếu ``ultralytics``, sai đường dẫn
        # ``--vehicle-model``, YOLO suy luận lỗi...) KHÔNG làm sập toàn bộ
        # pipeline video — nhất quán với Module 2 và Module 4/5/6 (đều đã được
        # bảo vệ + fallback). Khi lỗi: ghi cảnh báo qua ``self._warn``; nếu
        # detector bật chế độ "toàn khung" thì trả về 1 bbox phủ toàn frame
        # (giống lúc KHÔNG có model) để Module 3 + các module sau vẫn chạy được;
        # ngược lại thì bỏ qua frame này (an toàn, không sinh track rác).
        try:
            detections = self.vehicle_detector.detect(frame)
        except Exception as exc:  # noqa: BLE001 - lỗi model/thiếu ultralytics không làm sập pipeline
            self._warn(f"Lỗi dò phương tiện (Module 1): {exc}")
            if getattr(self.vehicle_detector, "full_frame_fallback", True):
                height, width = frame.shape[:2]
                detections = [
                    VehicleDetection(
                        bbox=(0, 0, int(width), int(height)),
                        confidence=1.0,
                        class_id=-1,
                        label="vehicle",
                    )
                ]
            else:
                detections = []

        # --- Module 3: tracking -------------------------------------------- #
        # Bọc ``try/except`` để lỗi nội bộ tracker (ma trận Kalman suy biến,
        # ``LinAlgError``...) KHÔNG làm sập cả pipeline video — đồng nhất với
        # Module 1/2/4/5/6 (đều đã được bảo vệ + fallback). Khi lỗi: ghi cảnh
        # báo, reset tracker để không lặp lỗi mỗi frame, và coi frame này như
        # không có track nào.
        try:
            tracks = self.tracker.update(detections, frame_index=frame_index, timestamp=timestamp)
        except Exception as exc:  # noqa: BLE001 - 1 lỗi tracker không làm sập video
            self._warn(f"Lỗi tracking (Module 3): {exc}")
            try:
                self.tracker.reset()
            except Exception:  # pragma: no cover - reset không nên lỗi
                pass
            tracks = []

        # Ghi nhận frame/timestamp ĐẦU TIÊN mà MỌI track đang sống xuất hiện.
        # ``SortTracker`` chỉ trả về track đã đủ ``min_hits`` (đã xác nhận), nên
        # nếu chỉ dựa vào ``tracks`` thì ``first_frame`` sẽ bị lệch về đúng frame
        # xác nhận (VD ``min_hits=3`` -> ``first_frame=2`` thay vì ``0``).
        # ``self.tracker.trackers`` giữ mọi track đang sống, kể cả track CHƯA xác
        # nhận, nên ta dùng nó để bắt đúng frame phương tiện xuất hiện lần đầu.
        for active_tracker in getattr(self.tracker, "trackers", []):
            tracker_id = getattr(active_tracker, "id", None)
            if tracker_id is not None:
                self._track_first_seen.setdefault(int(tracker_id), (frame_index, timestamp))

        # --- Module 2: dò biển số (một lần/frame, nếu bật đọc biển số) ----- #
        # Chỉ chạy Module 2 khi ``read_plates`` được bật; nếu người dùng đã tắt
        # đọc biển số thì bỏ qua hoàn toàn (không gọi ``detect``) để khớp với
        # tài liệu cấu hình (``False`` = chỉ Module 0/1/3).
        plate_detections: Optional[List[Any]] = None
        if self.config.video.read_plates and self.plate_detector is not None:
            try:
                plate_detections = list(self.plate_detector.detect(frame))
            except Exception as exc:  # noqa: BLE001
                self._warn(f"Lỗi dò biển số (Module 2): {exc}")
                plate_detections = []

        # --- Module 4/5/6: đọc biển số theo từng track --------------------- #
        best_frame_enabled = bool(self.config.best_frame.enabled) and bool(self.config.video.read_plates)
        readings: List[TrackReading] = []

        # Mỗi plate detection chỉ được gán cho ĐÚNG MỘT track trong 1 frame:
        # tập chỉ số này chia sẻ giữa các track để một biển không bị đọc lặp
        # (và không bị tính cho nhiều ``track_id`` khi voting).
        used_plate_indices: Set[int] = set()
        # Sắp track theo "điểm chứa biển" giảm dần TRƯỚC khi gán, để việc gán
        # plate -> track KHÔNG phụ thuộc thứ tự tracker trả về: xe chứa biển khớp
        # nhất sẽ "giành" biển trước, tránh biển chồng lấn bị gán cho xe xử lý
        # trước dù không khớp bằng.
        if plate_detections:
            ordered_tracks = sorted(
                tracks,
                key=lambda trk: (-self._best_plate_score(trk, plate_detections), trk.track_id),
            )
        else:
            ordered_tracks = list(tracks)

        if best_frame_enabled:
            # Pha THU THẬP: chỉ định vị bbox, cắt crop, chấm điểm (KHÔNG OCR).
            debug_legacy = bool(self.config.best_frame.debug)
            if debug_legacy:
                self._ensure_reader()
            debug_used_plate_indices: Set[int] = set()
            for track in ordered_tracks:
                self._update_track_meta(track, frame_index, timestamp)
                self._collect_best_candidate(
                    frame, track, plate_detections, used_plate_indices, frame_index, timestamp
                )
                if debug_legacy and self.reader is not None:
                    # Chạy song song legacy OCR (chỉ debug) để đối chiếu text.
                    outcome = self._read_plate(
                        frame, track, plate_detections, debug_used_plate_indices, frame_index, timestamp
                    )
                    if outcome is not None:
                        _record, legacy_candidates = outcome
                        for cand in legacy_candidates:
                            enriched = dict(cand)
                            enriched["track_id"] = int(track.track_id)
                            enriched["frame_index"] = int(frame_index)
                            enriched["timestamp"] = float(timestamp)
                            self._legacy_readings_by_track[track.track_id].append(enriched)
        else:
            # Pha đọc TRỰC TIẾP (hành vi cũ): OCR ngay trên mọi frame.
            reader = self._ensure_reader() if self.config.video.read_plates else None
            max_reading_frames = int(self.config.video.max_reading_frames)
            can_read = reader is not None and (max_reading_frames <= 0 or self._reading_frames < max_reading_frames)
            for track in ordered_tracks:
                meta = self._update_track_meta(track, frame_index, timestamp)
                if not can_read:
                    continue
                outcome = self._read_plate(
                    frame,
                    track,
                    plate_detections,
                    used_plate_indices,
                    frame_index,
                    timestamp,
                )
                if outcome is None:
                    continue
                record, candidates = outcome
                readings.append(record)
                meta["is_two_line"] = bool(meta.get("is_two_line", False)) or record.is_two_line
                for candidate in candidates:
                    enriched = dict(candidate)
                    enriched["track_id"] = int(track.track_id)
                    enriched["frame_index"] = int(frame_index)
                    enriched["timestamp"] = float(timestamp)
                    self._readings_by_track[track.track_id].append(enriched)
                    meta["readings"].append(enriched)

            if readings:
                self._reading_frames += 1

        # --- Module 7: overlay (tùy chọn) ---------------------------------- #
        if self.config.video.save_frame_overlays and tracks:
            if best_frame_enabled:
                # Chưa có text (OCR dời về cuối) -> lưu frame gốc + log để post-pass vẽ lại.
                self._overlay_log[int(frame_index)] = [
                    (int(track.track_id), [int(v) for v in track.bbox], str(track.label))
                    for track in tracks
                ]
                self._save_overlay(frame_index, frame)
            else:
                annotated = self._annotate(frame, tracks, readings)
                self._save_overlay(frame_index, annotated)

        return FramePipelineResult(
            frame_index=frame_index,
            timestamp=timestamp,
            tracks=list(tracks),
            readings=readings,
            frame=frame,
        )

    # ------------------------------------------------------------------ #
    def _finalize_tracks(self) -> List[TrackResult]:
        """Gộp (fusion) hoặc vote cả chuỗi theo ``track_id`` rồi đóng gói ``TrackResult``."""
        fusion_enabled = bool(self.config.fusion.enabled)
        outcomes: Dict[int, Any] = {}
        for track_id, items in self._readings_by_track.items():
            if not items:
                continue
            if fusion_enabled:
                outcomes[track_id] = fuse_readings(
                    items, self.config.fusion, self.config.postprocess
                )
            else:
                outcomes[track_id] = self.postprocessor.process(items, group_id=str(track_id))

        results: List[TrackResult] = []
        for track_id, meta in self._track_meta.items():
            outcome = outcomes.get(track_id)
            two_line = bool(meta.get("is_two_line", False))

            if fusion_enabled:
                fused = outcome if outcome is not None else FusionResult()
                text = str(fused.text or "")
                formatted = self.postprocessor.format(text, is_two_line=two_line) if text else ""
                confidence = float(fused.mean_confidence or 0.0)
                valid = bool(fused.valid)
                reliable = bool(fused.reliable)
                vote_count = int(fused.vote_count or 0)
                total_readings = int(fused.total_readings or 0)
                accepted_readings = int(fused.accepted_readings or 0)
                per_char_confidence = list(fused.per_char_confidence)
                supporting_frames = int(fused.supporting_frames or 0)
                fusion_notes = list(fused.notes)
                fusion_used = True
            else:
                vote = outcome
                text = str(getattr(vote, "text", "") or "")
                formatted = self.postprocessor.format(text, is_two_line=two_line) if text else ""
                confidence = float(getattr(vote, "mean_confidence", 0.0) or 0.0)
                valid = bool(getattr(vote, "valid", False))
                reliable = bool(getattr(vote, "reliable", False))
                vote_count = int(getattr(vote, "vote_count", 0) or 0)
                total_readings = int(getattr(vote, "total_readings", 0) or 0)
                accepted_readings = int(getattr(vote, "accepted_readings", 0) or 0)
                per_char_confidence = None
                supporting_frames = 0
                fusion_notes = []
                fusion_used = False

            results.append(
                TrackResult(
                    track_id=int(track_id),
                    text=text,
                    formatted=formatted,
                    is_two_line=two_line,
                    confidence=confidence,
                    valid=valid,
                    reliable=reliable,
                    vote_count=vote_count,
                    total_readings=total_readings,
                    accepted_readings=accepted_readings,
                    hits=int(meta.get("hits", 0)),
                    class_id=int(meta.get("class_id", -1)),
                    label=str(meta.get("label", "vehicle")),
                    first_frame=meta.get("first_frame"),
                    last_frame=meta.get("last_frame"),
                    first_timestamp=meta.get("first_timestamp"),
                    last_timestamp=meta.get("last_timestamp"),
                    vehicle_bbox=meta.get("vehicle_bbox"),
                    readings=list(meta.get("readings", [])),
                    per_char_confidence=per_char_confidence,
                    supporting_frames=supporting_frames,
                    fusion_notes=fusion_notes,
                    fusion_used=fusion_used,
                )
            )
        results.sort(
            key=lambda item: (
                item.first_frame if item.first_frame is not None else 0,
                item.track_id,
            )
        )
        return results

    # ------------------------------------------------------------------ #
    def run(self, video_path: str, keep_frame_images: bool = False) -> VideoPipelineResult:
        """Chạy toàn bộ pipeline video và trả kết quả đầy đủ.

        Args:
            video_path: Đường dẫn video đầu vào.
            keep_frame_images: Nếu ``True``, giữ ảnh frame trong mỗi
                :class:`FramePipelineResult` (tốn RAM với video dài). Mặc định
                ``False`` -> giải phóng ảnh ngay sau khi xử lý để tránh OOM.

        Returns:
            :class:`VideoPipelineResult` (tracks đã voting + thống kê + cảnh báo).

        Raises:
            FileNotFoundError: Nếu ``video_path`` không tồn tại.
            ValueError: Nếu không mở/đọc được video.
        """
        extraction = self.extractor.extract(video_path)
        self.reset()

        frames: List[FramePipelineResult] = []
        readings: List[TrackReading] = []
        for record in extraction.frames:  # type: FrameRecord
            frame_result = self.process_frame(record.frame, record.frame_index, record.timestamp)
            readings.extend(frame_result.readings)
            if not keep_frame_images:
                # Bỏ tham chiếu tới ảnh đầy đủ để KHÔNG giữ cả video trong RAM
                # (overlay đã được lưu ngay trong ``process_frame`` nếu bật
                # ``save_frame_overlays``). Tránh rò rỉ RAM / OOM với video dài:
                # ``FramePipelineResult`` không còn giữ ảnh và ``FrameRecord`` gốc
                # của Module 0 cũng được giải phóng dần.
                frame_result.frame = None
                record.frame = None
            frames.append(frame_result)

        self._flush_all_tracks()
        tracks = self._finalize_tracks()
        if self.config.video.save_frame_overlays and self.config.best_frame.enabled:
            self._annotate_overlays_post(tracks)
        if self.config.best_frame.debug:
            self._report_best_frame_debug(tracks)

        warnings = list(extraction.warnings)
        for warning in self._setup_warnings + self._warnings:
            if warning not in warnings:
                warnings.append(warning)

        stats: Dict[str, Any] = {
            "total_frames": int(extraction.stats.total_frames),
            "scanned_frames": int(extraction.stats.scanned_frames),
            "kept_frames": int(extraction.stats.kept_frames),
            "reading_frames": int(self._reading_frames),
            "detected_tracks": len(self._track_meta),
            "tracks_with_plate": sum(1 for track in tracks if track.text),
            "total_readings": sum(len(items) for items in self._readings_by_track.values()),
            "vehicle_detector": self.vehicle_detector.name,
            "plate_detector": getattr(self.plate_detector, "model_path", None),
            "tracker": "SORT",
        }

        return VideoPipelineResult(
            video_path=extraction.video_path,
            frames=frames,
            tracks=tracks,
            readings=readings,
            stats=stats,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    def iter_frames(self, video_path: str):
        """Generator xử lý từng frame và yield :class:`FramePipelineResult`.

        Hữu ích khi muốn xử lý streaming (không giữ toàn bộ video trong RAM).
        """
        self.reset()
        for record in self.extractor.iter_frames(video_path):  # type: FrameRecord
            yield self.process_frame(record.frame, record.frame_index, record.timestamp)


def build_video_pipeline(config: Optional[PipelineConfig] = None, **kwargs: Any) -> VideoPipeline:
    """Tiện dụng: tạo :class:`VideoPipeline` từ config (cho phép inject module)."""
    return VideoPipeline(config=config, **kwargs)


def run_video_pipeline(video_path: str, config: Optional[PipelineConfig] = None, **kwargs: Any) -> VideoPipelineResult:
    """Tiện dụng: chạy pipeline video 1 phát và trả kết quả."""
    return VideoPipeline(config=config, **kwargs).run(video_path)
