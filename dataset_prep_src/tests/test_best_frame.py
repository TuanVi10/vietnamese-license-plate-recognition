"""
tests/test_best_frame.py
========================
Pytest cho cơ chế chọn top-K crop biển số tốt nhất (``src/best_frame.py``).

Chạy bằng::

    python -m pytest tests/test_best_frame.py -v

Chỉ dùng numpy + opencv (ảnh tổng hợp), không cần dữ liệu ngoài.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.best_frame import PlateCandidate, TopKBuffer, score_crop  # noqa: E402
from src.config import BestFrameConfig, PipelineConfig  # noqa: E402


def make_plate_crop(width: int = 200, height: int = 50, blur_ksize: int = 0) -> np.ndarray:
    """Tạo 1 crop biển tổng hợp: nền sáng + các thanh dọc tối + 1 chấm."""
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    count = 6
    step = max(6, width // (count + 1))
    for index in range(count):
        x = step // 2 + index * step
        bar_w = max(3, width // 40)
        cv2.rectangle(image, (x, height // 6), (x + bar_w, height - height // 6), (20, 20, 20), -1)
    cv2.circle(image, (width - 12, height // 2), max(2, height // 10), (20, 20, 20), -1)
    if blur_ksize > 0:
        image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    return image


def _config(**kwargs) -> BestFrameConfig:
    config = BestFrameConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def _candidate(frame_index: int, score: float, source: str = "detection") -> PlateCandidate:
    det_conf = 0.0 if source == "fallback" else 0.8
    return PlateCandidate(
        frame_index=frame_index,
        crop=np.zeros((10, 10, 3), dtype=np.uint8),
        plate_bbox=[0, 0, 10, 10],
        det_conf=det_conf,
        score=score,
        source=source,
    )


# --------------------------------------------------------------------------- #
# score_crop
# --------------------------------------------------------------------------- #
def test_score_prefers_sharp_large_over_blurred_small():
    config = _config()
    sharp = make_plate_crop(200, 50)
    blurred = make_plate_crop(200, 50, blur_ksize=15)
    small = make_plate_crop(60, 15)

    sharp_score = score_crop(sharp, (0, 0, 200, 50), 0.9, 4.3, config).total
    blurred_score = score_crop(blurred, (0, 0, 200, 50), 0.9, 4.3, config).total
    small_score = score_crop(small, (0, 0, 60, 15), 0.9, 4.3, config).total

    assert sharp_score > blurred_score
    assert sharp_score > small_score


def test_blur_and_shrink_decrease_score():
    config = _config()
    base = make_plate_crop(220, 55)
    blurred = cv2.GaussianBlur(base, (21, 21), 0)
    shrunk = cv2.resize(base, (70, 18), interpolation=cv2.INTER_AREA)

    base_score = score_crop(base, (0, 0, 220, 55), 0.9, 4.3, config).total
    blur_score = score_crop(blurred, (0, 0, 220, 55), 0.9, 4.3, config).total
    shrink_score = score_crop(shrunk, (0, 0, 70, 18), 0.9, 4.3, config).total

    assert blur_score < base_score
    assert shrink_score < base_score


def test_score_components_clipped_to_01():
    config = _config()
    detail = score_crop(make_plate_crop(200, 50), (0, 0, 200, 50), 1.0, 4.3, config)
    for component in (
        detail.total,
        detail.sharpness,
        detail.height,
        detail.aspect,
        detail.confidence,
        detail.exposure,
    ):
        assert 0.0 <= component <= 1.0


def test_score_empty_crop_is_zero():
    config = _config()
    detail = score_crop(np.zeros((0, 0, 3), dtype=np.uint8), (0, 0, 0, 0), 0.9, 4.3, config)
    assert detail.total == 0.0


# --------------------------------------------------------------------------- #
# TopKBuffer
# --------------------------------------------------------------------------- #
def test_topk_keeps_k_and_gap():
    buffer = TopKBuffer(k=3, min_frame_gap=5)
    assert buffer.add(_candidate(10, 0.5))
    assert buffer.add(_candidate(20, 0.6))
    assert buffer.add(_candidate(30, 0.7))
    assert len(buffer) == 3
    # frame 12 cách frame 10 đúng 2 < gap 5 -> xung đột, điểm thấp -> bị loại.
    assert not buffer.add(_candidate(12, 0.1))
    assert len(buffer) == 3


def test_topk_replaces_lowest():
    buffer = TopKBuffer(k=2, min_frame_gap=0)
    buffer.add(_candidate(1, 0.3))
    buffer.add(_candidate(2, 0.6))
    assert buffer.add(_candidate(3, 0.5))  # 0.5 > 0.3 -> thay frame 1
    assert len(buffer) == 2
    scores = sorted(c.score for c in buffer.best())
    assert scores == [0.5, 0.6]


def test_topk_conflict_two_crops_replace_only_if_higher_than_all():
    buffer = TopKBuffer(k=3, min_frame_gap=10)
    buffer.add(_candidate(0, 0.5))
    buffer.add(_candidate(18, 0.6))  # |18-0|=18 >= 10 -> cùng tồn tại
    # frame 9 xung đột với CẢ frame 0 (|9-0|=9<10) và frame 18 (|9-18|=9<10).
    assert not buffer.add(_candidate(9, 0.55))  # không cao hơn cả 2 -> loại
    assert len(buffer) == 2
    assert buffer.add(_candidate(9, 0.7))  # cao hơn cả 2 -> xoá cả 2, thêm frame 9
    assert len(buffer) == 1
    assert buffer.best()[0].frame_index == 9


def test_fallback_rejected_when_detection_present():
    buffer = TopKBuffer(k=2, min_frame_gap=0)
    assert buffer.add(_candidate(1, 0.3, "detection"))
    # fallback điểm cao hơn vẫn thua detection thật.
    assert not buffer.add(_candidate(2, 0.9, "fallback"))
    assert all(c.source == "detection" for c in buffer.best())


def test_fallback_purged_when_detection_arrives():
    buffer = TopKBuffer(k=3, min_frame_gap=0)
    buffer.add(_candidate(1, 0.5, "fallback"))
    buffer.add(_candidate(2, 0.6, "fallback"))
    buffer.add(_candidate(3, 0.7, "detection"))
    assert all(c.source == "detection" for c in buffer.best())


# --------------------------------------------------------------------------- #
# Tích hợp video_pipeline: reader.detector=None + flow best-frame/legacy
# --------------------------------------------------------------------------- #
class _StubOCRItem:
    def __init__(self, text: str, confidence: float = 0.9):
        self.text = text
        self.raw_text = text
        self.confidence = confidence


class _StubOCRResult:
    def __init__(self, text: str = "51F12345"):
        self.text = text
        self.raw_text = text
        self.confidence = 0.9
        self.engine = "stub"
        self.items = [_StubOCRItem(text)]


class _StubEngine:
    name = "stub"

    def recognize(self, image):
        return _StubOCRResult()


def _make_scene(width: int = 320, height: int = 240) -> np.ndarray:
    """Khung cảnh tổng hợp có 1 vùng biển sáng ở nửa dưới."""
    frame = np.full((height, width, 3), 80, dtype=np.uint8)
    cv2.rectangle(frame, (80, 160), (240, 200), (200, 200, 200), -1)
    cv2.rectangle(frame, (80, 160), (240, 200), (0, 0, 0), 2)
    return frame


def _build_pipeline(enabled: bool):
    from src.detect_vehicle import VehicleDetector
    from src.plate_reader import PlateReader
    from src.video_pipeline import VideoPipeline

    config = PipelineConfig()
    config.best_frame.enabled = enabled
    config.best_frame.k = 3
    config.best_frame.min_frame_gap = 0
    config.tracking.min_hits = 1  # xác nhận track ngay để mọi frame đều có track
    config.output_dir = tempfile.mkdtemp(prefix="best_frame_test_")  # tránh ghi vào results/

    reader = PlateReader(config=config, ocr_engine=_StubEngine(), detector=None)
    pipeline = VideoPipeline(
        config=config,
        vehicle_detector=VehicleDetector(model_path=None),  # chế độ "toàn khung"
        reader=reader,
    )
    return pipeline


def test_reader_detector_is_none_when_reading_crop():
    from src.plate_reader import PlateReader

    reader = PlateReader(config=PipelineConfig(), ocr_engine=_StubEngine(), detector=None)
    assert reader.detector is None
    reading = reader.read(make_plate_crop(200, 50))  # bbox=None -> cả crop là biển
    assert reading is not None
    assert reading.text


def test_no_best_frame_reads_per_frame():
    pipeline = _build_pipeline(enabled=False)
    frame = _make_scene()
    for index in range(4):
        pipeline.process_frame(frame, frame_index=index, timestamp=index * 0.1)
    # Legacy: OCR ngay mỗi frame -> readings có dữ liệu, buffer rỗng.
    assert len(pipeline._best_buffers) == 0
    assert sum(len(v) for v in pipeline._readings_by_track.values()) > 0


def test_best_frame_defers_ocr_to_flush():
    pipeline = _build_pipeline(enabled=True)
    frame = _make_scene()
    for index in range(4):
        pipeline.process_frame(frame, frame_index=index, timestamp=index * 0.1)
    # Thu thập: buffer có dữ liệu, chưa OCR -> readings rỗng.
    assert len(pipeline._best_buffers) > 0
    assert sum(len(v) for v in pipeline._readings_by_track.values()) == 0
    pipeline._flush_all_tracks()
    assert sum(len(v) for v in pipeline._readings_by_track.values()) > 0


def test_best_frame_saves_crops_and_manifest(tmp_path):
    import json

    from src.detect_vehicle import VehicleDetector
    from src.plate_reader import PlateReader
    from src.video_pipeline import VideoPipeline

    config = PipelineConfig()
    config.best_frame.enabled = True
    config.best_frame.k = 3
    config.best_frame.min_frame_gap = 0
    config.tracking.min_hits = 1
    config.output_dir = str(tmp_path / "out")

    reader = PlateReader(config=config, ocr_engine=_StubEngine(), detector=None)
    pipeline = VideoPipeline(
        config=config,
        vehicle_detector=VehicleDetector(model_path=None),
        reader=reader,
    )
    frame = _make_scene()
    for index in range(4):
        pipeline.process_frame(frame, frame_index=index, timestamp=index * 0.1)
    pipeline._flush_all_tracks()

    crops_dir = Path(config.output_dir) / "plate_crops"
    crops = list(crops_dir.glob("track*_rank*.jpg"))
    manifest = crops_dir / "manifest.json"
    assert crops, "phải lưu ít nhất 1 crop"
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data
    for key in ("track_id", "rank", "frame_index", "score", "source", "text"):
        assert key in data[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
