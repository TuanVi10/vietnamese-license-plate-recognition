"""
tests/test_fusion_integration.py
================================
Pytest tích hợp fusion vào ``VideoPipeline._finalize_tracks`` (không cần model/OCR).

Chạy bằng::

    python -m pytest tests/test_fusion_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PipelineConfig  # noqa: E402
from src.video_pipeline import VideoPipeline  # noqa: E402


def _meta():
    return {
        "track_id": 1, "first_frame": 0, "first_timestamp": 0.0,
        "last_frame": 15, "last_timestamp": 1.5, "hits": 5,
        "class_id": 2, "label": "car", "vehicle_bbox": [0, 0, 100, 100],
        "readings": [], "is_two_line": False,
    }


def _pipeline(fusion_enabled, readings):
    cfg = PipelineConfig()
    cfg.fusion.enabled = fusion_enabled
    pipeline = VideoPipeline(cfg)
    pipeline._readings_by_track[1].extend(readings)
    pipeline._track_meta[1] = _meta()
    return pipeline


def test_finalize_without_fusion():
    readings = [
        {"text": "51F12345", "raw_text": "51F-12345", "confidence": 0.9, "frame_index": 0},
        {"text": "51F12345", "raw_text": "51F-12345", "confidence": 0.85, "frame_index": 5},
        {"text": "51F12346", "raw_text": "51F-12346", "confidence": 0.8, "frame_index": 10},
    ]
    res = _pipeline(False, readings)._finalize_tracks()
    assert len(res) == 1
    track = res[0]
    assert track.fusion_used is False
    assert track.per_char_confidence is None
    assert track.supporting_frames == 0
    assert track.text == "51F12345"


def test_finalize_with_fusion():
    readings = [
        {"text": "51F12345", "raw_text": "51F-12345", "confidence": 0.9, "frame_index": 0},
        {"text": "51F12345", "raw_text": "51F-12345", "confidence": 0.85, "frame_index": 5},
        {"text": "51F12346", "raw_text": "51F-12346", "confidence": 0.8, "frame_index": 10},
    ]
    res = _pipeline(True, readings)._finalize_tracks()
    assert len(res) == 1
    track = res[0]
    assert track.fusion_used is True
    assert track.text == "51F12345"
    assert track.supporting_frames == 2  # 2 crop khớp chuỗi cuối
    assert len(track.per_char_confidence) == 8
    assert track.valid is True


def test_finalize_with_fusion_fixes_error():
    """Mỗi lần đọc sai 1 vị trí khác nhau: fusion đúng, vote cũ sai."""
    readings = [
        {"text": "51F12346", "raw_text": "51F-12346", "confidence": 0.9, "frame_index": 0},
        {"text": "51F12355", "raw_text": "51F-12355", "confidence": 0.9, "frame_index": 5},
        {"text": "51F13345", "raw_text": "51F-13345", "confidence": 0.9, "frame_index": 10},
        {"text": "51F22345", "raw_text": "51F-22345", "confidence": 0.9, "frame_index": 15},
    ]
    res = _pipeline(True, readings)._finalize_tracks()
    assert res[0].text == "51F12345"
    assert res[0].fusion_used is True
