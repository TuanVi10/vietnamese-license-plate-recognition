"""
tools/test_e2e.py  (chạy .venv-datasets)
=======================================
Test end-to-end: VideoPipeline + CornerRegressor + PaddleOCR (subprocess, cách B)
trên videodemo2.mp4.
"""
from __future__ import annotations

import sys
from pathlib import Path

DSRC = Path(__file__).resolve().parent.parent.parent  # dataset_prep_src
sys.path.insert(0, str(DSRC))

from src.config import PipelineConfig
from src.corner_regressor import CornerRegressor
from src.ocr_subprocess import SubprocessPaddleEngine
from src.plate_reader import PlateReader
from src.video_pipeline import VideoPipeline

MODELS = Path(r"S:\M_AGENT\CV\models")
CORNER_BEST = r"S:\M_AGENT\CV\runs\corner_regression\corner_reg100\weights\best.pt"
VIDEO = r"S:\M_AGENT\CV\videodemo2.mp4"


def main() -> int:
    cfg = PipelineConfig()
    cfg.vehicle_detection.model_path = str(MODELS / "yolo11n.pt")
    cfg.vehicle_detection.device = "0"
    cfg.detection_model = str(MODELS / "plate_detector.pt")
    cfg.detection_device = "0"
    cfg.detection_conf = 0.25
    # quyết định đã chốt: crop có lề 15%
    cfg.preprocess.bbox_margin = 0.15
    # giới hạn để test nhanh
    cfg.frame_extraction.frame_interval = 5
    cfg.frame_extraction.max_duration = 60.0
    cfg.video.max_reading_frames = 50

    engine = SubprocessPaddleEngine()
    reader = PlateReader(cfg, ocr_engine=engine)
    pipeline = VideoPipeline(cfg, reader=reader)
    pipeline.set_corner_regressor(
        CornerRegressor(CORNER_BEST, conf=0.25, device="0", imgsz=160),
        angle_threshold=15.0)

    print(f"[i] chay pipeline tren {VIDEO} ...")
    result = pipeline.run(VIDEO, keep_frame_images=False)

    print("\n=== KET QUA (tracks) ===")
    for t in result.tracks:
        tid = getattr(t, "track_id", None)
        text = (getattr(t, "plate_text", None) or getattr(t, "plate_display", None)
                or getattr(t, "formatted", None) or "")
        conf = getattr(t, "confidence", None)
        print(f"track {tid}: text={text!r} conf={conf}")

    engine.close()
    print("\n[i] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
