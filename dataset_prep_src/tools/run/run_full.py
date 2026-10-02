"""
tools/run_full.py  (chạy .venv-datasets)
=======================================
Chạy pipeline FULL video videodemo2.mp4 (13 phút) với CornerRegressor + PaddleOCR
(subprocess, cách B). Xuất results.{json,csv,txt} + plate_crops/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2

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
OUT = Path(r"S:\M_AGENT\CV\runs\e2e_full")


def g(t, name, default=None):
    if hasattr(t, name):
        return getattr(t, name)
    if isinstance(t, dict):
        return t.get(name, default)
    return default


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig()
    cfg.vehicle_detection.model_path = str(MODELS / "yolo11n.pt")
    cfg.vehicle_detection.device = "0"
    cfg.detection_model = str(MODELS / "plate_detector.pt")
    cfg.detection_device = "0"
    cfg.detection_conf = 0.25
    cfg.preprocess.bbox_margin = 0.15
    cfg.output_dir = str(OUT)
    cfg.frame_extraction.frame_interval = 5
    cfg.video.save_frame_overlays = True

    engine = SubprocessPaddleEngine()
    reader = PlateReader(cfg, ocr_engine=engine)
    pipeline = VideoPipeline(cfg, reader=reader)
    pipeline.set_corner_regressor(
        CornerRegressor(CORNER_BEST, conf=0.25, device="0", imgsz=160),
        angle_threshold=15.0)

    print(f"[i] chay FULL video {VIDEO} ...", flush=True)
    result = pipeline.run(VIDEO, keep_frame_images=False)

    data = result.to_dict(include_frames=False)
    tracks = data["tracks"]
    (OUT / "results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [{f: g(t, f) for f in
             ("track_id", "plate_text", "plate_display", "confidence",
              "valid_format", "reliable", "is_two_line", "vote_count",
              "first_timestamp", "last_timestamp")} for t in tracks]
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    with_text = [t for t in tracks if (g(t, "plate_text") or g(t, "plate_display"))]
    lines = [f"Tong tracks: {len(tracks)} | co text: {len(with_text)}", ""]
    for t in with_text:
        lines.append(f"track {g(t, 'track_id')}: {g(t, 'plate_display') or g(t, 'plate_text')!r} "
                     f"conf={g(t, 'confidence')} reliable={g(t, 'reliable')}")
    (OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== KET QUA FULL VIDEO ===")
    print(f"tong tracks: {len(tracks)} | co text: {len(with_text)}")
    for t in with_text:
        print(f"  track {g(t, 'track_id')}: {g(t, 'plate_display') or g(t, 'plate_text')!r} "
              f"conf={g(t, 'confidence')}")

    # gộp frame overlay thành video
    frames_dir = OUT / "tracks_frames"
    files = sorted(list(frames_dir.glob("frame_*.png"))) if frames_dir.exists() else []
    if files:
        first = cv2.imread(str(files[0]))
        h, w = first.shape[:2]
        video_out = OUT / "output_demo.mp4"
        writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 30.0, (w, h))
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                writer.write(img)
        writer.release()
        print(f"[i] output video -> {video_out} ({len(files)} frames)")

    engine.close()
    print(f"\n[i] output -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
