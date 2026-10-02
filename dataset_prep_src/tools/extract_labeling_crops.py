"""
tools/extract_labeling_crops.py
===============================
Trích crop biển số từ 1 video để gán nhãn (4 góc + text) — KHÔNG chạy OCR.

Dùng lại cơ chế best-frame của ``VideoPipeline`` (dò xe + dò biển + tracking +
chấm điểm + TopKBuffer) nhưng BỎ qua bước OCR: chỉ lưu ảnh crop + manifest.

Chạy::

    python tools/extract_labeling_crops.py --video videodemo2.mp4 --output-dir labeling

Output:

    <output-dir>/images/000001.jpg ...   (tên tăng dần, dễ gửi cho nhóm labeling)
    <output-dir>/manifest.csv            (image, track_id, frame_index, source, score, w, h)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig  # noqa: E402
from src.image_utils import imwrite_unicode  # noqa: E402
from src.video_pipeline import VideoPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trích crop biển số để labeling (không OCR).")
    ap.add_argument("--video", required=True, help="đường dẫn video đầu vào")
    ap.add_argument("--output-dir", default=str(ROOT.parent / "labeling"), help="thư mục chứa images/ + manifest.csv")
    ap.add_argument("--vehicle-model", default=str(ROOT.parent / "models/yolo11n.pt"))
    ap.add_argument("--plate-model", default=str(ROOT.parent / "models/plate_detector.pt"))
    ap.add_argument("--device", default="0", help="0 = GPU, cpu = CPU")
    ap.add_argument("--frame-interval", type=int, default=10, help="bước nhảy frame (10 = ~6fps; 5 = ~12fps dày hơn)")
    ap.add_argument("--k", type=int, default=3, help="số crop giữ lại mỗi track")
    ap.add_argument("--margin", type=float, default=0.15, help="nới rộng bbox để thấy rõ 4 góc biển")
    ap.add_argument("--log-every", type=int, default=500, help="in tiến trình mỗi N frame")
    ap.add_argument("--flush-every", type=int, default=300, help="lưu ảnh tạm + progress.txt mỗi N frame")
    return ap.parse_args()


def _snapshot(pipeline: VideoPipeline, img_dir: Path, progress_path: Path, done_frames: int, total_frames: int) -> int:
    """Lưu tạm crop hiện có (tên track*) + ghi progress.txt; trả số crop đã lưu."""
    n_crops = 0
    for track_id in sorted(pipeline._best_buffers.keys()):
        for rank, cand in enumerate(pipeline._best_buffers[track_id].best()):
            if cand.crop is None or cand.crop.size == 0:
                continue
            name = f"track{int(track_id):05d}_r{rank}_f{int(cand.frame_index):06d}.jpg"
            imwrite_unicode(str(img_dir / name), cand.crop)
            n_crops += 1
    progress_path.write_text(
        f"{done_frames}/{total_frames} frame | {len(pipeline._best_buffers)} track | {n_crops} crop\n",
        encoding="utf-8",
    )
    return n_crops


def main() -> int:
    args = parse_args()

    out = Path(args.output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.txt"

    cfg = PipelineConfig()
    cfg.vehicle_detection.model_path = args.vehicle_model
    cfg.vehicle_detection.device = args.device
    cfg.detection_model = args.plate_model
    cfg.detection_device = args.device
    cfg.frame_extraction.frame_interval = args.frame_interval
    # Cần read_plates=True để chạy Module 2 (dò biển) + best-frame thu thập; nhưng ta
    # KHÔNG gọi _flush_all_tracks nên không có bước OCR nào chạy.
    cfg.video.read_plates = True
    cfg.best_frame.enabled = True
    cfg.best_frame.debug = False
    cfg.best_frame.k = int(args.k)
    cfg.preprocess.bbox_margin = float(args.margin)

    pipeline = VideoPipeline(cfg)
    pipeline.reset()

    # Pha THU THẬP (không OCR): lặp từng frame qua process_frame.
    extraction = pipeline.extractor.extract(args.video)
    total_frames = len(extraction.frames)
    for index, record in enumerate(extraction.frames):
        pipeline.process_frame(record.frame, record.frame_index, record.timestamp)
        done = index + 1
        if args.flush_every > 0 and done % args.flush_every == 0:
            n_crops = _snapshot(pipeline, img_dir, progress_path, done, total_frames)
            print(f"  {done}/{total_frames} frame | {len(pipeline._best_buffers)} track | {n_crops} crop", flush=True)

    # Xoá ảnh tạm (track*.jpg) của các lần flush, chuyển sang tên tuần tự.
    for stale in img_dir.glob("track*.jpg"):
        stale.unlink()

    # Lưu crop + manifest.
    fieldnames = ["image", "track_id", "rank", "frame_index", "source", "score", "width", "height"]
    manifest_path = out / "manifest.csv"
    seq = 0
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for track_id in sorted(pipeline._best_buffers.keys()):
            buffer = pipeline._best_buffers[track_id]
            for rank, cand in enumerate(buffer.best()):
                if cand.crop is None or cand.crop.size == 0:
                    continue
                seq += 1
                name = f"{seq:06d}.jpg"
                imwrite_unicode(str(img_dir / name), cand.crop)
                writer.writerow({
                    "image": name,
                    "track_id": int(track_id),
                    "rank": int(rank),
                    "frame_index": int(cand.frame_index),
                    "source": str(cand.source),
                    "score": round(float(cand.score), 4),
                    "width": int(cand.crop.shape[1]),
                    "height": int(cand.crop.shape[0]),
                })

    progress_path.write_text(f"HOÀN TẤT: {seq} crop\n", encoding="utf-8")
    print(f"Đã lưu {seq} crop vào {img_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Stats: {len(pipeline._best_buffers)} track có crop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
