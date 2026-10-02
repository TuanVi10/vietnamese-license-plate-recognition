"""
main.py
=======
CLI chạy pipeline nhận diện biển số xe Việt Nam qua video (Module 0 -> 7).

Ví dụ::

    python main.py --video data/samples/test_video.mp4 --device 0 --save-overlays
    python main.py --video in.mp4 --max-frames 60 --ocr-engine easyocr
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
# src/ (pipeline video) nằm trong dataset_prep_src/
sys.path.insert(0, str(ROOT / "dataset_prep_src"))

from src.config import PipelineConfig  # noqa: E402
from src.fusion import load_province_codes  # noqa: E402
from src.video_pipeline import VideoPipeline  # noqa: E402

CSV_FIELDS = [
    "track_id", "plate_text", "plate_display", "confidence", "valid_format",
    "reliable", "is_two_line", "vote_count", "total_readings",
    "accepted_readings", "hits", "class_id", "label",
    "first_frame", "last_frame", "first_timestamp", "last_timestamp",
    "vehicle_bbox", "per_char_confidence", "supporting_frames", "fusion_notes",
    "fusion_used",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALPR video pipeline (Module 0->7)")
    parser.add_argument("--video", required=True, help="path tới video đầu vào")
    parser.add_argument("--vehicle-model", default=str(ROOT / "models/yolo11n.pt"))
    parser.add_argument("--plate-model", default=str(ROOT / "models/plate_detector.pt"))
    parser.add_argument("--device", default="0", help="0 = GPU, cpu = CPU")
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0, help="giới hạn frame đọc biển (0 = tất cả)")
    parser.add_argument("--max-duration", type=float, default=0.0, help="chỉ xử lý N giây đầu video (0 = tất cả)")
    parser.add_argument("--conf-vehicle", type=float, default=0.25)
    parser.add_argument("--conf-plate", type=float, default=0.25)
    parser.add_argument("--ocr-engine", default="easyocr", choices=["auto", "paddle", "easyocr"])
    parser.add_argument("--ocr-gpu", action="store_true", help="chạy OCR trên GPU")
    parser.add_argument("--save-overlays", action="store_true", help="lưu frame đã vẽ + xuất output_demo.mp4")
    parser.add_argument("--no-plate-reading", action="store_true", help="bỏ OCR, chỉ detect + track")
    parser.add_argument("--best-k", type=int, default=None, help="số crop biển tốt nhất giữ mỗi track (mặc định theo config)")
    parser.add_argument("--min-frame-gap", type=int, default=None, help="khoảng cách frame tối thiểu giữa các crop giữ lại")
    parser.add_argument("--no-best-frame", action="store_true", help="tắt best-frame, quay về OCR mọi frame")
    parser.add_argument("--debug-best-frame", action="store_true", help="in chẩn đoán best-frame per-track (so sánh legacy)")
    parser.add_argument("--fusion", dest="fusion", action="store_true", default=None, help="bật fusion (gộp theo vị trí ký tự)")
    parser.add_argument("--no-fusion", dest="fusion", action="store_false", help="tắt fusion, dùng vote cả chuỗi cũ")
    parser.add_argument("--province-codes", default="", help="file mã tỉnh (tuỳ chọn) dùng làm tiêu chí phá hoà khi fusion")
    return parser.parse_args()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, tracks) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for track in tracks:
            data = track.to_dict()
            writer.writerow({field: data.get(field) for field in CSV_FIELDS})


def _fmt_track(track) -> list:
    """Định dạng 1 track thành danh sách dòng `key : value` (dễ đọc)."""
    data = track.to_dict()

    def seconds(value):
        return "—" if value is None else f"{value:.4f}s"

    def frame(value):
        return "—" if value is None else str(value)

    plate = data["plate_display"] or "(không đọc được)"
    lines = [
        f"track_id          : {data['track_id']}",
        f"label             : {data['label']} (class_id={data['class_id']})",
        f"plate_display     : {plate}",
        f"plate_text        : {data['plate_text'] or '(trống)'}",
        f"confidence        : {data['confidence']:.4f}",
        f"valid_format      : {data['valid_format']}",
        f"reliable          : {data['reliable']}",
        f"is_two_line       : {data['is_two_line']}",
        f"vote_count        : {data['vote_count']}",
        f"total_readings    : {data['total_readings']}",
        f"accepted_readings : {data['accepted_readings']}",
        f"hits              : {data['hits']}",
        f"first_frame       : {frame(data['first_frame'])}  ({seconds(data['first_timestamp'])})",
        f"last_frame        : {frame(data['last_frame'])}  ({seconds(data['last_timestamp'])})",
        f"vehicle_bbox      : {data['vehicle_bbox']}",
    ]
    if data.get("fusion_used"):
        lines += [
            f"fusion_used       : True",
            f"supporting_frames : {data.get('supporting_frames')}",
            f"per_char_confidence: {data.get('per_char_confidence')}",
            f"fusion_notes      : {data.get('fusion_notes')}",
        ]
    return lines


def write_report(path: Path, result) -> str:
    """Tạo bản tóm tắt dạng dọc (mỗi track 1 khối key: value)."""
    lines = ["=" * 62, "KẾT QUẢ NHẬN DIỆN BIỂN SỐ (theo từng xe)", "=" * 62, ""]
    for key, value in result.stats.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    if not result.tracks:
        lines.append("(không có xe nào được phát hiện)")
    for track in result.tracks:
        lines.append("-" * 62)
        lines.append(f"TRACK #{track.track_id}")
        lines.append("-" * 62)
        lines.extend(_fmt_track(track))
        lines.append("")
    for warning in result.warnings:
        lines.append(f"[warn] {warning}")
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text



def save_plate_crops(result, output_dir: Path) -> int:
    """Lưu 1 crop biển số tốt nhất (conf cao nhất) cho mỗi track."""
    crops_dir = output_dir / "plate_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    best_by_track = {}
    for reading in result.readings:
        if getattr(reading, "plate_bbox", None) is None:
            continue
        current = best_by_track.get(reading.track_id)
        if current is None or reading.confidence > current.confidence:
            best_by_track[reading.track_id] = reading

    saved = 0
    for track_id, reading in best_by_track.items():
        frame = None
        for fr in result.frames:
            if fr.frame_index == reading.frame_index and fr.frame is not None:
                frame = fr.frame
                break
        if frame is None:
            continue
        x1, y1, x2, y2 = reading.plate_bbox
        height, width = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        cv2.imwrite(str(crops_dir / f"track_{track_id}.jpg"), crop)
        saved += 1
    return saved


def build_video_from_overlays(output_dir: Path, fps: float = 30.0) -> Path:
    """Gộp các frame overlay (PNG) thành output_demo.mp4."""
    frames_dir = output_dir / "tracks_frames"
    files = sorted(list(frames_dir.glob("frame_*.png"))) if frames_dir.exists() else []
    if not files:
        return None
    first = cv2.imread(str(files[0]))
    height, width = first.shape[:2]
    out_path = output_dir / "output_demo.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for file in files:
        image = cv2.imread(str(file))
        if image is not None:
            writer.write(image)
    writer.release()
    return out_path


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"[!] Không tìm thấy video: {args.video}", file=sys.stderr)
        return 2

    cfg = PipelineConfig()
    cfg.vehicle_detection.model_path = args.vehicle_model
    cfg.vehicle_detection.device = args.device
    cfg.vehicle_detection.conf = args.conf_vehicle
    cfg.detection_model = args.plate_model
    cfg.detection_device = args.device
    cfg.detection_conf = args.conf_plate
    cfg.ocr.engine = args.ocr_engine
    cfg.ocr.use_gpu = args.ocr_gpu
    cfg.frame_extraction.frame_interval = args.frame_interval
    cfg.frame_extraction.max_duration = args.max_duration
    cfg.video.save_frame_overlays = args.save_overlays
    cfg.video.read_plates = not args.no_plate_reading
    cfg.video.max_reading_frames = args.max_frames
    if args.best_k is not None:
        cfg.best_frame.k = args.best_k
    if args.min_frame_gap is not None:
        cfg.best_frame.min_frame_gap = args.min_frame_gap
    if args.no_best_frame:
        cfg.best_frame.enabled = False
    if args.debug_best_frame:
        cfg.best_frame.debug = True
    if args.fusion is not None:
        cfg.fusion.enabled = args.fusion
    if args.province_codes:
        cfg.fusion.province_codes = load_province_codes(args.province_codes)
        cfg.fusion.province_codes_path = args.province_codes
    cfg.output_dir = args.output_dir

    print(f"[i] vehicle model: {cfg.vehicle_detection.model_path}")
    print(f"[i] plate model  : {cfg.detection_model}")
    print(f"[i] device       : {args.device} | ocr engine: {cfg.ocr.engine}")

    pipeline = VideoPipeline(cfg)
    result = pipeline.run(args.video, keep_frame_images=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "results.json", result.to_dict(include_frames=False))
    write_csv(output_dir / "results.csv", result.tracks)
    n_crops = save_plate_crops(result, output_dir)

    video_out = None
    if args.save_overlays:
        video_out = build_video_from_overlays(output_dir)

    report = write_report(output_dir / "results.txt", result)
    print(report)
    print("  outputs:")
    print(f"    results.txt   -> {output_dir / 'results.txt'}  (bản dễ đọc)")
    print(f"    results.json  -> {output_dir / 'results.json'}")
    print(f"    results.csv   -> {output_dir / 'results.csv'}")
    print(f"    plate_crops   -> {output_dir / 'plate_crops'} ({n_crops} crop)")
    if video_out:
        print(f"    output_demo   -> {video_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

