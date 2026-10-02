"""
train_detector.py
=================
Fine-tune YOLO11n phát hiện biển số VN trên dataset `datasets/dataset/`.

Bám đúng spec v2 mục 2.2 (Module 2 — Plate Detection):
  - model      : YOLO11n (pretrained COCO, `models/yolo11n.pt`)
  - optimizer  : AdamW
  - box        : 7.5 (box loss gain)
  - fliplr     : 0.0 (BẮT BUỘC TẮT — lật ngang sẽ đảo ngược thứ tự ký tự)
  - mosaic     : 1.0 (mặc định), hsv mặc định, degrees=5.0 (xoay nhẹ)
  - imgsz      : 640 (biển số là object nhỏ)

Mặc định epochs THẤP để chạy demo nhanh; tăng `--epochs` khi train thật.

Chạy::

    python train_detector.py --epochs 3 --batch 8 --device 0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n cho biển số VN")
    parser.add_argument("--data", type=str, default=str(ROOT / "datasets/dataset/data.yaml"))
    parser.add_argument("--model", type=str, default=str(ROOT / "models/yolo11n.pt"))
    parser.add_argument("--epochs", type=int, default=3, help="số epoch (thấp cho demo)")
    parser.add_argument("--batch", type=int, default=8, help="8 cho RTX 3050 6GB; 16 cho Colab T4")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default=str(ROOT / "runs/detect"))
    parser.add_argument("--name", type=str, default="vn_plate_demo")
    parser.add_argument("--output", type=str, default=str(ROOT / "models/plate_detector.pt"))
    parser.add_argument("--resume", action="store_true", help="tiếp tục từ checkpoint last.pt nếu có")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from ultralytics import YOLO

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("[warn] CUDA không sẵn có -> chạy CPU (chậm)")
        device = "cpu"
    if device != "cpu":
        print(f"[ok] device = {device} : {torch.cuda.get_device_name(int(device))}")

    last_pt = Path(args.project) / args.name / "weights" / "last.pt"
    if args.resume and last_pt.exists():
        print(f"[i] resume từ checkpoint: {last_pt}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            optimizer="AdamW",
            box=7.5,
            fliplr=0.0,
            degrees=5.0,
            mosaic=1.0,
            project=args.project,
            name=args.name,
            exist_ok=True,
            seed=42,
        )

    save_dir = Path(getattr(model.trainer, "save_dir", args.project))
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, args.output)
        print(f"[ok] best.pt -> {args.output}")

    metrics = getattr(model.trainer, "metrics", {}) or {}
    for key in (
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
    ):
        if key in metrics:
            print(f"{key} = {float(metrics[key]):.4f}")

    print(f"runs dir: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
