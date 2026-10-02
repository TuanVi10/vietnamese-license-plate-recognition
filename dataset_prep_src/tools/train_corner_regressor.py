"""
tools/train_corner_regressor.py
===============================
Train corner regressor (yolo11n-pose, 4 keypoint = 4 góc biển) trên CCPD-warped.

    python tools/train_corner_regressor.py --epochs 100 --batch 64 --imgsz 160
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO

DATA_YAML = "S:/M_AGENT/CV/datasets/corner_regression/data.yaml"
PROJECT = "S:/M_AGENT/CV/runs/corner_regression"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="corner_reg")
    args = ap.parse_args()

    model = YOLO("yolo11n-pose.pt")
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        project=PROJECT,
        patience=30,
        seed=0,
        exist_ok=True,
        plots=True,
        val=True,
    )
    print(f"[i] xong -> {PROJECT}/{args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
