"""
tools/train_pose_ccpd.py
========================
Train model YOLO-pose 4 keypoint (chỉnh góc biển số) trên dataset CCPD.

Hai chế độ so sánh:

* ``baseline``  : ``fliplr=0`` + augment MẶC ĐỊNH nhẹ của ultralytics
                  (``degrees=0``, ``perspective=0``) — không thêm xoay/phối cảnh.
* ``augmented`` : giống baseline nhưng bật ``degrees=20`` + ``perspective=0.001``
                  để bù việc 82% ảnh CCPD có |góc| < 5° (xem DATASET_CARD.md).

Cả hai đều ``fliplr=0.0`` (tắt lật ngang); ``data_pose.yaml`` đã khai báo
``flip_idx: [1,0,3,2]`` nên kể cả bật lật ngang cũng đúng thứ tự.

Chạy::

    python tools/train_pose_ccpd.py --mode baseline
    python tools/train_pose_ccpd.py --mode augmented
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

DATASET_YAML = "S:/M_AGENT/CV/datasets/ccpd_pose/data_pose.yaml"
PROJECT_DIR = "S:/M_AGENT/CV/runs/pose_ccpd"

#: Siêu tham số dùng CHUNG cho cả 2 chế độ (khác nhau chỉ ở degrees/perspective).
COMMON_HYP = dict(
    data=DATASET_YAML,
    imgsz=640,
    epochs=100,
    patience=25,
    batch=8,
    workers=4,
    device=0,
    seed=0,
    fliplr=0.0,          # TẮT lật ngang
    flipud=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    mosaic=1.0,
    close_mosaic=10,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    cache=False,
    plots=True,
    val=True,
    exist_ok=True,
    verbose=True,
)

#: Phần khác biệt của chế độ augmented.
AUGMENTED_ONLY = dict(degrees=20.0, perspective=0.001)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train YOLO-pose 4 keypoint trên CCPD.")
    ap.add_argument("--mode", choices=["baseline", "augmented"], required=True)
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Resume tu weights/last.pt (giu nguyen toan bo siêu tham số đã lưu).")
    args = ap.parse_args()

    hyp = dict(COMMON_HYP)
    hyp.update(epochs=args.epochs, batch=args.batch, device=args.device)
    hyp.update(degrees=0.0, perspective=0.0)
    if args.mode == "augmented":
        hyp.update(AUGMENTED_ONLY)

    name = args.name or f"ccpd_pose_{args.mode}"
    print(f"[i] mode={args.mode} name={name}")
    print(f"[i] degrees={hyp['degrees']} perspective={hyp['perspective']} "
          f"fliplr={hyp['fliplr']} epochs={hyp['epochs']} batch={hyp['batch']}")

    if args.resume:
        last = Path(PROJECT_DIR) / name / "weights" / "last.pt"
        if not last.exists():
            print(f"[error] Khong tim thay {last}")
            return 1
        print(f"[i] Resume tu {last} (epoch sau cung da luu; "
              f"giu nguyen degrees/perspective/batch/epochs cua checkpoint)")
        YOLO(str(last)).train(resume=str(last))
        print(f"[i] Xong. Ket qua tiep tuc tai {PROJECT_DIR}/{name}")
        return 0

    model = YOLO(args.model)
    model.train(name=name, project=PROJECT_DIR, **hyp)
    print(f"[i] Xong. Ket qua tai {PROJECT_DIR}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
