"""
tools/diagnose_pose_errors.py
=============================
CÔNG CỤ CHẨN ĐOÁN (READ-ONLY — KHÔNG sửa/không nối vào pipeline).

Trả lời 3 câu hỏi:

* **Q4** — quad_IoU thấp là do **(a)** dò ĐÚNG biển nhưng 4 điểm hồi quy sai,
  hay **(b)** dò NHẦM vật thể rồi mới tính góc trên vùng sai?
* **Q5** — ở ca IoU thấp, bbox suy TỪ 4 GÓC dự đoán có trùng vị trí biển thật
  không (nếu có ⇒ lỗi hồi quy góc thuần, không phải lỗi dò biển)?
* **Q6** — dùng thẳng bbox từ ``models/plate_detector.pt`` (cách CŨ, không qua
  pose) có tệ hơn 4 góc của pose model trên cùng tập val không?

Ba IoU tách biệt dùng để phân loại:
    ``iou_box``   = IoU(bbox do pose phát ra, bbox GT)  -> chất lượng DÒ
    ``iou_quad``  = IoU(tứ giác 4 góc, tứ giác GT)      -> chất lượng TỔNG THỂ
    ``iou_kpbox`` = IoU(bbox suy từ 4 góc, bbox GT)     -> trả lời Q5

Xuất: ``runs/pose_ccpd/diagnose_<tag>/`` gồm ``summary.json``, ``per_image.csv``
và ảnh minh hoạ ``examples_a_regression/``, ``examples_b_falsepos/``, ``examples_good/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_pose_bands import (  # noqa: E402
    IMAGES_DIR, LABEL_DIR, load_split, quad_iou, rows_with_plate,
)


def bbox_of(quad: np.ndarray) -> np.ndarray:
    """Bbox axis-aligned bao quanh 1 tứ giác: (x0, y0, x1, y1)."""
    return np.array([quad[:, 0].min(), quad[:, 1].min(),
                     quad[:, 0].max(), quad[:, 1].max()], np.float32)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU 2 bbox axis-aligned."""
    x0, y0 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x1, y1 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union) if union > 0 else 0.0


def bbox_as_quad(bb: np.ndarray) -> np.ndarray:
    """Bbox -> tứ giác (để tính quad IoU với GT)."""
    return np.array([[bb[0], bb[1]], [bb[2], bb[1]],
                     [bb[2], bb[3]], [bb[0], bb[3]]], np.float32)


def best_instance(result):
    """Trả (bbox xyxy, keypoints xy, conf) của instance tự tin nhất."""
    if result.boxes is None or len(result.boxes) == 0:
        return None, None, 0.0
    confs = result.boxes.conf.cpu().numpy()
    i = int(np.argmax(confs))
    bb = result.boxes.xyxy.cpu().numpy()[i].astype(np.float32)
    kp = None
    if result.keypoints is not None and result.keypoints.xy is not None:
        kxy = result.keypoints.xy.cpu().numpy()
        if len(kxy) > i:
            kp = kxy[i].astype(np.float32)
    return bb, kp, float(confs[i])


def draw_overlay(img, gt, pred_kp, pred_bb, old_bb, caption):
    """Vẽ GT (xanh), pose quad (đỏ), pose bbox (lam), bbox detector cũ (vàng)."""
    h, w = img.shape[:2]
    scale = int(max(1, min(6, round(220.0 / max(1, min(h, w))))))
    out = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def S(q):
        return (np.asarray(q, np.float32) * scale).astype(np.int32)

    cv2.polylines(out, [S(gt)], True, (0, 255, 0), 2)
    if old_bb is not None:
        cv2.rectangle(out, tuple(S(old_bb[:2])), tuple(S(old_bb[2:])), (0, 255, 255), 1)
    if pred_bb is not None:
        cv2.rectangle(out, tuple(S(pred_bb[:2])), tuple(S(pred_bb[2:])), (255, 0, 0), 1)
    if pred_kp is not None:
        cv2.polylines(out, [S(pred_kp)], True, (0, 0, 255), 2)
        for p in S(pred_kp):
            cv2.circle(out, tuple(p), 3, (0, 0, 255), -1)
    cv2.putText(out, caption, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, caption, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Chẩn đoán lỗi pose (read-only).")
    ap.add_argument("--split", default="val", choices=["val", "test", "all"])
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng cho CẢ pose và detector cũ")
    ap.add_argument("--pose", default=r"S:\M_AGENT\CV\runs\pose_ccpd\ccpd_pose_augmented\weights\best.pt")
    ap.add_argument("--pose-tag", default="augmented")
    ap.add_argument("--plate-model", default=r"S:\M_AGENT\CV\models\plate_detector.pt")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(r"S:\M_AGENT\CV\runs\pose_ccpd") / f"diagnose_{args.pose_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    pose = YOLO(args.pose)
    plate = YOLO(args.plate_model)

    rows = rows_with_plate(LABEL_DIR / "labeled.csv")
    split_map = load_split(LABEL_DIR / "split.csv")
    if args.split != "all":
        rows = [r for r in rows if split_map.get(r["image"]) == args.split]

    saved = {"good": 0, "a_regression": 0, "b_falsepos": 0, "miss": 0}
    cls_counter: Counter = Counter()
    per_image = []
    old_q, old_b = [], []
    pose_q, pose_b, pose_kb = [], [], []
    old_found = 0

    for row in rows:
        img = cv2.imread(str(IMAGES_DIR / row["image"]))
        if img is None:
            continue
        gt = row["quad"]
        gt_bb = bbox_of(gt)

        rp = pose.predict(img, conf=args.conf, imgsz=640, verbose=False, save=False)[0]
        p_bb, p_kp, p_conf = best_instance(rp)
        ro = plate.predict(img, conf=args.conf, imgsz=640, verbose=False, save=False)[0]
        o_bb, _, o_conf = best_instance(ro)

        iou_quad = quad_iou(p_kp, gt) if p_kp is not None else None
        iou_box = bbox_iou(p_bb, gt_bb) if p_bb is not None else None
        iou_kpbox = bbox_iou(bbox_of(p_kp), gt_bb) if p_kp is not None else None

        if p_kp is None:
            cls = "miss"
        elif iou_quad >= 0.5:
            cls = "good"
        elif iou_box is not None and iou_box >= 0.5:
            cls = "a_regression"
        else:
            cls = "b_falsepos"
        cls_counter[cls] += 1

        if iou_quad is not None:
            pose_q.append(iou_quad)
            pose_b.append(iou_box)
            pose_kb.append(iou_kpbox)
        if o_bb is not None:
            old_found += 1
            old_q.append(quad_iou(bbox_as_quad(o_bb), gt))
            old_b.append(bbox_iou(o_bb, gt_bb))

        caption = (f"{cls} c={p_conf:.2f} qIoU={0 if iou_quad is None else iou_quad:.2f} "
                   f"bIoU={0 if iou_box is None else iou_box:.2f} ang={row['angle']:.0f}")
        per_image.append({
            "image": row["image"], "cls": cls, "band": row["band"],
            "gt_angle": round(float(row["angle"]), 2),
            "pose_conf": round(p_conf, 4), "old_conf": round(o_conf, 4),
            "quad_iou": None if iou_quad is None else round(iou_quad, 4),
            "box_iou": None if iou_box is None else round(iou_box, 4),
            "kpbox_iou": None if iou_kpbox is None else round(iou_kpbox, 4),
            "old_quad_iou": None if o_bb is None else round(quad_iou(bbox_as_quad(o_bb), gt), 4),
            "old_box_iou": None if o_bb is None else round(bbox_iou(o_bb, gt_bb), 4),
        })

        if saved[cls] < args.n_examples:
            folder = out_dir / f"examples_{cls}"
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / row["image"]),
                        draw_overlay(img, gt, p_kp, p_bb, o_bb, caption))
            saved[cls] += 1

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    n = len(rows)
    summary = {
        "split": args.split, "conf": args.conf, "pose_tag": args.pose_tag,
        "n_images": n,
        "class_counts": dict(cls_counter),
        "class_pct": {k: round(100.0 * v / n, 2) for k, v in cls_counter.items()},
        "pose_model": {
            "detected": len(pose_q),
            "recall": round(len(pose_q) / n, 4) if n else 0.0,
            "mean_quad_iou": mean(pose_q),
            "mean_box_iou": mean(pose_b),
            "mean_kpbox_iou": mean(pose_kb),
        },
        "old_plate_detector": {
            "detected": old_found,
            "recall": round(old_found / n, 4) if n else 0.0,
            "mean_quad_iou": mean(old_q),
            "mean_box_iou": mean(old_b),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(per_image[0].keys()))
        wr.writeheader()
        wr.writerows(per_image)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[i] output -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
