"""
tools/eval_corner_regressor.py  (Stage 1, chạy .venv-datasets)
================================================================
Đánh giá corner regressor trên 213 ảnh VN (401-1701):
  * Đo IoU: iou_quad (4 góc vs GT), iou_kpbox (bbox suy từ 4 góc vs bbox GT).
  * Áp gate theo góc (>15° -> warpPerspective(4 góc), ngược lại affine) -> lưu
    ảnh đã căn chỉnh để Stage 2 (PaddleOCR, .venv-paddle) chấm sim/exact.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DSRC = HERE.parent.parent
sys.path.insert(0, str(DSRC))
sys.path.insert(0, str(HERE))

from eval_pose_bands import IMAGES_DIR, LABEL_DIR, quad_iou  # noqa: E402
from src.image_utils import crop_bbox, expand_bbox, rotate_image, trim_uniform_borders  # noqa: E402
from src.preprocess_plate import estimate_skew_angle  # noqa: E402

MODEL = r"S:\M_AGENT\CV\runs\corner_regression\corner_reg100\weights\best.pt"
OUT = Path(r"S:\M_AGENT\CV\runs\eval_corner")
ANGLE_TH = 15.0


def bbox_of(q):
    return np.array([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()],
                    np.float32)


def bbox_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def band_of(a):
    a = abs(a)
    return "<5" if a < 5 else ("5-15" if a < 15 else ">15")


def warp_by_corners(img, corners):
    tl, tr, br, bl = corners
    w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    w, h = max(8, w), max(8, h)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def old_align(img, corners, margin=0.0):
    h, w = img.shape[:2]
    bb = expand_bbox(bbox_of(corners), margin, w, h)
    crop = crop_bbox(img, bb)
    angle = estimate_skew_angle(crop)
    return trim_uniform_borders(rotate_image(crop, angle))


def load_text_rows(lo, hi):
    out = []
    with (LABEL_DIR / "labeled.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            num = int(r["image"][:6])
            if not (lo <= num <= hi):
                continue
            if r["plate_text"] in ("no_plate", "unreadable", ""):
                continue
            quad = np.array([[float(r["tl_x"]), float(r["tl_y"])],
                             [float(r["tr_x"]), float(r["tr_y"])],
                             [float(r["br_x"]), float(r["br_y"])],
                             [float(r["bl_x"]), float(r["bl_y"])]], np.float32)
            if not np.any(quad):
                continue
            tl, tr = quad[0], quad[1]
            r["quad"] = quad
            r["band"] = band_of(math.degrees(math.atan2(tr[1] - tl[1],
                                                        tr[0] - tl[0])))
            out.append(r)
    return out


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(MODEL)
    rows = load_text_rows(401, 1701)
    (OUT / "aligned").mkdir(parents=True, exist_ok=True)

    agg = {b: {"n": 0, "q": [], "kb": []} for b in ("<5", "5-15", ">15")}
    manifest = []

    for r in rows:
        img = cv2.imread(str(IMAGES_DIR / r["image"]))
        if img is None:
            continue
        gt = r["quad"]
        res = model.predict(img, conf=0.25, imgsz=160, verbose=False, save=False)[0]
        kp = None
        if res.keypoints is not None and res.keypoints.xy is not None \
                and len(res.keypoints.xy):
            kxy = res.keypoints.xy.cpu().numpy()
            confs = (res.keypoints.conf.cpu().numpy()
                     if res.keypoints.conf is not None
                     else np.ones((len(kxy), 4), np.float32))
            i = int(np.argmax(confs.mean(axis=1)))
            kp = kxy[i].astype(np.float32)

        angle = iou_q = iou_kb = None
        if kp is None:
            aligned = old_align(img, gt)
            mode = "affine_fallback"
        else:
            angle = math.degrees(math.atan2(kp[1][1] - kp[0][1],
                                            kp[1][0] - kp[0][0]))
            iou_q = quad_iou(kp, gt)
            iou_kb = bbox_iou(bbox_of(kp), bbox_of(gt))
            if abs(angle) > ANGLE_TH:
                aligned = warp_by_corners(img, kp)
                mode = "warp"
            else:
                aligned = old_align(img, gt)
                mode = "affine"

        cv2.imwrite(str(OUT / "aligned" / r["image"]), aligned)
        b = r["band"]
        a = agg[b]
        a["n"] += 1
        if iou_q is not None:
            a["q"].append(iou_q)
            a["kb"].append(iou_kb)
        manifest.append({
            "image": r["image"], "band": b, "gt": r["plate_text"], "mode": mode,
            "angle": None if angle is None else round(angle, 2),
            "iou_quad": None if iou_q is None else round(iou_q, 4),
            "iou_kpbox": None if iou_kb is None else round(iou_kb, 4),
        })

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    print(f"{'dai':<7}{'n':>5}{'iou_quad':>10}{'iou_kpbox':>11}{'warp':>6}")
    for b in ("<5", "5-15", ">15"):
        a = agg[b]
        n = a["n"]
        if not n:
            continue
        nwarp = sum(1 for m in manifest if m["band"] == b and m["mode"] == "warp")
        print(f"{b:<7}{n:>5}{mean(a['q']):>10}{mean(a['kb']):>11}{nwarp:>6}")

    n_all = sum(a["n"] for a in agg.values())
    all_q = [x for a in agg.values() for x in a["q"]]
    all_kb = [x for a in agg.values() for x in a["kb"]]
    print(f"{'TONG':<7}{n_all:>5}{mean(all_q):>10}{mean(all_kb):>11}"
          f"{sum(1 for m in manifest if m['mode'] == 'warp'):>6}")

    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"[i] manifest -> {OUT / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
