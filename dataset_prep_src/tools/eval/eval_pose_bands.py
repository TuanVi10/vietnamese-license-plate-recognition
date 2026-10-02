"""
tools/eval_pose_bands.py
========================
Đánh giá model YOLO-pose 4 keypoint trên **data thật đã gán tay** (401-1701),
tách riêng theo **dải góc nghiêng** (``<5``, ``5-15``, ``>15`` độ) thay vì chỉ
báo cáo một con số trung bình.

Cần 2 file do ``tools/unify_labeling.py`` sinh ra:
``labeling/labeled.csv`` và ``labeling/split.csv``.

Chỉ số mỗi dải:

* **detect_rate** : tỉ lệ ảnh model dò được biển (conf >= ``--conf``).
* **corner_err**  : sai số trung bình 4 góc so với nhãn tay, đơn vị PIXEL.
* **corner_err_norm** : sai số góc chia ``sqrt(diện tích biển GT)`` — không phụ
  thuộc kích thước ảnh, so sánh được giữa các dải.
* **quad_iou**    : IoU giữa tứ giác dự đoán và tứ giác nhãn tay.
* **angle_err**   : sai số góc nghiêng (độ) — góc cạnh trên của biển.

Chạy::

    python tools/eval_pose_bands.py --model runs/pose_ccpd/ccpd_pose_baseline/weights/best.pt \
        --tag baseline --split val
    python tools/eval_pose_bands.py --gt-only          # chỉ xem phân bố dải góc
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

LABEL_DIR = Path(r"S:\M_AGENT\CV\labeling")
IMAGES_DIR = LABEL_DIR / "images"
#: Biên các dải góc nghiêng (độ) theo yêu cầu.
BANDS = ((0.0, 5.0, "<5"), (5.0, 15.0, "5-15"), (15.0, 90.0, ">15"))


def band_of(angle: float) -> str:
    """Trả tên dải góc cho |angle|."""
    abs_angle = abs(angle)
    for low, high, name in BANDS:
        if low <= abs_angle < high:
            return name
    return BANDS[-1][2]


def rows_with_plate(labeled_path: Path) -> list[dict]:
    """Đọc labeled.csv, chỉ giữ ảnh CÓ biển (toạ độ khác 0)."""
    out: list[dict] = []
    with labeled_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = ["tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"]
            quad = np.array([[float(row[key[0]]), float(row[key[1]])],
                             [float(row[key[2]]), float(row[key[3]])],
                             [float(row[key[4]]), float(row[key[5]])],
                             [float(row[key[6]]), float(row[key[7]])]], np.float32)
            if not np.any(quad):
                continue
            row["quad"] = quad
            row["angle"] = math.degrees(math.atan2(quad[1][1] - quad[0][1],
                                                    quad[1][0] - quad[0][0]))
            row["band"] = band_of(row["angle"])
            out.append(row)
    return out


def load_split(path: Path | None) -> dict[str, str]:
    """image -> split ('val'/'test'); trả {} nếu không truyền."""
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {r["image"]: r["split"] for r in csv.DictReader(fh)}


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU hai tứ giác, tính bằng rasterize mask (đủ chính xác cho báo cáo)."""
    pts = np.vstack([a, b])
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    w, h = max(x1 - x0 + 2, 2), max(y1 - y0 + 2, 2)
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.round(a - [x0 - 1, y0 - 1]).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(b - [x0 - 1, y0 - 1]).astype(np.int32)], 1)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return inter / union if union else 0.0


def shoelace(quad: np.ndarray) -> float:
    """Diện tích đa giác (công thức dây giày)."""
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def predict_quad(model, img: np.ndarray, conf: float,
                 imgsz: int) -> tuple[np.ndarray | None, float]:
    """Dự đoán 4 góc (pixel ảnh gốc) của instance tự tin nhất."""
    result = model.predict(img, conf=conf, imgsz=imgsz, verbose=False, save=False)[0]
    if result.keypoints is None or result.keypoints.xy is None:
        return None, 0.0
    kxy = result.keypoints.xy.cpu().numpy()
    if len(kxy) == 0:
        return None, 0.0
    confs = (result.boxes.conf.cpu().numpy() if result.boxes is not None
             else np.ones(len(kxy)))
    best = int(np.argmax(confs))
    return kxy[best].astype(np.float32), float(confs[best])


def new_acc() -> dict:
    """Bộ tích luỹ chỉ số cho một dải."""
    return {"n": 0, "det": 0, "corner": [], "corner_norm": [], "iou": [], "ang": []}


def summarize(acc: dict) -> dict:
    """Chuyển bộ tích luỹ thành dict chỉ số đã tổng hợp."""
    det = acc["det"]
    return {
        "images": acc["n"],
        "detected": det,
        "detect_rate": round(det / acc["n"], 4) if acc["n"] else 0.0,
        "corner_px_mean": round(float(np.mean(acc["corner"])), 3) if det else None,
        "corner_px_p90": round(float(np.percentile(acc["corner"], 90)), 3) if det else None,
        "corner_norm_mean": round(float(np.mean(acc["corner_norm"])), 4) if det else None,
        "quad_iou_mean": round(float(np.mean(acc["iou"])), 4) if det else None,
        "angle_err_mean": round(float(np.mean(acc["ang"])), 3) if det else None,
        "angle_err_p90": round(float(np.percentile(acc["ang"], 90)), 3) if det else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Đánh giá pose theo dải góc trên data thật.")
    ap.add_argument("--model", default=None, help="Đường dẫn best.pt (bỏ trống nếu --gt-only).")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--labeled", default=str(LABEL_DIR / "labeled.csv"))
    ap.add_argument("--split-file", default=str(LABEL_DIR / "split.csv"))
    ap.add_argument("--split", default="all", choices=["all", "val", "test"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = rows_with_plate(Path(args.labeled))
    split_map = load_split(Path(args.split_file))
    if args.split != "all":
        rows = [r for r in rows if split_map.get(r["image"]) == args.split]
    if args.limit:
        rows = rows[: args.limit]

    gt_bands: Counter = Counter(r["band"] for r in rows)
    angles = [abs(r["angle"]) for r in rows]
    print(f"[i] {len(rows)} anh co bien | split={args.split} | "
          f"phan bo dai goc GT={dict(gt_bands)}")
    if angles:
        print("[i] |goc| GT (do): min=%.1f p50=%.1f p90=%.1f max=%.1f"
              % (min(angles), float(np.percentile(angles, 50)),
                 float(np.percentile(angles, 90)), max(angles)))
    if args.gt_only:
        return 0
    if not args.model:
        print("[error] Thiếu --model (hoặc dùng --gt-only).")
        return 1

    from ultralytics import YOLO

    model = YOLO(args.model)
    acc: dict[str, dict] = defaultdict(new_acc)
    for row in rows:
        img = cv2.imread(str(IMAGES_DIR / row["image"]))
        if img is None:
            continue
        gt = row["quad"]
        item = acc[row["band"]]
        item["n"] += 1
        pred, _ = predict_quad(model, img, args.conf, args.imgsz)
        if pred is None:
            continue
        item["det"] += 1
        item["corner"].append(float(np.linalg.norm(pred - gt, axis=1).mean()))
        item["corner_norm"].append(
            float(np.linalg.norm(pred - gt, axis=1).mean() / max(math.sqrt(shoelace(gt)), 1.0)))
        item["iou"].append(quad_iou(pred, gt))
        pred_angle = math.degrees(math.atan2(pred[1][1] - pred[0][1],
                                             pred[1][0] - pred[0][0]))
        item["ang"].append(abs(pred_angle - row["angle"]))

    overall = new_acc()
    for band in list(acc.values()):
        for key in ("n", "det"):
            overall[key] += band[key]
        for key in ("corner", "corner_norm", "iou", "ang"):
            overall[key].extend(band[key])

    result = {"tag": args.tag, "split": args.split, "conf": args.conf,
              "gt_band_counts": dict(gt_bands),
              "bands": {name: summarize(acc[name]) for _, _, name in BANDS},
              "overall": summarize(overall)}
    print()
    print(f"=== {args.tag} | split={args.split} | conf={args.conf} ===")
    header = (f"{'dai goc':<8}{'n':>5}{'det%':>8}{'corner_px':>11}"
              f"{'p90_px':>9}{'norm':>9}{'quad_IoU':>10}{'ang_err':>9}{'p90_ang':>9}")
    print(header)
    for _, _, name in BANDS + ((0, 0, "TONG"),):
        stat = result["bands"][name] if name != "TONG" else result["overall"]
        fmt = lambda v, p=2: ("%.*f" % (p, v)) if v is not None else "-"
        print(f"{name:<8}{stat['images']:>5}{stat['detect_rate'] * 100:>7.1f}%"
              f"{fmt(stat['corner_px_mean']):>11}{fmt(stat['corner_px_p90']):>9}"
              f"{fmt(stat['corner_norm_mean'], 3):>9}"
              f"{fmt(stat['quad_iou_mean'], 3):>10}"
              f"{fmt(stat['angle_err_mean']):>9}{fmt(stat['angle_err_p90']):>9}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        print(f"[i] JSON -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

