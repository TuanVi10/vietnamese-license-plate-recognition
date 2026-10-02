"""
tools/ceiling_test.py
=====================
CEILING TEST (độc lập pose model): dùng 4 góc NHÃN TAY (GT) từ 401-1701 để đo
TRẦN lợi ích của corner-detection đối với OCR.

So sánh 2 đường trên cùng 1 ảnh, cùng OCR (EasyOCR), chỉ khác bước CĂN CHỈNH:
  * CEILING : 4 góc GT -> ``cv2.warpPerspective``  (phối cảnh + góc tối ưu)
  * OLD     : bbox GT + ``estimate_skew_angle`` + ``rotate_image`` (pipeline hiện tại)

Kết quả tách theo dải góc (<5, 5-15, >15). Nếu CEILING không thắng OLD rõ rệt ở
dải nghiêng nặng -> corner detection không đáng đầu tư.

Chỉ số mỗi đường: tỉ lệ OCR KHỚP CHÍNH XÁC và độ tương đồng trung bình
(``difflib.SequenceMatcher.ratio``) giữa text OCR và text nhãn tay.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DSRC = HERE.parent.parent  # dataset_prep_src
sys.path.insert(0, str(DSRC))

from src.image_utils import (  # noqa: E402
    crop_bbox, expand_bbox, rotate_image, to_gray, trim_uniform_borders,
)
from src.ocr import EasyOCREngine  # noqa: E402
from src.preprocess_plate import estimate_skew_angle  # noqa: E402

LABEL_DIR = Path(r"S:\M_AGENT\CV\labeling")
IMAGES_DIR = LABEL_DIR / "images"
TW, TH = 320, 48


def bbox_of(q: np.ndarray) -> np.ndarray:
    return np.array([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()],
                    np.float32)


def fit(img: np.ndarray, tw: int = TW, th: int = TH) -> np.ndarray:
    """Resize giữ aspect vào trong (tw, th) rồi pad trắng cho đúng kích thước."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((th, tw), 255, np.uint8)
    s = min(float(tw) / w, float(th) / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((th, tw), 255, np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def finalize(img: np.ndarray) -> np.ndarray:
    """Hậu xử chung cho CẢ 2 đường: gray -> CLAHE -> resize/pad -> (TW,TH)."""
    return fit(clahe(to_gray(img)))


def warp_by_corners(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """CEILING: 4 góc GT -> warpPerspective, GIỮ aspect thật của biển."""
    tl, tr, br, bl = corners
    w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    w, h = max(8, w), max(8, h)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def old_align(img: np.ndarray, corners: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """OLD: bbox GT (axis-aligned) + affine deskew (heuristic) — như pipeline hiện tại."""
    h, w = img.shape[:2]
    bb = expand_bbox(bbox_of(corners), margin, w, h)
    crop = crop_bbox(img, bb)
    angle = estimate_skew_angle(crop)
    rotated = rotate_image(crop, angle)
    return trim_uniform_borders(rotated)


def band_of(angle_deg: float) -> str:
    a = abs(angle_deg)
    if a < 5:
        return "<5"
    if a < 15:
        return "5-15"
    return ">15"


#: Tên thư mục an toàn trên Windows (tránh ký tự < > trong tên).
BAND_DIR = {"<5": "lt5", "5-15": "5_15", ">15": "gt15"}


def montage(orig, old, ceil, caption):
    """Ghép 3 ảnh (gốc + GT | OLD | CEILING) ngang để đối chiếu."""
    orig_v = finalize(orig) if orig is not None else np.full((TH, TW), 255, np.uint8)
    old_v = finalize(old) if old is not None else np.full((TH, TW), 255, np.uint8)
    ceil_v = finalize(ceil) if ceil is not None else np.full((TH, TW), 255, np.uint8)
    gap = np.full((TH, 8), 255, np.uint8)
    row = cv2.hconcat([orig_v, gap, old_v, gap, ceil_v])
    row = cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
    cv2.putText(row, caption, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                cv2.LINE_AA)
    return row


def load_text_rows(lo: int, hi: int) -> list[dict]:
    """Các ảnh 401-1701 có text GT (không no_plate/unreadable) và có đủ 4 góc."""
    out = []
    cols = ["tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"]
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
            angle = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
            r["quad"] = quad
            r["angle"] = angle
            r["band"] = band_of(angle)
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Ceiling test (read-only).")
    ap.add_argument("--range-min", type=int, default=401)
    ap.add_argument("--range-max", type=int, default=1701)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--out-dir", default=r"S:\M_AGENT\CV\runs\ceiling_test")
    args = ap.parse_args()

    rows = load_text_rows(args.range_min, args.range_max)
    print(f"[i] {len(rows)} anh co text GT trong {args.range_min}-{args.range_max}")

    engine = EasyOCREngine(lang="en", use_gpu=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = {b: {"n": 0, "old_exact": 0, "ceil_exact": 0,
               "old_r": [], "ceil_r": [], "delta": []} for b in ("<5", "5-15", ">15")}
    per_image = []
    saved = Counter()

    for r in rows:
        img = cv2.imread(str(IMAGES_DIR / r["image"]))
        if img is None:
            continue
        gt_text = r["plate_text"].strip().upper()

        ceil_img = warp_by_corners(img, r["quad"])
        old_img = old_align(img, r["quad"])

        ceil_txt = engine.recognize(finalize(ceil_img)).text.upper()
        old_txt = engine.recognize(finalize(old_img)).text.upper()

        old_ratio = difflib.SequenceMatcher(None, gt_text, old_txt).ratio()
        ceil_ratio = difflib.SequenceMatcher(None, gt_text, ceil_txt).ratio()

        band = r["band"]
        a = agg[band]
        a["n"] += 1
        a["old_exact"] += int(old_txt == gt_text)
        a["ceil_exact"] += int(ceil_txt == gt_text)
        a["old_r"].append(old_ratio)
        a["ceil_r"].append(ceil_ratio)
        a["delta"].append(ceil_ratio - old_ratio)

        per_image.append({
            "image": r["image"], "band": band,
            "angle": round(float(r["angle"]), 2),
            "gt_text": gt_text, "old_text": old_txt, "ceil_text": ceil_txt,
            "old_ratio": round(old_ratio, 4), "ceil_ratio": round(ceil_ratio, 4),
            "delta": round(ceil_ratio - old_ratio, 4),
        })

        if saved[band] < args.n_examples:
            folder = out_dir / "examples" / BAND_DIR[band]
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"{r['image']}"),
                        montage(img, old_img, ceil_img,
                                f"{r['image']} | GT={gt_text} old={old_txt} ceil={ceil_txt}"))
            saved[band] += 1

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    summary = {"range": f"{args.range_min}-{args.range_max}", "n": len(rows)}
    print(f"\n{'dai':<7}{'n':>5}{'old_exact':>11}{'ceil_exact':>12}"
          f"{'old_mean':>10}{'ceil_mean':>11}{'delta':>8}")
    for b in ("<5", "5-15", ">15"):
        a = agg[b]
        n = a["n"]
        if n == 0:
            continue
        summary[b] = {
            "n": n,
            "old_exact_rate": round(a["old_exact"] / n, 4),
            "ceil_exact_rate": round(a["ceil_exact"] / n, 4),
            "old_mean_ratio": mean(a["old_r"]),
            "ceil_mean_ratio": mean(a["ceil_r"]),
            "mean_delta": mean(a["delta"]),
        }
        print(f"{b:<7}{n:>5}{a['old_exact'] / n:>11.1%}{a['ceil_exact'] / n:>12.1%}"
              f"{mean(a['old_r']):>10}{mean(a['ceil_r']):>11}{mean(a['delta']):>8}")

    # tổng
    all_old = [x for a in agg.values() for x in a["old_r"]]
    all_ceil = [x for a in agg.values() for x in a["ceil_r"]]
    all_old_exact = sum(a["old_exact"] for a in agg.values())
    all_ceil_exact = sum(a["ceil_exact"] for a in agg.values())
    n_all = sum(a["n"] for a in agg.values())
    summary["overall"] = {
        "n": n_all,
        "old_exact_rate": round(all_old_exact / n_all, 4),
        "ceil_exact_rate": round(all_ceil_exact / n_all, 4),
        "old_mean_ratio": mean(all_old),
        "ceil_mean_ratio": mean(all_ceil),
        "mean_delta": mean([c - o for o, c in zip(all_old, all_ceil)]),
    }
    print(f"{'TONG':<7}{n_all:>5}{all_old_exact / n_all:>11.1%}"
          f"{all_ceil_exact / n_all:>12.1%}{mean(all_old):>10}{mean(all_ceil):>11}"
          f"{mean([c - o for o, c in zip(all_old, all_ceil)]):>8}")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    print(f"\n[i] output -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
