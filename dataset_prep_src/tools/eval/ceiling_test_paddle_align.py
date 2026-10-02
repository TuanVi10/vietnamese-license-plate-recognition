"""
tools/ceiling_test_paddle_align.py
==================================
So PaddleOCR (PP-OCRv6) trên 2 đường căn chỉnh khác nhau của CÙNG 213 ảnh:
  * OLD    : bbox GT + estimate_skew_angle + rotate (như pipeline hiện tại)
  * CEILING: 4 góc GT + warpPerspective (trần corner-detection)
=> Biết corner detection còn đáng đầu tư không KHI OCR đã là PaddleOCR.
Chạy bằng .venv-paddle.
"""
from __future__ import annotations

import csv
import difflib
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

sys.path.insert(0, r"S:\M_AGENT\CV\dataset_prep_src")
from src.image_utils import crop_bbox, expand_bbox, rotate_image, trim_uniform_borders  # noqa: E402
from src.preprocess_plate import estimate_skew_angle  # noqa: E402

LABEL_DIR = Path(r"S:\M_AGENT\CV\labeling")
IMAGES_DIR = LABEL_DIR / "images"
TW, TH = 320, 48
CHARSET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def alnum(s: str) -> str:
    return "".join(c for c in s.upper() if c in CHARSET)


def band_of(a: float) -> str:
    a = abs(a)
    return "<5" if a < 5 else ("5-15" if a < 15 else ">15")


def bbox_of(q: np.ndarray) -> np.ndarray:
    return np.array([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()],
                    np.float32)


def load_text_rows(lo: int, hi: int) -> list:
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
            ang = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
            r["quad"] = quad
            r["band"] = band_of(ang)
            out.append(r)
    return out


def warp_by_corners(img, corners):
    tl, tr, br, bl = corners
    w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    w, h = max(8, w), max(8, h)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def old_align(img, corners, margin: float = 0.0):
    h, w = img.shape[:2]
    bb = expand_bbox(bbox_of(corners), margin, w, h)
    crop = crop_bbox(img, bb)
    angle = estimate_skew_angle(crop)
    return trim_uniform_borders(rotate_image(crop, angle))


def finalize(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    h, w = g.shape[:2]
    s = min(TW / w, TH / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    canvas = np.full((TH, TW), 255, np.uint8)
    y0, x0 = (TH - nh) // 2, (TW - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = g
    return canvas


def run_ocr(ocr, img) -> str:
    res = ocr.predict(cv2.cvtColor(finalize(img), cv2.COLOR_GRAY2BGR))
    texts = []
    for r in res:
        d = r.json if hasattr(r, "json") else r
        texts += d.get("res", {}).get("rec_texts", [])
    return alnum("".join(texts))


def main() -> int:
    rows = load_text_rows(401, 1701)
    print(f"[i] {len(rows)} anh co text GT")

    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)

    agg = {b: {"n": 0, "old_ex": 0, "ceil_ex": 0, "old_r": [], "ceil_r": []}
           for b in ("<5", "5-15", ">15")}
    per = []

    for r in rows:
        img = cv2.imread(str(IMAGES_DIR / r["image"]))
        if img is None:
            continue
        gt = alnum(r["plate_text"])
        old_t = run_ocr(ocr, old_align(img, r["quad"]))
        ceil_t = run_ocr(ocr, warp_by_corners(img, r["quad"]))
        or_ = difflib.SequenceMatcher(None, gt, old_t).ratio()
        cr_ = difflib.SequenceMatcher(None, gt, ceil_t).ratio()
        b = r["band"]
        a = agg[b]
        a["n"] += 1
        a["old_ex"] += int(old_t == gt)
        a["ceil_ex"] += int(ceil_t == gt)
        a["old_r"].append(or_)
        a["ceil_r"].append(cr_)
        per.append({"image": r["image"], "band": b, "gt": gt,
                    "old": old_t, "ceil": ceil_t,
                    "old_sim": round(or_, 4), "ceil_sim": round(cr_, 4)})

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    print(f"\n{'dai':<7}{'n':>5}{'OLD_exact':>11}{'CEIL_exact':>12}"
          f"{'OLD_sim':>10}{'CEIL_sim':>10}{'delta':>8}")
    for b in ("<5", "5-15", ">15"):
        a = agg[b]
        n = a["n"]
        if not n:
            continue
        d = mean(a["ceil_r"]) - mean(a["old_r"])
        print(f"{b:<7}{n:>5}{a['old_ex'] / n:>10.1%}{a['ceil_ex'] / n:>11.1%}"
              f"{mean(a['old_r']):>10}{mean(a['ceil_r']):>10}{d:>8.4f}")

    n_all = sum(a["n"] for a in agg.values())
    old_ex = sum(a["old_ex"] for a in agg.values())
    ceil_ex = sum(a["ceil_ex"] for a in agg.values())
    old_r = [x for a in agg.values() for x in a["old_r"]]
    ceil_r = [x for a in agg.values() for x in a["ceil_r"]]
    d_all = mean(ceil_r) - mean(old_r)
    print(f"{'TONG':<7}{n_all:>5}{old_ex / n_all:>10.1%}{ceil_ex / n_all:>11.1%}"
          f"{mean(old_r):>10}{mean(ceil_r):>10}{d_all:>8.4f}")

    out = Path(r"S:\M_AGENT\CV\runs\ceiling_test\paddle_align_per_image.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)
    print(f"\n[i] output -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
