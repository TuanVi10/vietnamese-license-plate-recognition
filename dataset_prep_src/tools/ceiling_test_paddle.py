"""
tools/ceiling_test_paddle.py
============================
Chạy PaddleOCR (PP-OCRv6) trên 213 ảnh ceiling test (warp 4 góc GT = CEILING),
so sim/exact với GT text và với EasyOCR CEILING (runs/ceiling_test/per_image.csv).
Chạy bằng .venv-paddle.
"""
from __future__ import annotations

import csv
import difflib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

LABEL_DIR = Path(r"S:\M_AGENT\CV\labeling")
IMAGES_DIR = LABEL_DIR / "images"
TW, TH = 320, 48
CHARSET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def alnum(s: str) -> str:
    return "".join(c for c in s.upper() if c in CHARSET)


def band_of(a: float) -> str:
    a = abs(a)
    return "<5" if a < 5 else ("5-15" if a < 15 else ">15")


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


def main() -> int:
    rows = load_text_rows(401, 1701)
    print(f"[i] {len(rows)} anh co text GT")

    easy = {}
    ep = Path(r"S:\M_AGENT\CV\runs\ceiling_test\per_image.csv")
    if ep.exists():
        for r in csv.DictReader(ep.open(encoding="utf-8")):
            easy[r["image"]] = r

    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)

    agg = {b: {"n": 0, "e_exact": 0, "p_exact": 0, "e_r": [], "p_r": []}
           for b in ("<5", "5-15", ">15")}
    per = []

    for r in rows:
        img = cv2.imread(str(IMAGES_DIR / r["image"]))
        if img is None:
            continue
        gt = alnum(r["plate_text"])
        ceil = warp_by_corners(img, r["quad"])
        res = ocr.predict(cv2.cvtColor(finalize(ceil), cv2.COLOR_GRAY2BGR))
        texts = []
        for rr in res:
            d = rr.json if hasattr(rr, "json") else rr
            texts += d.get("res", {}).get("rec_texts", [])
        pad = alnum("".join(texts))
        easy_t = alnum(easy[r["image"]]["ceil_text"]) if r["image"] in easy else ""
        er = difflib.SequenceMatcher(None, gt, easy_t).ratio()
        pr = difflib.SequenceMatcher(None, gt, pad).ratio()
        b = r["band"]
        a = agg[b]
        a["n"] += 1
        a["e_exact"] += int(easy_t == gt)
        a["p_exact"] += int(pad == gt)
        a["e_r"].append(er)
        a["p_r"].append(pr)
        per.append({"image": r["image"], "band": b, "gt": gt, "easy": easy_t,
                    "paddle": pad, "easy_sim": round(er, 4), "paddle_sim": round(pr, 4)})

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    print(f"\n{'dai':<7}{'n':>5}{'easy_exact':>12}{'paddle_exact':>13}"
          f"{'easy_sim':>10}{'paddle_sim':>11}")
    for b in ("<5", "5-15", ">15"):
        a = agg[b]
        n = a["n"]
        if not n:
            continue
        print(f"{b:<7}{n:>5}{a['e_exact'] / n:>11.1%}{a['p_exact'] / n:>12.1%}"
              f"{mean(a['e_r']):>10}{mean(a['p_r']):>11}")

    n_all = sum(a["n"] for a in agg.values())
    e_ex = sum(a["e_exact"] for a in agg.values())
    p_ex = sum(a["p_exact"] for a in agg.values())
    e_r = [x for a in agg.values() for x in a["e_r"]]
    p_r = [x for a in agg.values() for x in a["p_r"]]
    print(f"{'TONG':<7}{n_all:>5}{e_ex / n_all:>11.1%}{p_ex / n_all:>12.1%}"
          f"{mean(e_r):>10}{mean(p_r):>11}")

    out = Path(r"S:\M_AGENT\CV\runs\ceiling_test\paddle_per_image.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)
    print(f"\n[i] output -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
