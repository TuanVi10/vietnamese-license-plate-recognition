"""
tools/eval_corner_ocr.py (Stage 2, chạy .venv-paddle)
OCR các ảnh đã căn chỉnh (regressor + gate) bằng PaddleOCR, so sim/exact vs GT,
đối chiếu mốc OLD (affine) và CEILING (GT corners) từ ceiling_test_paddle_align.
"""
from __future__ import annotations

import csv
import difflib
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

MANIFEST = Path(r"S:\M_AGENT\CV\runs\eval_corner\manifest.csv")
ALIGNED = Path(r"S:\M_AGENT\CV\runs\eval_corner\aligned")
CHARSET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# (OLD_exact, CEIL_exact) theo band từ ceiling_test_paddle_align
BASE = {"<5": (0.821, 0.702), "5-15": (0.747, 0.733), ">15": (0.463, 0.833)}


def alnum(s):
    return "".join(c for c in s.upper() if c in CHARSET)


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    agg = {b: {"n": 0, "ex": 0, "r": []} for b in ("<5", "5-15", ">15")}

    for r in rows:
        img = cv2.imread(str(ALIGNED / r["image"]))
        if img is None:
            continue
        gt = alnum(r["gt"])
        res = ocr.predict(img)
        texts = []
        for rr in res:
            d = rr.json if hasattr(rr, "json") else rr
            texts += d.get("res", {}).get("rec_texts", [])
        txt = alnum("".join(texts))
        sim = difflib.SequenceMatcher(None, gt, txt).ratio()
        b = r["band"]
        a = agg[b]
        a["n"] += 1
        a["ex"] += int(txt == gt)
        a["r"].append(sim)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    print(f"{'dai':<7}{'n':>5}{'REG_exact':>11}{'REG_sim':>9}"
          f"{'OLD_exact':>11}{'CEIL_exact':>12}")
    for b in ("<5", "5-15", ">15"):
        a = agg[b]
        n = a["n"]
        if not n:
            continue
        oe, ce = BASE[b]
        print(f"{b:<7}{n:>5}{a['ex'] / n:>10.1%}{mean(a['r']):>9}"
              f"{oe:>10.1%}{ce:>11.1%}")

    n_all = sum(a["n"] for a in agg.values())
    ex = sum(a["ex"] for a in agg.values())
    r_all = [x for a in agg.values() for x in a["r"]]
    print(f"{'TONG':<7}{n_all:>5}{ex / n_all:>10.1%}{mean(r_all):>9}"
          f"{0.704:>10.1%}{0.746:>11.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
