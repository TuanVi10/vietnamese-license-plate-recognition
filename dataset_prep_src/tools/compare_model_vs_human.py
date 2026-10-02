"""
tools/compare_model_vs_human.py
===============================
So sánh số biển số MODEL DETECT vs số biển số CON NGƯỜI LABEL
trên 1701 crop (000001-001701).

- "model detect" = manifest.csv source == "detection" (Module 2 plate_detector
  tìm thấy biển số thật; "fallback" = không tìm thấy, dùng bbox xe xấp xỉ).
- "human label" = labeled.csv plate_text != no_plate (có biển: text/unreadable).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

MANIFEST = Path(r"S:\M_AGENT\CV\labeling\manifest.csv")
LABELED = Path(r"S:\M_AGENT\CV\labeling\labeled.csv")


def main() -> int:
    source_of: dict[str, str] = {}
    track_of: dict[str, int] = {}
    with MANIFEST.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            source_of[row["image"]] = row["source"]
            track_of[row["image"]] = int(row["track_id"])

    text_of: dict[str, str] = {}
    with LABELED.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            text_of[row["image"]] = row["plate_text"]

    images = sorted(set(source_of) | set(text_of))
    n_manifest = len(source_of)
    n_labeled = len(text_of)

    def is_detect(img: str) -> bool:
        return source_of.get(img) == "detection"

    def has_plate(img: str) -> bool:
        return text_of.get(img, "no_plate") != "no_plate"

    # --- Mức ẢNH (frame 0000-1700) ---
    model_detect = sum(1 for i in images if is_detect(i))
    human_plate = sum(1 for i in images if has_plate(i))
    both = model_only = human_only = neither = 0
    for i in images:
        m, h = is_detect(i), has_plate(i)
        if m and h:
            both += 1
        elif m and not h:
            model_only += 1
        elif h and not m:
            human_only += 1
        else:
            neither += 1

    # --- Mức TRACK (mỗi track = 1 xe = 1 biển) ---
    tm, th = defaultdict(bool), defaultdict(bool)
    for i in images:
        t = track_of.get(i)
        if t is None:
            continue
        if is_detect(i):
            tm[t] = True
        if has_plate(i):
            th[t] = True
    tracks = sorted(set(tm) | set(th))
    t_model = sum(1 for t in tracks if tm[t])
    t_human = sum(1 for t in tracks if th[t])

    print("=" * 62)
    print("SO SÁNH MODEL DETECT vs HUMAN LABEL  (000001-001701)")
    print("=" * 62)
    print(f"manifest.csv : {n_manifest} crop   | labeled.csv : {n_labeled} crop (đã gộp 1-200)")
    print("-" * 62)
    print("[MỨC ẢNH] (mỗi crop = 1 frame)")
    print(f"  model detect (source=detection): {model_detect}")
    print(f"  human label  (plate_text != no_plate): {human_plate}")
    print(f"  -> chênh lệch: {human_plate - model_detect:+d} biển")
    print(f"     (cùng đúng: {both} | model dư: {model_only} | human dư/miss: {human_only} | cả 2 no: {neither})")
    print("-" * 62)
    print(f"[MỨC TRACK] ({len(tracks)} track xe)")
    print(f"  model detect (>=1 crop detection): {t_model}")
    print(f"  human label  (>=1 crop có biển)  : {t_human}")
    print(f"  -> chênh lệch: {t_human - t_model:+d} biển")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
