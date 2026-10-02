"""
tools/eval_fusion.py
====================
Đánh giá fusion (gộp theo vị trí ký tự) so với vote cả chuỗi cũ trên bộ nhãn tay.

Đầu vào:
    * ``results.json``   — output của ``VideoPipeline`` (có ``tracks[].readings``).
    * ``labels.csv``     — nhãn tay, cột: ``video, track_id, ground_truth``.

Đầu ra (ra stdout):
    * Tỉ lệ đúng cả biển (accuracy).
    * CER — tỉ lệ sai từng ký tự (Levenshtein / độ dài ground-truth; dự đoán rỗng
      được tính là sai toàn bộ độ dài).
    * Tỉ lệ track không ra kết quả.
    * So sánh song song vote cũ vs fusion.

Chạy bằng::

    python tools/eval_fusion.py --results results/results.json --labels labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FusionConfig, PostprocessConfig  # noqa: E402
from src.fusion import fuse_readings  # noqa: E402
from src.postprocess import PostProcessor, normalize_plate_text  # noqa: E402


def levenshtein(a: str, b: str) -> int:
    """Khoảng cách Levenshtein (DP, O(|a|*|b|)), không thêm thư viện."""
    la, lb = len(a), len(b)
    row = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = row[0]
        row[0] = i
        for j in range(1, lb + 1):
            tmp = row[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            row[j] = min(row[j] + 1, row[j - 1] + 1, prev + cost)
            prev = tmp
    return row[lb]


def cer(pred: str, gt: str) -> float:
    """Tỉ lệ sai từng ký tự. Dự đoán rỗng = sai toàn bộ độ dài (CER = 1.0)."""
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


def load_labels(path: str) -> dict:
    """Đọc labels.csv -> ``{(video, track_id): ground_truth chuẩn hoá}``."""
    labels = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            video = (row.get("video") or "").strip()
            track_id = (row.get("track_id") or "").strip()
            gt = (row.get("ground_truth") or "").strip()
            if not track_id:
                continue
            labels[(video, track_id)] = normalize_plate_text(gt) if gt else ""
    return labels


def find_gt(labels: dict, video: str, track_id: str):
    """Tìm nhãn theo (video, track_id), fallback theo basename hoặc chỉ track_id."""
    for key in ((video, track_id), (os.path.basename(video), track_id), ("", track_id)):
        if key in labels:
            return labels[key]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="So sánh vote cũ vs fusion trên nhãn tay.")
    ap.add_argument("--results", required=True, help="Đường dẫn results.json.")
    ap.add_argument("--labels", required=True, help="Đường dẫn labels.csv (video,track_id,ground_truth).")
    ap.add_argument("--verbose", action="store_true", help="In chi tiết từng track.")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as fh:
        data = json.load(fh)
    labels = load_labels(args.labels)
    video = data.get("video", "")

    old_pp = PostProcessor()
    fusion_cfg = FusionConfig()
    post_cfg = PostprocessConfig()

    stats = {"old": {"n": 0, "ok": 0, "dist": 0, "gt_len": 0, "empty": 0},
             "fusion": {"n": 0, "ok": 0, "dist": 0, "gt_len": 0, "empty": 0}}
    rows = []

    for track in data.get("tracks", []):
        track_id = str(track.get("track_id"))
        gt = find_gt(labels, video, track_id)
        if gt is None:
            continue
        readings = track.get("readings") or []
        old_text = old_pp.process(readings).text
        fused_text = fuse_readings(readings, fusion_cfg, post_cfg).text
        old_text = normalize_plate_text(old_text) if old_text else ""
        fused_text = normalize_plate_text(fused_text) if fused_text else ""

        for key, pred in (("old", old_text), ("fusion", fused_text)):
            s = stats[key]
            s["n"] += 1
            s["ok"] += int(pred == gt)
            s["empty"] += int(not pred)
            s["dist"] += levenshtein(pred, gt)
            s["gt_len"] += len(gt)
        rows.append((track_id, gt, old_text, fused_text))

    def report(name: str, s: dict) -> None:
        n = s["n"] or 1
        acc = s["ok"] / n
        corpus_cer = (s["dist"] / s["gt_len"]) if s["gt_len"] else 0.0
        empty = s["empty"] / n
        print(
            f"  {name:<7} acc={acc:7.2%}  CER={corpus_cer:7.2%}  "
            f"empty={empty:6.2%}  n={s['n']}"
        )

    print(f"Video: {video}")
    print(f"Track có nhãn: {stats['old']['n']}")
    print("So sánh:")
    report("vote cũ", stats["old"])
    report("fusion", stats["fusion"])

    if args.verbose:
        print("\nChi tiết (track_id | GT | vote cũ | fusion):")
        for track_id, gt, old_text, fused_text in rows:
            mark = "" if old_text == fused_text else "  <-- khác"
            print(f"  {track_id:>6} | {gt:<12} | {old_text:<12} | {fused_text:<12}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
