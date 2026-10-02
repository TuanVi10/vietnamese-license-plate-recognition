"""
tools/eval_compare.py
=====================
Sau khi train xong (đủ 100 epoch), chạy MỘT lệnh để đánh giá cả baseline lẫn
augmented trên data thật 401-1701, theo dải góc (<5, 5-15, >15 độ).

    python tools/eval_compare.py val      # val set (mặc định)
    python tools/eval_compare.py test     # test set

Kết quả JSON ghi vào ``runs/pose_ccpd/eval_<tag>_<split>.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RUNS = Path(r"S:\M_AGENT\CV\runs\pose_ccpd")
TOOL = Path(__file__).resolve().parent / "eval_pose_bands.py"

MODELS = {
    "baseline": RUNS / "ccpd_pose_baseline" / "weights" / "best.pt",
    "augmented": RUNS / "ccpd_pose_augmented" / "weights" / "best.pt",
}


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def fmt(v, p=3):
    return "-" if v is None else f"{v:.{p}f}"


BANDS = ["<5", "5-15", ">15", "TONG"]


def run_one(tag, model, split):
    out = RUNS / f"eval_{tag}_{split}.json"
    cmd = [sys.executable, str(TOOL), "--model", str(model), "--tag", tag,
           "--split", split, "--out", str(out)]
    print("> " + " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def print_table(a, b):
    print(f"\n=== SO SANH ({a['split']}): baseline  vs  augmented ===")
    print(f"{'dai goc':<8}{'metric':<14}{'baseline':>11}{'augmented':>11}")
    print("-" * 46)
    for band in BANDS:
        sa = a["bands"][band] if band != "TONG" else a["overall"]
        sb = b["bands"][band] if band != "TONG" else b["overall"]
        rows = [
            ("n", str(sa["images"]), str(sb["images"])),
            ("det%", pct(sa["detect_rate"]), pct(sb["detect_rate"])),
            ("corner_px", fmt(sa["corner_px_mean"], 2), fmt(sb["corner_px_mean"], 2)),
            ("corner_norm", fmt(sa["corner_norm_mean"]), fmt(sb["corner_norm_mean"])),
            ("quad_IoU", fmt(sa["quad_iou_mean"]), fmt(sb["quad_iou_mean"])),
            ("angle_err", fmt(sa["angle_err_mean"], 2), fmt(sb["angle_err_mean"], 2)),
        ]
        for i, (metric, va, vb) in enumerate(rows):
            prefix = band if i == 0 else ""
            print(f"{prefix:<8}{metric:<14}{va:>11}{vb:>11}")
        print()


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    results = {}
    for tag, model in MODELS.items():
        if not model.exists():
            print(f"[skip] {tag}: chưa có {model}")
            continue
        results[tag] = run_one(tag, model, split)
    if len(results) == 2:
        print_table(results["baseline"], results["augmented"])
    print(f"\n[i] JSON: {[str(RUNS / f'eval_{t}_{split}.json') for t in results]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
