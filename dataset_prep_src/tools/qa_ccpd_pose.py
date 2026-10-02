"""
tools/qa_ccpd_pose.py
=====================
Kiểm tra chất lượng dataset YOLO-pose sinh từ CCPD trước khi train.

Các phép kiểm tra:
  1. Mỗi file nhãn có đúng 17 token (``cls cx cy w h`` + 4 x ``x y v``).
  2. Toạ độ nằm trong [0, 1] và mọi visibility = 2.
  3. Round-trip: giải chuẩn hoá keypoint rồi so với 4 đỉnh GỐC trong tên file CCPD
     (đã sắp lại TL,TR,BR,BL) — sai số phải <= 0.5 px.
  4. Bbox trong nhãn phải khớp bao min/max của 4 keypoint (sai số <= 1e-4).
  5. Thống kê phân bố góc nghiêng để chắc chắn độ đa dạng.

Chạy::

    python tools/qa_ccpd_pose.py --dataset S:/M_AGENT/CV/datasets/ccpd_pose
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

try:  # chạy trực tiếp: python tools/qa_ccpd_pose.py
    from ccpd_to_pose import order_corners_geometric, parse_ccpd_name
except ImportError:  # chạy dạng module: python -m tools.qa_ccpd_pose
    from tools.ccpd_to_pose import order_corners_geometric, parse_ccpd_name


def percentile(values: list[float], q: float) -> float:
    """Bách phân vị đơn giản."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def main() -> int:
    ap = argparse.ArgumentParser(description="QA dataset YOLO-pose từ CCPD.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tol-px", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.dataset)
    counts = {"files": 0, "bad_tokens": 0, "out_of_range": 0, "bad_vis": 0,
              "kpt_mismatch": 0, "bad_bbox": 0}
    splits = {"train": 0, "val": 0}
    max_kpt_err = 0.0
    max_bbox_err = 0.0
    angles: list[float] = []
    aspects: list[float] = []

    for split in splits:
        for lbl in sorted((root / "labels" / split).glob("*.txt")):
            counts["files"] += 1
            splits[split] += 1
            toks = lbl.read_text(encoding="utf-8").split()
            if len(toks) != 17:
                counts["bad_tokens"] += 1
                continue
            cx, cy, bw, bh = (float(t) for t in toks[1:5])
            pts = [(float(toks[5 + 3 * i]), float(toks[6 + 3 * i])) for i in range(4)]
            vis = [int(float(toks[7 + 3 * i])) for i in range(4)]

            if any(v < 0.0 or v > 1.0 for p in pts for v in p):
                counts["out_of_range"] += 1
            if any(v != 2 for v in vis):
                counts["bad_vis"] += 1

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox_err = max(
                abs((min(xs) + max(xs)) / 2 - cx),
                abs((min(ys) + max(ys)) / 2 - cy),
                abs((max(xs) - min(xs)) - bw),
                abs((max(ys) - min(ys)) - bh),
            )
            max_bbox_err = max(max_bbox_err, bbox_err)
            if bbox_err > 1e-4:
                counts["bad_bbox"] += 1

            info = parse_ccpd_name(lbl.stem + ".jpg")
            if info is not None:
                geo = order_corners_geometric(info["corners_raw"])
                for (gx, gy), (nx, ny) in zip(geo, pts):
                    err = max(abs(gx - nx * 720.0), abs(gy - ny * 1160.0))
                    max_kpt_err = max(max_kpt_err, err)
                    if err > args.tol_px:
                        counts["kpt_mismatch"] += 1
                        break

            angles.append(math.degrees(math.atan2(pts[1][1] - pts[0][1],
                                                   pts[1][0] - pts[0][0])))
            aspects.append((bw * 720.0) / (bh * 1160.0) if bh else 0.0)

    print(f"files={counts['files']} (train={splits['train']}, val={splits['val']})")
    print(f"bad_tokens={counts['bad_tokens']} out_of_range={counts['out_of_range']} "
          f"bad_vis={counts['bad_vis']}")
    print(f"kpt_mismatch(>{args.tol_px}px)={counts['kpt_mismatch']} "
          f"max_kpt_err={max_kpt_err:.4f}px")
    print(f"bad_bbox={counts['bad_bbox']} max_bbox_err={max_bbox_err:.2e}")

    if angles:
        print("angle(deg): min=%.1f p05=%.1f p50=%.1f p95=%.1f max=%.1f"
              % (min(angles), percentile(angles, 0.05), percentile(angles, 0.5),
                 percentile(angles, 0.95), max(angles)))
        bins: dict[int, int] = {}
        for angle in angles:
            key = min(int(abs(angle) // 5) * 5, 45)
            bins[key] = bins.get(key, 0) + 1
        print("|angle| buckets (deg):", dict(sorted(bins.items())))
        print("aspect w/h: p50=%.2f" % percentile(aspects, 0.5))

    ok = all(counts[k] == 0 for k in
             ("bad_tokens", "out_of_range", "bad_vis", "kpt_mismatch", "bad_bbox"))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

