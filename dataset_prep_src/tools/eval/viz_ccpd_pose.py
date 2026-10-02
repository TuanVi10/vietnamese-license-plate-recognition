"""
tools/viz_ccpd_pose.py
======================
Vẽ đè 4 keypoint (đánh số + tô màu khác nhau) lên ảnh CCPD để kiểm tra bằng mắt
thứ tự TL -> TR -> BR -> BL trước khi train (bước 3).

Chọn mẫu ĐA DẠNG: trải đều theo góc nghiêng, kèm vài ca cực trị về độ sáng / độ mờ.

Chạy::

    python tools/viz_ccpd_pose.py --dataset S:/M_AGENT/CV/datasets/ccpd_pose \
        --out S:/M_AGENT/CV/datasets/ccpd_pose/viz --samples 40 --cols 5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

#: Màu BGR cho từng góc (TL, TR, BR, BL) — khác nhau để nhìn là biết thứ tự.
CORNER_COLORS = [
    (0, 0, 255),      # 1 TL - đỏ
    (0, 255, 0),      # 2 TR - xanh lá
    (255, 0, 0),      # 3 BR - xanh dương
    (0, 165, 255),    # 4 BL - cam
]
CORNER_NAMES = ["1-TL", "2-TR", "3-BR", "4-BL"]
TILE_W, TILE_H, BAR_H = 460, 300, 26


def read_yolo_pose(path: Path) -> tuple[tuple[float, float], ...] | None:
    """Đọc 1 dòng nhãn YOLO-pose -> 4 điểm pixel (cần truyền width/height)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    toks = text.split()
    if len(toks) < 5 + 4 * 3:
        return None
    vals = [float(t) for t in toks[5:5 + 4 * 3]]
    return tuple((vals[i * 3], vals[i * 3 + 1]) for i in range(4))


def draw_overlay(img: np.ndarray, pts_norm: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Vẽ tứ giác + 4 keypoint có số thứ tự lên ảnh (trả về ảnh mới)."""
    out = img.copy()
    h, w = out.shape[:2]
    pts = [(int(round(x * w)), int(round(y * h))) for x, y in pts_norm]
    for i in range(4):
        cv2.line(out, pts[i], pts[(i + 1) % 4], (0, 255, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (x, y), 6, CORNER_COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, CORNER_COLORS[i], 1, cv2.LINE_AA)
    return out


def crop_around(pts_px: list[tuple[int, int]], w: int, h: int, margin: float = 0.55):
    """Cắt vùng quanh biển (nới lề) để nhìn rõ 4 góc."""
    xs = [p[0] for p in pts_px]
    ys = [p[1] for p in pts_px]
    bw = max(max(xs) - min(xs), 20)
    bh = max(max(ys) - min(ys), 12)
    x0 = int(max(0, min(xs) - bw * margin))
    x1 = int(min(w, max(xs) + bw * margin))
    y0 = int(max(0, min(ys) - bh * margin))
    y1 = int(min(h, max(ys) + bh * margin))
    return x0, y0, x1, y1


def fit_tile(img: np.ndarray) -> np.ndarray:
    """Đưa ảnh vào khung TILE_W x TILE_H (giữ tỉ lệ, viền đen)."""
    h, w = img.shape[:2]
    scale = min(TILE_W / w, TILE_H / h)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    oy = (TILE_H - resized.shape[0]) // 2
    ox = (TILE_W - resized.shape[1]) // 2
    canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    return canvas


def select_diverse(rows: list[dict], samples: int) -> list[dict]:
    """Chọn mẫu đa dạng: trải đều theo GÓC + thêm các ca cực trị sáng/mờ."""
    chosen: list[dict] = []
    seen: set[str] = set()

    def add(row: dict) -> None:
        if row["image"] not in seen:
            seen.add(row["image"])
            chosen.append(row)

    n_angle = max(1, int(samples * 0.6))
    by_angle = sorted(rows, key=lambda r: r["angle_deg"])
    step = max(1, len(by_angle) // n_angle)
    for i in range(0, len(by_angle), step):
        if len(chosen) >= n_angle:
            break
        add(by_angle[i])

    for key, reverse, k in (("blur", True, 4), ("blur", False, 2),
                            ("bright", False, 2), ("bright", True, 2)):
        for row in sorted(rows, key=lambda r: r[key], reverse=reverse)[:k]:
            add(row)
    return chosen[:samples]


def main() -> int:
    ap = argparse.ArgumentParser(description="Vẽ overlay 4 keypoint CCPD để kiểm tra.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--per-sheet", type=int, default=20)
    ap.add_argument("--individual", action="store_true",
                    help="Lưu thêm ảnh overlay nguyên kích thước cho từng mẫu.")
    args = ap.parse_args()

    root = Path(args.dataset)
    out_dir = Path(args.out) if args.out else root / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with (root / "manifest.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["angle_deg"] = float(row["angle_deg"])
            row["bright"] = int(row["bright"])
            row["blur"] = int(row["blur"])
            rows.append(row)

    chosen = select_diverse(rows, args.samples)
    ind_dir = out_dir / "individual"
    if args.individual:
        ind_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[np.ndarray] = []
    print(f"{'image':<44} {'angle':>7} {'bright':>7} {'blur':>6}")
    for row in chosen:
        split = row["split"]
        img = cv2.imread(str(root / "images" / split / row["image"]))
        pts_norm = read_yolo_pose(
            root / "labels" / split / (Path(row["image"]).stem + ".txt")
        )
        if img is None or pts_norm is None:
            continue
        h, w = img.shape[:2]
        pts_px = [(int(round(x * w)), int(round(y * h))) for x, y in pts_norm]
        overlay = draw_overlay(img, pts_norm)
        x0, y0, x1, y1 = crop_around(pts_px, w, h, margin=0.35)
        crop = overlay[y0:y1, x0:x1]
        if args.individual:
            name = f"{len(tiles) + 1:02d}_{row['image'][:34]}.png"
            cv2.imwrite(str(ind_dir / name), crop)
        tile = fit_tile(crop)
        bar = np.zeros((BAR_H, TILE_W, 3), np.uint8)
        caption = (f"{row['image'][:24]} ang={row['angle_deg']:.1f} "
                   f"br={row['bright']} bl={row['blur']}")
        cv2.putText(bar, caption, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([tile, bar]))
        print(f"{row['image']:<44} {row['angle_deg']:>7.2f} "
              f"{row['bright']:>7} {row['blur']:>6}")

    if not tiles:
        print("[error] Không dựng được tile nào.")
        return 1

    cols = args.cols
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    per_sheet = max(1, args.per_sheet // cols) * cols
    sheet = 0
    for start in range(0, len(tiles), per_sheet):
        block_rows = [np.hstack(tiles[i:i + cols])
                      for i in range(start, min(len(tiles), start + per_sheet), cols)]
        sheet += 1
        path = out_dir / f"sheet_{sheet:02d}.png"
        cv2.imwrite(str(path), np.vstack(block_rows))
        print(f"[i] {path}")
    print(f"[i] {len(tiles)} tile -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

