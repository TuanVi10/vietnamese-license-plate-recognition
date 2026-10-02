"""
tools/gen_ccpd_warped.py
========================
Bước (2) corner regressor: sinh dữ liệu train "crop biển số + 4 góc" từ CCPD.

Với mỗi ảnh CCPD (đã có 4 góc GT kiểu YOLO-pose):
  1. Pad ảnh thêm ``--margin`` (tỉ lệ so với kích thước biển) — mô phỏng crop có lề
     như ``expand_bbox`` của detector (KHÔNG crop sát biên).
  2. Áp homography ngẫu nhiên (rotation + perspective jitter + scale) lên cả crop.
  3. Biến đổi 4 góc qua cùng homography -> GT mới.
  4. LỌC BỎ các ca có góc rơi ra ngoài khung (gần biên) sau warp (đúng yêu cầu).

Đầu ra (architecture-agnostic):
  datasets/corner_regression/ccpd_warped/images/<idx>.jpg
  datasets/corner_regression/ccpd_warped/labels.csv   (image + 8 toạ độ pixel)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

CCPD = Path(r"S:\M_AGENT\CV\datasets\ccpd_pose")
OUT = Path(r"S:\M_AGENT\CV\datasets\corner_regression\ccpd_warped")
COLUMNS = ["image", "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"]


def parse_label(line: str, w: int, h: int):
    """YOLO-pose label -> 4 góc (pixel), thứ tự TL,TR,BR,BL."""
    parts = line.split()
    # parts: class cx cy bw bh  (x y v)*4
    kps = np.array([float(x) for x in parts[5:]], np.float32).reshape(4, 3)
    return kps[:, :2] * np.array([w, h], np.float32)


def random_homography(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(-40.0, 40.0)
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    cx, cy = w / 2.0, h / 2.0
    R = np.array([[c, -s, cx - c * cx + s * cy],
                  [s, c, cy - s * cx - c * cy],
                  [0, 0, 1]], np.float64)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jx = rng.uniform(-0.12, 0.12, 4) * w
    jy = rng.uniform(-0.12, 0.12, 4) * h
    dst = (src + np.stack([jx, jy], axis=1)).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    return P @ R


def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1))
    q = (H @ np.hstack([pts, ones]).T).T
    return (q[:, :2] / q[:, 2][:, None]).astype(np.float32)


def tight_crop(img: np.ndarray, corners: np.ndarray, pad: float = 0.05):
    """Crop sát vùng biển (bbox 4 góc + pad nhỏ). Trả (plate_img, corners_rel)."""
    h, w = img.shape[:2]
    bb = np.array([corners[:, 0].min(), corners[:, 1].min(),
                   corners[:, 0].max(), corners[:, 1].max()])
    dx = (bb[2] - bb[0]) * pad
    dy = (bb[3] - bb[1]) * pad
    x1 = int(round(max(0, bb[0] - dx)))
    y1 = int(round(max(0, bb[1] - dy)))
    x2 = int(round(min(w, bb[2] + dx)))
    y2 = int(round(min(h, bb[3] + dy)))
    return img[y1:y2, x1:x2].copy(), corners - np.array([x1, y1], np.float32)


def random_dest_quad(cw: int, ch: int, pw: int, ph: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Tứ giác đích (xoay ±40° + perspective jitter), LUÔN nằm trong canvas."""
    scale = 0.75
    w, h = pw * scale, ph * scale
    theta = rng.uniform(-40.0, 40.0)
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    quad = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                     [w / 2, h / 2], [-w / 2, h / 2]], np.float64)
    R = np.array([[c, -s], [s, c]])
    quad = (R @ quad.T).T + np.array([cw / 2, ch / 2])
    quad += rng.uniform(-0.08, 0.08, (4, 2)) * np.array([w, h])
    quad[:, 0] = np.clip(quad[:, 0], 3, cw - 3)
    quad[:, 1] = np.clip(quad[:, 1], 3, ch - 3)
    return quad.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--margin", type=float, default=0.15,
                    help="lề crop (tỉ lệ so với cạnh ngắn ảnh) — khớp expand_bbox")
    ap.add_argument("--warps-per-image", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = toàn bộ")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    img_dir = CCPD / "images" / args.split
    lbl_dir = CCPD / "labels" / args.split
    out_dir = Path(args.out)
    out_img = out_dir / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    label_files = sorted(lbl_dir.glob("*.txt"))
    if args.limit:
        label_files = label_files[: args.limit]

    rng = np.random.default_rng(0)
    rows = []
    idx = 0
    skipped = 0
    for lf in label_files:
        img_path = img_dir / (lf.stem + ".jpg")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = [ln for ln in lf.read_text().splitlines() if ln.strip()]
        if not lines:
            continue
        corners = parse_label(lines[0], w, h)  # TL,TR,BR,BL

        plate, corners_rel = tight_crop(img, corners)
        ph, pw = plate.shape[:2]
        if pw < 8 or ph < 8:
            continue

        # canvas = crop có lề (mô phỏng expand_bbox của detector)
        cw = int(round(pw * (1 + 2 * args.margin)))
        ch = int(round(ph * (1 + 2 * args.margin)))
        cw, ch = max(pw + 4, cw), max(ph + 4, ch)

        for _ in range(args.warps_per_image):
            dst = random_dest_quad(cw, ch, pw, ph, rng)
            H = cv2.getPerspectiveTransform(corners_rel.astype(np.float32), dst)
            warped = cv2.warpPerspective(plate, H, (cw, ch),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(255, 255, 255))

            # loại ca suy biến (diện tích tứ giác quá nhỏ)
            area = 0.5 * abs((dst[0, 0] * dst[1, 1] - dst[1, 0] * dst[0, 1])
                             + (dst[1, 0] * dst[2, 1] - dst[2, 0] * dst[1, 1])
                             + (dst[2, 0] * dst[3, 1] - dst[3, 0] * dst[2, 1])
                             + (dst[3, 0] * dst[0, 1] - dst[0, 0] * dst[3, 1]))
            if area < 0.2 * pw * ph:
                skipped += 1
                continue

            name = f"{idx:06d}.jpg"
            cv2.imwrite(str(out_img / name), warped)
            rows.append([name] + [int(round(v)) for p in dst for v in p])
            idx += 1

    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)

    print(f"[i] sinh {idx} anh warped (skip {skipped} ca goc ra ngoai khung)")
    print(f"[i] output -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
