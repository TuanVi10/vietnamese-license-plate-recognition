"""
tools/ccpd_to_pose.py
=====================
Chuyển CCPD2019 (nhãn nằm trong TÊN FILE) sang dataset YOLO-pose 4 keypoint.

Format tên file CCPD (7 field, cách nhau bởi ``-``)::

    <area>-<tiltH>_<tiltV>-<x1>&<y1>_<x2>&<y2>-<4 dinh>-<7 ky tu>-<bright>-<blur>.jpg

    vd: 00260416666667-90_87-377&340_480&375-470&377_376&371_378&339_472&345-1_0_32_32_4_32_25-122-14.jpg

QUAN TRỌNG: 4 đỉnh trong field 3 của CCPD **bắt đầu từ góc PHẢI-DƯỚI**, theo thứ tự
riêng của họ (BR, BL, TL, TR). Script KHÔNG dùng thứ tự gốc này mà **sắp xếp lại theo
vị trí hình học thực tế** về thứ tự cố định của mình: **TL -> TR -> BR -> BL**.

Thứ tự hình học dùng công thức kinh điển cho tứ giác lồi::

    TL = min(x + y)      BR = max(x + y)
    TR = max(x - y)      BL = min(x - y)

Đồng thời script **đối chiếu** kết quả hình học với thứ tự gốc của CCPD (xoay vòng
[BR,BL,TL,TR] -> [TL,TR,BR,BL]) để phát hiện ca bất đồng (biển xoay gần 45 độ...).

Đầu ra (YOLO-pose, ``kpt_shape: [4, 3]``, toạ độ chuẩn hoá theo kích thước ảnh)::

    images/train/*.jpg   labels/train/*.txt
    images/val/*.jpg     labels/val/*.txt
    data_pose.yaml       manifest.csv

Chạy::

    python tools/ccpd_to_pose.py --src S:/M_AGENT/CV/datasets_raw_ccpd/extracted \
        --out S:/M_AGENT/CV/datasets/ccpd_pose
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.utils.geometry_utils import is_convex, order_corners_geometric


def parse_ccpd_name(name: str) -> dict | None:
    """Tách 7 field trong tên file CCPD; trả ``None`` nếu không đúng format."""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    try:
        tilt_h, tilt_v = (int(v) for v in parts[1].split("_"))
        bx1, by1 = (int(v) for v in parts[2].split("_")[0].split("&"))
        bx2, by2 = (int(v) for v in parts[2].split("_")[1].split("&"))
        raw = [tuple(int(v) for v in p.split("&")) for p in parts[3].split("_")]
        bright, blur = int(parts[5]), int(parts[6])
    except ValueError:
        return None
    if len(raw) != 4 or any(len(p) != 2 for p in raw):
        return None
    return {
        "tilt_h": tilt_h,
        "tilt_v": tilt_v,
        "bbox": (bx1, by1, bx2, by2),
        "corners_raw": raw,          # thứ tự gốc CCPD: BR, BL, TL, TR
        "bright": bright,
        "blur": blur,
    }


def order_corners_from_ccpd(raw: list[tuple[int, int]]) -> tuple[tuple[float, float], ...]:
    """Sắp theo quy ước gốc của CCPD: [BR, BL, TL, TR] -> [TL, TR, BR, BL]."""
    br, bl, tl, tr = [(float(x), float(y)) for x, y in raw]
    return tl, tr, br, bl


def plate_angle(tl: tuple[float, float], tr: tuple[float, float]) -> float:
    """Góc nghiêng của cạnh trên biển (TL->TR) so với trục ngang, đơn vị độ."""
    return math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))


def to_yolo_pose_line(pts: tuple[tuple[float, float], ...], width: int, height: int) -> str:
    """Tạo 1 dòng nhãn YOLO-pose: ``0 cx cy w h x1 y1 v ... x4 y4 v`` (chuẩn hoá)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = max(0.0, min(xs)), min(float(width), max(xs))
    ymin, ymax = max(0.0, min(ys)), min(float(height), max(ys))
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height
    fields = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for x, y in pts:
        nx = min(max(x / width, 0.0), 1.0)
        ny = min(max(y / height, 0.0), 1.0)
        fields.append(f"{nx:.6f} {ny:.6f} 2")
    return " ".join(fields)


def place(src: Path, dst: Path) -> None:
    """Hardlink nếu cùng ổ đĩa, nếu không thì copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="CCPD -> YOLO-pose 4 keypoint.")
    ap.add_argument("--src", required=True, help="Thư mục đã giải nén (chứa trainn/ valn/).")
    ap.add_argument("--out", required=True, help="Thư mục dataset YOLO-pose đầu ra.")
    ap.add_argument("--splits", nargs="+", default=["trainn=train", "valn=val"],
                    help="Ánh xạ <thư mục nguồn>=<split đích>.")
    args = ap.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    manifest_rows: list[dict] = []
    stats = {"total": 0, "parsed": 0, "bad_name": 0, "unreadable": 0,
             "disagree": 0, "nonconvex": 0}

    for mapping in args.splits:
        src_name, split = mapping.split("=")
        img_dir = src_root / src_name
        out_img = out_root / "images" / split
        out_lbl = out_root / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for path in sorted(img_dir.glob("*.jpg")):
            stats["total"] += 1
            info = parse_ccpd_name(path.name)
            if info is None:
                stats["bad_name"] += 1
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                stats["unreadable"] += 1
                continue
            height, width = img.shape[:2]

            geo = order_corners_geometric(info["corners_raw"])
            ccpd = order_corners_from_ccpd(info["corners_raw"])
            if any(abs(g[0] - c[0]) > 1e-6 or abs(g[1] - c[1]) > 1e-6
                   for g, c in zip(geo, ccpd)):
                stats["disagree"] += 1
            if not is_convex(geo):
                stats["nonconvex"] += 1

            place(path, out_img / path.name)
            (out_lbl / f"{path.stem}.txt").write_text(
                to_yolo_pose_line(geo, width, height) + "\n", encoding="utf-8"
            )
            stats["parsed"] += 1
            manifest_rows.append({
                "image": path.name,
                "split": split,
                "angle_deg": round(plate_angle(geo[0], geo[1]), 3),
                "tilt_h": info["tilt_h"],
                "tilt_v": info["tilt_v"],
                "bright": info["bright"],
                "blur": info["blur"],
                "width": width,
                "height": height,
            })

    if not manifest_rows:
        print("[error] Không có mẫu nào được chuyển.")
        return 1

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    (out_root / "data_pose.yaml").write_text(
        "# Auto-generated: CCPD2019 -> YOLO-pose 4 keypoint (TL, TR, BR, BL)\n"
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: license_plate\n"
        "kpt_shape: [4, 3]\n"
        "flip_idx: [1, 0, 3, 2]\n",
        encoding="utf-8",
    )

    print(f"total={stats['total']} parsed={stats['parsed']} "
          f"bad_name={stats['bad_name']} unreadable={stats['unreadable']}")
    print(f"disagree_geo_vs_ccpd={stats['disagree']} nonconvex={stats['nonconvex']}")
    print(f"manifest + data_pose.yaml -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

