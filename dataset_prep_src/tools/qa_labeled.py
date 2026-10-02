"""
tools/qa_labeled.py
===================
Kiểm tra invariants của ``labeled.csv`` + ``split.csv`` (đầu ra của unify_labeling.py)
bằng số — thay cho kiểm tra bằng mắt (chạy được cả khi không mở được ảnh).

Các phép kiểm:

* ``no_plate`` phải có toạ độ = 0; ảnh có text phải có toạ độ != 0.
* Toạ độ nằm trong khung ảnh; tứ giác lồi và diện tích > 0.
* Phân bố aspect-ratio & tỉ lệ diện tích (bắt lỗi nhân/hệ số phần trăm SAI).
* ``split.csv``: ảnh cùng ``track_id`` phải chung 1 split (không rò rỉ).

Chạy::

    python tools/qa_labeled.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

LABEL_DIR = Path(r"S:\M_AGENT\CV\labeling")
IMAGES_DIR = LABEL_DIR / "images"


def quad_of(row: dict) -> np.ndarray:
    return np.array([[float(row["tl_x"]), float(row["tl_y"])],
                     [float(row["tr_x"]), float(row["tr_y"])],
                     [float(row["br_x"]), float(row["br_y"])],
                     [float(row["bl_x"]), float(row["bl_y"])]])


def area(q: np.ndarray) -> float:
    return 0.5 * abs(float(
        (q[0, 0] * q[1, 1] - q[1, 0] * q[0, 1])
        + (q[1, 0] * q[2, 1] - q[2, 0] * q[1, 1])
        + (q[2, 0] * q[3, 1] - q[3, 0] * q[2, 1])
        + (q[3, 0] * q[0, 1] - q[0, 0] * q[3, 1])))


def is_convex(q: np.ndarray) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = [cross(q[i], q[(i + 1) % 4], q[(i + 2) % 4]) for i in range(4)]
    return all(s >= -1e-6 for s in signs) or all(s <= 1e-6 for s in signs)


def bucket(v: float, edges: tuple[float, ...]) -> str:
    for e in edges:
        if v < e:
            return f"<{e}"
    return f">={edges[-1]}"


def main() -> int:
    rows = list(csv.DictReader((LABEL_DIR / "labeled.csv").open(encoding="utf-8")))
    print(f"[i] rows={len(rows)}")
    print(f"[i] plate_text={dict(Counter(r['plate_text'] for r in rows))}")

    problems: list[tuple[str, str]] = []
    n_plate = 0
    ar: Counter = Counter()
    frac: Counter = Counter()
    for r in rows:
        q = quad_of(r)
        is_no = r["plate_text"] == "no_plate"
        if is_no:
            if np.any(q):
                problems.append(("no_plate_has_coords", r["image"]))
            continue
        n_plate += 1
        if not np.any(q):
            problems.append(("text_zero_coords", r["image"]))
            continue
        w, h = Image.open(IMAGES_DIR / r["image"]).size
        if not (0 <= q[:, 0].min() and q[:, 0].max() <= w
                and 0 <= q[:, 1].min() and q[:, 1].max() <= h):
            problems.append(("out_of_bounds", f"{r['image']} ({w}x{h})"))
        a = area(q)
        frac[bucket(a / (w * h), (0.01, 0.05, 0.1, 0.2))] += 1
        if a <= 4:
            problems.append(("tiny_area", r["image"]))
        tl, tr, br, bl = q
        ww = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        hh = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        ar[bucket(ww / max(hh, 1e-6), (1.5, 2.0, 2.5, 3.0, 4.0, 5.0))] += 1
        if not is_convex(q):
            problems.append(("non_convex", r["image"]))

    print(f"[i] plate rows={n_plate}")
    print(f"[i] area/total buckets: {dict(frac)}")
    print(f"[i] aspect_ratio buckets: {dict(ar)}")
    print(f"[i] problems: {len(problems)}")
    for kind, name in problems[:40]:
        print(f"  [X] {kind}: {name}")

    split_rows = list(csv.DictReader((LABEL_DIR / "split.csv").open(encoding="utf-8")))
    track_split: dict[str, str] = {}
    bad_split = 0
    for r in split_rows:
        if r["track_id"] in track_split and track_split[r["track_id"]] != r["split"]:
            bad_split += 1
        track_split[r["track_id"]] = r["split"]
    print(f"[i] split rows={len(split_rows)}; track-split inconsistency={bad_split}")
    for name in ("val", "test"):
        n_img = sum(1 for r in split_rows if r["split"] == name)
        n_pl = sum(1 for r in split_rows if r["split"] == name and r["has_plate"] == "1")
        print(f"[i] split {name}: {n_img} anh, {n_pl} anh co bien")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
