"""
tools/ccpd_warped_to_yolo.py
===========================
Chuyển CCPD-warped (labels.csv: image + 8 toạ độ pixel) sang YOLO-pose (4 keypoint)
để train corner regressor. Tạo datasets/corner_regression/{images,labels}/{train,val}/
và data.yaml (kpt_shape [4, 3]).
"""
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SRC = Path(r"S:\M_AGENT\CV\datasets\corner_regression\ccpd_warped")
DST = Path(r"S:\M_AGENT\CV\datasets\corner_regression")


def main() -> int:
    rows = list(csv.DictReader((SRC / "labels.csv").open(encoding="utf-8")))
    rng = random.Random(0)
    rng.shuffle(rows)
    n_val = int(len(rows) * 0.05)
    val = {r["image"] for r in rows[:n_val]}

    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    n = {"train": 0, "val": 0}
    for r in rows:
        imgp = SRC / "images" / r["image"]
        img = cv2.imread(str(imgp))
        if img is None:
            continue
        h, w = img.shape[:2]
        q = np.array([[float(r["tl_x"]), float(r["tl_y"])],
                      [float(r["tr_x"]), float(r["tr_y"])],
                      [float(r["br_x"]), float(r["br_y"])],
                      [float(r["bl_x"]), float(r["bl_y"])]], np.float32)
        x0, x1 = q[:, 0].min(), q[:, 0].max()
        y0, y1 = q[:, 1].min(), q[:, 1].max()
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        kps = q / np.array([w, h], np.float32)
        line = (f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
                + " ".join(f"{p[0]:.6f} {p[1]:.6f} 2" for p in kps))
        split = "val" if r["image"] in val else "train"
        shutil.copy(str(imgp), str(DST / "images" / split / r["image"]))
        stem = r["image"].rsplit(".", 1)[0]
        (DST / "labels" / split / (stem + ".txt")).write_text(line + "\n",
                                                              encoding="utf-8")
        n[split] += 1

    yaml = (
        f"path: {DST.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: license_plate\n"
        "kpt_shape: [4, 3]\n"
    )
    (DST / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"[i] train {n['train']} | val {n['val']} -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
