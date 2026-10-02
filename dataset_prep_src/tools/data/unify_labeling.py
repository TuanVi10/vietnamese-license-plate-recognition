"""
tools/unify_labeling.py
=======================
Chuẩn hoá 4 file kết quả gán nhãn — vốn dùng **nhiều chuẩn toạ độ KHÁC NHAU** — về
MỘT file CSV thống nhất, toạ độ **PIXEL** trên ảnh crop.

Nguồn:

1. ``result/labels201-400.csv`` — toạ độ **PIXEL**, đúng schema đích, nhưng cột
   ``image`` là URL Label Studio (``?d=images%5C000201.jpg``) cần decode.
2. ``result/401-600.csv`` — export thô Label Studio: toạ độ **PHẦN TRĂM (0-100)**
   nằm trong cột ``label`` (JSON) kèm ``original_width/height``. Polygon có thể
   gồm 5 điểm (điểm cuối trùng điểm đầu) cần bỏ bớt.
3. ``result/labels_anh_601_800.csv`` — đã xử lý: toạ độ **PIXEL**, đúng schema đích.
4. ``result/801-1701.xlsx`` — đã xử lý nhưng SAI: cột tên ``unreadable_text`` và
   toạ độ là **PHẦN TRĂM (0-100)**.

Đầu ra ``labeled.csv`` — 10 cột, toạ độ pixel trên ``labeling/images/<image>``::

    image,plate_text,tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y

Thứ tự góc **luôn được sắp lại theo hình học** về TL, TR, BR, BL. Ảnh không có
biển -> ``plate_text=no_plate`` và toạ độ = 0.

Kèm ``split.csv`` chia tập **theo ``track_id``** (không chia theo ảnh, vì mỗi
track là nhiều khung hình liên tiếp của cùng một xe -> chia theo ảnh sẽ rò rỉ).

Chạy::

    python tools/unify_labeling.py
    python tools/unify_labeling.py --dump-overlays 12
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.utils.geometry_utils import dedupe_closing_point, order_corners_geometric

LABEL_DIR = Path(r"S:\M_AGENT\CV\labeling")
RESULT_DIR = LABEL_DIR / "result"
IMAGES_DIR = LABEL_DIR / "images"
MANIFEST = LABEL_DIR / "manifest.csv"

COLUMNS = ["image", "plate_text", "tl_x", "tl_y", "tr_x", "tr_y",
           "br_x", "br_y", "bl_x", "bl_y"]

#: Nguồn gốc từng dải ảnh (để báo cáo).
SOURCE_OF = {"1-200": "1-200.xlsx (pixel, dung)",
             "201-400": "labels201-400.csv (pixel, dung, URL-encode)",
             "401-600": "401-600.csv (Label Studio, phan tram)",
             "601-800": "labels_anh_601_800.csv (pixel, dung)",
             "801-1701": "801-1701.xlsx (tram, ten cot sai)"}


def norm_text(text: str) -> str:
    """Chuẩn hoá nhãn text: ``no plate`` -> ``no_plate``."""
    value = (text or "").strip()
    if value.lower() in ("no plate", "no_plate", "noplate"):
        return "no_plate"
    return value


def filename_from_ls_url(url: str) -> str:
    """``/data/local-files/?d=test%5C000407.jpg`` -> ``000407.jpg``."""
    raw = unquote(url.rsplit("?", 1)[-1]).split("=", 1)[-1]
    return raw.replace("\\", "/").rsplit("/", 1)[-1]


def image_size(name: str) -> tuple[int, int] | None:
    """(width, height) của ảnh crop; ``None`` nếu thiếu file."""
    path = IMAGES_DIR / name
    if not path.exists():
        return None
    with Image.open(path) as img:
        return img.size


def flat(quad: tuple[tuple[float, float], ...]) -> list[int]:
    """(TL,TR,BR,BL) -> 8 số nguyên theo thứ tự cột."""
    return [int(round(v)) for point in quad for v in point]


def zeros() -> list[int]:
    """8 toạ độ 0 cho ảnh không có biển."""
    return [0] * 8


def reorder_and_flat(pixel_points: list[tuple[float, float]],
                     stats: dict, key: str) -> list[int]:
    """Sắp lại theo hình học rồi làm phẳng; đếm số dòng bị đổi thứ tự."""
    given = [(float(x), float(y)) for x, y in pixel_points]
    quad = order_corners_geometric(given)
    if any(abs(g[0] - q[0]) > 1e-6 or abs(g[1] - q[1]) > 1e-6
           for g, q in zip(given, quad)):
        stats[f"reordered_{key}"] += 1
    return flat(quad)


def parse_pixel(pixel: list[tuple[float, float]], text: str, stats: dict,
                key: str) -> list[int]:
    """Nếu có biển thì sắp lại 4 góc; nếu không thì trả 8 số 0."""
    if text == "no_plate":
        return zeros()
    if not any(abs(x) > 1e-9 or abs(y) > 1e-9 for x, y in pixel):
        stats["has_text_no_poly"] += 1
        return zeros()
    return reorder_and_flat(pixel, stats, key)


def load_201_400(stats: dict) -> list[dict]:
    """Nguồn 201-400: toạ độ PIXEL, đúng schema; cột image là URL Label Studio."""
    out: list[dict] = []
    with (RESULT_DIR / "labels201-400.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = filename_from_ls_url(row["image"])
            text = norm_text(row["plate_text"])
            if image_size(name) is None:
                stats["missing_image"] += 1
                continue
            raw = [float(row[c]) for c in COLUMNS[2:]]
            pixel = [(raw[0], raw[1]), (raw[2], raw[3]), (raw[4], raw[5]), (raw[6], raw[7])]
            out.append(dict(zip(COLUMNS, [name, text] +
                                parse_pixel(pixel, text, stats, "201_400"))))
    return out


def load_401_600(stats: dict) -> list[dict]:
    """Nguồn 401-600: toạ độ PHẦN TRĂM trong JSON của Label Studio."""
    out: list[dict] = []
    with (RESULT_DIR / "401-600.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = filename_from_ls_url(row["image"])
            text = norm_text(row["plate_text"])
            size = image_size(name)
            if size is None:
                stats["missing_image"] += 1
                continue
            width, height = size
            pixel: list[tuple[float, float]] = []
            if text != "no_plate" and row["label"].strip():
                item = json.loads(row["label"])[0]
                if (item["original_width"], item["original_height"]) != (width, height):
                    stats["orig_size_mismatch"] += 1
                pts = dedupe_closing_point(item["points"])
                if len(pts) > 4:
                    stats["poly_gt4"] += 1
                pixel = [(float(x) / 100.0 * width, float(y) / 100.0 * height)
                         for x, y in pts[:4]]
            out.append(dict(zip(COLUMNS, [name, text] +
                                parse_pixel(pixel, text, stats, "401_600"))))
    return out


def load_601_800(stats: dict) -> list[dict]:
    """Nguồn 601-800: đã là PIXEL, đúng schema — chỉ chuẩn hoá text + sắp lại góc."""
    out: list[dict] = []
    with (RESULT_DIR / "labels_anh_601_800.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["image"]
            text = norm_text(row["plate_text"])
            if image_size(name) is None:
                stats["missing_image"] += 1
                continue
            raw = [float(row[c]) for c in COLUMNS[2:]]
            pixel = [(raw[0], raw[1]), (raw[2], raw[3]), (raw[4], raw[5]), (raw[6], raw[7])]
            out.append(dict(zip(COLUMNS, [name, text] +
                                parse_pixel(pixel, text, stats, "601_800"))))
    return out


def load_1_200(stats: dict) -> list[dict]:
    """Nguồn 1-200: xlsx, toạ độ PIXEL, đúng schema (image là tên file)."""
    from openpyxl import load_workbook

    out: list[dict] = []
    wb = load_workbook(RESULT_DIR / "1-200.xlsx", read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = ws.iter_rows(values_only=True)
    next(rows)  # bỏ header
    for row in rows:
        if not row or not row[0]:
            continue
        name = str(row[0]).strip()
        text = norm_text(str(row[1] or ""))
        if image_size(name) is None:
            stats["missing_image"] += 1
            continue
        raw = [float(v or 0.0) for v in row[2:10]]
        pixel = [(raw[0], raw[1]), (raw[2], raw[3]), (raw[4], raw[5]), (raw[6], raw[7])]
        out.append(dict(zip(COLUMNS, [name, text] +
                            parse_pixel(pixel, text, stats, "1_200"))))
    wb.close()
    return out


def load_801_1701(stats: dict) -> list[dict]:
    """Nguồn 801-1701: cột tên sai ``unreadable_text`` + toạ độ PHẦN TRĂM (0-100)."""
    from openpyxl import load_workbook

    out: list[dict] = []
    wb = load_workbook(RESULT_DIR / "801-1701.xlsx", read_only=True, data_only=True)
    ws = wb["Sheet1"] if "Sheet1" in wb.sheetnames else wb[wb.sheetnames[0]]
    rows = ws.iter_rows(values_only=True)
    header = [str(c) for c in next(rows)]
    idx_text = header.index("unreadable_text") if "unreadable_text" in header else 1
    for row in rows:
        if not row or not row[0]:
            continue
        name = str(row[0]).strip()
        text = norm_text(str(row[idx_text] or ""))
        size = image_size(name)
        if size is None:
            stats["missing_image"] += 1
            continue
        width, height = size
        raw = [float(v or 0.0) for v in row[2:10]]
        pixel = [(raw[0] / 100.0 * width, raw[1] / 100.0 * height),
                 (raw[2] / 100.0 * width, raw[3] / 100.0 * height),
                 (raw[4] / 100.0 * width, raw[5] / 100.0 * height),
                 (raw[6] / 100.0 * width, raw[7] / 100.0 * height)]
        out.append(dict(zip(COLUMNS, [name, text] +
                            parse_pixel(pixel, text, stats, "801_1701"))))
    return out


def source_of(image: str) -> str:
    """Suy ra nguồn gốc của ảnh từ số thứ tự trong tên file."""
    number = int(image[:6])
    if number <= 200:
        return "1-200"
    if number <= 400:
        return "201-400"
    if number <= 600:
        return "401-600"
    if number <= 800:
        return "601-800"
    return "801-1701"


def build_split(labeled: list[dict], val_frac: float, seed: int) -> list[dict]:
    """Chia val/test **theo track_id** (mỗi track = 1 xe, nhiều khung hình liên tiếp).

    Chia theo ảnh sẽ làm ảnh của cùng một xe rơi vào cả val lẫn test -> rò rỉ.
    """
    track_of: dict[str, int] = {}
    with MANIFEST.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            track_of[row["image"]] = int(row["track_id"])

    by_track: dict[int, list[str]] = {}
    text_of: dict[str, str] = {}
    for row in labeled:
        track_id = track_of.get(row["image"])
        if track_id is None:
            continue
        by_track.setdefault(track_id, []).append(row["image"])
        text_of[row["image"]] = row["plate_text"]

    ids = sorted(by_track)
    random.Random(seed).shuffle(ids)
    n_val = int(round(len(ids) * val_frac))
    val_ids = set(ids[:n_val])

    out: list[dict] = []
    for track_id in sorted(by_track):
        split = "val" if track_id in val_ids else "test"
        for image in by_track[track_id]:
            out.append({"image": image, "track_id": track_id, "split": split,
                        "has_plate": int(text_of[image] != "no_plate"),
                        "plate_text": text_of[image]})
    return out


def dump_overlays(labeled: list[dict], count: int, out_dir: Path, seed: int) -> int:
    """Vẽ đè 4 góc GT lên ảnh crop để kiểm tra bằng mắt (đủ cả 3 nguồn)."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = {"1-200": [], "201-400": [], "401-600": [],
                                     "601-800": [], "801-1701": []}
    for row in labeled:
        if row["plate_text"] != "no_plate":
            groups[source_of(row["image"])].append(row)

    rng = random.Random(seed)
    per = max(1, count // len(groups))
    picks: list[tuple[str, dict]] = []
    for key, rows in groups.items():
        rng.shuffle(rows)
        picks.extend((key, row) for row in rows[:per])

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 165, 255)]
    written = 0
    for key, row in picks:
        img = cv2.imread(str(IMAGES_DIR / row["image"]))
        if img is None:
            continue
        pts = [(int(row["tl_x"]), int(row["tl_y"])), (int(row["tr_x"]), int(row["tr_y"])),
               (int(row["br_x"]), int(row["br_y"])), (int(row["bl_x"]), int(row["bl_y"]))]
        for i in range(4):
            cv2.line(img, pts[i], pts[(i + 1) % 4], (0, 255, 255), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (x, y), 6, colors[i], -1, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, colors[i], 1, cv2.LINE_AA)
        caption = f"{row['image']} | {row['plate_text']} | nguon {key}"
        cv2.putText(img, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{key}_{row['image']}"), img)
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Chuẩn hoá 4 nguồn nhãn về 1 CSV pixel.")
    ap.add_argument("--out", default=str(LABEL_DIR / "labeled.csv"))
    ap.add_argument("--split-out", default=str(LABEL_DIR / "split.csv"))
    ap.add_argument("--val-frac", type=float, default=0.5,
                    help="Tỉ lệ TRACK dành cho val (phần còn lại là test).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-overlays", type=int, default=0,
                    help="Vẽ N ảnh overlay GT (chia đều 3 nguồn) để kiểm tra bằng mắt.")
    args = ap.parse_args()

    stats: dict[str, int] = defaultdict(int)
    rows = (load_1_200(stats) + load_201_400(stats) + load_401_600(stats)
            + load_601_800(stats) + load_801_1701(stats))
    rows.sort(key=lambda r: r["image"])

    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    by_src: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        text = row["plate_text"]
        key = text if text in ("no_plate", "unreadable") else "text"
        by_src[source_of(row["image"])][key] += 1

    print(f"[i] labeled.csv: {len(rows)} dong -> {out_path}")
    print(f"{'nguon':<10} {'no_plate':>9} {'unreadable':>11} {'text':>6} {'tong':>6}")
    for key in ("1-200", "201-400", "401-600", "601-800", "801-1701"):
        c = by_src[key]
        print(f"{key:<10} {c['no_plate']:>9} {c['unreadable']:>11} {c['text']:>6} "
              f"{sum(c.values()):>6}")
    total: Counter = Counter()
    for c in by_src.values():
        total.update(c)
    print(f"{'TONG':<10} {total['no_plate']:>9} {total['unreadable']:>11} "
          f"{total['text']:>6} {sum(total.values()):>6}")
    print(f"[i] Anh CO bien (model goc): {total['unreadable'] + total['text']}"
          f"  | CO text GT (OCR): {total['text']}")

    split = build_split(rows, args.val_frac, args.seed)
    split_path = Path(args.split_out)
    with split_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "track_id", "split",
                                                "has_plate", "plate_text"])
        writer.writeheader()
        writer.writerows(split)
    per_split: dict[str, Counter] = defaultdict(Counter)
    for row in split:
        per_split[row["split"]]["images"] += 1
        per_split[row["split"]]["plates"] += int(row["has_plate"])
    for name in ("val", "test"):
        print(f"[i] split {name}: {per_split[name]['images']} anh, "
              f"{per_split[name]['plates']} anh co bien")
    print(f"[i] split.csv -> {split_path}  (chia theo track_id)")
    for key in sorted(stats):
        print(f"[i] {key}: {stats[key]}")

    if args.dump_overlays:
        written = dump_overlays(rows, args.dump_overlays,
                                LABEL_DIR / "viz_labeled", args.seed)
        print(f"[i] {written} anh overlay -> {LABEL_DIR / 'viz_labeled'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



