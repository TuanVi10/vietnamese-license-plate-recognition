"""
tools/normalize_annguyen_labels.py
==================================
Sửa nhãn của nguồn ``datasets/raw/raw_annguyen`` (Roboflow "Vietnamese License
Plate - v8 Train6", MIT).

Nguồn này xuất khẩu nhãn KHÔNG chuẩn YOLO-detect: một số ảnh được gán nhãn dạng
polygon (``class x1 y1 x2 y2 ...``) viết trên MỘT dòng, trong khi các ảnh khác là
box chuẩn (``class cx cy w h``). Parser YOLO-detect chỉ hiểu 1 box/dòng nên sẽ
bỏ sót toàn bộ các polygon này (đếm được ~84 file, ~95 dòng polygon).

Script chỉ xử lý raw_annguyen, giữ nguyên các nguồn khác:

- dòng 5 token  (class + 4 số)            : giữ nguyên (box chuẩn).
- dòng >= 9 token (class + >= 8 số, chẵn) : polygon -> bounding box (min/max các
  đỉnh) rồi ghi lại thành ``class cx cy w h`` (1 box/dòng).
- các trường hợp khác: giữ nguyên + cảnh báo.

Chạy::

    python tools/normalize_annguyen_labels.py [<labels_root>]

Idempotent: chạy lại lần nữa không thay đổi gì thêm.
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_ROOT = r"S:\M_AGENT\CV\datasets\raw\raw_annguyen"


def polygon_to_box_line(tokens: list[str]) -> str:
    """Chuyển ``class x1 y1 x2 y2 ...`` thành 1 dòng box chuẩn ``class cx cy w h``."""
    cls = tokens[0]
    coords = [float(value) for value in tokens[1:]]
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    return f"{cls} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def normalize_file(path: Path) -> bool:
    """Rewrite one label file in place; return True if anything changed."""
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    output: list[str] = []
    changed = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue
        tokens = stripped.split()
        if len(tokens) == 5:
            output.append(line)
        elif len(tokens) >= 9 and (len(tokens) - 1) % 2 == 0:
            output.append(polygon_to_box_line(tokens))
            changed = True
        else:
            # Giữ nguyên, báo cáo để kiểm tra thủ công nếu phát sinh.
            output.append(line)
            print(f"  [warn] dong la, giu nguyen ({len(tokens)} token): {path.name}")
    if changed:
        path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_ROOT)
    if not root.exists():
        print(f"[error] khong tim thay {root}")
        return 1

    files = [p for p in sorted(root.rglob("*.txt")) if not p.name.startswith("README")]
    changed = 0
    for path in files:
        if normalize_file(path):
            changed += 1
    print(f"normalize_annguyen_labels: {len(files)} label file, da sua {changed} file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
