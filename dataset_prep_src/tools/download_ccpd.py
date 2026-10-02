"""
tools/download_ccpd.py
======================
Tải CCPD2019 từ HuggingFace mirror ``okita-souji/ccpd2019*`` rồi giải nén.

Nguồn (đã kiểm tra 2026-09-30):
  * ``okita-souji/ccpd2019balance`` -> ``trainn.zip`` (490 MB, 8,900 ảnh)
                                       ``valn.zip``   (487 MB, 8,874 ảnh)
  * ``okita-souji/ccpd2019train``   -> ``train.zip``  (5.47 GB, 100,000 ảnh)

Tên file giữ NGUYÊN format gốc CCPD, ví dụ::

  00260416666667-90_87-377&340_480&375-470&377_376&371_378&339_472&345-1_0_32_32_4_32_25-122-14.jpg
  [area_ratio] [tilt_h_tilt_v] [bbox x1&y1_x2&y2] [4 dinh, BAT DAU TU PHAI-DUOI] [7 ky tu] [bright] [blur]

7 field cách nhau bởi ``-``:
  0. area   : tỉ lệ diện tích biển / cả ảnh (bỏ dấu thập phân)
  1. tilt   : ``tilt_horizontal_tilt_vertical``
  2. bbox   : ``x1&y1_x2&y2`` (trái-trên & phải-dưới)
  3. corners: 4 đỉnh ``x&y``, **bắt đầu từ góc PHẢI-DƯỚI** theo thứ tự BR, BL, TL, TR
  4. number : 7 chỉ số ký tự (tỉnh, chữ, 5 ký tự)
  5. bright : độ sáng vùng biển
  6. blur   : độ mờ vùng biển  (đuôi ``.jpg``)

Chạy::

    python tools/download_ccpd.py --source balance
    python tools/download_ccpd.py --source train
    python tools/download_ccpd.py --source all

Script có resume: chạy lại sẽ tải tiếp file dở dang.
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

#: (repo_id, tên file trong repo) cho từng nguồn.
SOURCES: dict[str, list[tuple[str, str]]] = {
    "balance": [
        ("okita-souji/ccpd2019balance", "trainn.zip"),
        ("okita-souji/ccpd2019balance", "valn.zip"),
    ],
    "train": [
        ("okita-souji/ccpd2019train", "train.zip"),
    ],
}
SOURCES["all"] = SOURCES["balance"] + SOURCES["train"]


def download(repo_id: str, filename: str, out_dir: Path) -> Path:
    """Tải 1 file LFS (có resume) và trả về đường dẫn zip."""
    print(f"[i] Tải {repo_id}/{filename} ...", flush=True)
    started = time.time()
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(out_dir),
    )
    size_mb = Path(path).stat().st_size / 1e6
    print(
        f"[i] Xong {filename}: {size_mb:.1f} MB trong {time.time() - started:.1f}s",
        flush=True,
    )
    return Path(path)


def extract(zip_path: Path, extract_dir: Path) -> int:
    """Giải nén giữ nguyên cấu trúc thư mục bên trong; trả về số entry."""
    print(f"[i] Giải nén {zip_path.name} -> {extract_dir}", flush=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(extract_dir)
    print(f"[i] Đã giải nén {len(names)} entry", flush=True)
    return len(names)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tải + giải nén CCPD2019 từ HuggingFace.")
    ap.add_argument("--source", choices=sorted(SOURCES), default="balance")
    ap.add_argument("--out", default=r"S:\M_AGENT\CV\datasets_raw_ccpd", help="Thư mục lưu zip.")
    ap.add_argument("--no-extract", action="store_true", help="Chỉ tải, không giải nén.")
    ap.add_argument("--keep-zip", action="store_true", help="Giữ lại file zip sau khi giải nén.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = out_dir / "extracted"

    for repo_id, filename in SOURCES[args.source]:
        zip_path = download(repo_id, filename, out_dir)
        if args.no_extract:
            continue
        extract(zip_path, extract_dir)
        if not args.keep_zip:
            zip_path.unlink()
            print(f"[i] Đã xoá {zip_path.name} (dùng --keep-zip để giữ)", flush=True)

    print("[i] Hoàn tất.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
