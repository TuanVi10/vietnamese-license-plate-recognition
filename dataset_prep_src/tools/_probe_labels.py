"""Probe tạm: xem cấu trúc 3 file kết quả label trước khi viết script unify."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

RES = Path(r"S:\M_AGENT\CV\labeling\result")

# ---- 401-600.csv (Label Studio export) -------------------------------
print("=" * 70)
print("401-600.csv")
rows = list(csv.DictReader(RES.joinpath("401-600.csv").open(encoding="utf-8")))
print("cols :", list(rows[0].keys()))
print("rows :", len(rows))
print("plate_text values:", Counter(r["plate_text"] for r in rows))
withplate = [r for r in rows if r["plate_text"] not in ("", "no plate", "no_plate")]
print("rows with plate:", len(withplate))
for r in withplate[:2]:
    print("  image      :", r["image"])
    print("  plate_text :", r["plate_text"])
    print("  label      :", r["label"][:600])
    try:
        obj = json.loads(r["label"])
        print("  label keys :", list(obj[0].keys()))
        for item in obj:
            print("   - type:", item.get("type"), "| keys:", sorted(item.keys()))
            if "value" in item:
                print("     value keys:", list(item["value"].keys()))
                print("     value     :", str(item["value"])[:400])
    except Exception as exc:
        print("  [json err]", exc)
    print()

# ---- labels_anh_601_800.csv ------------------------------------------
print("=" * 70)
print("labels_anh_601_800.csv")
rows2 = list(csv.DictReader(RES.joinpath("labels_anh_601_800.csv").open(encoding="utf-8")))
print("cols :", list(rows2[0].keys()))
print("rows :", len(rows2))
print("plate_text values:", Counter(r["plate_text"] for r in rows2))
wp2 = [r for r in rows2 if r["plate_text"] not in ("no_plate", "no plate", "")]
for r in wp2[:2]:
    print("  ", r)
print()

# ---- 801-1701.xlsx ---------------------------------------------------
print("=" * 70)
print("801-1701.xlsx")
from openpyxl import load_workbook

wb = load_workbook(RES / "801-1701.xlsx", read_only=True, data_only=True)
print("sheets:", wb.sheetnames)
ws = wb["labels"] if "labels" in wb.sheetnames else wb[wb.sheetnames[0]]
it = ws.iter_rows(values_only=True)
header = next(it)
print("header:", header)
data = [r for r in it if r and r[0]]
print("rows :", len(data))
txt_col = 1
print("text values (first 15):", Counter(str(r[txt_col]) for r in data[:400]))
for r in data[:3]:
    print("  ", r)
nums = []
for r in data:
    for v in r[2:10]:
        if isinstance(v, (int, float)):
            nums.append(float(v))
print("coord min/max:", min(nums), max(nums))
