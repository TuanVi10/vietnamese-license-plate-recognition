"""
tools/analyze_angle_dist.py
===========================
Trả lời: 54/213 ảnh >15° trong tập nhãn tay (401-1701) có ĐẠI DIỆN phân bố góc
thật của video, hay do CÁCH CHỌN MẪU gán nhãn (lệch)?

Phân tích:
  * Số track_id chứa ảnh >15°: tập trung vài track = artifact; trải nhiều = đại diện.
  * Phân bố dải góc: toàn bộ ảnh có biển vs chỉ ảnh có text.
  * frame_index của ảnh >15°: có dồn về 1 đoạn video không.
"""

from __future__ import annotations

import csv
import math
from collections import Counter

LABEL = r"S:\M_AGENT\CV\labeling\labeled.csv"
MANIFEST = r"S:\M_AGENT\CV\labeling\manifest.csv"


def band(a: float) -> str:
    a = abs(a)
    return "<5" if a < 5 else ("5-15" if a < 15 else ">15")


def main() -> int:
    mani = {}
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        mani[r["image"]] = r

    plates = []  # (image, band, angle, has_text, track_id, frame_index)
    for r in csv.DictReader(open(LABEL, encoding="utf-8")):
        num = int(r["image"][:6])
        if not (401 <= num <= 1701):
            continue
        if r["plate_text"] == "no_plate":
            continue
        q = [(float(r["tl_x"]), float(r["tl_y"])),
             (float(r["tr_x"]), float(r["tr_y"]))]
        ang = math.degrees(math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]))
        has_text = r["plate_text"] not in ("unreadable", "")
        m = mani.get(r["image"], {})
        plates.append((r["image"], band(ang), ang, has_text,
                       m.get("track_id", "?"), int(m.get("frame_index", 0) or 0)))

    all_plates = Counter(b for _, b, _, _, _, _ in plates)
    text_plates = Counter(b for _, b, _, t, _, _ in plates if t)
    print("TOAN BO anh co bien (401-1701):", dict(all_plates), "n =", len(plates))
    print("Chi anh CO TEXT               :", dict(text_plates),
          "n =", sum(text_plates.values()))

    gt15 = [p for p in plates if p[1] == ">15"]
    tracks_gt15 = Counter(p[4] for p in gt15)
    all_tracks = set(p[4] for p in plates)
    n_tracks_with_gt15 = len([t for t in all_tracks if t in tracks_gt15])
    print(f"\n[>15] so anh = {len(gt15)} | so track chua >=1 anh >15 = "
          f"{n_tracks_with_gt15} / {len(all_tracks)} track")
    print("[>15] top track:", tracks_gt15.most_common(10))

    fi = sorted(p[5] for p in gt15)
    if fi:
        print(f"[>15] frame_index min/median/max = {fi[0]} / {fi[len(fi)//2]} / {fi[-1]}")

    track_n = Counter(p[4] for p in plates)
    print("\n[all] top track theo so anh:", track_n.most_common(10))

    # Moi track: ti le anh >15 tren tong anh cua track (xem co track nao bi lech)
    per_track_gt15 = Counter()
    per_track_all = Counter()
    for p in plates:
        per_track_all[p[4]] += 1
        if p[1] == ">15":
            per_track_gt15[p[4]] += 1
    ratios = []
    for t in per_track_all:
        if per_track_all[t] >= 3:
            ratios.append((per_track_gt15[t] / per_track_all[t], t, per_track_all[t]))
    ratios.sort(reverse=True)
    print("\n[track co >=3 anh] ti le >15 (track):")
    for rr, t, n in ratios[:12]:
        print(f"  track {t}: {rr:.0%} ({per_track_gt15[t]}/{n})")
    if ratios:
        import statistics
        print("  trung binh ti le >15/track (>=3 anh): %.0f%%"
              % (100 * statistics.mean(r[0] for r in ratios)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
