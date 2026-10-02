"""
tools/stop_at_epoch.py
======================
Theo dõi 2 job train (baseline + augmented) trên CCPD-pose:

* Dừng mỗi job khi đã train đủ ``--epochs`` (mặc định 50).
* Nếu job bị crash (CUDA OOM ...) trước khi đủ epoch -> tự ``--resume`` lại
  (tối đa ``--max-resume`` lần).
* Khi cả 2 đã dừng -> chạy ``eval_compare.py`` trên val + test để so sánh.

Chạy nền (khuyến nghị)::

    python tools/stop_at_epoch.py --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RUNS = Path(r"S:\M_AGENT\CV\runs\pose_ccpd")
HERE = Path(__file__).resolve().parent

MODES = {
    "baseline": RUNS / "ccpd_pose_baseline",
    "augmented": RUNS / "ccpd_pose_augmented",
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def last_epoch(results_csv: Path) -> int:
    """Epoch lớn nhất đã hoàn thành (dòng cuối results.csv)."""
    if not results_csv.exists():
        return 0
    try:
        with results_csv.open() as fh:
            rows = list(csv.DictReader(fh))
        return int(rows[-1]["epoch"]) if rows else 0
    except Exception:
        return 0


def get_pids(mode: str) -> list[int]:
    """PID của các tiến trình python đang chạy mode này."""
    ps = (
        'Get-CimInstance Win32_Process -Filter "Name = \'python.exe\'" | '
        f'Where-Object {{ $_.CommandLine -match "--mode {mode}" }} | '
        'Select-Object -ExpandProperty ProcessId'
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=90)
        return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception:
        return []


def kill(pids: list[int]) -> None:
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)


def resume(mode: str) -> None:
    py = sys.executable
    ts = int(time.time())
    with (RUNS / f"{mode}_resume_{ts}.out.txt").open("w") as fo, \
         (RUNS / f"{mode}_resume_{ts}.err.txt").open("w") as fe:
        subprocess.Popen([py, str(HERE / "train_pose_ccpd.py"),
                          "--mode", mode, "--resume"],
                         cwd=str(HERE.parent), stdout=fo, stderr=fe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-resume", type=int, default=3)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    state = {m: {"done": False, "resumed": 0} for m in MODES}
    log(f"giam sat, muc tieu {args.epochs} epoch/mode, poll {args.poll}s")

    while True:
        for mode, rundir in MODES.items():
            st = state[mode]
            if st["done"]:
                continue
            ep = last_epoch(rundir / "results.csv")
            pids = get_pids(mode)
            if pids:
                if ep >= args.epochs:
                    kill(pids)
                    st["done"] = True
                    log(f"{mode}: dat epoch {ep} -> DUNG (kill {len(pids)} tien trinh)")
                else:
                    log(f"{mode}: epoch {ep}/{args.epochs} (dang chay)")
            elif ep >= args.epochs:
                st["done"] = True
                log(f"{mode}: da dung (epoch {ep})")
            elif st["resumed"] < args.max_resume:
                st["resumed"] += 1
                resume(mode)
                log(f"{mode}: KHONG con tien trinh (epoch {ep}) -> resume lan {st['resumed']}")
            else:
                st["done"] = True
                log(f"{mode}: het lan resume (epoch {ep}) - CAN KIEM TRA THU CONG!")

        if all(st["done"] for st in state.values()):
            break
        time.sleep(args.poll)

    log("ca 2 job da dung -> danh gia so sanh ...")
    if not args.no_eval:
        for split in ("val", "test"):
            r = subprocess.run([sys.executable, str(HERE.parent / "eval" / "eval_compare.py"), split])
            log(f"eval {split}: rc={r.returncode}")
    log("HOAN TAT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
