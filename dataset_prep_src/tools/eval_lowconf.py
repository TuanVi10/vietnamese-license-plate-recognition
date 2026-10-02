"""
tools/eval_lowconf.py
=====================
Chạy đánh giá pose ở NHIỀU ngưỡng confidence thấp (mặc định 0.001, 0.05) để khảo
sát cận dưới của recall khi model gặp domain gap (CCPD -> data thật 401-1701).

Kết quả: runs/pose_ccpd/eval_<tag>_<split>_c<conf>.json
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RUNS = Path(r"S:\M_AGENT\CV\runs\pose_ccpd")
TOOL = Path(r"S:\M_AGENT\CV\dataset_prep_src\tools\eval_pose_bands.py")
CONFS = ["0.001", "0.05"]
SPLITS = ["val", "test"]


def main() -> int:
    for tag, d in (("baseline", "ccpd_pose_baseline"),
                   ("augmented", "ccpd_pose_augmented")):
        model = RUNS / d / "weights" / "best.pt"
        for split in SPLITS:
            for conf in CONFS:
                out = RUNS / f"eval_{tag}_{split}_c{conf.replace('.', '')}.json"
                cmd = [sys.executable, str(TOOL), "--model", str(model),
                       "--tag", tag, "--split", split, "--conf", conf,
                       "--out", str(out)]
                print(">", " ".join(cmd), flush=True)
                subprocess.run(cmd)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
