"""
src/corner_regressor.py
========================
Module mới (bước 4): hồi quy 4 góc biển TRÊN CROP (đã biết chắc có biển).
Dùng ``yolo11n-pose`` (4 keypoint = TL, TR, BR, BL).

Kết hợp với gate theo góc (>15°): nếu |angle| vượt ngưỡng thì warpPerspective(4 góc),
ngược lại dùng bbox + affine deskew như cũ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CornerResult:
    """Kết quả hồi quy 4 góc trên 1 crop."""

    corners: np.ndarray  # (4, 2) TL, TR, BR, BL (pixel trong crop)
    conf: float          # trung bình conf 4 keypoint
    angle: float         # góc nghiêng (độ), suy từ cạnh trên TL->TR


class CornerRegressor:
    """Bọc ``ultralytics.YOLO`` (pose 4 keypoint) để đo 4 góc biển trên crop."""

    def __init__(self, model_path: str, conf: float = 0.25,
                 device: Optional[str] = None, imgsz: int = 160) -> None:
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.device = device
        self.imgsz = int(imgsz)

    def predict(self, crop: np.ndarray) -> Optional[CornerResult]:
        """Trả 4 góc của biển trong ``crop`` (BGR), hoặc ``None`` nếu không dò được."""
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        kwargs = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False, "save": False}
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(crop, **kwargs)[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return None
        kxy = result.keypoints.xy.cpu().numpy()
        if len(kxy) == 0:
            return None
        if result.keypoints.conf is not None:
            confs = result.keypoints.conf.cpu().numpy()
        else:
            confs = np.ones((len(kxy), 4), np.float32)
        best = int(np.argmax(confs.mean(axis=1)))
        kp = kxy[best].astype(np.float32)
        conf = float(confs[best].mean())
        tl, tr = kp[0], kp[1]
        angle = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
        return CornerResult(corners=kp, conf=conf, angle=angle)
