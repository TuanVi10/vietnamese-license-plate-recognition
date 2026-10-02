"""
detector.py
===========
Wrapper tùy chọn cho Module 2 — dò biển số trong 1 ảnh tĩnh bằng YOLO.

Module 2 trong spec cần một model YOLO fine-tune trên dataset biển số VN
(``plate_detector.pt``). Ở đây ta bọc ``ultralytics.YOLO`` để dùng model đó nếu
có. Nếu KHÔNG truyền model, pipeline sẽ coi toàn bộ ảnh là vùng biển số — rất
tiện khi đầu vào đã là ảnh crop sẵn (trường hợp phổ biến của bài toán ảnh tĩnh).

``ultralytics`` là phụ thuộc tùy chọn; chỉ cần cài khi thực sự dùng detector.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Detection:
    """Một bbox biển số/phương tiện được phát hiện."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int = 0
    label: str = "license_plate"

    def to_dict(self) -> dict:
        """Serialize thành dict (dùng cho JSON output)."""
        return {
            "bbox": [int(v) for v in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "class_id": int(self.class_id),
            "label": self.label,
        }


class PlateDetector:
    """Bọc ``ultralytics.YOLO`` để dò biển số trong 1 ảnh."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        device: Optional[str] = None,
        imgsz: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RuntimeError(
                "Cần cài ultralytics để dùng detector: pip install ultralytics"
            ) from exc

        self.model_path = model_path
        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.device = device
        self.imgsz = int(imgsz)

    # ------------------------------------------------------------------ #
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Chạy detect trên 1 ảnh, trả danh sách :class:`Detection` (giảm dần conf)."""
        if image is None or getattr(image, "size", 0) == 0:
            return []

        kwargs: dict = {"conf": self.conf, "verbose": False, "imgsz": self.imgsz}
        if self.device:
            kwargs["device"] = self.device

        results = self.model.predict(source=image, **kwargs)
        names = getattr(self.model, "names", {}) or {}
        detections: List[Detection] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                coordinates = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                if isinstance(names, dict):
                    label = str(names.get(class_id, class_id))
                else:
                    label = str(class_id)
                detections.append(
                    Detection(
                        bbox=(
                            int(round(coordinates[0])),
                            int(round(coordinates[1])),
                            int(round(coordinates[2])),
                            int(round(coordinates[3])),
                        ),
                        confidence=confidence,
                        class_id=class_id,
                        label=label,
                    )
                )

        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------ #
    def best_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Trả bbox tự tin nhất, hoặc ``None`` nếu không phát hiện được gì."""
        detections = self.detect(image)
        return detections[0].bbox if detections else None
