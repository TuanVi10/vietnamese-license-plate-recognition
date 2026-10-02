"""
detect_vehicle.py
=================
Module 1 — Phát hiện phương tiện (Vehicle Detection) bằng YOLOv8n.

Module này là mắt xích thứ hai của pipeline video, ngay sau Module 0 (trích &
lọc frame) và trước Module 3 (tracking):

    Module 0 (frame)
        -> Module 1 (vehicle detect)      <- file này
        -> Module 2 (plate detect)
        -> Module 3 (tracking)
        -> Module 4/5/6 (đọc biển số)
        -> Module 7 (output)

Theo spec v2, Module 1 dùng YOLOv8n (``ultralytics``) để tìm bbox phương tiện
trong frame. Bbox phương tiện chính là **vùng tìm kiếm** cho Module 2 (dò biển
số) và là đầu vào cho Module 3 (gán ``track_id``).

``ultralytics`` là phụ thuộc TÙY CHỌN. Khi KHÔNG truyền ``model_path``, lớp
:class:`VehicleDetector` chạy ở chế độ "toàn khung" (coi cả frame là 1 phương
tiện) để toàn bộ pipeline video vẫn chạy được mà chưa cần model — rất tiện khi
thử nghiệm/kiểm thử logic Module 3.

Ví dụ::

    from src.detect_vehicle import VehicleDetector

    detector = VehicleDetector("yolov8n.pt")
    detections = detector.detect(frame)
    for det in detections:
        print(det.label, det.confidence, det.bbox)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import VehicleDetectionConfig


#: Nhóm lớp COCO coi là "phương tiện" (khớp mặc định của
#: :class:`~src.config.VehicleDetectionConfig`). Chỉ gồm phương tiện CƠ GIỚI:
#: car, motorcycle, bus, truck (KHÔNG có ``bicycle``) để ``VehicleDetector()``
#: và ``VehicleDetector.from_config(VehicleDetectionConfig())`` cho kết quả
#: giống nhau.
COCO_VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

#: Nhãn mặc định cho detection ở chế độ "toàn khung" (không có model).
FULL_FRAME_LABEL = "vehicle"


def ultralytics_available() -> bool:
    """True nếu đã cài ``ultralytics`` (cần cho YOLOv8n của Module 1)."""
    try:  # pragma: no cover - phụ thuộc môi trường
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def clip_bbox(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Kẹp bbox ``(x1, y1, x2, y2)`` vào trong biên ảnh, đảm bảo rộng/cao >= 1px.

    Chấp nhận cả bbox hơi "méo" từ module trước (mảng ``(1, 4)``/``(4, 1)``,
    thừa phần tử, hoặc chứa ``NaN``/``inf``) mà KHÔNG ném ``ValueError`` — chỉ
    lấy 4 giá trị đầu, thay giá trị không hữu hạn bằng 0 rồi kẹp vào biên ảnh.
    Nhờ vậy dữ liệu bẩn từ Module 2 bị "làm sạch" thay vì làm sập pipeline.
    """
    try:
        flat = np.asarray(bbox, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        flat = np.zeros(4, dtype=float)
    if flat.size < 4:
        flat = np.concatenate([flat, np.zeros(4 - flat.size, dtype=float)])
    values = np.nan_to_num(flat[:4], nan=0.0, posinf=0.0, neginf=0.0)
    x1 = int(round(float(values[0])))
    y1 = int(round(float(values[1])))
    x2 = int(round(float(values[2])))
    y2 = int(round(float(values[3])))
    x1 = max(0, min(x1, max(0, width - 1)))
    y1 = max(0, min(y1, max(0, height - 1)))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def bbox_area(bbox: Sequence[float]) -> float:
    """Diện tích bbox ``(x1, y1, x2, y2)`` (>= 0)."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class VehicleDetection:
    """Một bbox phương tiện được Module 1 phát hiện trong 1 frame."""

    #: Bbox ``(x1, y1, x2, y2)`` trong hệ toạ độ frame (pixel, int).
    bbox: Tuple[int, int, int, int]
    #: Độ tin cậy của detection (0-1).
    confidence: float
    #: Id lớp (COCO; ``-1`` cho chế độ "toàn khung").
    class_id: int = -1
    #: Nhãn lớp (``"car"``, ``"motorcycle"``, ``"bus"``, ``"truck"``...).
    label: str = FULL_FRAME_LABEL

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thành dict (dùng cho JSON output)."""
        return {
            "bbox": [int(v) for v in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "class_id": int(self.class_id),
            "label": str(self.label),
        }


class VehicleDetector:
    """Bọc ``ultralytics.YOLO`` để dò phương tiện trong 1 frame (Module 1)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None,
        imgsz: int = 640,
        classes: Optional[Sequence[int]] = None,
        max_det: int = 100,
        full_frame_fallback: bool = True,
    ) -> None:
        """Khởi tạo Module 1.

        Args:
            model_path: Đường dẫn model YOLOv8 (``.pt``). ``None`` -> chế độ
                "toàn khung" (không cần ``ultralytics``).
            conf: Ngưỡng confidence tối thiểu.
            iou: Ngưỡng IoU cho NMS của YOLO.
            device: Thiết bị suy luận (``"cpu"`` / ``"cuda"`` / ``"0"``...).
            imgsz: Kích thước ảnh đầu vào cho YOLO.
            classes: Nhóm lớp cần giữ. ``None`` -> dùng mặc định của
                :class:`~src.config.VehicleDetectionConfig` (tức
                :data:`COCO_VEHICLE_CLASSES`). Truyền list/tuple RỖNG sẽ được
                giữ nguyên là rỗng (KHÔNG bị ghi đè bằng mặc định); khi đó
                ``detect`` KHÔNG truyền ``classes`` cho YOLO, nghĩa là KHÔNG lọc
                theo lớp — YOLO sẽ trả về MỌI lớp COCO (cả person/bicycle/sign...).
                Muốn chỉ dò phương tiện, hãy để ``classes=None`` (mặc định).
            max_det: Số detection tối đa mỗi frame.
            full_frame_fallback: Khi ``model_path is None``, trả về 1 bbox phủ
                toàn khung (coi cả frame là 1 xe).
        """
        self.model_path = model_path
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.imgsz = int(imgsz)
        # Chỉ khi ``classes`` KHÁC ``None`` mới ghi đè; khi ``classes is None``
        # dùng đúng mặc định của VehicleDetectionConfig để ``VehicleDetector()``
        # và ``VehicleDetector.from_config(...)`` luôn nhất quán. Lưu ý: list/tuple
        # RỖNG vẫn là giá trị hợp lệ và được giữ nguyên (không ghi đè mặc định).
        self.classes = (
            tuple(int(c) for c in classes)
            if classes is not None
            else tuple(VehicleDetectionConfig().classes)
        )
        self.max_det = int(max_det)
        self.full_frame_fallback = bool(full_frame_fallback)
        self._model: Any = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: VehicleDetectionConfig) -> "VehicleDetector":
        """Tạo :class:`VehicleDetector` từ :class:`~src.config.VehicleDetectionConfig`."""
        return cls(
            model_path=config.model_path,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            imgsz=config.imgsz,
            classes=config.classes,
            max_det=config.max_det,
            full_frame_fallback=config.full_frame_fallback,
        )

    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Tên ngắn của detector (dùng khi log/ghi manifest)."""
        if self.model_path:
            return f"yolov8({os.path.basename(self.model_path)})"
        return "full_frame_fallback"

    @property
    def uses_model(self) -> bool:
        """True nếu detector thực sự dùng model YOLO (không phải chế độ toàn khung)."""
        return bool(self.model_path)

    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> Any:
        """Nạp model YOLO (lazy). Ném ``RuntimeError`` nếu thiếu ``ultralytics``."""
        if self._model is not None:
            return self._model
        if not self.model_path:
            raise RuntimeError(
                "VehicleDetector không có model_path — hãy truyền model YOLOv8n (.pt)."
            )
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RuntimeError(
                "Cần cài ultralytics để dùng Module 1: pip install ultralytics"
            ) from exc
        self._model = YOLO(self.model_path)
        return self._model

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        """Dò phương tiện trong 1 frame, trả danh sách giảm dần theo confidence.

        Args:
            frame: Ảnh BGR của frame.

        Returns:
            Danh sách :class:`VehicleDetection` (rỗng nếu không có gì). Ở chế độ
            "toàn khung" (``model_path is None``), trả về đúng 1 detection phủ
            toàn bộ frame.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        height, width = frame.shape[:2]

        # Chế độ "toàn khung": chưa cần model, coi cả frame là 1 phương tiện.
        if not self.model_path:
            if not self.full_frame_fallback:
                return []
            return [
                VehicleDetection(
                    bbox=(0, 0, int(width), int(height)),
                    confidence=1.0,
                    class_id=-1,
                    label=FULL_FRAME_LABEL,
                )
            ]

        model = self._ensure_model()
        kwargs: Dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        # ``classes == ()`` (người dùng cố ý truyền rỗng) => KHÔNG lọc theo lớp,
        # đúng như docstring của ``__init__`` (rỗng KHÔNG bị ghi đè bằng mặc định).
        # Hệ quả: YOLO trả về MỌI lớp COCO.
        if len(self.classes) > 0:
            kwargs["classes"] = list(self.classes)

        results = model.predict(source=frame, **kwargs)
        names = getattr(model, "names", {}) or {}
        detections: List[VehicleDetection] = []

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
                else:  # pragma: no cover - một số bản ultralytics dùng list
                    label = str(names[class_id]) if class_id < len(names) else str(class_id)
                detections.append(
                    VehicleDetection(
                        bbox=clip_bbox(coordinates, width, height),
                        confidence=confidence,
                        class_id=class_id,
                        label=label,
                    )
                )

        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------ #
    def best(self, frame: np.ndarray) -> Optional[VehicleDetection]:
        """Trả detection tự tin nhất, hoặc ``None`` nếu không phát hiện được gì."""
        detections = self.detect(frame)
        return detections[0] if detections else None


def build_vehicle_detector(config: Optional[VehicleDetectionConfig] = None) -> VehicleDetector:
    """Tạo :class:`VehicleDetector` từ config (mặc định: chế độ toàn khung)."""
    return VehicleDetector.from_config(config or VehicleDetectionConfig())


def detect_vehicles(
    frame: np.ndarray,
    detector: Optional[VehicleDetector] = None,
    config: Optional[VehicleDetectionConfig] = None,
) -> List[VehicleDetection]:
    """Tiện dụng: dò phương tiện trong 1 frame bằng ``detector`` (hoặc config)."""
    active = detector or build_vehicle_detector(config)
    return active.detect(frame)
