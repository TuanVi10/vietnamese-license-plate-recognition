"""
tracker.py
==========
Module 3 — Tracking (SORT: Kalman filter + Hungarian assignment + IoU).

Module này gán một ``track_id`` ổn định cho mỗi phương tiện qua nhiều frame, để
các module sau (dò biển số + OCR + voting) gom được **nhiều lần đọc của cùng
một xe** rồi bỏ phiếu (đúng tinh thần "voting theo ``track_id``" của spec 3.6).

Thuật toán: **SORT** (Simple Online and Realtime Tracking)

    1. Mỗi track giữ một bộ lọc Kalman (mô hình vận tốc không đổi) trên vector
       trạng thái ``[u, v, s, r, u', v', s']`` — tâm ``(u, v)``, diện tích ``s``,
       tỉ lệ ``r = w/h`` và vận tốc tương ứng.
    2. ``predict`` ước lượng bbox hiện tại của mọi track.
    3. Ghép detection với track bằng **IoU** + **Hungarian** (thuật toán gán tối ưu).
    4. Track không được cập nhật quá ``max_age`` frame thì bị xoá; track đủ
       ``min_hits`` lần khớp liên tiếp mới được coi là "đã xác nhận".

Toàn bộ chỉ dùng ``numpy`` + ``math`` (KHÔNG cần ``scipy``/``filterpy``/
``motpy``), nên Module 3 chạy được ngay trong môi trường tối thiểu.

Ví dụ::

    from src.tracker import SortTracker

    tracker = SortTracker(TrackingConfig(max_age=30, min_hits=2))
    for frame_index, detections in enumerate(stream):
        tracks = tracker.update(detections, frame_index=frame_index)
        for track in tracks:
            print(track.track_id, track.bbox)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import TrackingConfig


# --------------------------------------------------------------------------- #
# Hình học: IoU
# --------------------------------------------------------------------------- #
def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU (Intersection over Union) của 2 bbox ``(x1, y1, x2, y2)``."""
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in box_b[:4])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _boxes_to_array(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    """Chuẩn hoá danh sách bbox thành mảng ``(n, 4)`` float (bỏ phần tử dư)."""
    rows: List[List[float]] = []
    for box in boxes:
        values = [float(v) for v in list(box)[:4]]
        if len(values) < 4:
            values = (values + [0.0, 0.0, 0.0, 0.0])[:4]
        rows.append(values)
    if not rows:
        return np.empty((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)


def iou_matrix(boxes_a: Sequence[Sequence[float]], boxes_b: Sequence[Sequence[float]]) -> np.ndarray:
    """Ma trận IoU kích thước ``(len(boxes_a), len(boxes_b))``.

    Tính vector hoá bằng numpy (thay cho 2 vòng lặp Python trước đây) để tránh
    nghẽn cổ chai ``O(n_det·n_trk)`` mỗi frame trên cảnh đông xe.
    """
    a = _boxes_to_array(boxes_a)
    b = _boxes_to_array(boxes_b)
    n_a, n_b = a.shape[0], b.shape[0]
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=float)

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    intersection = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, intersection / union, 0.0)
    iou = np.where(intersection <= 0.0, 0.0, iou)
    return np.asarray(iou, dtype=float)


# --------------------------------------------------------------------------- #
# Thuật toán gán tối ưu (Hungarian / Kuhn-Munkres)
# --------------------------------------------------------------------------- #
def _hungarian_square(cost: np.ndarray) -> List[int]:
    """Gán tối ưu cho ma trận VUÔNG, trả ``row_to_col`` (độ dài ``n``).

    Cài đặt theo "thuật toán Hungary" O(n^3) với thế vị ``u``/``v`` (biến thể
    e-maxx). Yêu cầu ``cost`` là ma trận vuông ``n x n`` với chi phí hữu hạn.
    """
    n = int(cost.shape[0])
    if n == 0:
        return []

    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = float(cost[i0 - 1, j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_col = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            row_to_col[p[j] - 1] = j - 1
    return row_to_col


def linear_assignment(cost_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Gán tối ưu cho ma trận chi phí (có thể CHỮ NHẬT).

    Args:
        cost_matrix: Ma trận chi phí (list/np.ndarray 2 chiều). Giá trị càng nhỏ
            càng tốt.

    Returns:
        Mảng ``(k, 2)`` các cặp ``(row, col)`` được gán. Các hàng/cột dư (không
        ghép được) sẽ không xuất hiện.
    """
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("linear_assignment cần ma trận 2 chiều.")
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty((0, 2), dtype=int)

    # Làm sạch NaN/inf -> chi phí rất lớn (coi như không ghép được).
    big = 1e6
    cost = np.where(np.isfinite(cost), cost, big)

    size = max(n_rows, n_cols)
    padded = np.full((size, size), float(big), dtype=float)
    padded[:n_rows, :n_cols] = cost
    # Ma trận phải VUÔNG cho thuật toán Hungary: bù các hàng/cột "giả" bằng chi
    # phí 0 để chúng không tranh chấp với các cặp thật và vùng đệm không còn giữ
    # sentinel ``big``. (Bản cũ dùng điều kiện ``size > n_rows and size > n_cols``
    # nên KHÔNG BAO GIỜ chạy — vì ``size`` luôn bằng 1 trong 2 giá trị — khiến
    # vùng đệm giữ nguyên ``big`` và logic ghép trở nên mong manh.)
    if n_rows < n_cols:
        padded[n_rows:, :] = 0.0
    elif n_cols < n_rows:
        padded[:, n_cols:] = 0.0

    row_to_col = _hungarian_square(padded)

    pairs: List[Tuple[int, int]] = []
    for row in range(n_rows):
        col = row_to_col[row]
        if 0 <= col < n_cols:
            pairs.append((row, col))
    return np.array(pairs, dtype=int).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# Kalman filter (mô hình tuyến tính, kích thước nhỏ)
# --------------------------------------------------------------------------- #
class KalmanFilter:
    """Bộ lọc Kalman tuyến tính tối giản với ma trận chuyển ``F`` / quan sát ``H``."""

    def __init__(self, dim_x: int, dim_z: int) -> None:
        """Khởi tạo với số chiều trạng thái ``dim_x`` và số chiều quan sát ``dim_z``."""
        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)
        self.x = np.zeros((self.dim_x, 1), dtype=float)
        self.F = np.eye(self.dim_x, dtype=float)
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=float)
        self.P = np.eye(self.dim_x, dtype=float)
        self.R = np.eye(self.dim_z, dtype=float)
        self.Q = np.eye(self.dim_x, dtype=float)

    def predict(self) -> np.ndarray:
        """Bước dự đoán: ``x = F x``, ``P = F P F^T + Q``."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z: Sequence[float]) -> np.ndarray:
        """Bước cập nhật với quan sát ``z`` (độ dài ``dim_z``)."""
        measurement = np.asarray(z, dtype=float).reshape(self.dim_z, 1)
        innovation = measurement - self.H @ self.x
        innovation_cov = self.H @ self.P @ self.H.T + self.R
        # ``innovation_cov`` về lý thuyết luôn xác định dương (nhờ ``R``), nhưng
        # bbox suy biến (rộng/cao ~0) hoặc trùng lặp đo có thể khiến nó suy biến
        # -> ``np.linalg.inv`` ném ``LinAlgError`` làm sập cả pipeline video.
        # Dùng ``pinv`` (giả nghịch đảo) làm đường lui an toàn về mặt số học.
        try:
            innovation_cov_inv = np.linalg.inv(innovation_cov)
        except np.linalg.LinAlgError:
            innovation_cov_inv = np.linalg.pinv(innovation_cov)
        gain = self.P @ self.H.T @ innovation_cov_inv
        self.x = self.x + gain @ innovation
        identity = np.eye(self.dim_x, dtype=float)
        self.P = (identity - gain @ self.H) @ self.P
        return self.x


def convert_bbox_to_z(bbox: Sequence[float]) -> np.ndarray:
    """Đổi bbox ``(x1, y1, x2, y2)`` sang vector trạng thái quan sát ``[u, v, s, r]``.

    ``u, v`` là tâm; ``s`` là diện tích; ``r`` là tỉ lệ ``w/h``.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    width = max(x2 - x1, 1e-3)
    height = max(y2 - y1, 1e-3)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    area = width * height
    ratio = width / height
    return np.array([[center_x], [center_y], [area], [ratio]], dtype=float)


def convert_x_to_bbox(x: Sequence[float]) -> np.ndarray:
    """Đổi vector trạng thái Kalman ``[u, v, s, r, ...]`` về bbox ``(x1, y1, x2, y2)``."""
    values = np.asarray(x, dtype=float).reshape(-1)
    area = max(values[2], 1e-6)
    ratio = max(values[3], 1e-6)
    width = math.sqrt(area * ratio)
    height = area / width if width > 1e-9 else 1e-3
    center_x, center_y = values[0], values[1]
    return np.array(
        [center_x - width / 2.0, center_y - height / 2.0, center_x + width / 2.0, center_y + height / 2.0],
        dtype=float,
    )


# --------------------------------------------------------------------------- #
# SORT track
# --------------------------------------------------------------------------- #
class KalmanBoxTracker:
    """Một track SORT: bộ lọc Kalman trên bbox + thống kê vòng đời."""

    def __init__(self, bbox: Sequence[float], track_id: int, config: Optional[TrackingConfig] = None) -> None:
        """Khởi tạo track từ bbox đầu tiên."""
        self.config = config or TrackingConfig()
        self.id = int(track_id)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Ma trận chuyển: thêm vận tốc cho u, v, s (mô hình vận tốc không đổi).
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )

        # Hiệp phương sai nhiễu quan sát / quá trình (theo bản SORT gốc).
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # vận tốc ban đầu rất không chắc chắn
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.confidence = 0.0
        self.class_id = -1
        self.label = "vehicle"

    # ------------------------------------------------------------------ #
    def update(self, bbox: Sequence[float], confidence: Optional[float] = None,
               class_id: Optional[int] = None, label: Optional[str] = None) -> None:
        """Cập nhật track với bbox mới của frame hiện tại."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if confidence is not None:
            self.confidence = float(confidence)
        if class_id is not None:
            self.class_id = int(class_id)
        if label is not None:
            self.label = str(label)
        self.kf.update(convert_bbox_to_z(bbox))

    # ------------------------------------------------------------------ #
    def predict(self) -> np.ndarray:
        """Dự đoán bbox frame kế tiếp và cập nhật bộ đếm vòng đời."""
        # Nếu diện tích dự đoán âm thì triệt tiêu vận tốc diện tích (theo SORT gốc).
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    # ------------------------------------------------------------------ #
    def get_state(self) -> np.ndarray:
        """Bbox hiện tại ước lượng từ bộ lọc Kalman."""
        return convert_x_to_bbox(self.kf.x)


@dataclass
class TrackedObject:
    """Một track đã được xác nhận tại frame hiện tại (kết quả của Module 3)."""

    #: Id track (ổn định qua các frame).
    track_id: int
    #: Bbox ``(x1, y1, x2, y2)`` tại frame hiện tại.
    bbox: Tuple[int, int, int, int]
    #: Confidence của detection mới nhất.
    confidence: float = 0.0
    #: Id lớp (COCO).
    class_id: int = -1
    #: Nhãn lớp.
    label: str = "vehicle"
    #: Tổng số lần track được cập nhật.
    hits: int = 0
    #: Số frame kể từ khi track được tạo.
    age: int = 0
    #: Số frame kể từ lần cập nhật gần nhất.
    time_since_update: int = 0
    #: Trạng thái track. ``SortTracker`` chỉ trả về track ĐÃ XÁC NHẬN
    #: (``hits >= min_hits``) nên giá trị luôn là ``"confirmed"``.
    state: str = "confirmed"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thành dict (dùng cho JSON output)."""
        return {
            "track_id": int(self.track_id),
            "bbox": [int(v) for v in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "class_id": int(self.class_id),
            "label": str(self.label),
            "hits": int(self.hits),
            "age": int(self.age),
            "time_since_update": int(self.time_since_update),
            "state": self.state,
        }


def _parse_detection(detection: Any) -> Optional[Tuple[Tuple[int, int, int, int], float, int, str]]:
    """Chuẩn hoá 1 detection về ``(bbox, confidence, class_id, label)``.

    Chấp nhận: đối tượng có ``.bbox`` (VD: :class:`~src.detect_vehicle.VehicleDetection`),
    dict có ``"bbox"``/``"box"``, hoặc tuple/list 4 số.
    """
    if detection is None:
        return None

    confidence = 1.0
    class_id = -1
    label = "vehicle"
    bbox: Any = None

    if isinstance(detection, dict):
        bbox = detection.get("bbox", detection.get("box"))
        confidence = float(detection.get("confidence", detection.get("conf", 1.0)))
        class_id = int(detection.get("class_id", detection.get("cls", -1)))
        label = str(detection.get("label", "vehicle"))
    elif isinstance(detection, (list, tuple, np.ndarray)):
        sequence = list(detection)
        if len(sequence) < 4:
            return None
        bbox = sequence[:4]
    else:
        bbox = getattr(detection, "bbox", None)
        if bbox is None:
            return None
        confidence = float(getattr(detection, "confidence", getattr(detection, "conf", 1.0)))
        class_id = int(getattr(detection, "class_id", -1))
        label = str(getattr(detection, "label", "vehicle"))

    if bbox is None:
        return None
    try:
        values = [float(v) for v in list(bbox)[:4]]
    except (TypeError, ValueError):
        return None
    if len(values) < 4:
        return None

    box = (
        int(round(min(values[0], values[2]))),
        int(round(min(values[1], values[3]))),
        int(round(max(values[0], values[2]))),
        int(round(max(values[1], values[3]))),
    )
    return box, confidence, class_id, label


class SortTracker:
    """Triển khai Module 3 — SORT tracker (Kalman + Hungarian + IoU)."""

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        """Khởi tạo tracker với cấu hình cho trước."""
        self.config = config or TrackingConfig()
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self._next_id = 1

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Xoá toàn bộ track và đếm frame (dùng lại tracker cho video khác)."""
        self.trackers = []
        self.frame_count = 0
        self._next_id = 1

    # ------------------------------------------------------------------ #
    def _associate(self, iou_scores: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Ghép detection với track theo IoU + Hungarian.

        Returns:
            ``(matched, unmatched_detections, unmatched_trackers)`` với ``matched``
            là danh sách cặp ``(detection_index, tracker_index)``.
        """
        n_det, n_trk = iou_scores.shape
        unmatched_dets = list(range(n_det))
        unmatched_trks = list(range(n_trk))
        matched: List[Tuple[int, int]] = []

        if n_det == 0 or n_trk == 0:
            return matched, unmatched_dets, unmatched_trks

        cost = 1.0 - iou_scores
        pairs = linear_assignment(cost)
        threshold_cost = 1.0 - float(self.config.iou_threshold)

        for row, col in pairs:
            row = int(row)
            col = int(col)
            if cost[row, col] > threshold_cost:
                continue
            matched.append((row, col))
            if row in unmatched_dets:
                unmatched_dets.remove(row)
            if col in unmatched_trks:
                unmatched_trks.remove(col)
        return matched, unmatched_dets, unmatched_trks

    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: Sequence[Any],
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[TrackedObject]:
        """Nạp các detection của 1 frame, trả danh sách track đã xác nhận.

        Args:
            detections: Danh sách detection (đối tượng/dict/tuple, xem
                :func:`_parse_detection`).
            frame_index: Chỉ số frame gốc (chỉ dùng cho log; không bắt buộc).
            timestamp: Thời điểm frame (chỉ dùng cho log; không bắt buộc).

        Returns:
            Danh sách :class:`TrackedObject` của frame hiện tại.
        """
        del frame_index, timestamp  # hiện chưa dùng tới, giữ chữ ký cho tiện mở rộng

        # Chấp nhận ``detections=None`` như danh sách rỗng để tránh ``TypeError``
        # lan ra ngoài (một số nguồn gọi có thể truyền ``None`` thay vì ``[]``).
        if detections is None:
            detections = []
        parsed: List[Tuple[Tuple[int, int, int, int], float, int, str]] = []
        for detection in detections:
            item = _parse_detection(detection)
            if item is not None:
                parsed.append(item)

        # Tracking tắt: mỗi detection là 1 track sống đúng 1 frame.
        if not self.config.enabled:
            outputs: List[TrackedObject] = []
            for box, confidence, class_id, label in parsed:
                outputs.append(
                    TrackedObject(
                        track_id=self._next_id,
                        bbox=box,
                        confidence=confidence,
                        class_id=class_id,
                        label=label,
                        hits=1,
                        age=0,
                        time_since_update=0,
                        state="confirmed",
                    )
                )
                self._next_id += 1
            self.frame_count += 1
            return outputs

        self.frame_count += 1

        # 1) Dự đoán bbox hiện tại của mọi track đang sống.
        predicted: List[Tuple[int, int, int, int]] = []
        for tracker in self.trackers:
            state = tracker.predict()
            predicted.append(
                (
                    int(round(state[0])),
                    int(round(state[1])),
                    int(round(state[2])),
                    int(round(state[3])),
                )
            )

        detection_boxes = [item[0] for item in parsed]
        if detection_boxes and predicted:
            iou_scores = iou_matrix(detection_boxes, predicted)
        else:
            iou_scores = np.zeros((len(detection_boxes), len(predicted)), dtype=float)

        matched, unmatched_dets, _ = self._associate(iou_scores)

        # 2) Cập nhật các track được ghép.
        for det_index, trk_index in matched:
            box, confidence, class_id, label = parsed[det_index]
            self.trackers[trk_index].update(box, confidence=confidence, class_id=class_id, label=label)

        # 3) Tạo track mới cho detection chưa ghép được.
        for det_index in unmatched_dets:
            box, confidence, class_id, label = parsed[det_index]
            tracker = KalmanBoxTracker(box, track_id=self._next_id, config=self.config)
            # Cập nhật NGAY khi tạo để ``hits``/``hit_streak``/``confidence`` phản ánh
            # đúng việc detection đầu tiên đã khớp (``min_hits=1`` -> xác nhận tức thì,
            # ``min_hits>1`` -> phải khớp liên tiếp đủ số lần mới trả về).
            tracker.update(box, confidence=confidence, class_id=class_id, label=label)
            self.trackers.append(tracker)
            self._next_id += 1

        # 4) Gom kết quả các track "vừa được cập nhật" ở frame này.
        outputs = []
        for tracker in self.trackers:
            if tracker.time_since_update != 0:
                continue
            # Chỉ xác nhận khi track đã khớp liên tiếp đủ ``min_hits`` lần. KHÔNG "gia hạn"
            # theo ``frame_count`` — nếu không, mọi track mới trong các frame đầu sẽ bị trả về
            # dù chưa từng được cập nhật, làm mất tác dụng của ``min_hits``.
            confirmed = tracker.hit_streak >= self.config.min_hits
            if not confirmed:
                continue
            state = tracker.get_state()
            outputs.append(
                TrackedObject(
                    track_id=tracker.id,
                    bbox=(
                        int(round(state[0])),
                        int(round(state[1])),
                        int(round(state[2])),
                        int(round(state[3])),
                    ),
                    confidence=tracker.confidence,
                    class_id=tracker.class_id,
                    label=tracker.label,
                    hits=tracker.hits,
                    age=tracker.age,
                    time_since_update=tracker.time_since_update,
                    state="confirmed",
                )
            )

        # 5) Xoá các track "chết" (không cập nhật quá max_age frame).
        self.trackers = [
            tracker for tracker in self.trackers if tracker.time_since_update <= self.config.max_age
        ]
        return outputs

    # ------------------------------------------------------------------ #
    @property
    def active_tracks(self) -> int:
        """Số track đang được tracker giữ (kể cả chưa xác nhận)."""
        return len(self.trackers)


#: Bí danh chung cho người dùng (SORT là thuật toán mặc định của Module 3).
Tracker = SortTracker


def track_detections(
    detection_frames: Sequence[Sequence[Any]],
    config: Optional[TrackingConfig] = None,
) -> List[List[TrackedObject]]:
    """Tiện dụng: chạy tracker qua một chuỗi frame detection.

    Args:
        detection_frames: Danh sách theo frame, mỗi phần tử là danh sách detection.
        config: Cấu hình Module 3.

    Returns:
        Danh sách theo frame, mỗi phần tử là danh sách :class:`TrackedObject`.
    """
    tracker = SortTracker(config)
    return [
        tracker.update(detections, frame_index=index)
        for index, detections in enumerate(detection_frames)
    ]
