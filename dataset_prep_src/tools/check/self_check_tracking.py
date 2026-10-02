"""
tools/self_check_tracking.py
============================
Tự kiểm tra **Module 1** (detect_vehicle), **Module 3** (tracker) và **file glue**
(video_pipeline) — KHÔNG cần ``ultralytics`` / ``paddleocr`` / ``easyocr`` / codec
video, nên chạy được ngay trong môi trường tối thiểu (chỉ ``numpy`` + ``opencv``).

Nội dung kiểm tra:

* Hình học: ``box_iou`` / ``iou_matrix``.
* Gán tối ưu: ``linear_assignment`` (Hungarian) trên ma trận vuông & chữ nhật.
* Module 1: chế độ "toàn khung" khi không có model YOLO.
* Module 3: SORT giữ ``track_id`` khi vật thể di chuyển, tách 2 vật thể, và xoá
  track sau ``max_age`` frame không cập nhật.
* Module 3: ngưỡng xác nhận ``min_hits`` (track nhiễu 1–2 frame KHÔNG bị coi là xác nhận).
* File glue: nối Module 0 (extractor giả) -> 1 -> 3 -> 4/5/6 (reader giả) và
  voting theo ``track_id`` ra kết quả biển số cuối cùng.
* File glue: ``first_frame`` / ``first_timestamp`` là frame ĐẦU TIÊN xe xuất hiện,
  KHÔNG phải frame mà track đủ ``min_hits`` mới được tracker xác nhận.
* File glue: Module 1 ném lỗi (thiếu ``ultralytics`` / model hỏng) KHÔNG làm sập
  pipeline — chỉ ghi cảnh báo rồi fallback về "toàn khung".

Chạy::

    python tools/self_check_tracking.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

# Cho phép chạy trực tiếp từ thư mục gốc project.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (  # noqa: E402
    PipelineConfig,
    PostprocessConfig,
    TrackingConfig,
    VehicleDetectionConfig,
    VideoPipelineConfig,
)
from src.detect_vehicle import VehicleDetector  # noqa: E402
from src.frame_extractor import (  # noqa: E402
    FrameExtractionResult,
    FrameExtractionStats,
    FrameRecord,
)
from src.tracker import SortTracker, box_iou, iou_matrix, linear_assignment  # noqa: E402
from src.video_pipeline import VideoPipeline  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
FAILURES: List[str] = []


def check(name: str, actual, expected) -> None:
    """So sánh ``actual`` với ``expected`` và ghi nhận kết quả."""
    ok = actual == expected
    if not ok:
        FAILURES.append(f"{name}: nhận {actual!r}, mong đợi {expected!r}")
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name}: {actual!r}")


def close(name: str, actual: float, expected: float, tol: float = 1e-6) -> None:
    """So sánh 2 số thực với dung sai ``tol``."""
    ok = abs(float(actual) - float(expected)) <= tol
    if not ok:
        FAILURES.append(f"{name}: nhận {actual!r}, mong đợi {expected!r} (± {tol})")
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name}: {actual!r}")


def _make_records(count: int, width: int = 200, height: int = 120) -> List[FrameRecord]:
    """Sinh ``count`` FrameRecord giả (ảnh đen) để test glue không cần video thật."""
    records: List[FrameRecord] = []
    for index in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        records.append(
            FrameRecord(
                frame_index=index,
                frame=frame,
                timestamp=index / 30.0,
                brightness=100.0,
                blur_score=200.0,
            )
        )
    return records


class StubExtractor:
    """Thay Module 0: trả sẵn danh sách FrameRecord (không cần codec video)."""

    def __init__(self, records: List[FrameRecord]) -> None:
        self._records = records

    def extract(self, video_path: str) -> FrameExtractionResult:
        """Trả kết quả Module 0 giả lập."""
        stats = FrameExtractionStats(
            total_frames=len(self._records),
            scanned_frames=len(self._records),
            kept_frames=len(self._records),
            fps=30.0,
            duration=len(self._records) / 30.0,
        )
        return FrameExtractionResult(
            video_path=video_path,
            frames=list(self._records),
            stats=stats,
            warnings=[],
        )


class StubReading:
    """Thay ``PlateReading`` của Module 4/5/6."""

    def __init__(self, text: str, confidence: float = 0.9, raw_text: Optional[str] = None) -> None:
        self.text = text
        self.confidence = confidence
        self.engine = "stub"
        self.valid = True
        self.readings = [
            {
                "source": "merged",
                "text": text,
                "raw_text": raw_text or text,
                "confidence": confidence,
                "engine": "stub",
            }
        ]


class StubReader:
    """Thay ``PlateReader``: luôn trả cùng 1 chuỗi biển số (không cần OCR)."""

    def __init__(self, text: str = "51F12345", raw_text: Optional[str] = "51F-12345") -> None:
        self.text = text
        self.raw_text = raw_text
        self.calls = 0

    def read(self, frame, bbox=None) -> StubReading:  # noqa: ANN001 - ký tự giống PlateReader
        """Giả lập đọc biển số và đếm số lần được gọi."""
        self.calls += 1
        return StubReading(self.text, raw_text=self.raw_text)


class FailingVehicleDetector:
    """Thay Module 1 khi model LỖI: ``detect`` luôn ném ngoại lệ.

    Dùng để kiểm tra glue bọc lỗi Module 1 đúng cách (không làm sập pipeline),
    mô phỏng các tình huống thực tế: thiếu ``ultralytics``, sai đường dẫn
    ``--vehicle-model``, model hỏng, hoặc suy luận YOLO lỗi.
    """

    def __init__(self, full_frame_fallback: bool = True) -> None:
        self.full_frame_fallback = full_frame_fallback

    @property
    def name(self) -> str:
        """Tên detector (khớp ``VehicleDetector`` để dùng cho stats/log)."""
        return "failing_vehicle_detector"

    def detect(self, frame):  # noqa: ANN001 - chữ ký giống ``VehicleDetector.detect``
        """Luôn ném lỗi để mô phỏng model hỏng / thiếu ultralytics."""
        raise RuntimeError("giả lập lỗi Module 1 (model hỏng / thiếu ultralytics)")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_geometry() -> None:
    """Kiểm tra IoU."""
    print("\n== Hình học: box_iou / iou_matrix ==")
    close("IoU 2 box trùng nhau", box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
    close("IoU 2 box rời nhau", box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
    # Giao 5x10 = 50; hợp 100 + 100 - 50 = 150 -> 1/3.
    close("IoU chồng lấn một nửa", box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1.0 / 3.0)

    scores = iou_matrix([(0, 0, 10, 10), (20, 20, 30, 30)], [(0, 0, 10, 10), (100, 100, 110, 110)])
    check("iou_matrix shape", scores.shape, (2, 2))
    close("iou_matrix[0,0]", scores[0, 0], 1.0)
    close("iou_matrix[1,1]", scores[1, 1], 0.0)


def test_hungarian() -> None:
    """Kiểm tra thuật toán gán tối ưu (Hungarian)."""
    print("\n== Gán tối ưu: linear_assignment ==")
    square = np.array([[1.0, 2.0], [2.0, 1.0]])
    pairs = {tuple(int(v) for v in row) for row in linear_assignment(square)}
    check("ma trận vuông 2x2 -> gán chéo", pairs, {(0, 0), (1, 1)})

    # Ma trận chữ nhật 2x3: tối ưu là (0,0) và (1,1).
    rectangular = np.array([[0.1, 0.9, 1.0], [0.9, 0.2, 0.8]])
    pairs = {tuple(int(v) for v in row) for row in linear_assignment(rectangular)}
    check("ma trận chữ nhật 2x3", pairs, {(0, 0), (1, 1)})

    check("ma trận rỗng -> rỗng", linear_assignment(np.zeros((0, 3))).shape, (0, 2))


def test_vehicle_detector_fallback() -> None:
    """Kiểm tra Module 1 ở chế độ 'toàn khung' (không cần ultralytics)."""
    print("\n== Module 1: VehicleDetector (toàn khung) ==")
    detector = VehicleDetector()
    check("tên detector", detector.name, "full_frame_fallback")
    check("dùng model?", detector.uses_model, False)

    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    check("số detection toàn khung", len(detections), 1)
    check("bbox toàn khung", detections[0].bbox, (0, 0, 200, 120))
    close("confidence toàn khung", detections[0].confidence, 1.0)
    check("không có model -> không dùng ultralytics", detector.detect(frame)[0].class_id, -1)


def test_sort_tracker() -> None:
    """Kiểm tra Module 3: giữ id khi di chuyển, tách vật thể, xoá theo max_age."""
    print("\n== Module 3: SortTracker (SORT) ==")
    config = TrackingConfig(min_hits=1, max_age=2, iou_threshold=0.3)
    tracker = SortTracker(config)

    seen_ids: List[int] = []
    for index in range(6):
        offset = 5 * index
        tracks = tracker.update([(10 + offset, 10, 60 + offset, 60)], frame_index=index)
        if len(tracks) != 1:
            FAILURES.append(f"tracking 1 vật thể: frame {index} có {len(tracks)} track")
        else:
            seen_ids.append(tracks[0].track_id)
    check("một vật thể -> 1 track_id ổn định", len(set(seen_ids)), 1)

    tracker = SortTracker(TrackingConfig(min_hits=1, max_age=5, iou_threshold=0.3))
    first = tracker.update([(0, 0, 50, 50), (100, 100, 150, 150)])
    check("hai vật thể -> 2 track", len(first), 2)
    ids_first = {track.track_id for track in first}
    second = tracker.update([(1, 1, 51, 51), (101, 101, 151, 151)])
    ids_second = {track.track_id for track in second}
    check("id giữ nguyên khi di chuyển nhẹ", ids_second, ids_first)

    tracker = SortTracker(TrackingConfig(min_hits=1, max_age=2, iou_threshold=0.3))
    tracker.update([(0, 0, 50, 50)], frame_index=0)
    tracker.update([(0, 0, 50, 50)], frame_index=1)
    for index in range(2, 5):
        tracker.update([], frame_index=index)
    check("track bị xoá sau max_age frame không cập nhật", tracker.active_tracks, 0)


def test_min_hits_confirmation() -> None:
    """Kiểm tra Module 3: ``min_hits`` xác nhận đúng, KHÔNG bị 'gia hạn' theo frame đầu."""
    print("\n== Module 3: ngưỡng xác nhận min_hits ==")

    # min_hits=3: detection chỉ xuất hiện 1 frame KHÔNG được xác nhận.
    tracker = SortTracker(TrackingConfig(min_hits=3, max_age=10, iou_threshold=0.3))
    check("min_hits=3: frame đầu chưa xác nhận", len(tracker.update([(10, 10, 60, 60)], frame_index=0)), 0)
    for index in range(1, 6):
        check(
            f"min_hits=3: frame {index} (mất detection) vẫn chưa xác nhận",
            len(tracker.update([], frame_index=index)),
            0,
        )

    # min_hits=3: khớp liên tiếp đủ 3 frame mới xác nhận (từ frame thứ 3).
    tracker = SortTracker(TrackingConfig(min_hits=3, max_age=10, iou_threshold=0.3))
    counts = [len(tracker.update([(10, 10, 60, 60)], frame_index=index)) for index in range(5)]
    check("min_hits=3: chỉ xác nhận từ frame thứ 3", counts, [0, 0, 1, 1, 1])

    # min_hits=1: xác nhận ngay ở frame đầu tiên.
    tracker = SortTracker(TrackingConfig(min_hits=1, max_age=10, iou_threshold=0.3))
    check("min_hits=1: xác nhận ngay frame đầu", len(tracker.update([(10, 10, 60, 60)])), 1)


def test_first_frame_lifecycle() -> None:
    """Glue ghi ``first_frame`` là frame ĐẦU TIÊN, KHÔNG phải frame xác nhận.

    Với ``min_hits=3`` và 3 frame liên tiếp có cùng detection, tracker chỉ trả về
    track ở frame thứ 3 (index 2) — nhưng ``VideoPipeline`` phải ghi nhận phương
    tiện xuất hiện từ frame 0. Nếu (SAI) lấy frame track được trả về thì
    ``first_frame`` sẽ là 2, làm hỏng thông tin vòng đời track trong output.
    """
    print("\n== File glue: first_frame = frame đầu tiên (không phải frame xác nhận) ==")
    config = PipelineConfig(
        vehicle_detection=VehicleDetectionConfig(model_path=None, full_frame_fallback=True),
        tracking=TrackingConfig(min_hits=3, max_age=10, iou_threshold=0.3),
        video=VideoPipelineConfig(read_plates=False),
    )
    pipeline = VideoPipeline(config=config, extractor=StubExtractor(_make_records(3)))
    result = pipeline.run("stub.mp4")

    check("min_hits=3 -> đúng 1 track", len(result.tracks), 1)
    check(
        "first_frame = frame đầu tiên (0), KHÔNG phải frame xác nhận (2)",
        result.tracks[0].first_frame,
        0,
    )
    check("last_frame = frame cuối (2)", result.tracks[0].last_frame, 2)
    close("first_timestamp = timestamp frame đầu (0.0)", result.tracks[0].first_timestamp, 0.0)
    close(
        "last_timestamp = timestamp frame cuối (2/30)",
        result.tracks[0].last_timestamp,
        2 / 30.0,
    )


def test_glue_tracking_only() -> None:
    """Kiểm tra glue chạy Module 0 -> 1 -> 3 mà KHÔNG cần OCR."""
    print("\n== File glue: VideoPipeline (chỉ tracking) ==")
    config = PipelineConfig(
        vehicle_detection=VehicleDetectionConfig(model_path=None, full_frame_fallback=True),
        tracking=TrackingConfig(min_hits=1, max_age=5),
        video=VideoPipelineConfig(read_plates=False),
    )
    pipeline = VideoPipeline(config=config, extractor=StubExtractor(_make_records(3)))
    result = pipeline.run("stub.mp4")

    check("có phát hiện track", result.stats["detected_tracks"] >= 1, True)
    check("danh sách track không rỗng", len(result.tracks) >= 1, True)
    check("track không có biển (không OCR)", result.tracks[0].text, "")
    check("không gọi OCR", result.stats["total_readings"], 0)


def test_glue_full_read() -> None:
    """Kiểm tra glue nối tới Module 4/5/6 (reader giả) + voting theo track_id."""
    print("\n== File glue: VideoPipeline (đọc biển số + voting) ==")
    config = PipelineConfig(
        postprocess=PostprocessConfig(min_confidence=0.3, min_readings=1),
        vehicle_detection=VehicleDetectionConfig(model_path=None, full_frame_fallback=True),
        tracking=TrackingConfig(min_hits=1, max_age=5),
        video=VideoPipelineConfig(read_plates=True),
    )
    reader = StubReader("51F12345", raw_text="51F-12345")
    pipeline = VideoPipeline(config=config, extractor=StubExtractor(_make_records(3)), reader=reader)
    result = pipeline.run("stub.mp4")

    check("có track được đọc biển", result.stats["tracks_with_plate"] >= 1, True)
    check("reader được gọi", reader.calls >= 1, True)
    check("biển số sau voting", result.tracks[0].text, "51F12345")
    check("chuỗi hiển thị", result.tracks[0].formatted, "51F-12345")
    check("hợp lệ format", result.tracks[0].valid, True)

    # Serialize phải không lỗi và không chứa ảnh (frame không JSON-serializable).
    payload = result.to_dict()
    check("to_dict có khóa tracks", "tracks" in payload, True)
    check("to_dict có khóa frames", "frames" in payload, True)

    # Hồi quy: ``raw_text`` ở tầng frame PHẢI là chuỗi OCR thô còn dấu phân cách
    # (bug cũ lấy nhầm ``reading.text`` đã chuẩn hóa nên ``51F-12345`` -> ``51F12345``).
    frame_raw_texts = [
        reading.get("raw_text")
        for frame in payload.get("frames", [])
        for reading in frame.get("readings", [])
    ]
    check("raw_text tầng frame giữ dấu phân cách", "51F-12345" in frame_raw_texts, True)


def test_module1_error_fallback() -> None:
    """File glue: Module 1 ném lỗi KHÔNG được làm sập cả pipeline video.

    Mô phỏng lỗi thực tế khi ``--vehicle-model`` hỏng / thiếu ``ultralytics``:
    ``VehicleDetector.detect`` ném ngoại lệ. Glue PHẢI bắt lỗi, ghi cảnh báo và
    (nếu bật ``full_frame_fallback``) rơi về detection "toàn khung" để Module 3 +
    các module sau vẫn chạy — thay vì để ngoại lệ thoát ra ngoài và sập cả video.
    """
    print("\n== File glue: Module 1 ném lỗi KHÔNG làm sập pipeline ==")

    config = PipelineConfig(
        vehicle_detection=VehicleDetectionConfig(model_path="broken_model.pt"),
        tracking=TrackingConfig(min_hits=1, max_age=5, iou_threshold=0.3),
        video=VideoPipelineConfig(read_plates=False),
    )

    # Nhánh 1: bật fallback "toàn khung" -> pipeline chạy tiếp + sinh track.
    pipeline = VideoPipeline(
        config=config,
        vehicle_detector=FailingVehicleDetector(full_frame_fallback=True),
        extractor=StubExtractor(_make_records(3)),
    )
    result = pipeline.run("stub.mp4")
    check("Module 1 lỗi: pipeline vẫn chạy (KHÔNG crash)", len(result.tracks) >= 1, True)
    check(
        "Module 1 lỗi: có ghi cảnh báo",
        any("Lỗi dò phương tiện (Module 1)" in w for w in result.warnings),
        True,
    )

    # Nhánh 2: tắt fallback -> bỏ qua frame, KHÔNG track nào, nhưng vẫn chạy an toàn.
    pipeline_no_fallback = VideoPipeline(
        config=config,
        vehicle_detector=FailingVehicleDetector(full_frame_fallback=False),
        extractor=StubExtractor(_make_records(3)),
    )
    result_no_fallback = pipeline_no_fallback.run("stub.mp4")
    check("Module 1 lỗi + tắt fallback: không track nào", len(result_no_fallback.tracks), 0)
    check(
        "Module 1 lỗi + tắt fallback: vẫn ghi cảnh báo",
        any("Lỗi dò phương tiện (Module 1)" in w for w in result_no_fallback.warnings),
        True,
    )


def main() -> int:
    """Chạy toàn bộ self-check Module 1/3 + glue và trả mã thoát."""
    test_geometry()
    test_hungarian()
    test_vehicle_detector_fallback()
    test_sort_tracker()
    test_min_hits_confirmation()
    test_first_frame_lifecycle()
    test_glue_tracking_only()
    test_glue_full_read()
    test_module1_error_fallback()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"[KẾT QUẢ] {len(FAILURES)} kiểm tra THẤT BẠI:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("[KẾT QUẢ] Tất cả kiểm tra ĐỀU ĐẠT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
