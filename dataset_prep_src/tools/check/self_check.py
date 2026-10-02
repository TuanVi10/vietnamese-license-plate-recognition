"""
tools/self_check.py
===================
Kiểm tra nhanh Module 6 (hậu xử lý luật biển VN + voting) và Module 4 (các hàm
thuần OpenCV) mà KHÔNG cần cài OCR engine. Hữu ích để xác nhận logic đúng trước
khi cài PaddleOCR/EasyOCR nặng.

Module 5 cũng có một test riêng cho ``PaddleOCREngine._parse`` để đảm bảo đọc
được kết quả của CẢ PaddleOCR 2.x lẫn 3.x — cũng không cần cài ``paddleocr``.

Module 0 (trích & lọc frame từ video) có test ``test_frame_extractor`` — sinh
video giả lập trong thư mục tạm rồi kiểm chứng bộ lọc tối/mờ và chỉ số frame gốc.

Chạy::

    python tools/self_check.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.postprocess import (  # noqa: E402
    PostProcessor,
    VotingAggregator,
    format_plate,
    interpolate_linear,
    normalize_plate_text,
    validate_plate,
)
from src.config import PostprocessConfig  # noqa: E402


class _FakeModernPaddleResult:
    """Giả lập *result object* của PaddleOCR 3.x (chỉ có thuộc tính ``.json``).

    Dùng để test ``PaddleOCREngine._parse`` mà KHÔNG cần cài ``paddleocr``.
    """

    def __init__(self, texts, scores, polys) -> None:
        self._payload = {
            "res": {"rec_texts": texts, "rec_scores": scores, "rec_polys": polys}
        }

    @property
    def json(self):
        """Trả về dict giống ``OCRResult.json`` của PaddleOCR 3.x."""
        return self._payload


class _FakeMappingPaddleResult:
    """Giả lập *result object* 3.x hỗ trợ ``obj["res"]``."""

    def __init__(self, texts, scores, polys) -> None:
        self._res = {"rec_texts": texts, "rec_scores": scores, "rec_polys": polys}

    def __getitem__(self, key):
        if key == "res":
            return self._res
        raise KeyError(key)


FAILURES = []


def check(description: str, actual, expected) -> None:
    """So sánh ``actual`` với ``expected`` và ghi nhận nếu sai."""
    status = "OK  " if actual == expected else "FAIL"
    if actual != expected:
        FAILURES.append(description)
    print(f"[{status}] {description}: {actual!r} (mong đợi {expected!r})")


def test_normalize() -> None:
    """Kiểm tra chuẩn hoá + sửa ký tự dễ nhầm theo vị trí."""
    print("\n== normalize_plate_text ==")
    check("bỏ dấu gạch ngang", normalize_plate_text("51F-12345"), "51F12345")
    check("bỏ dấu chấm", normalize_plate_text("51F-123.45"), "51F12345")
    check("O -> 0 ở vị trí số", normalize_plate_text("5OF-12345"), "50F12345")
    check("I -> 1 ở vị trí số", normalize_plate_text("5IF-12345"), "51F12345")
    check("l -> 1 (chữ L thường)", normalize_plate_text("5lF-l2345"), "51F12345")
    check("S -> 5 ở vị trí số", normalize_plate_text("S1F-12345"), "51F12345")
    check("giữ series 2 chữ cái", normalize_plate_text("51LD-1234"), "51LD1234")
    check("series 1 chữ + 1 số (xe máy)", normalize_plate_text("29-B1 12345"), "29B112345")
    check("chuỗi rỗng", normalize_plate_text(""), "")
    check("bỏ ký tự tiếng Việt", normalize_plate_text("51F-12ấ"), "51F12")

    # --- Dấu phân cách quyết định series 1 hay 2 chữ cái (bug Module 6 đã sửa) ---
    check(
        "raw_text giữ dấu phân cách (text đã lọc)",
        normalize_plate_text("5LFL2345", raw_text="5lF-l2345"),
        "51F12345",
    )
    check(
        "series 2 chữ giữ nguyên khi có dấu phân cách",
        normalize_plate_text("51LD1234", raw_text="51-LD 1234"),
        "51LD1234",
    )
    check(
        "không có dấu phân cách: chọn phương án hợp lệ ít sửa nhất",
        normalize_plate_text("S1F12345"),
        "51F12345",
    )
    check(
        "dấu phân cách ngay sau mã tỉnh (xe máy)",
        normalize_plate_text("29B112345", raw_text="29-B1 12345"),
        "29B112345",
    )

    # --- Bug Module 6 (IndexError khi không có phương án thay thế nào) --------
    # Trước đây ``normalize_plate_text`` kết thúc bằng ``return candidates[1]``.
    # Với chuỗi sai định dạng mà bảng sửa ký tự theo vị trí KHÔNG đổi được gì,
    # mọi phương án đều trùng ``cleaned`` nên ``candidates`` chỉ có ĐÚNG 1 phần
    # tử -> ``candidates[1]`` ném IndexError. Sau khi sửa, hàm phải trả về an
    # toàn chuỗi đã lọc charset để Module 6 loại vì sai định dạng.
    check(
        "không crash với chuỗi sai định dạng không sửa được",
        normalize_plate_text("ABC123"),
        "ABC123",
    )
    check(
        "1 phương án (mọi thay thế trùng cleaned) không crash",
        normalize_plate_text("AAC123"),
        "AAC123",
    )
    check(
        "chuỗi dài sai định dạng không crash",
        normalize_plate_text("ABCDEFGH"),
        "ABCDEFGH",
    )


def test_validate() -> None:
    """Kiểm tra luật định dạng biển trắng dân sự VN."""
    print("\n== validate_plate ==")
    check("biển ô tô hợp lệ", validate_plate("51F12345"), True)
    check("biển 4 số hợp lệ", validate_plate("29H1234"), True)
    check("biển xe máy 2 dòng", validate_plate("29B112345"), True)
    check("quá ngắn", validate_plate("51F12"), False)
    check("thiếu chữ series", validate_plate("51123456"), False)
    check("có ký tự lạ", validate_plate("51F-1234"), False)
    check("quá dài", validate_plate("51F1234567"), False)


def test_format() -> None:
    """Kiểm tra sinh chuỗi hiển thị."""
    print("\n== format_plate ==")
    check("ô tô 1 dòng", format_plate("51F12345"), "51F-12345")
    check("xe máy 2 dòng", format_plate("29B112345", is_two_line=True), "29-B1 12345")
    check("chuỗi không hợp lệ", format_plate("ABC"), "ABC")


def test_voting() -> None:
    """Kiểm tra lọc + bỏ phiếu của Module 6."""
    print("\n== VotingAggregator ==")
    aggregator = VotingAggregator(min_confidence=0.4, min_readings=2)
    readings = [
        ("51F-12345", 0.95),
        ("51F12345", 0.90),
        ("5IF-l2345", 0.88),   # l -> 1, I -> 1  => 51F12345
        ("51F-1234S", 0.35),   # dưới ngưỡng confidence -> loại
        ("ABC123", 0.99),      # sai định dạng -> loại
        ("", 0.99),            # rỗng -> loại
    ]
    accepted = aggregator.add_many(readings)
    result = aggregator.result()
    check("số lần đọc được chấp nhận", accepted, 3)
    check("chuỗi thắng", result.text, "51F12345")
    check("số phiếu", result.vote_count, 3)
    check("hợp lệ", result.valid, True)
    check("đáng tin cậy", result.reliable, True)
    check("loại do confidence thấp", result.rejected.get("low_confidence"), 1)
    check("loại do sai định dạng", result.rejected.get("invalid_format"), 1)
    check("loại do rỗng", result.rejected.get("empty"), 1)

    # --- Bug Module 6 (IndexError) -----------------------------------------
    # Chuỗi sai định dạng nhưng KHÔNG sửa được bằng bảng ký tự dễ nhầm (VD
    # "ABC123") trước đây khiến ``normalize_plate_text`` ném IndexError vì
    # ``candidates[1]`` khi ``candidates`` chỉ có 1 phần tử. Sau khi sửa phải
    # bị loại AN TOÀN (trả False), không vào voting và KHÔNG crash.
    print("\n== chuỗi sai định dạng không sửa được (bug IndexError đã sửa) ==")
    check(
        "chuỗi sai định dạng không sửa được bị loại, không crash",
        aggregator.add("ABC123", 0.99),
        False,
    )

    invalid_aggregator = VotingAggregator(min_confidence=0.4, min_readings=1)
    check(
        "thêm chuỗi sai định dạng không sửa được trả False",
        invalid_aggregator.add("ABC123", 0.99),
        False,
    )
    check(
        "chuỗi sai định dạng không sửa được không vào voting",
        invalid_aggregator.result().accepted_readings,
        0,
    )
    check(
        "lý do loại là invalid_format",
        invalid_aggregator.result().rejected.get("invalid_format"),
        1,
    )

    print("\n== track quá ngắn -> không đáng tin cậy ==")
    short = VotingAggregator(min_confidence=0.4, min_readings=3)
    short.add_many([("29H-1234", 0.9), ("29H-1234", 0.8)])
    short_result = short.result()
    check("chuỗi", short_result.text, "29H1234")
    check("không đáng tin cậy", short_result.reliable, False)

    print("\n== raw_text đi kèm khi voting (bug series 1 vs 2 chữ) ==")
    raw_aggregator = VotingAggregator(min_confidence=0.4, min_readings=1)
    raw_aggregator.add_many(
        [
            {"text": "5LFL2345", "raw_text": "5lF-l2345", "confidence": 0.9},
            {"text": "51F12345", "raw_text": "51F-12345", "confidence": 0.8},
        ]
    )
    raw_result = raw_aggregator.result()
    check("dùng raw_text để suy series 1 chữ", raw_result.text, "51F12345")
    check("gộp đúng 2 phiếu", raw_result.vote_count, 2)


def test_postprocessor() -> None:
    """Kiểm tra PostProcessor qua API cấp cao."""
    print("\n== PostProcessor ==")
    processor = PostProcessor(PostprocessConfig(min_confidence=0.3, min_readings=1))
    result = processor.process(
        [{"text": "29-B1 12345", "confidence": 0.92}, {"text": "29B112345", "confidence": 0.85}],
        group_id="sample",
    )
    check("chuỗi thắng", result.text, "29B112345")
    check("hiển thị", processor.format(result.text, is_two_line=True), "29-B1 12345")


def test_interpolate() -> None:
    """Kiểm tra nội suy frame thiếu (dùng khi mở rộng sang video)."""
    print("\n== interpolate_linear ==")
    check("nội suy giữa", interpolate_linear([0.0, None, 10.0]), [0.0, 5.0, 10.0])
    check("kéo đầu/cuối", interpolate_linear([None, 4.0, None]), [4.0, 4.0, 4.0])


def test_frame_extractor() -> None:
    """Kiểm tra Module 0: trích frame, lọc tối/mờ, giới hạn số lượng, chỉ số gốc."""
    print("\n== Module 0 (Frame Extraction & Filtering) ==")
    try:
        from src.config import FrameExtractionConfig
        from src.frame_extractor import extract_frames, iter_valid_frames
        from tools.sample.make_sample_video import build_synthetic_video
    except ImportError as exc:
        # CHỈ bỏ qua khi THIẾU thư viện (VD: chưa cài cv2/numpy). Lỗi import do code
        # Module 0 hỏng (SyntaxError, AttributeError...) vẫn phải nổi lên để KHÔNG
        # âm thầm che giấu lỗi thật dưới nhãn [SKIP].
        print(f"[SKIP] Không import được Module 0 (thiếu thư viện?): {exc}")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "test_video.mp4")
        try:
            build_synthetic_video(
                video_path,
                total_frames=60,
                fps=30.0,
                dark_frames=(10, 11, 40),
                blur_frames=(20, 21, 50),
            )
        except RuntimeError as exc:  # noqa: BLE001 - môi trường thiếu codec mp4v
            print(f"[SKIP] Không ghi được video test (thiếu codec mp4v?): {exc}")
            return

        result = extract_frames(video_path, FrameExtractionConfig(frame_interval=5))
        check("tổng frame đọc được", result.stats.total_frames, 60)
        check("có giữ được frame", result.stats.kept_frames > 0, True)
        check("bỏ được frame tối", result.stats.skipped_dark > 0, True)
        check("bỏ được frame mờ", result.stats.skipped_blurry > 0, True)
        check("fps đọc đúng", round(result.stats.fps), 30)

        check(
            "mọi frame đều là ảnh hợp lệ",
            all(f.frame is not None and f.frame.size > 0 for f in result.frames),
            True,
        )
        check("mọi frame đều đủ sáng", all(f.brightness >= 50.0 for f in result.frames), True)
        check("mọi frame đều đủ nét", all(f.blur_score >= 100.0 for f in result.frames), True)
        check(
            "frame_index là bội của frame_interval",
            all(f.frame_index % 5 == 0 for f in result.frames),
            True,
        )
        check(
            "frame_index tăng dần",
            [f.frame_index for f in result.frames] == sorted(f.frame_index for f in result.frames),
            True,
        )
        check(
            "timestamp = frame_index / fps",
            all(abs(f.timestamp - f.frame_index / 30.0) < 1e-6 for f in result.frames),
            True,
        )
        check(
            "các frame tối/mờ bị loại",
            {f.frame_index for f in result.frames} & {10, 11, 40, 20, 21, 50},
            set(),
        )
        check(
            "max_frames cắt đúng",
            len(extract_frames(video_path, FrameExtractionConfig(frame_interval=5, max_frames=3)).frames),
            3,
        )
        check(
            "tắt bộ lọc chất lượng thì giữ nhiều hơn",
            len(extract_frames(video_path, FrameExtractionConfig(frame_interval=5, enable_quality_filter=False)).frames)
            > result.stats.kept_frames,
            True,
        )

        missing_path = os.path.join(tmp_dir, "khong_ton_tai.mp4")
        try:
            extract_frames(missing_path, FrameExtractionConfig())
            exc = None
        except Exception as error:  # noqa: BLE001 - bắt mọi loại lỗi để kiểm tra kiểu
            exc = error
        check("file không tồn tại -> FileNotFoundError", isinstance(exc, FileNotFoundError), True)

        check(
            "iter_valid_frames khớp extract_frames",
            len(list(iter_valid_frames(video_path, FrameExtractionConfig(frame_interval=5)))),
            result.stats.kept_frames,
        )


def test_preprocess_helpers() -> None:
    """Kiểm tra Module 4 có chạy được với ảnh giả lập hay không."""
    print("\n== Module 4 (preprocess) ==")
    try:
        from src.sample_data import add_degradation, make_single_line_plate, make_two_line_plate
        from src.preprocess_plate import PlatePreprocessor
    except Exception as exc:  # noqa: BLE001 - thiếu opencv thì bỏ qua
        print(f"[SKIP] Không import được Module 4 ({exc}).")
        return

    preprocessor = PlatePreprocessor()

    single = add_degradation(make_single_line_plate("51F", "12345"), angle=6.0, seed=0)
    single_result = preprocessor.preprocess(single)
    check("biển 1 dòng không bị coi là 2 dòng", single_result.is_two_line, False)
    check("không quá nhỏ", single_result.too_small, False)
    check("ảnh ra đúng kích thước chuẩn", single_result.image.shape[:2], (48, 320))
    check("đã bù góc nghiêng", abs(single_result.skew_angle) > 0.5, True)

    two_line = make_two_line_plate("29-B1", "12345")
    two_result = preprocessor.preprocess(two_line)
    check("biển 2 dòng được tách", two_result.is_two_line, True)
    check("tách được 2 dòng", len(two_result.lines), 2)

    tiny = make_single_line_plate("51F", "12345")[10:22, 10:45]
    tiny_result = preprocessor.preprocess(tiny)
    check("ảnh nhỏ bị đánh dấu too_small", tiny_result.too_small, True)
    check("không sinh biến thể OCR khi too_small", len(list(preprocessor.iter_variants(tiny_result))), 0)


def test_plate_reader_joined_text() -> None:
    """Voting phải dùng chuỗi GHÉP khi OCR tách ảnh biển thành nhiều text-box.

    Tái hiện lỗi cũ trong ``PlateReader.read``: chỉ từng OCR item rời được đưa
    vào voting. Với biển 2 dòng, engine trả 2 item ``"29B1"`` và ``"12345"``; cả
    hai đều ngắn hơn ``MIN_PLATE_LENGTH`` nên bị Module 6 loại vì sai định dạng,
    khiến ``ocr_result.text`` (``"29B112345"``) — vốn hoàn toàn hợp lệ — bị bỏ
    qua và kết quả trả về RỖNG. Sau khi sửa, chuỗi ghép cũng được đưa vào voting.
    """
    print("\n== PlateReader: chuỗi ghép nhiều OCR item (biển 2 dòng) ==")
    try:
        from src.ocr import BaseOCREngine, OCRItem
        from src.plate_reader import PlateReader
        from src.sample_data import make_two_line_plate
    except Exception as exc:  # noqa: BLE001 - thiếu opencv/numpy thì bỏ qua
        print(f"[SKIP] Không import được PlateReader ({exc}).")
        return

    class FakeTwoItemOCR(BaseOCREngine):
        """Engine giả lập: luôn trả 2 text-box tách rời như biển 2 dòng."""

        name = "fake_two_item"

        def _run(self, image):  # noqa: ANN001 - engine giả cho test
            return [
                OCRItem(text="29-B1", confidence=0.9),
                OCRItem(text="12345", confidence=0.9),
            ]

    plate = make_two_line_plate("29-B1", "12345")
    reader = PlateReader(ocr_engine=FakeTwoItemOCR())
    result = reader.read(plate)

    check("ghép đúng chuỗi biển 2 dòng", result.text, "29B112345")
    check("hợp lệ theo luật biển VN", result.valid, True)
    check("đủ tin cậy (>= min_readings)", result.reliable, True)
    check(
        "có lần đọc chuỗi ghép trong danh sách",
        any(str(reading.get("text")) == "29B112345" for reading in result.readings),
        True,
    )
    # Mỗi biến thể tiền xử lý sinh 3 lần đọc: "29B1", "12345", "29B112345".
    # Chỉ chuỗi ghép vượt được bộ lọc định dạng -> accepted == total / 3.
    check(
        "chỉ chuỗi ghép được chấp nhận vào voting",
        result.accepted_readings,
        result.total_readings // 3,
    )
    check("có nhiều hơn 1 biến thể OCR", result.total_readings > 3, True)


def test_paddle_parse_compat() -> None:
    """``PaddleOCREngine._parse`` phải đọc được cả kết quả PaddleOCR 2.x lẫn 3.x.

    Đây là bug chính của Module 5: bản cũ chỉ đọc dict qua ``entry.get``, còn
    PaddleOCR 3.x trả về *result object* (``.json``/``["res"]``) nên parse ra
    rỗng dù khởi tạo thành công. Test không cần cài ``paddleocr`` vì ``_parse``
    là ``staticmethod``.
    """
    print("\n== PaddleOCREngine._parse (2.x & 3.x) ==")
    try:
        from src.ocr import PaddleOCREngine
    except Exception as exc:  # noqa: BLE001 - thiếu numpy thì bỏ qua
        print(f"[SKIP] Không import được src.ocr ({exc}).")
        return

    polys = [
        [[0, 0], [10, 0], [10, 5], [0, 5]],
        [[0, 6], [12, 6], [12, 11], [0, 11]],
    ]
    modern = _FakeModernPaddleResult(["29-B1", "12345"], [0.97, 0.93], polys)
    modern_items = PaddleOCREngine._parse([modern])
    check("3.x (.json) đọc đúng 2 text", [i.text for i in modern_items], ["29-B1", "12345"])
    check(
        "3.x (.json) đọc đúng confidence",
        [round(i.confidence, 2) for i in modern_items],
        [0.97, 0.93],
    )
    check("3.x (.json) có box", all(i.box is not None for i in modern_items), True)

    mapping = _FakeMappingPaddleResult(
        ["51F12345"], [0.9], [[[1, 2], [3, 2], [3, 4], [1, 4]]]
    )
    mapping_items = PaddleOCREngine._parse([mapping])
    check("3.x (['res']) đọc được text", [i.text for i in mapping_items], ["51F12345"])

    flat_items = PaddleOCREngine._parse(
        [{"rec_texts": ["29H-1234"], "rec_scores": [0.8]}]
    )
    check("3.x (dict phẳng) đọc được text", [i.text for i in flat_items], ["29H-1234"])
    check("3.x (dict phẳng) đọc đúng score", flat_items[0].confidence, 0.8)

    no_score = PaddleOCREngine._parse([{"rec_texts": ["51F12345"]}])
    check("3.x thiếu rec_scores -> mặc định 1.0", no_score[0].confidence, 1.0)

    legacy = [[[[0, 0], [10, 0], [10, 5], [0, 5]], ("51F-12345", 0.95)]]
    legacy_items = PaddleOCREngine._parse(legacy)
    check("2.x (nested list) đọc được text", [i.text for i in legacy_items], ["51F-12345"])
    check("2.x (nested list) đọc đúng confidence", legacy_items[0].confidence, 0.95)

    check("raw None -> rỗng, không crash", PaddleOCREngine._parse(None), [])


def main() -> int:
    """Chạy toàn bộ self-check và trả mã thoát."""
    test_normalize()
    test_validate()
    test_format()
    test_voting()
    test_postprocessor()
    test_interpolate()
    test_frame_extractor()
    test_preprocess_helpers()
    test_plate_reader_joined_text()
    test_paddle_parse_compat()

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
