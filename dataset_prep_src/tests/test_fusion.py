"""
tests/test_fusion.py
====================
Pytest cho ``src/fusion.py`` — gộp nhiều lần đọc 1 track theo vị trí ký tự.

Chạy bằng::

    python -m pytest tests/test_fusion.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FusionConfig, PostprocessConfig  # noqa: E402
from src.fusion import FusionResult, fuse_readings, load_province_codes  # noqa: E402
from src.postprocess import PostProcessor, validate_plate  # noqa: E402


def rd(text, conf=0.9, frame=0, quality=1.0):
    """Tạo 1 lần đọc (dict) với frame_index + quality_score rõ ràng."""
    return {
        "text": text,
        "raw_text": text,
        "confidence": conf,
        "frame_index": frame,
        "quality_score": quality,
    }


def test_fixes_single_char_error():
    """5 lần đọc cùng biển, 1 lần nhầm 5 thành 6 -> vẫn ra chuỗi đúng."""
    readings = [rd("51F12345", frame=i) for i in range(4)]
    readings.append(rd("51F12346", frame=4))
    assert fuse_readings(readings).text == "51F12345"


def test_majority_length():
    """Các lần đọc khác độ dài -> chọn độ dài đa số."""
    readings = [rd("51F1234", frame=i) for i in range(3)]
    readings += [rd("51F12345", frame=3 + i) for i in range(2)]
    assert fuse_readings(readings).text == "51F1234"


def test_fallback_on_invalid_format():
    """Fusion ghép ra chuỗi sai định dạng -> lùi về vote cả chuỗi cũ."""
    readings = [rd("51F12A4", frame=i) for i in range(3)]
    readings.append(rd("51F1234", frame=3))
    res = fuse_readings(readings)
    assert res.fallback_to_old_vote is True
    assert res.text == "51F1234"  # vote cũ lọc sai định dạng, giữ chuỗi hợp lệ
    assert res.per_char_confidence == []  # fallback -> bỏ confidence từng vị trí
    assert any("sai định dạng" in n for n in res.notes)


def test_tie_broken_by_weight():
    """Hoà số phiếu -> chọn theo tổng trọng số cao hơn."""
    readings = [rd("51F12345", conf=0.3, frame=i) for i in range(2)]
    readings += [rd("51F12346", conf=0.9, frame=2 + i) for i in range(2)]
    assert fuse_readings(readings).text == "51F12346"


def test_fragment_does_not_affect_length_vote():
    """Mảnh vụn biển 2 dòng (độ dài < MIN) không ảnh hưởng vote độ dài."""
    readings = [rd("29B112345", frame=i) for i in range(3)]
    readings += [rd("29B1", frame=3), rd("12345", frame=4), rd("29B1", frame=5)]
    assert fuse_readings(readings).text == "29B112345"


def test_one_crop_many_readings_does_not_win():
    """1 crop có 4 reading trùng KHÔNG thắng 3 crop khác."""
    readings = [rd("51F12345", frame=0) for _ in range(4)]
    readings += [rd("51F12346", frame=1 + i) for i in range(3)]
    assert fuse_readings(readings).text == "51F12346"


def test_two_plates_not_hybrid():
    """Track chứa 2 biển khác nhau -> ra 1 biển thật, không phải chuỗi lai."""
    readings = [rd("51F12345", conf=0.95, frame=i) for i in range(3)]
    readings += [rd("29A67890", conf=0.9, frame=3 + i) for i in range(3)]
    res = fuse_readings(readings)
    assert res.text == "51F12345"
    assert res.reliable is False
    assert any("trộn" in n for n in res.notes)


def test_each_reading_wrong_at_different_position():
    """Mỗi lần đọc sai 1 vị trí khác nhau, vote cũ ra sai/hoà, fusion ra đúng."""
    readings = [
        rd("51F12346", frame=0),
        rd("51F12355", frame=1),
        rd("51F13345", frame=2),
        rd("51F22345", frame=3),
    ]
    assert fuse_readings(readings).text == "51F12345"
    old = PostProcessor().process(readings)
    assert old.text != "51F12345"


def test_never_returns_invalid_format():
    """Fusion không bao giờ trả chuỗi sai định dạng (lùi về vote cũ nếu ghép sai)."""
    cases = [
        [rd("111111", frame=i) for i in range(3)],
        [rd("51F12A4", frame=i) for i in range(3)] + [rd("51F1234", frame=3)],
        [rd("ABCDEF", frame=i) for i in range(3)],
    ]
    for readings in cases:
        res = fuse_readings(readings)
        if res.text:
            assert validate_plate(res.text), f"chuỗi sai định dạng: {res.text!r}"


def test_empty_readings():
    """Không có lần đọc nào hợp lệ -> text rỗng."""
    res = fuse_readings([rd("12", frame=0)])
    assert res.text == ""
    assert res.total_readings == 1


def test_province_tiebreak():
    """Mã tỉnh chỉ dùng để phá hoà (không loại cứng)."""
    readings = [rd("50F1234", frame=i) for i in range(2)]
    readings += [rd("56F1234", frame=2 + i) for i in range(2)]
    # Không có bảng mã tỉnh -> phá hoà bằng ký tự nhỏ hơn -> '0'.
    assert fuse_readings(readings, FusionConfig()).text == "50F1234"
    # Có bảng {"56"} -> ưu tiên '6' để thành mã tỉnh hợp lệ.
    assert fuse_readings(readings, FusionConfig(province_codes=("56",))).text == "56F1234"


def test_load_province_codes():
    """Nạp mã tỉnh từ file, chỉ giữ token 2 chữ số."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("29, 30\n51\n\nAB  abc 123 45\n")
        path = fh.name
    try:
        assert load_province_codes(path) == ("29", "30", "51", "45")
        assert load_province_codes(None) is None
    finally:
        Path(path).unlink()


def test_char_probs_layer_b():
    """Tầng B (chỗ cắm): char_probs thay cho vote khi cùng độ dài."""
    high = [{"5": 0.9, "6": 0.1}, {"1": 0.9}, {"F": 0.9}, {"1": 0.9},
            {"2": 0.9}, {"3": 0.9}, {"4": 0.9}]
    low = [{"5": 0.4, "6": 0.6}, {"1": 0.9}, {"F": 0.9}, {"1": 0.9},
           {"2": 0.9}, {"3": 0.9}, {"4": 0.9}]
    readings = [
        {**rd("51F1234", frame=0), "char_probs": high},
        {**rd("51F1234", frame=1), "char_probs": high},
        {**rd("61F1234", frame=2), "char_probs": low},
    ]
    res = fuse_readings(readings)
    # 2 lần '5' log-prob cao + 1 lần '6' -> '5' thắng ở vị trí 0.
    assert res.text == "51F1234"
    assert any("Tầng B" in n for n in res.notes)


def test_to_dict():
    res = fuse_readings([rd("51F12345", frame=i) for i in range(3)])
    d = res.to_dict()
    assert d["text"] == "51F12345"
    assert d["supporting_frames"] == 3
    assert len(d["per_char_confidence"]) == 8
