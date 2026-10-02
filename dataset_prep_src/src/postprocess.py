"""
postprocess.py
==============
Module 6 — Hậu xử lý luật biển số Việt Nam + voting theo nhóm.

Bao gồm:
    * Chuẩn hoá chuỗi đọc được (uppercase, bỏ ký tự ngoài charset 0-9/A-Z).
    * Sửa lỗi ký tự dễ nhầm theo **vị trí kỳ vọng** (O<->0, I<->1, B<->8,
      S<->5...). Đây là cách giảm lỗi OCR hiệu quả mà không cần train lại model.
    * Kiểm tra định dạng biển trắng dân sự VN: ``[2 số][1-2 chữ][3-6 số]``.
    * Lọc confidence thấp + lọc sai định dạng **TRƯỚC** khi voting, để không làm
      nhiễu kết quả "xuất hiện nhiều nhất".
    * Bỏ phiếu cho chuỗi chuẩn hoá xuất hiện nhiều nhất (hoà thì ưu tiên
      confidence trung bình cao hơn).
    * Đánh dấu "độ tin cậy thấp" khi số lần đọc hợp lệ < ``min_readings``
      (tương ứng luật "track quá ngắn" của spec 3.6).

Với ảnh tĩnh, "nhóm voting" chính là 1 ảnh. Khi mở rộng sang video, dùng
:func:`group_readings_by_track` để nhóm theo ``track_id`` rồi gọi
:class:`PostProcessor` cho từng nhóm.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import DEFAULT_CHARSET, PostprocessConfig, filter_charset


# --------------------------------------------------------------------------- #
# Bảng ký tự dễ nhầm
# --------------------------------------------------------------------------- #
#: Chữ cái bị OCR nhầm thành chữ số ở vị trí lẽ ra phải là số.
#: Chỉ giữ các cặp nhầm lẫn phổ biến và ít rủi ro (spec 3.6: O<->0, I<->1,
#: B<->8, S<->5) cộng thêm ``L -> 1`` (rất hay gặp vì ``l`` viết thường).
#: Cố tình KHÔNG map các cặp mơ hồ như ``A -> 4``, ``D -> 0``, ``Q -> 0`` vì
#: chúng dễ biến một chuỗi rác thành chuỗi "trông hợp lệ".
LETTER_TO_DIGIT: Dict[str, str] = {
    "O": "0",
    "I": "1",
    "L": "1",
    "B": "8",
    "S": "5",
}

#: Chữ số bị OCR nhầm thành chữ cái ở vị trí lẽ ra phải là chữ.
DIGIT_TO_LETTER: Dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}

#: Cấu trúc biển trắng dân sự đã bỏ dấu phân cách:
#:   [2 số mã tỉnh] [1-2 chữ series] [3-6 số đăng ký]
PLATE_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]{3,6}$")

#: Giới hạn độ dài hợp lệ của chuỗi biển số đã chuẩn hoá.
MIN_PLATE_LENGTH = 6
MAX_PLATE_LENGTH = 10


# --------------------------------------------------------------------------- #
# Chuẩn hoá & kiểm tra định dạng
# --------------------------------------------------------------------------- #
def build_position_template(length: int, ambiguous_series_index: bool = True) -> Optional[List[str]]:
    """Sinh "khuôn" ký tự kỳ vọng cho chuỗi độ dài ``length``.

    Returns:
        Danh sách nhãn ``'D'`` (bắt buộc số), ``'L'`` (bắt buộc chữ),
        ``'A'`` (chấp nhận cả hai) hoặc ``None`` nếu chuỗi quá ngắn.
    """
    if length < MIN_PLATE_LENGTH:
        return None
    template = ["D", "D", "L", "A" if ambiguous_series_index else "D"]
    while len(template) < length:
        template.append("D")
    return template[:length]


def _apply_position_template(text: str, ambiguous_series_index: bool = True) -> str:
    """Áp bảng sửa lỗi ký tự theo vị trí kỳ vọng."""
    template = build_position_template(len(text), ambiguous_series_index)
    if template is None:
        return text
    chars = list(text)
    for index, char in enumerate(chars):
        expected = template[index]
        if expected == "D" and char.isalpha():
            chars[index] = LETTER_TO_DIGIT.get(char, char)
        elif expected == "L" and char.isdigit():
            chars[index] = DIGIT_TO_LETTER.get(char, char)
    return "".join(chars)


#: Các ký tự được coi là dấu phân cách trên biển số (dùng để suy cấu trúc).
SEPARATOR_CHARS = frozenset({"-", ".", " ", "_", "/", ":"})


def _is_separator(char: str) -> bool:
    """True nếu ``char`` là dấu phân cách (bao gồm mọi loại khoảng trắng)."""
    return char in SEPARATOR_CHARS or char.isspace()


def _infer_ambiguous_series_index(raw: str, charset: str = DEFAULT_CHARSET) -> Optional[bool]:
    """Suy ra series có 1 hay 2 chữ cái dựa vào VỊ TRÍ dấu phân cách của chuỗi thô.

    Biển VN thường viết ``<mã tỉnh><series>-<số>`` (ô tô) hoặc ``<mã tỉnh>-<series><số>``
    (xe máy 2 dòng). Đếm số ký tự thuộc ``charset`` trước dấu phân cách đầu tiên nằm
    ngay sau mã tỉnh:

    * 3 ký tự (``51F-...``)  -> series **1 chữ cái** -> vị trí ngay sau dấu phân cách
      bắt buộc là SỐ (``ambiguous_series_index=False``).
    * 4 ký tự (``51LD-...``) -> series **2 chữ cái** -> cho phép vị trí đó là chữ cái
      (``ambiguous_series_index=True``).

    Dấu phân cách nằm ngay sau mã tỉnh (kiểu ``29-B1 12345``) bị bỏ qua, chờ dấu phân
    cách tiếp theo giữa series và khối số.

    Returns:
        ``False`` (series 1 chữ cái), ``True`` (series 2 chữ cái) hoặc ``None`` nếu
        không đủ thông tin để kết luận (VD: chuỗi không có dấu phân cách hữu ích).
    """
    if not raw:
        return None
    upper = str(raw).upper()
    cleaned_count = 0
    for char in upper:
        if _is_separator(char):
            if cleaned_count == 3:
                return False
            if cleaned_count == 4:
                return True
            # Dấu phân cách ngay sau mã tỉnh (kiểu xe máy) -> chưa kết luận được.
            continue
        if char in charset:
            cleaned_count += 1
    return None


def normalize_plate_text(
    text: str,
    charset: str = DEFAULT_CHARSET,
    apply_position_fix: bool = True,
    raw_text: Optional[str] = None,
) -> str:
    """Chuẩn hoá chuỗi OCR thô thành chuỗi biển số "sạch".

    Quy trình:
        1. Xác định cấu trúc từ chuỗi THÔ (``raw_text`` nếu có, ngược lại ``text``)
           bằng vị trí dấu phân cách: series **1 hay 2 chữ cái**.
        2. Viết hoa + chỉ giữ ký tự trong ``charset`` (bỏ ``-``, ``.``, khoảng trắng).
        3. Nếu ``apply_position_fix``: áp bảng sửa lỗi ký tự theo vị trí kỳ vọng.
           Vị trí ngay sau dấu phân cách chỉ được coi là chữ cái khi đã xác định
           series có 2 chữ cái; ngược lại nó là số.

    Việc dùng dấu phân cách để "chốt" độ dài series là mấu chốt: nếu thử phương án
    series 2 chữ cái trước, chuỗi ``"5lF-l2345"`` sẽ bị chuẩn hoá nhầm thành
    ``"51FL2345"`` (vẫn hợp lệ về định dạng nên rất khó phát hiện).

    Args:
        text: Chuỗi OCR (đã hoặc chưa lọc charset).
        charset: Tập ký tự hợp lệ.
        apply_position_fix: Bật sửa ký tự dễ nhầm theo vị trí.
        raw_text: Chuỗi thô còn dấu phân cách tương ứng với ``text`` (nếu có).

    Returns:
        Chuỗi biển số đã chuẩn hoá (chỉ gồm ký tự trong ``charset``).
    """
    # ``raw_text`` (nếu có) giữ dấu phân cách để suy cấu trúc; ngược lại dùng ``text``.
    base = raw_text if raw_text and str(raw_text).strip() else text
    cleaned = filter_charset(base, charset)
    if not cleaned:
        cleaned = filter_charset(text, charset)
    if not cleaned:
        return ""
    if not apply_position_fix or len(cleaned) < MIN_PLATE_LENGTH:
        return cleaned

    # (1) Chốt độ dài series từ vị trí dấu phân cách của chuỗi thô.
    ambiguity = _infer_ambiguous_series_index(base, charset)
    if ambiguity is not None:
        preferred = _apply_position_template(cleaned, ambiguous_series_index=ambiguity)
        if validate_plate(preferred, charset):
            return preferred

    # (2) Không chốt được (hoặc phương án theo dấu phân cách không hợp lệ):
    #     chấm điểm các phương án, ưu tiên phương án hợp lệ với ít sửa đổi nhất.
    variants: List[bool] = [] if ambiguity is None else [bool(ambiguity)]
    for fallback in (True, False):
        if fallback not in variants:
            variants.append(fallback)

    candidates: List[str] = [cleaned]
    for flag in variants:
        candidate = _apply_position_template(cleaned, ambiguous_series_index=flag)
        if candidate not in candidates:
            candidates.append(candidate)

    best_valid: Optional[str] = None
    best_valid_substitutions: Optional[int] = None
    for candidate in candidates:
        if not validate_plate(candidate, charset):
            continue
        substitutions = sum(1 for a, b in zip(cleaned, candidate) if a != b)
        if best_valid_substitutions is None or substitutions < best_valid_substitutions:
            best_valid_substitutions = substitutions
            best_valid = candidate
    if best_valid is not None:
        return best_valid

    # (3) Không phương án nào hợp lệ -> trả chuỗi đã lọc charset (best effort).
    #
    # TRƯỚC ĐÂY: ``return candidates[1]``. Với các chuỗi sai định dạng mà bảng
    # sửa ký tự theo vị trí KHÔNG đổi được ký tự nào (VD: "AAC123", "ABCDEFGH"),
    # mọi phương án thay thế đều trùng ``cleaned`` nên ``candidates`` chỉ có ĐÚNG
    # 1 phần tử -> ``candidates[1]`` ném IndexError và làm sập toàn bộ pipeline
    # ngay tại Module 6, thay vì loại kết quả sai định dạng như spec yêu cầu.
    #
    # Trả ``cleaned`` (chuỗi gốc sau khi lọc charset, chưa thay thế gì) là "best
    # effort" an toàn nhất: :meth:`VotingAggregator.add` vẫn kiểm tra
    # ``validate_plate`` nên kết quả sai định dạng sẽ bị loại TRƯỚC khi voting.
    return cleaned


def validate_plate(text: str, charset: str = DEFAULT_CHARSET) -> bool:
    """Kiểm tra chuỗi đã chuẩn hoá có khớp cấu trúc biển trắng dân sự VN không.

    Cấu trúc chấp nhận (sau khi bỏ dấu phân cách): ``[2 số][1-2 chữ][3-6 số]``,
    tổng độ dài 6-10 ký tự.
    """
    if not text:
        return False
    if not (MIN_PLATE_LENGTH <= len(text) <= MAX_PLATE_LENGTH):
        return False
    if any(char not in charset for char in text):
        return False
    if not (text[0].isdigit() and text[1].isdigit()):
        return False
    return PLATE_PATTERN.match(text) is not None


def format_plate(text: str, is_two_line: bool = False) -> str:
    """Sinh chuỗi hiển thị "best effort" cho biển số.

    Ví dụ:
        * Ô tô 1 dòng: ``51F12345`` -> ``51F-12345``
        * Xe máy 2 dòng: ``29B112345`` -> ``29-B1 12345``

    Nếu chuỗi không hợp lệ thì trả về nguyên chuỗi đã chuẩn hoá.
    """
    if not text:
        return ""
    if not validate_plate(text):
        return text

    province = text[:2]
    index = 2
    while index < len(text) and text[index].isalpha():
        index += 1
    series = text[2:index]
    numbers = text[index:]
    if not numbers:
        return text

    if is_two_line and series.isalpha() and len(numbers) > 4:
        # Biển xe máy: ký tự số đầu tiên của hàng dưới thuộc về series (VD: B1).
        series = series + numbers[0]
        numbers = numbers[1:]
        return f"{province}-{series} {numbers}"
    return f"{province}{series}-{numbers}"


# --------------------------------------------------------------------------- #
# Voting
# --------------------------------------------------------------------------- #
@dataclass
class VoteResult:
    """Kết quả bỏ phiếu cho 1 nhóm (1 ảnh tĩnh, hoặc 1 track_id với video)."""

    text: str = ""
    vote_count: int = 0
    total_readings: int = 0
    accepted_readings: int = 0
    mean_confidence: float = 0.0
    valid: bool = False
    reliable: bool = False
    candidates: List[Tuple[str, int]] = field(default_factory=list)
    rejected: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thành dict (dùng cho JSON output)."""
        return {
            "text": self.text,
            "vote_count": int(self.vote_count),
            "total_readings": int(self.total_readings),
            "accepted_readings": int(self.accepted_readings),
            "mean_confidence": round(float(self.mean_confidence), 4),
            "valid": bool(self.valid),
            "reliable": bool(self.reliable),
            "candidates": [[text, int(count)] for text, count in self.candidates],
            "rejected": dict(self.rejected),
            "notes": list(self.notes),
        }


class VotingAggregator:
    """Gom nhiều lần đọc OCR và bỏ phiếu cho chuỗi chuẩn hoá xuất hiện nhiều nhất.

    Luồng theo spec 3.6:
        1. Bỏ qua lần đọc có confidence < ``min_confidence``.
        2. Chuẩn hoá + (tuỳ chọn) yêu cầu khớp định dạng biển VN.
        3. Đếm phiếu theo chuỗi chuẩn hoá; chọn chuỗi nhiều phiếu nhất, hoà thì
           ưu tiên confidence trung bình cao hơn.
    """

    def __init__(
        self,
        min_confidence: float = 0.35,
        min_readings: int = 2,
        charset: str = DEFAULT_CHARSET,
        require_valid_format: bool = True,
        apply_position_fix: bool = True,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_readings = int(min_readings)
        self.charset = charset
        self.require_valid_format = bool(require_valid_format)
        self.apply_position_fix = bool(apply_position_fix)

        self._counter: Counter = Counter()
        self._confidences: Dict[str, List[float]] = defaultdict(list)
        self._rejected: Counter = Counter()
        self.total = 0

    # ------------------------------------------------------------------ #
    def add(
        self,
        text: Optional[str],
        confidence: Optional[float],
        raw_text: Optional[str] = None,
    ) -> bool:
        """Thêm 1 lần đọc. Trả về True nếu lần đọc này được chấp nhận vào voting.

        Args:
            text: Chuỗi OCR (đã hoặc chưa lọc charset).
            confidence: Độ tin cậy của lần đọc.
            raw_text: Chuỗi thô còn dấu phân cách tương ứng (nếu có); được dùng để
                suy ra series 1 hay 2 chữ cái ở :func:`normalize_plate_text`.
        """
        self.total += 1

        if text is None or not str(text).strip():
            self._rejected["empty"] += 1
            return False

        numeric_confidence = float(confidence) if confidence is not None else 0.0
        if numeric_confidence < self.min_confidence:
            self._rejected["low_confidence"] += 1
            return False

        normalized = normalize_plate_text(
            str(text), self.charset, self.apply_position_fix, raw_text=raw_text
        )
        if not normalized:
            self._rejected["empty_after_normalize"] += 1
            return False

        if self.require_valid_format and not validate_plate(normalized, self.charset):
            self._rejected["invalid_format"] += 1
            return False

        self._counter[normalized] += 1
        self._confidences[normalized].append(numeric_confidence)
        return True

    def add_many(self, readings: Iterable[Any]) -> int:
        """Thêm nhiều lần đọc. Chấp nhận cả đối tượng có ``.text``/``.confidence``
        lẫn tuple/list ``(text, confidence[, raw_text])`` hay dict
        ``{"text": ..., "confidence": ..., "raw_text": ...}``.

        Returns:
            Số lần đọc được chấp nhận.
        """
        accepted = 0
        for reading in readings:
            text: Optional[str] = None
            confidence: Optional[float] = None
            raw_text: Optional[str] = None

            if isinstance(reading, dict):
                text = reading.get("text")
                confidence = reading.get("confidence")
                raw_text = reading.get("raw_text")
            elif hasattr(reading, "text"):
                text = getattr(reading, "text")
                confidence = getattr(reading, "confidence", None)
                raw_text = getattr(reading, "raw_text", None)
            elif isinstance(reading, (tuple, list)) and reading:
                text = reading[0]
                if len(reading) > 1:
                    confidence = reading[1]
                if len(reading) > 2:
                    raw_text = reading[2]

            if self.add(text, confidence, raw_text=raw_text):
                accepted += 1
        return accepted

    def result(self) -> VoteResult:
        """Tổng hợp phiếu bầu và trả về :class:`VoteResult`."""
        result = VoteResult(total_readings=self.total, rejected=dict(self._rejected))
        if not self._counter:
            result.notes.append("Không có lần đọc nào vượt được bộ lọc (confidence/định dạng).")
            return result

        def sort_key(item: Tuple[str, int]) -> Tuple[int, float]:
            text, count = item
            scores = self._confidences.get(text) or [0.0]
            return (int(count), float(np.mean(scores)))

        ranked = sorted(self._counter.items(), key=sort_key, reverse=True)
        best_text, best_count = ranked[0]

        result.text = best_text
        result.vote_count = int(best_count)
        result.accepted_readings = int(sum(self._counter.values()))
        result.mean_confidence = float(np.mean(self._confidences[best_text]))
        result.candidates = [(text, int(count)) for text, count in ranked[:5]]
        result.valid = validate_plate(best_text, self.charset)
        result.reliable = result.vote_count >= self.min_readings

        if not result.reliable:
            result.notes.append(
                f"Chỉ có {result.vote_count} lần đọc hợp lệ (< {self.min_readings}) — "
                "độ tin cậy thấp."
            )
        return result


# --------------------------------------------------------------------------- #
# Bộ hậu xử lý
# --------------------------------------------------------------------------- #
class PostProcessor:
    """Triển khai Module 6 cho 1 nhóm voting (mặc định: 1 ảnh tĩnh)."""

    def __init__(self, config: Optional[PostprocessConfig] = None) -> None:
        self.config = config or PostprocessConfig()

    def process(self, readings: Iterable[Any], group_id: str = "image") -> VoteResult:
        """Lọc + bỏ phiếu cho danh sách lần đọc của 1 nhóm.

        Args:
            readings: Các lần đọc (đối tượng/tuple/dict có ``text`` + ``confidence``).
            group_id: Nhãn nhóm (với ảnh tĩnh là tên ảnh; với video là ``track_id``).

        Returns:
            :class:`VoteResult`.
        """
        aggregator = VotingAggregator(
            min_confidence=self.config.min_confidence,
            min_readings=self.config.min_readings,
            charset=self.config.charset,
            require_valid_format=self.config.require_valid_format,
            apply_position_fix=self.config.apply_position_fix,
        )
        aggregator.add_many(readings)
        result = aggregator.result()
        result.notes.append(f"Nhóm voting: {group_id}")
        return result

    def format(self, text: str, is_two_line: bool = False) -> str:
        """Sinh chuỗi hiển thị từ chuỗi đã chuẩn hoá."""
        return format_plate(text, is_two_line=is_two_line)


# --------------------------------------------------------------------------- #
# Tiện ích mở rộng sang video
# --------------------------------------------------------------------------- #
def group_readings_by_track(readings: Iterable[Any]) -> Dict[Any, List[Any]]:
    """Nhóm các lần đọc theo ``track_id`` (dùng khi mở rộng pipeline sang video).

    Mỗi phần tử trong ``readings`` là dict hoặc đối tượng có thuộc tính
    ``track_id`` (và ``text``/``confidence``).
    """
    groups: Dict[Any, List[Any]] = defaultdict(list)
    for reading in readings:
        if isinstance(reading, dict):
            track_id = reading.get("track_id", "unknown")
        else:
            track_id = getattr(reading, "track_id", "unknown")
        groups[track_id].append(reading)
    return dict(groups)


def vote_by_track(
    readings: Iterable[Any],
    config: Optional[PostprocessConfig] = None,
) -> Dict[Any, VoteResult]:
    """Bỏ phiếu riêng cho từng ``track_id`` — API tiện dụng cho pipeline video."""
    processor = PostProcessor(config)
    grouped = group_readings_by_track(readings)
    return {
        track_id: processor.process(items, group_id=str(track_id))
        for track_id, items in grouped.items()
    }


def interpolate_linear(values: Sequence[Optional[float]]) -> List[float]:
    """Nội suy tuyến tính các giá trị thiếu (``None``) trong một chuỗi.

    Dùng cho logic "nội suy frame thiếu" của Module 6 khi mở rộng sang video
    (tương tự ``add_missing_data.py`` của repo tham khảo): các frame model bỏ sót
    detect sẽ được lấp bằng nội suy giữa 2 mốc có dữ liệu gần nhất.
    """
    result: List[float] = [float(v) if v is not None else float("nan") for v in values]
    if not result:
        return result

    known = [i for i, v in enumerate(result) if not np.isnan(v)]
    if not known:
        return [0.0] * len(result)
    # Nội suy cho các đoạn giữa 2 mốc đã biết.
    for left, right in zip(known, known[1:]):
        if right - left <= 1:
            continue
        step = (result[right] - result[left]) / float(right - left)
        for offset in range(1, right - left):
            result[left + offset] = result[left] + step * offset
    # Các mốc ở đầu/cuối chưa biết -> kéo giá trị gần nhất.
    first, last = known[0], known[-1]
    for i in range(first):
        result[i] = result[first]
    for i in range(last + 1, len(result)):
        result[i] = result[last]
    return [float(v) for v in result]
