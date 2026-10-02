"""
fusion.py
=========
Gộp nhiều lần đọc của 1 track thành 1 biển số cuối theo **vị trí ký tự**.

Khác với ``VotingAggregator`` (vote theo CẢ CHUỖI), module này bỏ phiếu cho từng
VỊ TRÍ ký tự để một ký tự sai lẻ tẻ (vd: 5 đọc thành 6) không làm chia nhỏ phiếu.

Tầng A (mặc định, không cần char_probs):
    1. Chuẩn hoá bằng ``normalize_plate_text``.
    2. Lọc mảnh vụn: độ dài trong ``[MIN_PLATE_LENGTH, MAX_PLATE_LENGTH]`` và
       ``confidence >= min_confidence``.
    3. Vote độ dài chuỗi (đa số).
    4. Gom cụm (chống trộn nhiều biển): cùng độ dài, khác nhau tối đa
       ``max_cluster_distance`` ký tự. Chỉ fuse trong cụm lớn nhất.
    5. Vote từng vị trí, trọng số ``confidence^a * quality^b``; mỗi crop
       (``frame_index``) đóng góp tổng trọng số bằng nhau (chia đều).
    6. Chuỗi ghép phải qua ``validate_plate``; sai định dạng -> lùi về vote cả chuỗi.

Tầng B (chỗ cắm, chưa chạy): nếu lần đọc có ``char_probs`` thì cộng log-prob theo
vị trí thay cho vote (chỉ dùng khi cùng độ dài).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import FusionConfig, PostprocessConfig
from .postprocess import (
    MAX_PLATE_LENGTH,
    MIN_PLATE_LENGTH,
    VotingAggregator,
    normalize_plate_text,
    validate_plate,
)


@dataclass
class FusionResult:
    """Kết quả fusion cho 1 track.

    ``vote_count``/``supporting_frames`` tính theo CROP (``frame_index``), không theo
    số lần đọc — một crop có nhiều reading không được bỏ nhiều phiếu.
    """

    text: str = ""
    per_char_confidence: List[float] = field(default_factory=list)
    supporting_frames: int = 0
    notes: List[str] = field(default_factory=list)
    valid: bool = False
    reliable: bool = False
    fallback_to_old_vote: bool = False
    vote_count: int = 0
    total_readings: int = 0
    accepted_readings: int = 0
    mean_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thành dict (dùng cho JSON output)."""
        return {
            "text": self.text,
            "per_char_confidence": [round(float(c), 4) for c in self.per_char_confidence],
            "supporting_frames": int(self.supporting_frames),
            "notes": list(self.notes),
            "valid": bool(self.valid),
            "reliable": bool(self.reliable),
            "fallback_to_old_vote": bool(self.fallback_to_old_vote),
            "vote_count": int(self.vote_count),
            "total_readings": int(self.total_readings),
            "accepted_readings": int(self.accepted_readings),
            "mean_confidence": round(float(self.mean_confidence), 4),
        }


@dataclass
class _Reading:
    """Một lần đọc đã chuẩn hoá + trọng số, sẵn sàng cho fusion."""

    text: str
    confidence: float
    quality: float
    weight: float
    frame_key: Any
    char_probs: Optional[List[Dict[str, float]]] = None


def _unpack(reading: Any) -> Tuple[Any, ...]:
    """Bóc ``(text, raw_text, confidence, quality, frame_index, char_probs)``.

    Chấp nhận dict / đối tượng có thuộc tính / tuple-list. ``quality`` lấy từ
    ``quality_score`` (fallback ``score``, mặc định 1.0).
    """
    text = confidence = raw_text = quality = frame_index = char_probs = None
    if isinstance(reading, dict):
        text = reading.get("text")
        confidence = reading.get("confidence")
        raw_text = reading.get("raw_text")
        quality = reading.get("quality_score", reading.get("score", 1.0))
        frame_index = reading.get("frame_index")
        char_probs = reading.get("char_probs")
    elif hasattr(reading, "text"):
        text = getattr(reading, "text", None)
        confidence = getattr(reading, "confidence", None)
        raw_text = getattr(reading, "raw_text", None)
        quality = getattr(reading, "quality_score", getattr(reading, "score", 1.0))
        frame_index = getattr(reading, "frame_index", None)
        char_probs = getattr(reading, "char_probs", None)
    elif isinstance(reading, (tuple, list)) and reading:
        text = reading[0]
        if len(reading) > 1:
            confidence = reading[1]
        if len(reading) > 2:
            raw_text = reading[2]
        if len(reading) > 3:
            quality = reading[3]
        if len(reading) > 4:
            char_probs = reading[4]
    return text, raw_text, confidence, quality, frame_index, char_probs


def _hamming(a: str, b: str) -> int:
    """Số vị trí khác nhau giữa 2 chuỗi CÙNG độ dài."""
    return sum(1 for x, y in zip(a, b) if x != y)


def load_province_codes(path: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Nạp mã tỉnh từ file văn bản (mỗi dòng 1 mã, hoặc ngăn cách dấu phẩy/khoảng trắng).

    Chỉ giữ token gồm ĐÚNG 2 chữ số. Trả ``None`` nếu ``path`` rỗng/không hợp lệ.
    """
    if not path:
        return None
    codes: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            for token in line.replace(",", " ").split():
                token = token.strip()
                if token.isdigit() and len(token) == 2:
                    codes.append(token)
    return tuple(dict.fromkeys(codes)) or None


# --------------------------------------------------------------------------- #
# Các bước fusion
# --------------------------------------------------------------------------- #
def _extract_readings(
    readings: List[Any],
    fusion_cfg: FusionConfig,
    postprocess_cfg: PostprocessConfig,
) -> List[_Reading]:
    """Chuẩn hoá + lọc mảnh vụn (độ dài + confidence)."""
    out: List[_Reading] = []
    for idx, reading in enumerate(readings):
        text, raw_text, confidence, quality, frame_index, char_probs = _unpack(reading)
        if text is None or not str(text).strip():
            continue
        conf = float(confidence) if confidence is not None else 0.0
        if conf < fusion_cfg.min_confidence:
            continue
        normalized = normalize_plate_text(
            str(text),
            postprocess_cfg.charset,
            postprocess_cfg.apply_position_fix,
            raw_text=raw_text,
        )
        if not normalized:
            continue
        if not (MIN_PLATE_LENGTH <= len(normalized) <= MAX_PLATE_LENGTH):
            continue
        qual = float(quality) if quality is not None else 1.0
        weight = (max(conf, 0.0) ** fusion_cfg.confidence_exponent) * (
            max(qual, 0.0) ** fusion_cfg.quality_exponent
        )
        frame_key: Any = frame_index if frame_index is not None else ("single", idx)
        out.append(
            _Reading(
                text=normalized,
                confidence=conf,
                quality=qual,
                weight=weight,
                frame_key=frame_key,
                char_probs=char_probs,
            )
        )
    return out


def _vote_length(readings: List[_Reading]) -> List[_Reading]:
    """Vote độ dài chuỗi: giữ các lần đọc có độ dài đa số.

    Hoà -> tổng trọng số cao hơn; vẫn hoà -> độ dài dài hơn (nhiều thông tin hơn).
    """
    if not readings:
        return []
    length_counts: Dict[int, int] = defaultdict(int)
    length_weights: Dict[int, float] = defaultdict(float)
    for r in readings:
        length_counts[len(r.text)] += 1
        length_weights[len(r.text)] += r.weight
    best_length = max(
        length_counts.keys(),
        key=lambda L: (length_counts[L], length_weights[L], L),
    )
    return [r for r in readings if len(r.text) == best_length]


def _cluster(readings: List[_Reading], max_distance: int) -> List[List[_Reading]]:
    """Gom lần đọc thành cụm (union-find) theo độ giống.

    Hai lần đọc cùng cụm nếu cùng độ dài và khác nhau tối đa ``max_distance`` ký tự
    (khoảng cách Hamming). Trả cụm sắp theo (kích thước, tổng trọng số) giảm dần.
    """
    n = len(readings)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = readings[i], readings[j]
            if len(ri.text) == len(rj.text) and _hamming(ri.text, rj.text) <= max_distance:
                union(i, j)

    groups: Dict[int, List[_Reading]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(readings[i])
    clusters = list(groups.values())
    clusters.sort(key=lambda c: (len(c), sum(r.weight for r in c)), reverse=True)
    return clusters


def _pick_winner(
    totals: Dict[str, float],
    confs: Dict[str, List[float]],
    position: int,
    other_province_char: Optional[str],
    province_codes: Optional[Sequence[str]],
) -> str:
    """Chọn ký tự thắng tại 1 vị trí.

    Thứ tự phá hoà: tổng trọng số cao hơn -> mã tỉnh hợp lệ (vị trí 1, nếu có bảng)
    -> confidence trung bình cao hơn -> ký tự nhỏ hơn (deterministic).
    """
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return ""
    best_char, best_weight = ranked[0]
    tol = 1e-9
    tied = [c for c, w in ranked if abs(w - best_weight) <= tol]
    if len(tied) == 1:
        return best_char

    # Phá hoà bằng mã tỉnh: chỉ ở vị trí 1 (đã biết chữ số tỉnh đầu tiên).
    if province_codes and position == 1 and other_province_char is not None:
        valid = [c for c in tied if (other_province_char + c) in province_codes]
        if valid:
            tied = valid
    if len(tied) == 1:
        return tied[0]

    def _mean_conf(c: str) -> float:
        vals = confs.get(c)
        return float(np.mean(vals)) if vals else 0.0

    return max(tied, key=lambda c: (_mean_conf(c), -ord(c)))


def _fuse_by_vote(
    cluster: List[_Reading],
    fusion_cfg: FusionConfig,
    postprocess_cfg: PostprocessConfig,
    province_codes: Optional[Sequence[str]],
) -> Tuple[str, List[float]]:
    """Tầng A: vote từng vị trí, mỗi crop đóng góp tổng trọng số bằng nhau."""
    if not cluster:
        return "", []
    crops: Dict[Any, List[_Reading]] = defaultdict(list)
    for r in cluster:
        crops[r.frame_key].append(r)

    # Chuẩn hoá trọng số trong từng crop để mỗi crop đóng góp đúng 1.0 tổng trọng số.
    weighted: List[Tuple[str, float, float]] = []  # (text, weight, confidence)
    for rs in crops.values():
        total = sum(r.weight for r in rs)
        if total <= 0:
            per = 1.0 / len(rs)
            for r in rs:
                weighted.append((r.text, per, r.confidence))
        else:
            for r in rs:
                weighted.append((r.text, r.weight / total, r.confidence))

    length = len(cluster[0].text)
    pos_weights: List[Dict[str, float]] = [defaultdict(float) for _ in range(length)]
    pos_confs: List[Dict[str, List[float]]] = [defaultdict(list) for _ in range(length)]
    for text, w, conf in weighted:
        for i, ch in enumerate(text):
            pos_weights[i][ch] += w
            pos_confs[i][ch].append(conf)

    chars: List[str] = []
    per_char: List[float] = []
    for i in range(length):
        totals = pos_weights[i]
        total_w = sum(totals.values())
        winner = _pick_winner(
            totals,
            pos_confs[i],
            position=i,
            other_province_char=(chars[0] if i == 1 and chars else None),
            province_codes=province_codes,
        )
        chars.append(winner)
        per_char.append((totals.get(winner, 0.0) / total_w) if total_w > 0 else 0.0)
    return "".join(chars), per_char


def _fuse_by_probs(
    cluster: List[_Reading],
    fusion_cfg: FusionConfig,
) -> Tuple[str, List[float]]:
    """Tầng B (chỗ cắm, chưa chạy): cộng log-prob theo vị trí thay cho vote.

    Chỉ gọi khi mọi lần đọc trong cụm đều có ``char_probs`` cùng độ dài với chuỗi.
    """
    length = len(cluster[0].char_probs)
    chars: List[str] = []
    per_char: List[float] = []
    for i in range(length):
        log_scores: Dict[str, float] = defaultdict(float)
        for r in cluster:
            for ch, p in r.char_probs[i].items():
                log_scores[ch] += r.weight * math.log(max(float(p), 1e-12))
        if not log_scores:
            chars.append("")
            per_char.append(0.0)
            continue
        keys = list(log_scores.keys())
        xs = np.array([log_scores[k] for k in keys], dtype=float)
        xs = xs - xs.max()
        exps = np.exp(xs)
        total = float(exps.sum())
        best_idx = int(np.argmax(exps))
        chars.append(keys[best_idx])
        per_char.append(float(exps[best_idx] / total) if total else 0.0)
    return "".join(chars), per_char


def _supporting_frames(cluster: List[_Reading], final_text: str) -> int:
    """Số crop khác nhau có lần đọc đại diện (trọng số cao nhất) khớp chuỗi cuối."""
    crops: Dict[Any, List[_Reading]] = defaultdict(list)
    for r in cluster:
        crops[r.frame_key].append(r)
    support = 0
    for rs in crops.values():
        rep = max(rs, key=lambda r: (r.weight,))
        if rep.text == final_text:
            support += 1
    return support


def _mean_supporting_confidence(cluster: List[_Reading], final_text: str) -> float:
    """Confidence trung bình của các lần đọc khớp đúng chuỗi cuối."""
    confs = [r.confidence for r in cluster if r.text == final_text]
    if confs:
        return float(np.mean(confs))
    return float(np.mean([r.confidence for r in cluster])) if cluster else 0.0


def _fallback_to_old_vote(readings: List[Any], postprocess_cfg: PostprocessConfig) -> Any:
    """Vote cả chuỗi kiểu cũ (``VotingAggregator``) để lùi về khi fusion sai định dạng."""
    agg = VotingAggregator(
        min_confidence=postprocess_cfg.min_confidence,
        min_readings=postprocess_cfg.min_readings,
        charset=postprocess_cfg.charset,
        require_valid_format=postprocess_cfg.require_valid_format,
        apply_position_fix=postprocess_cfg.apply_position_fix,
    )
    agg.add_many(readings)
    return agg.result()


# --------------------------------------------------------------------------- #
# API công khai
# --------------------------------------------------------------------------- #
def fuse_readings(
    readings: Iterable[Any],
    fusion_cfg: Optional[FusionConfig] = None,
    postprocess_cfg: Optional[PostprocessConfig] = None,
) -> FusionResult:
    """Gộp nhiều lần đọc của 1 track thành 1 biển số cuối theo vị trí ký tự.

    Args:
        readings: Danh sách lần đọc. Mỗi phần tử là dict/đối tượng có ``text``,
            ``raw_text``, ``confidence``, tuỳ chọn ``quality_score`` (hoặc ``score``),
            ``frame_index`` và ``char_probs``.
        fusion_cfg: Cấu hình fusion (:class:`~config.FusionConfig`).
        postprocess_cfg: Cấu hình hậu xử lý (:class:`~config.PostprocessConfig`).

    Returns:
        :class:`FusionResult`.
    """
    fusion_cfg = fusion_cfg or FusionConfig()
    postprocess_cfg = postprocess_cfg or PostprocessConfig()
    province_codes = fusion_cfg.province_codes

    result = FusionResult()
    raw_readings = list(readings) if readings is not None else []
    result.total_readings = len(raw_readings)

    filtered = _extract_readings(raw_readings, fusion_cfg, postprocess_cfg)
    if not filtered:
        result.notes.append("Không có lần đọc nào qua bộ lọc (độ dài/confidence).")
        return result

    majority = _vote_length(filtered)
    if not majority:
        result.notes.append("Không xác định được độ dài đa số.")
        return result

    clusters = _cluster(majority, fusion_cfg.max_cluster_distance)
    if not clusters:
        result.notes.append("Không tạo được cụm lần đọc nào.")
        return result

    largest = clusters[0]
    result.accepted_readings = len(largest)

    suspected_mixing = False
    if len(clusters) >= 2:
        total = sum(len(c) for c in clusters)
        share = (len(clusters[1]) / total) if total else 0.0
        if share >= fusion_cfg.cluster_min_share:
            suspected_mixing = True
            result.notes.append(
                f"Nghi trộn nhiều biển: cụm lớn thứ 2 chiếm {share:.0%} "
                f"({len(clusters[1])}/{total}) lần đọc."
            )

    # Tầng B nếu đủ điều kiện; ngược lại Tầng A.
    has_probs = all(r.char_probs is not None for r in largest)
    if has_probs:
        prob_lengths = {len(r.char_probs) for r in largest}
        if len(prob_lengths) == 1 and next(iter(prob_lengths)) == len(largest[0].text):
            fused_text, per_char = _fuse_by_probs(largest, fusion_cfg)
            result.notes.append("Dùng Tầng B (log-prob theo vị trí từ char_probs).")
        else:
            fused_text, per_char = _fuse_by_vote(
                largest, fusion_cfg, postprocess_cfg, province_codes
            )
    else:
        fused_text, per_char = _fuse_by_vote(
            largest, fusion_cfg, postprocess_cfg, province_codes
        )

    result.per_char_confidence = per_char

    # Chuỗi ghép sai định dạng -> lùi về vote cả chuỗi cũ (không trả chuỗi sai định dạng).
    if not fused_text or not validate_plate(fused_text, postprocess_cfg.charset):
        old = _fallback_to_old_vote(raw_readings, postprocess_cfg)
        result.text = old.text
        result.valid = bool(old.valid)
        result.reliable = bool(old.reliable)
        result.vote_count = int(old.vote_count)
        result.accepted_readings = int(old.accepted_readings)
        result.mean_confidence = float(old.mean_confidence)
        result.fallback_to_old_vote = True
        # Chuỗi ghép bị loại -> confidence từng vị trí không còn ý nghĩa.
        result.per_char_confidence = []
        result.notes.append(
            f"Fusion ghép ra chuỗi '{fused_text}' sai định dạng — lùi về vote cả chuỗi cũ."
        )
        return result

    result.text = fused_text
    result.valid = True
    result.supporting_frames = _supporting_frames(largest, fused_text)
    result.vote_count = result.supporting_frames
    result.mean_confidence = _mean_supporting_confidence(largest, fused_text)
    result.reliable = result.supporting_frames >= postprocess_cfg.min_readings
    if suspected_mixing:
        result.reliable = False
    return result
