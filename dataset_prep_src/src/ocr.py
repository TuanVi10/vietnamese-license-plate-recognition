"""
ocr.py
======
Module 5 — OCR biển số.

Engine chính: **PaddleOCR (PP-OCRv6)** — theo spec 3.5, phù hợp với biển 2 dòng
của xe máy VN và hỗ trợ tốt ký tự Latin (0-9, A-Z).

Lưu ý triển khai quan trọng (spec 3.5):
    * Trên máy CPU phải set ``enable_mkldnn=False`` khi khởi tạo PaddleOCR để
      tránh crash trên một số cấu hình.
    * Giới hạn charset chỉ còn ``0-9`` và ``A-Z`` để giảm nhiễu kết quả.
    * Nếu setup PaddleOCR gặp khó, dùng **EasyOCR** làm fallback để có pipeline
      baseline chạy được trước, không để việc setup OCR chặn tiến độ.

Wrapper PaddleOCR dưới đây hỗ trợ **cả API 2.x lẫn 3.x** (xem
:class:`PaddleOCREngine`), tránh lỗi "cài 3.x nhưng code chỉ biết 2.x".
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import DEFAULT_CHARSET, OCRConfig, filter_charset


@dataclass
class OCRItem:
    """Một đơn vị text do OCR trả về (có thể là 1 dòng hoặc 1 ký tự)."""

    text: str
    confidence: float
    box: Optional[np.ndarray] = None
    #: Chuỗi THÔ do engine trả về, GIỮ NGUYÊN dấu phân cách (``-``, ``.``,
    #: khoảng trắng) trước khi lọc charset. Cần thiết cho Module 6 vì vị trí
    #: dấu phân cách giúp xác định series có 1 hay 2 chữ cái (spec 3.6).
    raw_text: str = ""
    #: (Tầng B, tuỳ chọn) xác suất từng ký tự theo vị trí ``[{"char": prob, ...}, ...]``.
    #: Mặc định ``None`` = OCR chưa cung cấp. Chỉ dùng khi cùng độ dài với ``text``.
    char_probs: Optional[List[Dict[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thành dict (dùng cho JSON output)."""
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "confidence": round(float(self.confidence), 4),
            "char_probs": self.char_probs,
            "box": None
            if self.box is None
            else np.asarray(self.box, dtype=float).round(2).tolist(),
        }


@dataclass
class OCRResult:
    """Kết quả OCR cho 1 ảnh."""

    text: str
    confidence: float
    items: List[OCRItem] = field(default_factory=list)
    engine: str = ""
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thành dict (dùng cho JSON output)."""
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "engine": self.engine,
            "raw_text": self.raw_text,
            "items": [item.to_dict() for item in self.items],
        }


# --------------------------------------------------------------------------- #
# Hàm hỗ trợ
# --------------------------------------------------------------------------- #
def _box_array(box: Any) -> Optional[np.ndarray]:
    """Chuẩn hoá box về ``np.ndarray`` shape ``(N, 2)``; trả ``None`` nếu không hợp lệ."""
    if box is None:
        return None
    array = np.asarray(box, dtype=float)
    if array.ndim == 1:
        if array.size == 4:
            return array.reshape(2, 2)
        if array.size >= 4 and array.size % 2 == 0:
            return array.reshape(-1, 2)
        return None
    if array.ndim >= 2:
        return array.reshape(-1, 2)
    return None


def sort_reading_order(items: Sequence[OCRItem]) -> List[OCRItem]:
    """Sắp xếp các item OCR theo thứ tự đọc (trên -> dưới, trái -> phải).

    Dùng tâm box; các item gần cùng hàng (chênh lệch y nhỏ hơn ~0.6 chiều cao
    trung vị) được gộp thành 1 hàng rồi sắp theo trục x.
    """
    boxed = [item for item in items if item.box is not None]
    others = [item for item in items if item.box is None]
    if len(boxed) < 2:
        return list(items)

    centers: List[Tuple[float, float]] = []
    heights: List[float] = []
    for item in boxed:
        box = np.asarray(item.box, dtype=float).reshape(-1, 2)
        centers.append((float(box[:, 0].mean()), float(box[:, 1].mean())))
        heights.append(max(1.0, float(box[:, 1].max() - box[:, 1].min())))

    tolerance = 0.6 * float(np.median(heights))
    order = list(np.argsort([center[1] for center in centers]))

    rows: List[Dict[str, Any]] = []
    for index in order:
        placed = False
        for row in rows:
            if abs(centers[index][1] - row["y"]) <= tolerance:
                row["items"].append(index)
                row["y"] = float(np.mean([centers[i][1] for i in row["items"]]))
                placed = True
                break
        if not placed:
            rows.append({"y": centers[index][1], "items": [index]})

    rows.sort(key=lambda row: row["y"])
    ordered: List[OCRItem] = []
    for row in rows:
        for index in sorted(row["items"], key=lambda i: centers[i][0]):
            ordered.append(boxed[index])
    ordered.extend(others)
    return ordered


def aggregate_confidence(items: Sequence[OCRItem]) -> float:
    """Confidence trung bình có trọng số theo độ dài text của từng item."""
    if not items:
        return 0.0
    total = 0.0
    weight = 0.0
    for item in items:
        w = max(1, len(item.text.strip()))
        total += float(item.confidence) * w
        weight += w
    return total / weight if weight else 0.0


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class BaseOCREngine(ABC):
    """Giao diện chung cho mọi OCR engine."""

    name = "base"

    def __init__(self, charset: str = DEFAULT_CHARSET, drop_empty: bool = True) -> None:
        self.charset = charset
        self.drop_empty = drop_empty

    @abstractmethod
    def _run(self, image: np.ndarray) -> List[OCRItem]:
        """Chạy OCR thô, trả danh sách :class:`OCRItem` (chưa lọc charset)."""
        raise NotImplementedError

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Chạy OCR + lọc charset + sắp thứ tự đọc.

        Args:
            image: Ảnh biển số đã tiền xử lý (BGR hoặc grayscale).

        Returns:
            :class:`OCRResult` với ``text`` chỉ gồm ký tự trong charset.
        """
        if image is None or getattr(image, "size", 0) == 0:
            return OCRResult(text="", confidence=0.0, items=[], engine=self.name, raw_text="")

        raw_items = self._run(image)
        cleaned: List[OCRItem] = []
        for item in raw_items:
            # Lưu ``raw_text`` TRƯỚC khi lọc charset để Module 6 còn thấy được
            # dấu phân cách (``-``, ``.``, khoảng trắng) nhằm suy ra độ dài series.
            item_raw_text = item.raw_text or item.text
            text = filter_charset(item_raw_text, self.charset)
            if not text and self.drop_empty:
                continue
            cleaned.append(
                OCRItem(
                    text=text,
                    confidence=float(item.confidence),
                    box=item.box,
                    raw_text=item_raw_text,
                )
            )

        cleaned = sort_reading_order(cleaned)
        text = "".join(item.text for item in cleaned)
        raw_text = "".join((item.raw_text or item.text) for item in raw_items)
        return OCRResult(
            text=text,
            confidence=aggregate_confidence(cleaned),
            items=cleaned,
            engine=self.name,
            raw_text=raw_text,
        )


class PaddleOCREngine(BaseOCREngine):
    """Wrapper PaddleOCR (PP-OCRv6), tương thích cả API 2.x và 3.x.

    Hai dòng API khác nhau ở cả cách khởi tạo lẫn cách gọi:

    * **2.x** — ``PaddleOCR(...).ocr(img)`` trả về list lồng nhau
      ``[[box, (text, score)], ...]``; tham số khởi tạo dùng ``use_gpu``,
      ``use_angle_cls``, ``enable_mkldnn``, ``show_log``.
    * **3.x** — ``PaddleOCR(...).predict(img)`` trả về list các *result object*;
      mỗi object có ``.json`` (dict chứa ``res.rec_texts`` / ``res.rec_scores``
      / ``res.rec_polys``), hỗ trợ ``obj["res"]``; tham số khởi tạo dùng
      ``device``, ``use_textline_orientation``, ``use_doc_orientation_classify``,
      ``use_doc_unwarping`` và KHÔNG nhận ``use_gpu`` / ``use_angle_cls`` /
      ``show_log``.

    Wrapper tự dò major version để thử đúng bộ tham số trước, nhưng vẫn thử lần
    lượt các ứng viên còn lại nên vẫn chạy được cả khi không đọc được version
    (bản build tùy biến) — khắc phục lỗi "chỉ tương thích 2.x".
    """

    name = "paddleocr"

    #: Khoá chỉ có ở API 2.x — dùng để nhận diện bản 3.x qua signature.
    _LEGACY_ONLY_KWARGS = ("use_gpu", "use_angle_cls", "show_log")
    #: Khoá đặc trưng của API 3.x.
    _MODERN_ONLY_KWARGS = ("device", "use_textline_orientation", "use_doc_unwarping")

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        enable_mkldnn: bool = False,
        use_angle_cls: bool = True,
        charset: str = DEFAULT_CHARSET,
        drop_empty: bool = True,
    ) -> None:
        super().__init__(charset=charset, drop_empty=drop_empty)
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RuntimeError(
                "Không import được paddleocr. Cài đặt bằng: "
                "pip install paddleocr paddlepaddle"
            ) from exc

        self.lang = lang
        self.use_gpu = bool(use_gpu)
        self.enable_mkldnn = enable_mkldnn
        self.use_angle_cls = bool(use_angle_cls)
        #: Major version dò được (2, 3, ...) hoặc ``None`` nếu bất định.
        self._api_major = self._detect_api_major(PaddleOCR)
        #: Lỗi gần nhất khi gọi OCR (rỗng nếu thành công).
        self.last_error = ""
        self._ocr = self._build(PaddleOCR)

    # ------------------------------------------------------------------ #
    # Dò version & signature
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_int(value: Any) -> Optional[int]:
        """Lấy số nguyên đầu tiên trong chuỗi version (VD: ``"3.0.0"`` -> 3)."""
        match = re.search(r"(\d+)", str(value or ""))
        return int(match.group(1)) if match else None

    @classmethod
    def _detect_api_major(cls, paddle_ocr_cls: Any) -> Optional[int]:
        """Suy ra major version PaddleOCR (2, 3, ...); ``None`` nếu bất định.

        Ưu tiên ``__version__`` của package/class/module; nếu không có thì soi
        signature ``PaddleOCR.__init__`` để đoán. Khi bất định, wrapper vẫn thử
        lần lượt mọi ứng viên tham số nên không bị chặn.
        """
        version_sources: List[Any] = []
        try:
            import paddleocr  # type: ignore

            version_sources.append(getattr(paddleocr, "__version__", None))
        except Exception:  # noqa: BLE001
            pass
        version_sources.append(getattr(paddle_ocr_cls, "__version__", None))
        module = sys.modules.get(getattr(paddle_ocr_cls, "__module__", "") or "")
        if module is not None:
            version_sources.append(getattr(module, "__version__", None))
        for source in version_sources:
            major = cls._first_int(source)
            if major is not None:
                return major

        accepts = cls._accepted_init_kwargs(paddle_ocr_cls)
        if accepts is None:
            return None
        has_modern = any(key in accepts for key in cls._MODERN_ONLY_KWARGS)
        has_legacy = any(key in accepts for key in cls._LEGACY_ONLY_KWARGS)
        if has_modern and not has_legacy:
            return 3
        if has_legacy and not has_modern:
            return 2
        return None

    @staticmethod
    def _accepted_init_kwargs(paddle_ocr_cls: Any) -> Optional[Set[str]]:
        """Tập tên tham số ``PaddleOCR.__init__``; ``None`` nếu không xác định."""
        try:
            signature = inspect.signature(paddle_ocr_cls.__init__)
        except (TypeError, ValueError):
            return None
        params = signature.parameters
        if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return None
        return set(params)

    # ------------------------------------------------------------------ #
    # Sinh ứng viên tham số khởi tạo
    # ------------------------------------------------------------------ #
    def _device_value(self) -> str:
        """Giá trị ``device`` cho API 3.x (``"gpu"``/``"cpu"``)."""
        return "gpu" if self.use_gpu else "cpu"

    def _modern_candidates(self) -> List[Dict[str, Any]]:
        """Ứng viên cho PaddleOCR >= 3.x (API mới)."""
        base: Dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": self.use_angle_cls,
            "device": self._device_value(),
        }
        # Có/không ``lang``: vài bản 3.x đã đủ mặc định, bản khác cần ``lang``.
        return [{"lang": self.lang, **base}, dict(base)]

    def _legacy_candidates(self) -> List[Dict[str, Any]]:
        """Ứng viên cho PaddleOCR 2.x (tham số cũ)."""
        device_variants: List[Dict[str, Any]] = [{"use_gpu": self.use_gpu}]
        if self.use_gpu:
            # Vài bản 2.x chỉ nhận bool; thử cả hướng còn lại cho chắc.
            device_variants.append({"use_gpu": False})
        device_variants.append({})

        candidates: List[Dict[str, Any]] = []
        for device_kwargs in device_variants:
            base: Dict[str, Any] = {"lang": self.lang, **device_kwargs}
            if self.enable_mkldnn is not None:
                base["enable_mkldnn"] = self.enable_mkldnn
            candidates.append({**base, "use_angle_cls": self.use_angle_cls})
            candidates.append({**base, "show_log": False})
            candidates.append(dict(base))
        return candidates

    def _candidate_kwargs(self) -> List[Dict[str, Any]]:
        """Danh sách kwargs để thử khởi tạo, xếp theo API dự đoán.

        LUÔN bao gồm ứng viên cho cả hai API và một ứng viên **tối giản**
        (không ``lang``/``enable_mkldnn``/``use_gpu``/``use_angle_cls``) để
        không bao giờ rơi vào tình trạng "mọi candidate đều dính tham số
        legacy" như bản cũ.
        """
        modern = self._modern_candidates()
        legacy = self._legacy_candidates()
        minimal: List[Dict[str, Any]] = [{"lang": self.lang}, {}]

        if self._api_major is not None and self._api_major >= 3:
            ordered = modern + legacy + minimal
        elif self._api_major == 2:
            ordered = legacy + modern + minimal
        else:
            # Bất định: ưu tiên API mới rồi tới API cũ rồi tối giản.
            ordered = modern + legacy + minimal

        # Loại trùng nhưng giữ nguyên thứ tự ưu tiên.
        seen: Set[tuple] = set()
        unique: List[Dict[str, Any]] = []
        for kwargs in ordered:
            key = tuple(sorted(kwargs.items()))
            if key in seen:
                continue
            seen.add(key)
            unique.append(kwargs)
        return unique

    def _build(self, paddle_ocr_cls: Any) -> Any:
        """Khởi tạo PaddleOCR, thử lần lượt các bộ tham số tương thích."""
        candidates = self._candidate_kwargs()
        last_error: Optional[Exception] = None
        for kwargs in candidates:
            try:
                return paddle_ocr_cls(**kwargs)
            except Exception as exc:  # noqa: BLE001 - cần thử nhiều biến thể
                last_error = exc
                continue
        raise RuntimeError(
            f"Không khởi tạo được PaddleOCR (đã thử {len(candidates)} bộ tham số): {last_error}"
        )

    # ------------------------------------------------------------------ #
    # Gọi OCR
    # ------------------------------------------------------------------ #
    def _call_predict(self, image: np.ndarray) -> Any:
        """Gọi API 3.x ``ocr.predict(image)``."""
        predict_fn = getattr(self._ocr, "predict", None)
        if predict_fn is None:
            raise AttributeError("PaddleOCR không có phương thức 'predict'")
        return predict_fn(image)

    def _call_ocr(self, image: np.ndarray) -> Any:
        """Gọi API 2.x ``ocr.ocr(image)`` (kèm biến thể có ``cls``)."""
        ocr_fn = getattr(self._ocr, "ocr", None)
        if ocr_fn is None:
            raise AttributeError("PaddleOCR không có phương thức 'ocr'")
        try:
            return ocr_fn(image, cls=self.use_angle_cls)
        except TypeError:
            return ocr_fn(image)

    def _run(self, image: np.ndarray) -> List[OCRItem]:
        ocr = self._ocr
        # Ưu tiên phương thức khớp API đã dò, nhưng vẫn thử phương thức còn lại
        # để không phụ thuộc hoàn toàn vào việc dò version.
        if self._api_major is not None and self._api_major >= 3:
            call_order = [self._call_predict, self._call_ocr]
        elif self._api_major == 2:
            call_order = [self._call_ocr, self._call_predict]
        elif hasattr(ocr, "predict"):
            call_order = [self._call_predict, self._call_ocr]
        else:
            call_order = [self._call_ocr, self._call_predict]

        errors: List[str] = []
        for call in call_order:
            try:
                raw = call(image)
            except Exception as exc:  # noqa: BLE001 - thử phương thức kế tiếp
                errors.append(f"{call.__name__}: {exc}")
                continue
            if raw is None:
                errors.append(f"{call.__name__}: trả về None")
                continue
            items = self._parse(raw)
            if items:
                self.last_error = ""
                return items
            errors.append(f"{call.__name__}: không phân tích được kết quả")
        self.last_error = "; ".join(errors)
        return []

    # ------------------------------------------------------------------ #
    # Phân tích kết quả
    # ------------------------------------------------------------------ #
    @staticmethod
    def _entry_to_mapping(entry: Any) -> Optional[Dict[str, Any]]:
        """Chuẩn hoá 1 entry kết quả 3.x (object/dict) về ``dict``.

        PaddleOCR 3.x trả về *result object* chứ không phải dict thuần, nên bản
        cũ dùng ``entry.get("rec_texts")`` không bao giờ khớp. Hàm này bóc lần
        lượt: ``.json`` (property/method, dict hoặc chuỗi JSON) -> ``["res"]``
        -> thuộc tính ``.res``.
        """
        if entry is None:
            return None
        if isinstance(entry, dict):
            return entry

        try:
            data = getattr(entry, "json", None)
        except Exception:  # noqa: BLE001 - property có thể raise
            data = None
        if callable(data):
            try:
                data = data()
            except Exception:  # noqa: BLE001
                data = None
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:  # noqa: BLE001
                data = None
        if isinstance(data, dict):
            return data

        try:  # object kiểu mapping: entry["res"]
            res = entry["res"]
        except Exception:  # noqa: BLE001
            res = getattr(entry, "res", None)
        if isinstance(res, dict):
            return {"res": res}
        return None

    @staticmethod
    def _mapping_texts(mapping: Dict[str, Any]) -> Optional[Tuple[Any, Any, Any]]:
        """Lấy ``(texts, scores, polys)`` từ mapping 3.x (có/không bọc ``res``)."""
        if not isinstance(mapping, dict):
            return None
        sources: List[Dict[str, Any]] = [mapping]
        nested = mapping.get("res")
        if isinstance(nested, dict):
            sources.append(nested)
        for source in sources:
            texts = source.get("rec_texts")
            if texts is None:
                texts = source.get("rec_text")
            if texts is None:
                continue
            if isinstance(texts, str):
                texts = [texts]
            scores = source.get("rec_scores")
            if scores is None:
                scores = source.get("rec_score")
            if scores is None:
                scores = [1.0] * len(texts)
            elif isinstance(scores, (int, float)):
                scores = [float(scores)]
            polys = source.get("rec_polys")
            if polys is None:
                polys = source.get("dt_polys")
            if polys is None:
                polys = source.get("rec_boxes")
            return texts, scores, polys
        return None

    @staticmethod
    def _parse(raw: Any) -> List[OCRItem]:
        """Phân tích kết quả PaddleOCR cho cả 2.x (nested list) và 3.x (object).

        Với 3.x, kết quả là list các *result object* (không phải dict thuần),
        nên phải bóc qua ``.json`` / ``["res"]`` trước khi đọc ``rec_texts`` —
        đây là nguyên nhân bản cũ parse ra rỗng dù khởi tạo thành công.
        """
        items: List[OCRItem] = []
        if raw is None:
            return items

        entries = list(raw) if isinstance(raw, (list, tuple)) else [raw]

        # --- PaddleOCR >= 3.x: object/dict có 'rec_texts' (trong 'res') --- #
        for entry in entries:
            mapping = PaddleOCREngine._entry_to_mapping(entry)
            if mapping is None:
                continue
            extracted = PaddleOCREngine._mapping_texts(mapping)
            if extracted is None:
                continue
            texts, scores, polys = extracted
            for index, text in enumerate(texts):
                score = 1.0
                if index < len(scores):
                    try:
                        score = float(scores[index])
                    except (TypeError, ValueError):
                        score = 1.0
                box = None
                if polys is not None and index < len(polys):
                    box = _box_array(polys[index])
                items.append(OCRItem(text=str(text), confidence=score, box=box))
        if items:
            return items

        # --- PaddleOCR 2.x: [[ [box], (text, conf) ], ...] ---------------- #
        def walk(node: Any) -> None:
            if isinstance(node, (list, tuple)):
                if (
                    len(node) == 2
                    and isinstance(node[1], (list, tuple))
                    and len(node[1]) == 2
                    and isinstance(node[1][0], str)
                ):
                    items.append(
                        OCRItem(
                            text=str(node[1][0]),
                            confidence=float(node[1][1]),
                            box=_box_array(node[0]),
                        )
                    )
                    return
                for child in node:
                    walk(child)

        walk(raw)
        return items


class EasyOCREngine(BaseOCREngine):
    """Wrapper EasyOCR — dùng làm fallback khi PaddleOCR gặp khó (spec 3.5)."""

    name = "easyocr"

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        charset: str = DEFAULT_CHARSET,
        drop_empty: bool = True,
    ) -> None:
        super().__init__(charset=charset, drop_empty=drop_empty)
        try:
            import easyocr  # type: ignore
        except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RuntimeError("Không import được easyocr. Cài đặt: pip install easyocr") from exc

        languages = [lang] if isinstance(lang, str) else list(lang)
        self._reader = easyocr.Reader(languages, gpu=bool(use_gpu), verbose=False)

    def _run(self, image: np.ndarray) -> List[OCRItem]:
        results = self._reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=list(self.charset),
        )
        items: List[OCRItem] = []
        for entry in results:
            if len(entry) < 3:
                continue
            box, text, confidence = entry[0], entry[1], entry[2]
            items.append(
                OCRItem(text=str(text), confidence=float(confidence), box=_box_array(box))
            )
        return items


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_ocr_engine(
    config: Optional[OCRConfig] = None,
    prefer: Optional[str] = None,
) -> BaseOCREngine:
    """Tạo OCR engine theo cấu hình, tự động fallback PaddleOCR -> EasyOCR.

    Args:
        config: :class:`OCRConfig`; dùng mặc định nếu ``None``.
        prefer: Ghi đè ``config.engine`` ("paddle" | "easyocr" | "auto").

    Returns:
        Instance :class:`BaseOCREngine`.

    Raises:
        RuntimeError: Nếu không engine nào khởi tạo được (kèm chi tiết lỗi).
    """
    cfg = config or OCRConfig()
    preference = (prefer or cfg.engine or "auto").lower()

    if preference == "paddle":
        order = ["paddle"]
    elif preference in ("easyocr", "easy"):
        order = ["easyocr"]
    else:
        order = ["paddle", "easyocr"]

    errors: Dict[str, str] = {}
    for name in order:
        try:
            if name == "paddle":
                return PaddleOCREngine(
                    lang=cfg.lang,
                    use_gpu=cfg.use_gpu,
                    enable_mkldnn=cfg.enable_mkldnn,
                    use_angle_cls=cfg.use_angle_cls,
                    charset=cfg.charset,
                    drop_empty=cfg.drop_empty,
                )
            return EasyOCREngine(
                lang=cfg.lang,
                use_gpu=cfg.use_gpu,
                charset=cfg.charset,
                drop_empty=cfg.drop_empty,
            )
        except Exception as exc:  # noqa: BLE001
            errors[name] = str(exc)

    raise RuntimeError(
        "Không khởi tạo được OCR engine nào. Chi tiết: "
        + json.dumps(errors, ensure_ascii=False)
    )


def available_engines() -> Dict[str, bool]:
    """Kiểm tra nhanh thư viện OCR nào đã cài trong môi trường."""
    availability: Dict[str, bool] = {}
    try:
        import paddleocr  # noqa: F401

        availability["paddle"] = True
    except Exception:  # noqa: BLE001
        availability["paddle"] = False
    try:
        import easyocr  # noqa: F401

        availability["easyocr"] = True
    except Exception:  # noqa: BLE001
        availability["easyocr"] = False
    return availability
