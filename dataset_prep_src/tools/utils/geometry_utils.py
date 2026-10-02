"""
tools/geometry_utils.py
=======================
Tiện ích hình học dùng chung cho các script gán nhãn 4 góc biển số
(``ccpd_to_pose.py``, ``unify_labeling.py``).
"""

from __future__ import annotations

#: Thứ tự góc cố định của dự án: trên-trái -> trên-phải -> dưới-phải -> dưới-trái.
CORNER_ORDER = ("tl", "tr", "br", "bl")


def order_corners_geometric(raw: list[tuple[float, float]]
                            ) -> tuple[tuple[float, float], ...]:
    """Sắp 4 điểm về (TL, TR, BR, BL) CHỈ dựa vào vị trí hình học thực tế.

    Công thức kinh điển cho tứ giác lồi::

        TL = min(x + y)      BR = max(x + y)
        TR = max(x - y)      BL = min(x - y)
    """
    pts = [(float(x), float(y)) for x, y in raw]
    by_sum = sorted(pts, key=lambda p: p[0] + p[1])
    by_diff = sorted(pts, key=lambda p: p[0] - p[1])
    tl, br = by_sum[0], by_sum[-1]
    bl, tr = by_diff[0], by_diff[-1]
    return tl, tr, br, bl


def is_convex(pts: tuple[tuple[float, float], ...]) -> bool:
    """Kiểm tra tứ giác lồi (không đổi dấu qua 4 cạnh)."""
    signs = []
    for i in range(4):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 4]
        cx, cy = pts[(i + 2) % 4]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) < 1e-6:
            return False
        signs.append(cross > 0)
    return all(s == signs[0] for s in signs)


def dedupe_closing_point(points: list, eps: float = 1e-6) -> list:
    """Bỏ điểm cuối nếu trùng điểm đầu (polygon khép kín do Label Studio trả về)."""
    if len(points) > 4:
        first, last = points[0], points[-1]
        if abs(float(first[0]) - float(last[0])) < eps and \
                abs(float(first[1]) - float(last[1])) < eps:
            return points[:-1]
    return points
