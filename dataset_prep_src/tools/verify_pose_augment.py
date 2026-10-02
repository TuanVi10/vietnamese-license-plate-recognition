"""
tools/verify_pose_augment.py
============================
KIỂM CHỨNG thực nghiệm: augment xoay / phối cảnh / lật của ultralytics có biến đổi
**đúng cả 4 keypoint** (không chỉ bbox) hay không.

Cách kiểm chứng (không tin cảm tính):

* **A. Cùng một ma trận homography cho ảnh và keypoint**
  Lấy ``M`` từ ``RandomPerspective.get_params`` rồi tự warp ảnh bằng ``cv2.warpPerspective``
  và so với ảnh ultralytics tạo ra → phải trùng khít.

* **B. Keypoint do ultralytics trả về = homography áp lên keypoint gốc**
  Tự tính ``cv2.perspectiveTransform(kpt_goc, M)`` rồi so với keypoint ultralytics trả về.

* **C. Lật ngang + flip_idx**: kiểm tra ``keypoint[0] (TL)`` sau lật đúng bằng ảnh gương của
  ``keypoint[1] (TR)`` gốc, tức ``flip_idx=[1,0,3,2]`` hoạt động đúng.

* **D. ``fliplr=0.0`` thì không lật** — keypoint phải giữ nguyên.

* **E. Hàng rào cấu hình**: ``v8_transforms`` phải chấp nhận ``flip_idx`` dài đúng bằng
  ``kpt_shape[0]`` và **không** tự tắt augment.

Chạy::

    python tools/verify_pose_augment.py --dataset S:/M_AGENT/CV/datasets/ccpd_pose \
        --out S:/M_AGENT/CV/datasets/ccpd_pose/viz_augment
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.augment import RandomFlip, RandomPerspective, v8_transforms
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.instance import Instances

#: Màu BGR theo thứ tự TL, TR, BR, BL (giống viz_ccpd_pose.py).
CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 165, 255)]
KPTS = 4


def load_sample(img_path: Path, lbl_path: Path):
    """Đọc ảnh + 4 keypoint (pixel) + cls từ nhãn YOLO-pose."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    toks = lbl_path.read_text(encoding="utf-8").split()
    cls = np.array([float(toks[0])], dtype=np.float32)
    kpts = np.zeros((1, KPTS, 3), dtype=np.float32)
    for i in range(KPTS):
        kpts[0, i, 0] = float(toks[5 + 3 * i]) * w
        kpts[0, i, 1] = float(toks[6 + 3 * i]) * h
        kpts[0, i, 2] = float(toks[7 + 3 * i])
    return img, cls, kpts


def make_labels(img: np.ndarray, cls: np.ndarray, kpts: np.ndarray) -> dict:
    """Dựng dict ``labels`` đúng dạng mà augment của ultralytics mong đợi."""
    h, w = img.shape[:2]
    bbox = np.array(
        [[kpts[0, :, 0].min(), kpts[0, :, 1].min(),
          kpts[0, :, 0].max(), kpts[0, :, 1].max()]], dtype=np.float32
    )
    segs = np.zeros((0, 1000, 2), dtype=np.float32)
    return {
        "img": img.copy(),
        "cls": cls.copy(),
        "instances": Instances(bbox, segs, kpts.copy(),
                               bbox_format="xyxy", normalized=False),
        "ori_shape": (h, w),
        "resized_shape": (h, w),
        "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
        "im_file": "",
        "shape": (h, w),
    }


def draw_kpts(img: np.ndarray, pts: np.ndarray, extra: str = "") -> np.ndarray:
    """Vẽ 4 keypoint (đánh số + tô màu) lên ảnh pixel."""
    out = img.copy()
    pts = [(int(round(x)), int(round(y))) for x, y in pts]
    for i in range(KPTS):
        cv2.line(out, pts[i], pts[(i + 1) % KPTS], (0, 255, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (x, y), 7, CORNER_COLORS[i], -1, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, CORNER_COLORS[i], 1, cv2.LINE_AA)
    if extra:
        cv2.putText(out, extra, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, extra, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Kiểm chứng augment keypoint của ultralytics.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    root = Path(args.dataset)
    out_dir = Path(args.out) if args.out else root / "viz_augment"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_names = sorted(p.name for p in (root / "labels" / "val").glob("*.txt"))
    names = all_names[:: max(1, len(all_names) // args.samples)][: args.samples]
    fails: list[str] = []

    # ---------------------------------------------------------------- TEST 1
    # Hình học thuần: giữ nguyên kích thước, không dịch/không scale để 4 góc
    # luôn nằm trong khung -> phép so sánh số học phải khớp tuyệt đối.
    print("== TEST 1: kiem chung so hoc (khong dich, khong scale) ==")
    rp_exact = RandomPerspective(degrees=25.0, translate=0.0, scale=(1.0, 1.0),
                                 shear=0.0, perspective=0.001, size=(720, 1160))
    for name in names:
        img, cls, kpts = load_sample(root / "images" / "val" / (name[:-4] + ".jpg"),
                                     root / "labels" / "val" / name)
        labels = make_labels(img, cls, kpts)
        params = rp_exact.get_params(copy.deepcopy(labels))
        lab = rp_exact.apply_image(copy.deepcopy(labels), params)
        lab = rp_exact.apply_instances(lab, params)
        M, size = params["M"], params["size"]

        if rp_exact.perspective:
            ref_img = cv2.warpPerspective(img, M, dsize=size,
                                          borderValue=(114, 114, 114, 114))
        else:
            ref_img = cv2.warpAffine(img, M[:2], dsize=size,
                                     borderValue=(114, 114, 114, 114))
        diff_a = int(np.abs(ref_img.astype(np.int16) - lab["img"].astype(np.int16)).max())

        ref_k = cv2.perspectiveTransform(kpts[..., :2].astype(np.float64), M)[0]
        got = lab["instances"].keypoints[0]
        got_k, got_v = got[:, :2], got[:, 2]
        # Ultralytics có CLIP keypoint vào khung ảnh -> chỉ so các điểm nằm trong khung,
        # còn điểm vượt khung thì kiểm tra visibility phải bị đặt về 0.
        inside = ((ref_k[:, 0] >= 0) & (ref_k[:, 0] <= size[0])
                  & (ref_k[:, 1] >= 0) & (ref_k[:, 1] <= size[1]))
        n_out = int((~inside).sum())
        diff_b = float(np.abs(ref_k[inside] - got_k[inside]).max()) if inside.any() else 0.0
        vis_ok = bool(np.all(got_v[~inside] == 0)) if n_out else True

        print(f"  {name[:34]:34s} |A|anh={diff_a:<3d} |B|kpt_trong_khung={diff_b:.6f}px "
              f"| ngoai_khung={n_out} vis0_dung:{vis_ok}")
        if diff_a != 0:
            fails.append(f"TEST1-A {name}")
        if diff_b > 0.01:
            fails.append(f"TEST1-B {name} ({diff_b:.4f}px)")
        if n_out and not vis_ok:
            fails.append(f"TEST1-vis {name}")

    # ---------------------------------------------------------------- TEST 2
    # Cấu hình THẬT khi train -> xuất ảnh overlay để soi bằng mắt.
    print(f"== TEST 2: cau hinh train that (imgsz={args.imgsz}, degrees=20, "
          f"perspective=0.001) ==")
    rp_train = RandomPerspective(degrees=20.0, translate=0.1, scale=0.5, shear=0.0,
                                 perspective=0.001, size=(args.imgsz, args.imgsz))
    clipped = 0
    for i, name in enumerate(names):
        img, cls, kpts = load_sample(root / "images" / "val" / (name[:-4] + ".jpg"),
                                     root / "labels" / "val" / name)
        labels = make_labels(img, cls, kpts)
        params = rp_train.get_params(copy.deepcopy(labels))
        lab = rp_train.apply_image(copy.deepcopy(labels), params)
        lab = rp_train.apply_instances(lab, params)
        inst = lab["instances"]
        if len(inst.bboxes) == 0:
            print(f"  {name[:36]:36s} -> instance bi loai boi box_candidates")
            continue
        pts = inst.keypoints[0][:, :2]
        vis = inst.keypoints[0][:, 2]
        if any(v != 2 for v in vis):
            clipped += 1
        ang = float(np.degrees(np.arctan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])))
        cv2.imwrite(str(out_dir / f"persp_{i}_rot{ang:+.0f}deg.png"),
                    draw_kpts(lab["img"], pts, f"deg=20 persp=0.001 rot={ang:+.1f}"))
    print(f"  -> {len(names)} anh -> {out_dir} (so mau co kpt bi clip ra ngoai khung: {clipped})")

    # ---------------------------------------------------------------- TEST 3
    print("== TEST 3: lat ngang + flip_idx=[1,0,3,2] ==")
    img, cls, kpts = load_sample(root / "images" / "val" / (names[0][:-4] + ".jpg"),
                                 root / "labels" / "val" / names[0])
    h, w = img.shape[:2]
    base = make_labels(img, cls, kpts)
    rf = RandomFlip(direction="horizontal", p=1.0, flip_idx=[1, 0, 3, 2])
    p3 = rf.get_params(base)
    lab = rf.apply_image(copy.deepcopy(base), p3)
    lab = rf.apply_instances(lab, p3)
    got = lab["instances"].keypoints[0][:, :2]
    exp = np.stack([np.array([w - kpts[0, j, 0], kpts[0, j, 1]]) for j in [1, 0, 3, 2]])
    diff_c = float(np.abs(exp - got).max())
    print(f"  kpt sau lat vs cong thuc (w-x) theo flip_idx: maxdiff={diff_c:.6f} px")
    print("  y nghia: kpt[0]=TL sau lat == anh guong cua kpt[1]=TR goc")
    if diff_c > 0.01:
        fails.append(f"TEST3 ({diff_c:.4f}px)")
    cv2.imwrite(str(out_dir / "flip_horizontal.png"),
                draw_kpts(lab["img"], got, "fliplr=1.0 flip_idx=[1,0,3,2]"))

    # ---------------------------------------------------------------- TEST 4
    rf0 = RandomFlip(direction="horizontal", p=0.0, flip_idx=[1, 0, 3, 2])
    base0 = make_labels(img, cls, kpts)
    p4 = rf0.get_params(base0)
    lab0 = rf0.apply_instances(rf0.apply_image(copy.deepcopy(base0), p4), p4)
    diff_d = float(np.abs(lab0["instances"].keypoints[0][:, :2] - kpts[0, :, :2]).max())
    print(f"== TEST 4: fliplr=0.0 -> keypoint khong doi: maxdiff={diff_d:.6f} px")
    if diff_d > 0.01:
        fails.append(f"TEST4 ({diff_d:.4f}px)")

    # ---------------------------------------------------------------- TEST 5
    print("== TEST 5: hang rao cau hinh cua v8_transforms ==")
    from ultralytics.cfg import DEFAULT_CFG

    class FakeDS:
        names = {0: "license_plate"}
        use_keypoints = True
        cache = "ram"

        def __init__(self, data):
            self.data = data

    def build(data, fliplr):
        hyp = IterableSimpleNamespace(**{
            **vars(DEFAULT_CFG), "degrees": 20.0, "perspective": 0.001, "fliplr": fliplr,
            "flipud": 0.0, "mosaic": 1.0, "mixup": 0.0, "cutmix": 0.0,
            "copy_paste": 0.0, "copy_paste_mode": "flip", "augmentations": None})
        v8_transforms(FakeDS(data), args.imgsz, hyp)
        return hyp

    hyp_ok = build({"kpt_shape": [4, 3], "flip_idx": [1, 0, 3, 2], "names": FakeDS.names}, 0.0)
    print(f"  flip_idx dai dung 4 == kpt_shape[0] -> OK, fliplr giu nguyen = {hyp_ok.fliplr}")
    if hyp_ok.fliplr != 0.0:
        fails.append("TEST5 fliplr bi doi")

    hyp_noidx = build({"kpt_shape": [4, 3], "names": FakeDS.names}, 0.5)
    print(f"  KHONG co flip_idx + fliplr=0.5 -> tu tat: fliplr={hyp_noidx.fliplr}")

    try:
        build({"kpt_shape": [4, 3], "flip_idx": [1, 0, 3], "names": FakeDS.names}, 0.0)
        print("  flip_idx sai do dai -> KHONG bao loi (rui ro!)")
        fails.append("TEST5 thieu guard do dai flip_idx")
    except ValueError as exc:
        print(f"  flip_idx sai do dai -> bao loi dung: {exc}")

    print()
    print("RESULT:", "PASS" if not fails else f"FAIL -> {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())


