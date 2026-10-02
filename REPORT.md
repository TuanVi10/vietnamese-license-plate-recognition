# Project Report — Vietnamese License Plate Recognition (ALPR)

> A narrative of the full journey: from the problem, through the challenges we hit, to the
> baseline improvements (with numbers) — written as supporting material for internship applications.

---

## Demo (video)

Full pipeline output on real traffic video — vehicle detection + SORT tracking + plate OCR overlays:

<video src="docs/assets/demo.mp4" controls width="100%"></video>

> If the video does not render inline on GitHub, open [`docs/assets/demo.mp4`](docs/assets/demo.mp4).

---

## 1. Project overview

Build a pipeline that reads **Vietnamese license plates from video** (real traffic footage, 13 minutes,
~60 fps). Output is a **list of plates per vehicle (track)** with confidence scores.

The project spans the full lifecycle of a computer-vision system:

```
data collection → labeling → model training → evaluation → integration → end-to-end
```

---

## 2. Pipeline architecture (8 modules)

```
video.mp4
  │
  ├─ M0  FrameExtractor            filter / extract frames (keep original index + timestamp)
  ├─ M1  VehicleDetector (YOLO11n)  detect vehicles
  ├─ M3  SortTracker → track_id     track each vehicle across frames
  │
  ├─ M2  PlateDetector (YOLO11n finetuned)  LOCALIZE the plate inside the vehicle box
  │        └─ bbox + conf
  │
  ├─ crop the plate with a 15% margin (bbox_margin = 0.15)
  │
  ├─ ★ CornerRegressor (YOLO11n-pose, 4 keypoints)  [NEW]
  │        └─ 4 corners TL/TR/BR/BL + inferred tilt angle
  │             ├─ |angle| > 15°  → warpPerspective(4 corners) → flat plate
  │             └─ otherwise      → bbox + affine deskew (old path)
  │
  ├─ ★ PaddleOCR (PP-OCRv6) → text   [NEW, replaces EasyOCR]
  ├─ M6  PostProcessor / fusion       merge multiple reads per track, voting
  └─ M7  Output                       JSON + CSV + txt + video overlay
```

---

## 3. Journey & challenges (chronological narrative)

### 3.1. No standard Vietnamese-plate dataset → collect + hand-label ourselves
- The task needs a **Vietnamese-plate detector**, but there is almost no clean public dataset for VN plates.
- **Solution**: gathered VN plate images from multiple sources → normalized to YOLO format, deduplicated,
  and produced a stratified train/val/test split. Result: **18,619 images** (train 14,897 / val 1,862 / test 1,862).
- We also hand-built a **1,701-crop real ground-truth set (836 vehicle tracks)** for evaluation — laborious,
  but the only thing that lets us measure "how good the model is on REAL data".

### 3.2. Generic OCR reads Vietnamese plates very poorly (first domain gap)
- Tried **EasyOCR** (a popular engine) on 213 real VN plates with text GT → only **6.6% exact / 0.614 sim**.
- Cause: EasyOCR is trained mostly on ordinary Latin text and struggles with VN plate fonts
  (specific characters, 2-line plates, noise, skew).
- **Solution**: switched to **PaddleOCR PP-OCRv6** → jumped to **74.6% exact / 0.917 sim**
  (≈ 11× the exact-match rate).

### 3.3. Skewed / perspective plates → need geometric "de-skew" via 4 corners (keypoints)
- OCR on plates tilted >15° was very poor (only ~46% exact). We needed a **geometric alignment** step before OCR.
- **Solution**: added a **CornerRegressor** (YOLO11n-pose, `kpt_shape=[4,3]`) that predicts the **4 corners**
  TL/TR/BR/BL, then `warpPerspective` to "flatten" the plate.
- **Sub-challenge**: CCPD's 4-corner labels use a native order `BR,BL,TL,TR` (starting from bottom-right).
  We had to **reorder geometrically** to TL→TR→BR→BL and **verified 17,774 / 17,774 images with zero
  mismatch** (max 0.0006 px).

### 3.4. Second domain gap — the corner model trained on Chinese data
- Used **CCPD2019** (Chinese plates, 17,774 images) to train the corner model because there was no
  equivalent VN data.
- **Incident**: on CCPD val the model looked "great" (`mAP50-95(P) = 0.995`, saturating from epoch ~10),
  but on **real VN images it nearly failed** — at `conf=0.25` it only detected **1.4%**.
- **Diagnosis (important)**: this was not "the model lost its ability" — the **confidence shifted down
  when switching domains** (domain gap). At `conf=0.001` the model still detected **27%**.
- **Solution**: stronger **augmentation** for tilt (`degrees=20`, `perspective=0.001`) because 82% of
  CCPD images have |angle| < 5° (few tilted samples). Detection rate at `conf=0.001` rose
  **27.2% → 83.9%**.
- **Remaining limitation (identified)**: to work at high confidence thresholds we need to **fine-tune on
  the real 201–1701 data** (664 hand-labeled plates) instead of training on CCPD alone.

### 3.5. CUDA environment conflict between PyTorch and PaddlePaddle
- The pipeline uses **PyTorch** (torch cu126 / cuDNN 9.10) for YOLO, while **PaddleOCR** needs paddle
  cu118 / cuDNN 8.9 → **they cannot be imported in the same process**.
- **Solution**: keep two separate virtualenvs (`.venv-datasets`, `.venv-paddle`) and have the pipeline
  call OCR through a **worker subprocess** (avoids GPU-library crashes).

### 3.6. Other engineering issues
- **Tracking**: used SORT; handled ID switches, selected a **best-frame** per track (prefer crops with a
  real detection, block fallback/junk), and used voting/fusion across reads to output one plate per track.
- **Output video codec**: `mp4v` (legacy MPEG-4) wouldn't open on Windows → re-encoded to **H.264**
  (`avc1`) with `imageio-ffmpeg` (OpenCV ships no OpenH264 DLL).

---

## 4. Baseline improvements (measured on 213 real images)

| Item | Before (baseline) | After (improved) | How |
|---|---|---|---|
| **OCR engine** | EasyOCR **6.6%** exact / 0.614 sim | PaddleOCR **74.6%** exact / 0.917 sim | switch engine → PP-OCRv6 |
| **Skewed plates (>15°)** | bbox + affine: **46.3%** exact | corner-warp: **83.3%** exact | CornerRegressor + warpPerspective |
| **End-to-end (detect → align → OCR)** | **70.4%** exact / 0.886 sim | **79.8%** exact / 0.923 sim | added the alignment stage |
| **Plate detection** | — | **mAP50-95 = 0.73**, mAP50 = 0.99 | YOLO11n, 50 epochs, 18.6k images |
| **Corner model — det% on real images** | baseline **27.2%** @ conf 0.001 | augmented **83.9%** @ conf 0.001 | degrees=20 + perspective=0.001 |

**Key insight** — breakdown by tilt band (213 images with text GT):

| Tilt band | EasyOCR exact | PaddleOCR exact | End-to-end (corner-warp) |
|---|---|---|---|
| < 5° | 8.3% | 70.2% | 78.6% |
| 5°–15° | 6.7% | 73.3% | 78.7% |
| > 15° | 3.7% | 83.3% | **83.3%** (up from 46.3% with the old path) |

→ Adding the **alignment stage** especially **rescues heavily-tilted plates (>15°)**: from 46.3% to 83.3%,
while lifting the overall score from 70.4% to 79.8%.

---

## 5. Final results & side-by-side measurements

- **End-to-end run** on the 13-minute real video: **1,248 vehicle tracks**, of which **142 plates had
  readable text** (after voting/fusion, with format + reliability checks).
- **Model vs. human labeler** on 1,701 crops (836 tracks):
  - Model detected: **575 plates** | Human labeled: **664 plates** → model **missed 89** (recall ≈ 86.6%).
  - 169 crops were false positives (human marked `no_plate`), 258 crops were missed by the model.
- **Self-built data**: 18,619 detection images + 1,701 ground-truth crops + 17,774 CCPD + 35,443 CCPD-warped.

---

## 6. Tech stack & skills used

- **Python**, **PyTorch**, **Ultralytics YOLO11** (detection + pose).
- **PaddlePaddle / PaddleOCR** (PP-OCRv6), **EasyOCR**.
- **OpenCV** (homography, `warpPerspective`, affine, crop/overlay, video writer).
- **SORT** tracking, track/ID handling, best-frame selection, fusion/voting.
- **NumPy**, pandas, openpyxl, imageio / imageio-ffmpeg.
- Multi-virtualenv + subprocess management (handling torch/paddle CUDA conflicts).
- **Model evaluation**: mAP, IoU, keypoint/angle error, per-band detection rate, OCR exact/similarity.
- **Dataset building & QA**: dedup, stratified split, label verification (round-trip, convex-quad check).

---

## 7. Lessons learned (interview talking points)

1. **mAP on train/val does NOT reflect real quality.** The corner model hit 0.995 on CCPD yet nearly
   failed on VN images → always evaluate on the problem's real data.
2. **Domain gap ≠ a bad model.** A "shifted-down confidence" symptom needs proper diagnosis
   (sweep thresholds) rather than a hasty conclusion.
3. **Data quality caps performance.** Missing tilted samples made the model weak in exactly that band;
   the right augmentation (degrees/perspective) clearly helped.
4. **Decompose the problem into modules** (detect → align → OCR → fusion) to isolate faults and measure
   each layer independently.
5. **The "angle gate"** (only warp when >15°) is a practical precision/recall trade-off instead of paying
   for an expensive transform on every image.
