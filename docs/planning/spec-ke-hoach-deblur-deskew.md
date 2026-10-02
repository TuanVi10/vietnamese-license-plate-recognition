# SPEC: XỬ LÝ ẢNH BIỂN SỐ - DEBLUR & DESKEW

**Mục đích:** Bổ sung 2 module tiền xử lý ảnh biển số vào pipeline ALPR:
1. **Deblur** — Khử làm mờ cho ảnh biển số mờ
2. **Deskew** — Chỉnh góc nghiêng ảnh biển số

---

## 1. VỊ TRÍ TRONG PIPELINE

Dựa trên kiến trúc pipeline hiện tại, đề xuất vị trí đặt như sau:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Module 4: Tiền xử lý ảnh biển số (HIỆN TẠI)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. crop = _crop_plate(image, bbox)          # Đã có                       │
│  2. deskew = rectify_plate(crop)             # THÊM MỚI (deskew sớm)       │
│  3. deblur = deblur_plate(desked)            # THÊM MỚI (deblur)           │
│  4. two_line = split_two_line(deblurred)     # Đã có (tách dòng)           │
│  5. normalize = resize_with_padding(...)     # Đã có                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Module 5: OCR (PaddleOCR)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Module 4 (tiếp): Skew Retry Ladder                                         │
│  - should_retry_skewed()              # Đã có                              │
│  - retry_skewed_variants()            # Đã có (retry sau OCR fail)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Lưu ý quan trọng:**
- **Deskew sớm** (ngay sau crop, trước OCR lần đầu) giúp OCR đọc chính xác hơn từ đầu
- **Deskew retry** (sau OCR fail) vẫn giữ lại như cơ chế dự phòng
- **Deblur** đặt **sau deskew** để đảm bảo ảnh đã được căn chỉnh phẳng trước khi khử mờ

---

## 2. MODULE DEBLUR (KHỬ LÀM MỜ)

### 2.1 Bối cảnh vấn đề

- Ảnh biển số có thể bị mờ do: camera rung, xe di chuyển nhanh, khoảng cách xa, điều kiện ánh sáng kém
- OCR (PaddleOCR/EasyOCR) giảm độ chính xác đáng kể khi đầu vào bị mờ
- Không có module deblur trong pipeline hiện tại

### 2.2 Thuật toán đề xuất

**Phương pháp 1: Wiener Filter (Khuyến nghị cho CPU)**

```python
def wiener_deblur(image: np.ndarray, kernel_size: int = 5, noise_var: float = 0.01) -> np.ndarray:
    """
    Khử mờ bằng Wiener filter - nhẹ, chạy nhanh trên CPU.
    
    Args:
        image: Ảnh đầu vào (grayscale hoặc BGR)
        kernel_size: Kích thước kernel Gaussian (lẻ)
        noise_var: Ước tính nhiễu (điều chỉnh theo thực tế)
    
    Returns:
        Ảnh đã khử mờ
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    
    # Tạo kernel motion blur (giả định blur do di chuyển ngang)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    
    # Wiener filter trong frequency domain
    psf = np.fft.fft2(kernel, s=gray.shape)
    psf_conj = np.conj(psf)
    img_fft = np.fft.fft2(gray)
    
    # SNR = signal power / noise power
    nsr = noise_var / np.mean(np.abs(img_fft) ** 2)
    
    # Wiener filter: G = H* / (|H|² + NSR)
    deconv = psf_conj / (psf * psf_conj + nsr) * img_fft
    result = np.fft.ifft2(deconv)
    result = np.real(result)
    
    # Normalize về [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result
```

**Phương pháp 2: Sharpening Kernel (Fallback đơn giản)**

```python
def sharpen_image(image: np.ndarray) -> np.ndarray:
    """
    Tăng độ nét bằng unsharp masking - nhanh nhưng ít hiệu quả với mờ nặng.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    
    # Tạo ảnh mờ Gaussian
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Unsharp mask: sharpened = original + (original - blurred) * amount
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else sharpened
```

**Phương pháp 3: Blind Deconvolution (Nâng cao)**

```python
def blind_deblur(image: np.ndarray, max_iterations: int = 10) -> np.ndarray:
    """
    Tự động ước tính kernel mờ - phức tạp hơn, chạy chậm hơn.
    Chỉ dùng khi Wiener không hiệu quả.
    """
    # Đây là phương pháp phức tạp, cần OpenCV xử lý từng bước
    # Khuyến nghị: dùng thư viện chuyên dụng như OpenCV deblur API
    # hoặc simple- deblur GAN nếu cần
    pass
```

### 2.3 Phát hiện mờ (Blur Detection)

Trước khi deblur, cần xác định ảnh có cần deblur không:

```python
def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Phát hiện ảnh có bị mờ không dùng Laplacian variance.
    
    Args:
        image: Ảnh đầu vào
        threshold: Ngưỡng - thấp = mờ, cao = rõ (default 100 phù hợp cho ảnh biển số)
    
    Returns:
        True nếu ảnh mờ (cần deblur)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold
```

**Ngưỡng đề xuất:**
| Loại ảnh | Laplacian variance threshold |
|----------|------------------------------|
| Rõ nét   | > 150                        |
| Chấp nhận được | 100-150              |
| Mờ nhẹ   | 50-100                       |
| Mờ nặng  | < 50                         |

### 2.4 Tham số cần tinh chỉnh

| Tham số | Giá trị khởi điểm | Cần tinh chỉnh theo |
|---------|-------------------|---------------------|
| `blur_threshold` | 100 (Laplacian variance) | Video thật |
| `noise_var` (Wiener) | 0.01 | Thử giá trị 0.001 - 0.1 |
| `kernel_size` | 5 | Lẻ, từ 3-9 |
| `enable_deblur` | True | Có thể tắt nếu video chất lượng tốt |

---

## 3. MODULE DESKEW (CHỈNH GÓC NGHIÊNG)

### 3.1 Bối cảnh vấn đề

- Ảnh biển số có thể bị nghiêng do: góc camera, xe nghiêng khi đi qua
- OCR yêu cầu ảnh ngang để đọc chính xác
- Pipeline hiện tại đã có **skew retry ladder** (xử lý sau OCR fail), nhưng nên xử lý **sớm** (trước OCR lần đầu)

### 3.2 Thuật toán đề xuất

**Deskew sớm — thêm vào Module 4 (trước OCR):**

```python
def deskew_plate(image: np.ndarray) -> np.ndarray:
    """
    Phát hiện và chỉnh góc nghiêng của biển số.
    
    Thuật toán:
    1. Tìm góc nghiêng bằng Hough Lines hoặc minAreaRect
    2. Xoay ảnh để căn chỉnh theo trục ngang
    
    Args:
        image: Ảnh biển số đã crop
    
    Returns:
        Ảnh đã được căn chỉnh (deskewed)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    
    # Nhị phân hóa ảnh
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image  # Không tìm thấy contour, trả về ảnh gốc
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tìm bounding rect có góc xoay
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]  # Góc nghiêng ước tính
    
    # Điều chỉnh góc: OpenCV trả về góc [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Nếu góc nhỏ quá (< 1 độ), bỏ qua
    if abs(angle) < 1.0:
        return image
    
    # Xoay ảnh
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Xoay với border đặc biệt để giữ nguyên thông tin biển số
    rotated = cv2.warpAffine(image, M, (w, h), 
                              borderMode=cv2.BORDER_REPLICATE)
    
    return rotated
```

**Hoặc dùng phương pháp đơn giản hơn (project profile):**

```python
def deskew_by_projection(image: np.ndarray) -> np.ndarray:
    """
    Deskew đơn giản bằng projection profile - thử nhiều góc, chọn góc tốt nhất.
    Chậm hơn nhưng chính xác hơn với biển số có text rõ.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    best_angle = 0
    best_score = 0
    
    # Thử các góc từ -30 đến +30 độ
    for angle in range(-30, 31, 1):
        rotated = rotate_image(gray, angle)
        
        # Tính projection profile theo chiều ngang
        projection = np.sum(rotated, axis=1)
        
        # Score = variance của projection (cao = text phân bố đều = ngang)
        score = np.var(projection)
        
        if score > best_score:
            best_score = score
            best_angle = angle
    
    if best_angle != 0:
        return rotate_image(image, best_angle)
    return image
```

### 3.3 Kết hợp với skew retry ladder hiện có

Pipeline hiện tại đã có cơ chế retry với các góc cố định:

```python
# Trong ai/inference/pipeline.py (đã có)
def retry_skewed_variants(crop, recognizer):
    variants = [
        ("upright", 0.0),
        ("slight_left", -2.5),
        ("slight_right", 2.5),
        ("upside_down", 180.0),
    ]
    # ... retry logic
```

**Đề xuất cải tiến:** Thêm vào danh sách variants:
- ±5 độ (cho góc nghiêng lớn hơn)
- ±10 độ (nếu cần)

### 3.4 Tham số cần tinh chỉnh

| Tham số | Giá trị khởi điểm | Cần tinh chỉnh theo |
|---------|-------------------|---------------------|
| `deskew_angle_threshold` | 1.0 độ | Bỏ qua nếu góc nhỏ hơn |
| `max_skew_angle` | 30 độ | Giới hạn góc xoay tối đa |
| `enable_early_deskew` | True | Bật để xử lý sớm |

---

## 4. THỨ TỰ XỬ LÝ TRONG MODULE 4

```python
# Module 4: preprocess_plate.py (đề xuất mới)

def preprocess_plate(crop: np.ndarray, config: InferenceConfig) -> np.ndarray:
    """
    Tiền xử lý ảnh biển số theo thứ tự:
    1. Deskew (chỉnh nghiêng) 
    2. Tách dòng (nếu 2 dòng)
    3. Deblur (khử mờ) - nếu cần
    4. Resize & normalize
    """
    
    # Bước 1: Deskew sớm
    if config.enable_early_deskew:
        deskewed = deskew_plate(crop)
    else:
        deskewed = crop
    
    # Bước 2: Tách dòng (nếu biển 2 dòng) - logic hiện có
    line_count = estimate_line_count(deskewed)
    if line_count == 2:
        processed = merge_two_line(deskewed)  # Ghép 2 dòng thành 1
    else:
        processed = deskewed
    
    # Bước 3: Deblur (nếu phát hiện mờ)
    if config.enable_deblur:
        if is_blurry(processed, config.blur_threshold):
            processed = wiener_deblur(processed, 
                                       kernel_size=config.deblur_kernel_size,
                                       noise_var=config.deblur_noise_var)
    
    # Bước 4: Resize với padding - giữ aspect ratio
    resized = resize_with_padding(processed, target_size=(640, 64))
    
    return resized
```

---

## 5. TÍCH HỢP VÀO PIPELINE CHÍNH

### 5.1 Sửa đổi file `ai/inference/pipeline.py`

```python
# Thêm vào imports
from ai.inference.deblur import wiener_deblur, is_blurry
from ai.inference.deskew import deskew_plate

# Sửa method process() - thêm bước tiền xử lý sau crop
def process(self, image, read_text=True):
    # ... existing detection code ...
    
    for detection in detections:
        crop = self._crop_plate(image, detection.bbox)
        
        # THÊM MỚI: Deskew sớm
        if self._config.enable_early_deskew:
            crop = deskew_plate(crop)
        
        # THÊM MỚI: Deblur nếu cần
        if self._config.enable_deblur and is_blurry(crop, self._config.blur_threshold):
            crop = wiener_deblur(crop)
        
        # Tiếp tục OCR như cũ
        recognition = self._recognizer.recognize(crop)
        # ...
```

### 5.2 Cập nhật InferenceConfig

```python
# Trong ai/inference/config.py
class InferenceConfig:
    # ... existing fields ...
    
    # Deblur settings
    enable_deblur: bool = True
    blur_threshold: float = 100.0  # Laplacian variance
    deblur_kernel_size: int = 5
    deblur_noise_var: float = 0.01
    
    # Early deskew settings  
    enable_early_deskew: bool = True
    deskew_angle_threshold: float = 1.0  # độ
```

---

## 6. KẾ HOẠCH TRIỂN KHAI

| Tuần | Công việc | Đầu ra |
|------|-----------|--------|
| 4 (mở rộng) | Viết hàm `is_blurry()`, `wiener_deblur()`, `deskew_plate()` | File `deblur.py`, `deskew.py` |
| 4 (mở rộng) | Tích hợp vào Module 4 (`preprocess_plate`) | Pipeline xử lý đúng thứ tự |
| 5 | Test trên video thật, tinh chỉnh ngưỡng blur_threshold, noise_var | Ngưỡng tối ưu cho dataset |
| 5 | So sánh: có deblur vs không deblur, có early deskew vs không | Đo lường improvement |

---

## 7. METRIC ĐÁNH GIÁ

| Metric | Công thức | Cách đo |
|--------|-----------|---------|
| **Blur Detection Accuracy** | (số ảnh mờ đúng) / (tổng ảnh mờ) | So sánh với nhãn thủ công |
| **Deblur Improvement** | (OCR acc sau deblur) - (OCR acc trước deblur) | Đo trên tập ảnh mờ |
| **Deskew Success Rate** | (số ảnh nghiêng được căn chỉnh đúng góc) / (tổng ảnh nghiêng) | Đo góc sau deskew |
| **OCR Accuracy với early deskew** | So sánh với skew retry ladder | A/B test |

---

## 8. FILES CẦN TẠO/SỬA

| File | Hàm cần tạo | Ghi chú |
|------|-------------|---------|
| `ai/inference/deblur.py` | `is_blurry()`, `wiener_deblur()`, `sharpen_image()` | Module mới |
| `ai/inference/deskew.py` | `deskew_plate()`, `deskew_by_projection()` | Module mới |
| `ai/inference/config.py` | Thêm các field config mới | Cập nhật |
| `ai/inference/pipeline.py` | Tích hợp vào `process()` | Cập nhật |
| `ai/inference/preprocess_plate.py` | Gọi các hàm mới | Cập nhật |

---

## 9. LƯU Ý QUAN TRỌNG

1. **Thứ tự xử lý:** Deskew → Tách dòng → Deblur → Resize. Làm ngược thứ tự sẽ gây ra kết quả không chính xác.

2. **Không phải lúc nào cũng cần deblur:** Nếu ảnh đã rõ nét, deblur có thể làm hỏng thay vì cải thiện. Luôn kiểm tra `is_blurry()` trước.

3. **Tinh chỉnh ngưỡng theo video thật:** Các ngưỡng blur_threshold, noise_var cần được điều chỉnh dựa trên video test cụ thể.

4. **Fallback:** Nếu Wiener filter không hiệu quả, dùng sharpening kernel đơn giản hơn.

5. **Performance:** Deblur + Deskew tăng thời gian xử lý mỗi plate khoảng 5-20ms trên CPU. Cân nhắc enable/disable theo yêu cầu tốc độ.

---

*Spec được viết ngày 20/09/2026, bổ sung vào spec kế hoạch dự án nhận diện biển số xe v2.*