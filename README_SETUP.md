# ComparisonDetector - TF2 Compatible Version

Phát hiện đối tượng dựa trên so sánh few-shot sử dụng FPN, RPN và Fast R-CNN. Code đã được cập nhật để tương thích với **TensorFlow 2.19 + tf.compat.v1** và loại bỏ `tf.contrib`.

---

## 🚀 Cài đặt nhanh (Windows)

### 1. Clone repo và tạo môi trường ảo
```powershell
git clone https://github.com/dungdinhhaha/AIDetect.git
cd AIDetect

# Tạo venv (dùng Python 3.10 hoặc 3.11)
py -3.10 -m venv .venv
.\.venv\Scripts\activate
```

### 2. Cài đặt dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Chạy smoke test (không cần GPU/data)
```powershell
python .\tools\smoke_test.py
```

Nếu thấy `✓ SMOKE TEST PASSED`, code đã sẵn sàng!

---

## 📦 Yêu cầu
- **Python**: 3.10 hoặc 3.11
- **TensorFlow**: 2.19.0 (với tf.compat.v1 mode)
- **tf_slim**: 1.1.0 (thay thế tf.contrib.slim)
- **NumPy**: 1.26.4 (tránh ABI issues với NumPy 2.x)
- **OpenCV**: opencv-python-headless 4.7.0.72

Xem đầy đủ trong `requirements.txt`.

---

## 🏋️ Train trên máy local

### 1. Chuẩn bị dữ liệu
Đặt TFRecord files và `labels.tsv` vào `data/tct/`:
```
data/
  tct/
    train.tfrecord
    test.tfrecord
    labels.tsv
```

### 2. Cấu hình
Chỉnh trong `configs/config.py`:
- `DATA_DIR`: đường dẫn đến thư mục data
- `CHECKPOINT_DIR`: pretrained ResNet checkpoint (nếu có)
- `MODLE_DIR`: nơi lưu model output

### 3. Chạy training
```powershell
python .\tools\train.py
```

**Lưu ý**: Training cần GPU; nếu máy yếu, dùng Colab (xem phần dưới).

---

## ☁️ Train trên Google Colab

### Cách 1: Mở notebook trực tiếp
Tạo một Colab notebook mới và paste code từ hướng dẫn dưới đây.

### Cách 2: Upload từ GitHub
1. Trong Colab: **File → Open notebook → GitHub**
2. Nhập URL repo: `https://github.com/dungdinhhaha/AIDetect`
3. Chọn notebook (nếu đã tạo `.ipynb` trong repo)

### Setup trong Colab (copy vào cell đầu):
```python
# Cài đặt dependencies
!pip install -q tensorflow==2.19.0 tf_slim==1.1.0 numpy==1.26.4 \\
    opencv-python-headless==4.7.0.72 scikit-image scipy matplotlib tqdm

# Disable eager execution (quan trọng!)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Clone repo
!git clone https://github.com/dungdinhhaha/AIDetect.git
%cd AIDetect

# Mount Google Drive để lưu model/data
from google.colab import drive
drive.mount('/content/drive')

# Upload data hoặc copy từ Drive vào /content/data/tct/
```

Sau đó chạy:
```python
!python tools/train.py
```

---

## 🔧 Thay đổi so với code gốc

### ✅ Đã sửa
- **tf.contrib.slim** → **tf_slim** (tất cả network files)
- **scipy.misc.imresize** → **skimage.transform.resize** (trong `reference.py`)
- **tf.contrib.estimator** fallbacks (trong `tools/train.py`)
- Pinned NumPy < 2.0 để tránh `_ARRAY_API` ABI error

### 📁 Files đã patch
- `tools/train.py`
- `libs/networks/nets/resnet_v1.py`, `resnet_v2.py`, `resnet_utils.py`
- `libs/networks/nets/vgg.py`
- `libs/networks/nets/overfeat.py`, `overfeat_test.py`
- `libs/networks/nets/pix2pix.py`
- `reference.py`

---

## 🐛 Xử lý lỗi Git config

Nếu gặp lỗi `Permission denied` khi dùng `git config --global`:

### Giải pháp 1: Chạy PowerShell **as Administrator**
1. Chuột phải PowerShell → **Run as Administrator**
2. Chạy lại:
   ```powershell
   git config --global user.name "dungdinhhaha"
   git config --global user.email "dungdinh542004@gmail.com"
   ```

### Giải pháp 2: Dùng config local (chỉ trong repo này)
```powershell
cd D:\ComparisonDetector
git config user.name "dungdinhhaha"
git config user.email "dungdinh542004@gmail.com"
```

### Giải pháp 3: Sửa file .gitconfig thủ công
1. Mở `C:\Users\ADMIN\.gitconfig` bằng Notepad **as Admin**
2. Thêm:
   ```ini
   [user]
       name = dungdinhhaha
       email = dungdinh542004@gmail.com
   ```
3. Lưu lại

---

## 📚 Tham khảo
- **Paper gốc**: [Comparison Detector for Few-Shot Object Detection](https://arxiv.org/abs/...)
- **TensorFlow Slim**: https://github.com/google-research/tf-slim
- **Repo gốc**: https://github.com/CVIU-CSU/ComparisonDetector

---

## 📝 License
Giữ nguyên license của repo gốc (nếu có).

---

**Tác giả migration**: dungdinhhaha  
**Email**: dungdinh542004@gmail.com
