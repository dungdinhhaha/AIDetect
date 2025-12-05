# 🔧 HƯỚNG DẪN XỬ LÝ SAU KHI TRAIN TRÊN COLAB

## 📌 Vấn đề bạn gặp phải

Training dừng ở epoch 7 vì:
1. **Bạn bấm Ctrl+C** (chủ động dừng)
2. **Dataset hết data** - Warning: `Your input ran out of data`
3. **`steps_per_epoch=500` quá lớn** cho dataset

---

## ✅ GIẢI PHÁP ĐÃ FIX

### 1. Update `train_keras.py`:
- ✅ Tự động tính `steps_per_epoch` dựa trên dataset size
- ✅ Thêm `.repeat()` để dataset không bao giờ hết
- ✅ Thêm callbacks tốt hơn: `best_model`, `reduce_lr`
- ✅ Save cả final model và best model

### 2. Tạo `resume_training.py`:
- ✅ Resume từ checkpoint cuối cùng
- ✅ Tự động detect epoch đã train
- ✅ Continue từ epoch tiếp theo

---

## 🚀 CÁCH SỬ DỤNG TRÊN COLAB

### **Option 1: Train từ đầu (lần đầu)**

```python
# Cell 1: Clone & Setup
!git clone https://github.com/dungdinhhaha/AIDetect.git
%cd AIDetect
!pip install -q -r requirements.txt

# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Start Training
!python train_keras.py
```

**Kết quả:**
- Model sẽ tự động save mỗi epoch vào `/content/drive/MyDrive/comparison_detector_models_v2/checkpoints/`
- Best model save vào `/content/drive/MyDrive/comparison_detector_models_v2/best_model.h5`
- Final model save vào `/content/drive/MyDrive/comparison_detector_models_v2/final_model.keras`

---

### **Option 2: Resume training (nếu bị ngắt giữa chừng)**

Nếu Colab disconnect hoặc bạn bấm Ctrl+C:

```python
# Cell 1: Clone & Setup (nếu session mới)
!git clone https://github.com/dungdinhhaha/AIDetect.git
%cd AIDetect
!pip install -q -r requirements.txt

# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Resume từ checkpoint
!python resume_training.py
```

**Script sẽ:**
- ✅ Tìm checkpoint cuối cùng (vd: `ckpt_07.weights.h5`)
- ✅ Load weights
- ✅ Continue từ epoch 8 đến epoch 20

---

## 📊 THEO DÕI TRAINING

### **1. TensorBoard trong Colab:**

```python
# Load TensorBoard extension
%load_ext tensorboard

# Start TensorBoard
%tensorboard --logdir /content/drive/MyDrive/comparison_detector_models_v2/logs
```

### **2. Kiểm tra checkpoints:**

```python
!ls -lh /content/drive/MyDrive/comparison_detector_models_v2/checkpoints/
```

### **3. Kiểm tra model đã save:**

```python
import os
model_dir = '/content/drive/MyDrive/comparison_detector_models_v2'
print("📁 Saved files:")
for f in os.listdir(model_dir):
    path = os.path.join(model_dir, f)
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"  - {f}: {size_mb:.1f} MB")
```

---

## 💾 SAU KHI TRAINING XONG

### **1. Save kết quả cuối cùng:**

```python
# Cell: Post-Training - Save all artifacts
import json
import shutil
from pathlib import Path

# Paths
model_dir = Path('/content/drive/MyDrive/comparison_detector_models_v2')
archive_dir = model_dir / 'archive_2025_12_05'
archive_dir.mkdir(exist_ok=True)

# Copy models
print("📦 Archiving models...")
shutil.copy(model_dir / 'final_model.keras', archive_dir / 'final_model.keras')
shutil.copy(model_dir / 'best_model.h5', archive_dir / 'best_model.h5')

# Save training config
config_info = {
    'date': '2025-12-05',
    'epochs': 20,
    'batch_size': 2,
    'backbone': 'resnet50',
    'image_size': [640, 640],
    'num_classes': 12,
    'final_loss': float(history.history['loss'][-1]),
    'final_accuracy': float(history.history['accuracy'][-1])
}

with open(archive_dir / 'training_info.json', 'w') as f:
    json.dump(config_info, f, indent=2)

print(f"✅ Archived to: {archive_dir}")
```

---

### **2. Export model cho deployment:**

```python
# Cell: Export for deployment

# 1. TensorFlow SavedModel (cho FastAPI)
import tensorflow as tf

model = tf.keras.models.load_model('/content/drive/MyDrive/comparison_detector_models_v2/best_model.h5')
export_dir = '/content/drive/MyDrive/comparison_detector_models_v2/saved_model'
model.save(export_dir, save_format='tf')
print(f"✅ SavedModel exported to: {export_dir}")

# 2. TF Lite (cho mobile/edge devices - optional)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = '/content/drive/MyDrive/comparison_detector_models_v2/model_quantized.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ TFLite model exported to: {tflite_path}")

# 3. Check model size
import os
for name, path in [
    ('Final Keras', '/content/drive/MyDrive/comparison_detector_models_v2/final_model.keras'),
    ('Best H5', '/content/drive/MyDrive/comparison_detector_models_v2/best_model.h5'),
    ('TFLite', tflite_path)
]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"  {name}: {size_mb:.1f} MB")
```

---

### **3. Evaluate model:**

```python
# Cell: Evaluation
from data.loader_tf2 import build_dataset
import numpy as np

# Load test data
test_paths = ['/content/drive/MyDrive/content/data/tct/test.tfrecord']
test_ds = build_dataset(test_paths, image_size=(640, 640), batch_size=2)

# Map to labels
def extract_label(img, tgt):
    return img, tgt['labels'][:, 0]

test_ds = test_ds.map(extract_label).take(100)  # Take 100 batches

# Evaluate
results = model.evaluate(test_ds)
print(f"\n📊 Test Results:")
print(f"  Loss: {results[0]:.4f}")
print(f"  Accuracy: {results[1]:.4f}")

# Save results
metrics = {
    'test_loss': float(results[0]),
    'test_accuracy': float(results[1])
}

with open('/content/drive/MyDrive/comparison_detector_models_v2/test_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

---

### **4. Visualize predictions (sample):**

```python
# Cell: Visualize predictions
import matplotlib.pyplot as plt

# Get one batch
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(2):
        axes[i].imshow(images[i])
        pred_class = np.argmax(predictions[i])
        true_class = labels[i].numpy()
        axes[i].set_title(f'True: {true_class}, Pred: {pred_class}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/comparison_detector_models_v2/sample_predictions.png', dpi=150)
    plt.show()

print("✅ Saved sample predictions")
```

---

## 📥 DOWNLOAD VỀ MÁY LOCAL (WINDOWS)

### **Option 1: Download qua Google Drive UI**
1. Mở Google Drive
2. Vào folder `MyDrive/comparison_detector_models_v2/`
3. Download các file:
   - `best_model.h5` (hoặc `final_model.keras`)
   - `saved_model/` (cả folder)
   - `test_metrics.json`

### **Option 2: Download bằng code:**

```python
# Cell: Prepare download links
from google.colab import files

# Zip all models
!cd /content/drive/MyDrive/comparison_detector_models_v2 && \
  zip -r models_trained_2025_12_05.zip \
    best_model.h5 \
    final_model.keras \
    test_metrics.json \
    sample_predictions.png

# Download (nếu file nhỏ < 100MB)
# files.download('/content/drive/MyDrive/comparison_detector_models_v2/models_trained_2025_12_05.zip')

print("✅ Models zipped! Download from Drive:")
print("   /content/drive/MyDrive/comparison_detector_models_v2/models_trained_2025_12_05.zip")
```

---

## 🚀 SỬ DỤNG MODEL LOCAL (WINDOWS)

### **1. Setup local environment:**

```powershell
# PowerShell
cd d:\ComparisonDetector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### **2. Copy model về:**

```powershell
# Tạo folder models nếu chưa có
New-Item -ItemType Directory -Force -Path "d:\ComparisonDetector\trained_models"

# Copy từ Google Drive (sau khi download)
# Giả sử bạn download về Downloads folder
Copy-Item "$env:USERPROFILE\Downloads\best_model.h5" -Destination "d:\ComparisonDetector\trained_models\"
```

### **3. Test model:**

```python
# test_model.py
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('trained_models/best_model.h5')
print("✅ Model loaded!")

# Load test image
img = Image.open('test_image.jpg').convert('RGB')
img = img.resize((640, 640))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, 0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

Chạy:
```powershell
python test_model.py
```

---

## 🐛 TROUBLESHOOTING

### **Vấn đề 1: Training dừng giữa chừng**
**Giải pháp:** Chạy `resume_training.py`

### **Vấn đề 2: Colab disconnect**
**Giải pháp:** 
- Free tier: 12 hours max, cần resume
- Pro tier: 24 hours
- Sử dụng Colab Pro hoặc chia nhỏ training (10 epochs/lần)

### **Vấn đề 3: Out of memory**
**Giải pháp:** Giảm `BATCH_SIZE` trong `configs/config_v2.py`:
```python
BATCH_SIZE = 1  # Thay vì 2
```

### **Vấn đề 4: Dataset không tìm thấy**
**Giải pháp:** Check path trong config:
```python
# configs/config_v2.py
DATA_DIR = "/content/drive/MyDrive/content/data/tct"  # Đúng path
```

---

## ✅ CHECKLIST SAU KHI TRAIN XONG

```
[ ] Training hoàn thành 20 epochs
[ ] Best model saved (best_model.h5)
[ ] Final model saved (final_model.keras)
[ ] TensorBoard logs có đầy đủ
[ ] Test metrics calculated
[ ] Sample predictions visualized
[ ] Models exported (SavedModel, TFLite)
[ ] Models archived with config
[ ] Models downloaded về local
[ ] Test model trên local works
```

---

**Bạn ready để train lại chưa?** 🚀

**Lệnh chạy lại trên Colab:**
```python
!python train_keras.py  # Train từ đầu
# HOẶC
!python resume_training.py  # Resume từ checkpoint
```
