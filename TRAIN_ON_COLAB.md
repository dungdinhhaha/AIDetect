# 🚀 Hướng dẫn Training ComparisonDetector trên Google Colab

## Bước 1: Mở Google Colab

1. Truy cập: https://colab.research.google.com/
2. Tạo notebook mới: File → New notebook
3. Chọn GPU Runtime: Runtime → Change runtime type → GPU (T4 hoặc cao hơn)

---

## Bước 2: Kiểm tra GPU

```python
# Cell 1: Kiểm tra GPU
!nvidia-smi

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
```

---

## Bước 3: Mount Google Drive và Clone Repository

```python
# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted")
```

```python
# Cell 3: Clone repository
!git clone https://github.com/dungdinhhaha/AIDetect.git /content/ComparisonDetector
%cd /content/ComparisonDetector
!git pull
!ls -la
```

---

## Bước 4: Chuẩn bị dữ liệu

### Option A: Upload TFRecord từ Google Drive

```python
# Cell 4A: Copy TFRecord từ Drive
!mkdir -p /content/data/tct

# THAY ĐỔI đường dẫn này theo vị trí data của bạn trên Drive
!cp /content/drive/MyDrive/TCT_Data/*.tfrecord /content/data/tct/ 2>/dev/null || echo "No TFRecords found"

# Kiểm tra
!ls -lh /content/data/tct/
```

### Option B: Tạo Dummy Data để Test

```python
# Cell 4B: Tạo dummy TFRecord
import tensorflow as tf
import numpy as np

def create_dummy_tfrecord(output_path, num_samples=100):
    """Tạo TFRecord giả với image và bounding boxes"""
    with tf.io.TFRecordWriter(output_path) as writer:
        for i in range(num_samples):
            # Dummy image (640x640x3)
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_bytes = tf.io.encode_jpeg(img).numpy()
            
            # Dummy boxes và labels
            num_boxes = np.random.randint(1, 5)
            boxes = np.random.rand(num_boxes, 4).astype(np.float32)
            labels = np.random.randint(1, 12, num_boxes, dtype=np.int64)
            
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                'boxes': tf.train.Feature(float_list=tf.train.FloatList(value=boxes.flatten())),
                'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
                'num_boxes': tf.train.Feature(int64_list=tf.train.Int64List(value=[num_boxes])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    print(f"✓ Created {output_path} with {num_samples} samples")

# Tạo train và test TFRecord
!mkdir -p /content/data/tct
create_dummy_tfrecord('/content/data/tct/train.tfrecord', num_samples=500)
create_dummy_tfrecord('/content/data/tct/test.tfrecord', num_samples=100)

!ls -lh /content/data/tct/
```

---

## Bước 5: Xem và Tùy chỉnh Config

```python
# Cell 5: Xem config hiện tại
from configs.config_v2 import ConfigV2

cfg = ConfigV2()
print("📋 Current Configuration:")
print(f"  Data Dir: {cfg.DATA_DIR}")
print(f"  Model Dir: {cfg.MODEL_DIR}")
print(f"  Batch Size: {cfg.BATCH_SIZE}")
print(f"  Epochs: {cfg.EPOCHS}")
print(f"  Learning Rate: {cfg.LEARNING_RATE}")
print(f"  Backbone: {cfg.BACKBONE}")
print(f"  Image Size: {cfg.IMAGE_SIZE}")
print(f"  Num Classes: {cfg.NUM_CLASSES}")
```

```python
# Cell 6: Override config (optional)
cfg.BATCH_SIZE = 1  # Giảm batch size nếu GPU nhỏ
cfg.EPOCHS = 10     # Số epochs muốn train
cfg.LEARNING_RATE = 5e-4  # Fine-tune learning rate

print("✓ Config updated for Colab")
```

---

## Bước 6: Test Data Pipeline và Model

```python
# Cell 7: Test data loader
import os
from data.loader_tf2 import build_dataset

tfrecord_paths = tf.io.gfile.glob(os.path.join(cfg.DATA_DIR, '*.tfrecord'))
print(f"Found {len(tfrecord_paths)} TFRecord files")

if tfrecord_paths:
    ds = build_dataset(tfrecord_paths, image_size=cfg.IMAGE_SIZE, batch_size=cfg.BATCH_SIZE)
    
    for images, targets in ds.take(1):
        print(f"✓ Data pipeline working")
        print(f"  Images shape: {images.shape}")
        print(f"  Targets keys: {targets.keys()}")
else:
    print("⚠ No TFRecords found - will use dummy dataset")
```

```python
# Cell 8: Test model architecture
from models.detector import ComparisonDetector

detector = ComparisonDetector(
    num_classes=cfg.NUM_CLASSES,
    backbone_name=cfg.BACKBONE,
    backbone_weights=cfg.BACKBONE_WEIGHTS
)

dummy_input = tf.random.uniform((1, 640, 640, 3))
boxes, scores = detector(dummy_input, training=False)

print(f"✓ Model test passed")
print(f"  Output boxes: {boxes.shape}")
print(f"  Output scores: {scores.shape}")
```

---

## Bước 7: Bắt đầu Training 🚀

### Option 1: Chạy script training

```python
# Cell 9A: Chạy train_keras.py
!python train_keras.py
```

### Option 2: Train trực tiếp trong notebook

```python
# Cell 9B: Train trong notebook
import os
import tensorflow as tf
from tensorflow.keras import optimizers
from configs.config_v2 import ConfigV2
from data.loader_tf2 import build_dataset
from models.backbone_keras import build_backbone

# Config
cfg = ConfigV2()
os.makedirs(cfg.MODEL_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR, exist_ok=True)
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

# Distribution strategy
strategy = tf.distribute.MirroredStrategy() if cfg.USE_DISTRIBUTE else tf.distribute.get_strategy()

with strategy.scope():
    # Build model
    backbone = build_backbone(cfg.BACKBONE, cfg.BACKBONE_WEIGHTS)
    inputs = backbone.input
    features = backbone(inputs)[-1]
    x = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='comparison_detector_v2')
    
    # Compile
    opt = optimizers.SGD(learning_rate=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dataset
tfrecord_paths = tf.io.gfile.glob(os.path.join(cfg.DATA_DIR, '*.tfrecord'))
if not tfrecord_paths:
    print('⚠ No TFRecords found, using dummy dataset...')
    dummy_images = tf.random.uniform((cfg.BATCH_SIZE, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], 3))
    dummy_labels = tf.random.uniform((cfg.BATCH_SIZE,), minval=0, maxval=cfg.NUM_CLASSES, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels)).repeat().batch(cfg.BATCH_SIZE)
else:
    ds = build_dataset(tfrecord_paths, image_size=cfg.IMAGE_SIZE, batch_size=cfg.BATCH_SIZE)
    ds = ds.map(lambda img, tgt: (img, tf.zeros((), dtype=tf.int32))).repeat()

# Callbacks
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(cfg.CHECKPOINT_DIR, 'ckpt_{epoch:02d}.weights.h5'),
    save_weights_only=True,
    save_freq='epoch'
)
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=cfg.LOG_DIR)

# Train
print("\n🚀 Starting training...\n")
history = model.fit(
    ds, 
    epochs=cfg.EPOCHS, 
    steps_per_epoch=100,  # Điều chỉnh theo dataset size
    callbacks=[ckpt_cb, tb_cb]
)

# Save
model.save(os.path.join(cfg.MODEL_DIR, 'model.keras'))
print('\n✅ Training completed!')
print(f'Model saved to: {cfg.MODEL_DIR}/model.keras')
```

---

## Bước 8: TensorBoard Monitoring (Optional)

```python
# Cell 10: Load TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/comparison_detector_models_v2/logs
```

---

## Bước 9: Đánh giá Model

```python
# Cell 11: Load và test model
model_path = os.path.join(cfg.MODEL_DIR, 'model.keras')
if os.path.exists(model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Test inference
    test_img = tf.random.uniform((1, 640, 640, 3))
    pred = loaded_model(test_img, training=False)
    print(f"  Prediction shape: {pred.shape}")
    print(f"  Predicted class: {tf.argmax(pred[0]).numpy()}")
else:
    print(f"⚠ Model not found at {model_path}")
```

---

## Bước 10: Download Model về Local

```python
# Cell 12: Zip và download
!cd /content/drive/MyDrive && zip -r comparison_detector_models_v2.zip comparison_detector_models_v2/

from google.colab import files
files.download('/content/drive/MyDrive/comparison_detector_models_v2.zip')
```

---

## 📝 Troubleshooting

### ❌ Out of Memory
```python
# Giảm batch size và image size
cfg.BATCH_SIZE = 1
cfg.IMAGE_SIZE = (512, 512)
cfg.BACKBONE = 'resnet50'  # Thay vì resnet101
```

### ❌ No TFRecords Found
- Kiểm tra đường dẫn trong Cell 4A
- Hoặc dùng Option B để tạo dummy data

### ❌ Model Not Converging
```python
# Giảm learning rate
cfg.LEARNING_RATE = 1e-4

# Xem TensorBoard để debug
%tensorboard --logdir /content/drive/MyDrive/comparison_detector_models_v2/logs
```

### ❌ GPU Disconnected
- Colab free có giới hạn thời gian
- Lưu checkpoint thường xuyên
- Nâng cấp Colab Pro nếu cần train lâu

---

## 💾 Model được lưu tại:

- **Google Drive**: `/content/drive/MyDrive/comparison_detector_models_v2/`
- **Checkpoints**: Mỗi epoch tại `/checkpoints/`
- **TensorBoard logs**: `/logs/`
- **Final model**: `model.keras`

---

## 📊 Theo dõi Training:

1. **Terminal output**: Loss và accuracy mỗi epoch
2. **TensorBoard**: Graphs chi tiết
3. **Google Drive**: Auto-save checkpoints

---

**Repository**: https://github.com/dungdinhhaha/AIDetect  
**Author**: dungdinhhaha | dungdinh542004@gmail.com  
**Email**: dungdinh542004@gmail.com

---

## 🎯 Quick Start (Copy-Paste tất cả):

```python
# Setup nhanh - paste tất cả vào 1 cell
!git clone https://github.com/dungdinhhaha/AIDetect.git /content/ComparisonDetector
%cd /content/ComparisonDetector

from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/data/tct
# Uncomment dòng dưới nếu có data thật
# !cp /content/drive/MyDrive/TCT_Data/*.tfrecord /content/data/tct/

# Train
!python train_keras.py
```

Copy code trên vào Google Colab và chạy để bắt đầu training ngay! 🚀
