# 🔍 KIỂM TRA TRẠNG THÁI GIAI ĐOẠN 1

**Ngày kiểm tra:** December 3, 2025  
**Mục tiêu:** mAP 26.3% → 50-60%  
**Thời gian dự kiến:** 3 tháng

---

## 📊 TỔNG KẾT TRẠNG THÁI

### ✅ ĐÃ CÓ (COMPLETED)

| Component | Status | Notes |
|-----------|--------|-------|
| **Data Pipeline** | ✅ | TFRecord parser working, padding implemented |
| **Backbone** | ✅ | ResNet50/101 via Keras Applications |
| **FPN** | ✅ | Basic FPN implemented |
| **RPN** | ✅ | Region Proposal Network ready |
| **Detector** | ✅ | Main detector model functional |
| **Training Script** | ✅ | train_keras.py working on Colab |
| **Basic Losses** | ✅ | RPN + RCNN losses implemented |

### ⚠️ CẦN CẢI THIỆN (NEEDS IMPROVEMENT)

| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| **Data Augmentation** | ❌ None | Advanced (Albumentations) | 🔴 HIGH |
| **Loss Functions** | ⚠️ Basic | Focal Loss + GIoU | 🔴 HIGH |
| **Backbone** | ⚠️ ResNet50 | EfficientNetB4/V2 | 🟡 MEDIUM |
| **FPN** | ⚠️ Basic | BiFPN/PANet | 🟡 MEDIUM |
| **Learning Rate** | ⚠️ Fixed | Cosine + Warmup | 🔴 HIGH |
| **Optimizer** | ⚠️ Adam | AdamW | 🟢 LOW |
| **Data Quality** | ❓ Unknown | Audit needed | 🔴 HIGH |
| **Class Balance** | ❓ Unknown | Check needed | 🔴 HIGH |

---

## 📁 CHI TIẾT CÁC FILE

### 1. **config.py & configs/config_v2.py**

#### ✅ Đã có:
```python
# configs/config_v2.py
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-3
BACKBONE = "resnet50"
IMAGE_SIZE = (640, 640)
NUM_CLASSES = 12
```

#### ❌ Thiếu:
```python
# Cần thêm:
USE_AUGMENTATION = True
USE_FOCAL_LOSS = True
USE_GIOU_LOSS = True
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-4

# Learning rate schedule
USE_WARMUP = True
WARMUP_STEPS = 500
USE_COSINE_DECAY = True
MIN_LR = 1e-6

# Data quality
DATA_AUDIT_REQUIRED = True
CLASS_BALANCE_THRESHOLD = 3.0  # Max imbalance ratio
```

---

### 2. **data/loader_tf2.py**

#### ✅ Đã có:
- ✅ TFRecord parsing (raw bytes format)
- ✅ Box padding (max 100 boxes)
- ✅ Basic preprocessing (resize, normalize)
- ✅ Batch creation

#### ❌ Thiếu:
- ❌ **Data augmentation** (CRITICAL!)
- ❌ Multi-scale training
- ❌ Copy-paste augmentation
- ❌ Mosaic augmentation

#### 📝 Cần implement:
```python
# Thêm vào loader_tf2.py:

def augment_advanced(image, boxes, labels):
    """Advanced augmentation using Albumentations"""
    import albumentations as A
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.5),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3
    ))
    
    # Convert TF tensors to numpy
    img_np = image.numpy()
    boxes_np = boxes.numpy()
    labels_np = labels.numpy()
    
    # Apply augmentation
    augmented = transform(image=img_np, bboxes=boxes_np, labels=labels_np)
    
    return augmented['image'], augmented['bboxes'], augmented['labels']
```

---

### 3. **losses_tf2.py**

#### ✅ Đã có:
```python
def rpn_objectness_loss(logits, labels):
    """Binary cross-entropy for RPN"""
    
def rpn_bbox_loss(pred_deltas, target_deltas, weights=None):
    """Smooth L1 loss for RPN boxes"""
    
def rcnn_cls_loss(logits, labels):
    """Sparse categorical cross-entropy for classification"""
    
def rcnn_bbox_loss(pred_deltas, target_deltas):
    """Smooth L1 loss for RCNN boxes"""
```

#### ❌ Thiếu:
- ❌ **Focal Loss** (for class imbalance)
- ❌ **GIoU Loss** (better box regression)
- ❌ **Quality Focal Loss** (advanced)

#### 📝 Cần thêm vào losses_tf2.py:

```python
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for imbalanced classification
    
    Paper: RetinaNet (ICCV 2017)
    Improvement: +3-5% mAP
    """
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # Convert to one-hot
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(y_true, num_classes)
    
    # Cross-entropy
    ce = -y_true_one_hot * tf.math.log(y_pred)
    
    # Focal term
    p_t = y_true_one_hot * y_pred + (1 - y_true_one_hot) * (1 - y_pred)
    focal_weight = tf.pow(1 - p_t, gamma)
    
    # Alpha weighting
    alpha_t = y_true_one_hot * alpha + (1 - y_true_one_hot) * (1 - alpha)
    
    loss = alpha_t * focal_weight * ce
    
    return tf.reduce_sum(loss, axis=-1)


def giou_loss(pred_boxes, target_boxes):
    """
    Generalized IoU Loss
    
    Paper: GIoU (CVPR 2019)
    Improvement: +2-3% mAP vs Smooth L1
    """
    # pred_boxes, target_boxes: [N, 4] in format [y1, x1, y2, x2]
    
    # Intersection
    y1_inter = tf.maximum(pred_boxes[:, 0], target_boxes[:, 0])
    x1_inter = tf.maximum(pred_boxes[:, 1], target_boxes[:, 1])
    y2_inter = tf.minimum(pred_boxes[:, 2], target_boxes[:, 2])
    x2_inter = tf.minimum(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = tf.maximum(0., y2_inter - y1_inter) * \
                 tf.maximum(0., x2_inter - x1_inter)
    
    # Union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    # Smallest enclosing box
    y1_enclosing = tf.minimum(pred_boxes[:, 0], target_boxes[:, 0])
    x1_enclosing = tf.minimum(pred_boxes[:, 1], target_boxes[:, 1])
    y2_enclosing = tf.maximum(pred_boxes[:, 2], target_boxes[:, 2])
    x2_enclosing = tf.maximum(pred_boxes[:, 3], target_boxes[:, 3])
    
    enclosing_area = (y2_enclosing - y1_enclosing) * \
                     (x2_enclosing - x1_enclosing)
    
    # GIoU
    giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-7)
    
    # Loss = 1 - GIoU
    loss = 1.0 - giou
    
    return tf.reduce_mean(loss)
```

---

### 4. **models/detector.py**

#### ✅ Đã có:
- ✅ Backbone integration
- ✅ FPN
- ✅ RPN
- ✅ Basic inference

#### ❌ Thiếu:
- ❌ Dropout layers
- ❌ Batch normalization
- ❌ Better ROI Align
- ❌ Proper training mode

#### 📝 Cần update:
```python
class DetectorV2(tf.keras.Model):
    def __init__(self, num_classes, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        
        # Existing components
        self.backbone = build_backbone('resnet50', 'imagenet')
        self.fpn = FPN(channels=256)
        self.rpn = RPN(channels=256, num_anchors=9)
        
        # ADD: Regularization
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bn = tf.keras.layers.BatchNormalization()
        
        # ADD: Better classification head
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
```

---

### 5. **train_keras.py**

#### ✅ Đã có:
- ✅ Basic training loop
- ✅ TensorBoard logging
- ✅ Model saving

#### ❌ Thiếu:
- ❌ **Learning rate schedule** (CRITICAL!)
- ❌ **AdamW optimizer**
- ❌ **Mixed precision training**
- ❌ **Early stopping with mAP**
- ❌ **Gradient clipping**

#### 📝 Cần update:
```python
# train_keras.py - UPDATE

# 1. Learning Rate Schedule
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=1e-3, warmup_steps=500, 
                 total_steps=10000, min_lr=1e-6):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        # Warmup
        warmup_lr = self.initial_lr * step / self.warmup_steps
        
        # Cosine decay
        decay_steps = self.total_steps - self.warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(
            np.pi * (step - self.warmup_steps) / decay_steps
        ))
        decayed_lr = (self.initial_lr - self.min_lr) * cosine_decay + self.min_lr
        
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: decayed_lr
        )

# 2. AdamW Optimizer
from tensorflow_addons.optimizers import AdamW

total_steps = cfg.EPOCHS * steps_per_epoch
lr_schedule = WarmupCosineDecay(
    initial_lr=1e-3,
    warmup_steps=500,
    total_steps=total_steps,
    min_lr=1e-6
)

optimizer = AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-4
)

# 3. Mixed Precision (faster training)
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 4. Gradient Clipping
@tf.function
def train_step(model, images, targets, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(predictions, targets)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Clip gradients
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss
```

---

## 🔧 CÔNG CỤ KIỂM TRA DỮ LIỆU

### Tool 1: Dataset Audit

```python
# tools/audit_dataset.py - TẠO MỚI

import tensorflow as tf
import numpy as np
from collections import defaultdict
import json

def audit_dataset(tfrecord_path):
    """Comprehensive dataset audit"""
    
    stats = {
        'total_images': 0,
        'total_boxes': 0,
        'class_distribution': defaultdict(int),
        'box_sizes': [],
        'images_per_num_boxes': defaultdict(int),
        'issues': {
            'empty_images': [],
            'tiny_boxes': [],
            'huge_boxes': [],
            'invalid_boxes': []
        }
    }
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    for idx, record in enumerate(dataset):
        # Parse
        features = {
            'img': tf.io.FixedLenFeature([], tf.string),
            'img_height': tf.io.FixedLenFeature([], tf.int64),
            'img_width': tf.io.FixedLenFeature([], tf.int64),
            'gtboxes_and_label': tf.io.FixedLenFeature([], tf.string),
        }
        parsed = tf.io.parse_single_example(record, features)
        
        # Decode boxes
        gtboxes_and_label = tf.io.decode_raw(parsed['gtboxes_and_label'], tf.int32)
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])
        
        boxes = gtboxes_and_label[:, :4].numpy()
        labels = gtboxes_and_label[:, 4].numpy()
        
        height = int(parsed['img_height'].numpy())
        width = int(parsed['img_width'].numpy())
        
        stats['total_images'] += 1
        num_boxes = len(boxes)
        stats['total_boxes'] += num_boxes
        stats['images_per_num_boxes'][num_boxes] += 1
        
        # Empty images
        if num_boxes == 0:
            stats['issues']['empty_images'].append(idx)
        
        # Check each box
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # Class distribution
            stats['class_distribution'][int(label)] += 1
            
            # Box size
            stats['box_sizes'].append((w, h))
            
            # Tiny boxes
            if w < 10 or h < 10:
                stats['issues']['tiny_boxes'].append((idx, i, box))
            
            # Huge boxes
            if w > width * 0.8 or h > height * 0.8:
                stats['issues']['huge_boxes'].append((idx, i, box))
            
            # Invalid boxes
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                stats['issues']['invalid_boxes'].append((idx, i, box))
    
    # Calculate metrics
    avg_boxes_per_image = stats['total_boxes'] / stats['total_images']
    
    # Class imbalance
    class_counts = list(stats['class_distribution'].values())
    if class_counts:
        imbalance_ratio = max(class_counts) / min(class_counts)
    else:
        imbalance_ratio = 0
    
    # Print report
    print("="*60)
    print("DATASET AUDIT REPORT")
    print("="*60)
    print(f"\n📊 BASIC STATS:")
    print(f"Total images: {stats['total_images']}")
    print(f"Total boxes: {stats['total_boxes']}")
    print(f"Avg boxes/image: {avg_boxes_per_image:.2f}")
    
    print(f"\n📦 CLASS DISTRIBUTION:")
    total = sum(stats['class_distribution'].values())
    for class_id in sorted(stats['class_distribution'].keys()):
        count = stats['class_distribution'][class_id]
        pct = count / total * 100
        print(f"  Class {class_id:2d}: {count:5d} ({pct:5.2f}%)")
    
    print(f"\n⚖️ CLASS IMBALANCE:")
    print(f"  Ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 5:
        print(f"  ⚠️ SEVERE IMBALANCE - Need balancing!")
    elif imbalance_ratio > 3:
        print(f"  ⚠️ MODERATE IMBALANCE - Consider balancing")
    else:
        print(f"  ✅ GOOD balance")
    
    print(f"\n⚠️ ISSUES:")
    for issue_type, items in stats['issues'].items():
        if items:
            print(f"  {issue_type}: {len(items)}")
    
    # Box size distribution
    if stats['box_sizes']:
        box_sizes = np.array(stats['box_sizes'])
        print(f"\n📏 BOX SIZE STATS:")
        print(f"  Width:  Mean={box_sizes[:,0].mean():.1f}, Std={box_sizes[:,0].std():.1f}")
        print(f"  Height: Mean={box_sizes[:,1].mean():.1f}, Std={box_sizes[:,1].std():.1f}")
        print(f"  Min size: {box_sizes.min():.0f}")
        print(f"  Max size: {box_sizes.max():.0f}")
    
    # Save report
    with open('dataset_audit_report.json', 'w') as f:
        # Convert defaultdict to dict for JSON
        stats_json = {
            'total_images': stats['total_images'],
            'total_boxes': stats['total_boxes'],
            'avg_boxes_per_image': float(avg_boxes_per_image),
            'class_distribution': dict(stats['class_distribution']),
            'imbalance_ratio': float(imbalance_ratio),
            'issues': {k: len(v) for k, v in stats['issues'].items()}
        }
        json.dump(stats_json, f, indent=2)
    
    print(f"\n✅ Report saved to dataset_audit_report.json")
    
    return stats

# Run audit
if __name__ == '__main__':
    train_stats = audit_dataset('tfdata/tct/train.tfrecord')
    test_stats = audit_dataset('tfdata/tct/test.tfrecord')
```

---

## 📋 ACTION PLAN - TUẦN 1-2

### Week 1: Data Audit & Quality

```bash
# DAY 1-2: Kiểm tra dữ liệu
cd d:\ComparisonDetector
python tools/audit_dataset.py

# Output expected:
# - dataset_audit_report.json
# - Identify class imbalance
# - Find data quality issues

# DAY 3-4: Install dependencies
pip install albumentations
pip install tensorflow-addons

# DAY 5: Implement advanced augmentation
# - Create data/augmentation_advanced.py
# - Update loader_tf2.py to use augmentation
```

### Week 2: Loss Functions & Training

```bash
# DAY 1-2: Update losses
# - Add focal_loss() to losses_tf2.py
# - Add giou_loss() to losses_tf2.py

# DAY 3-4: Update training
# - Add WarmupCosineDecay schedule
# - Switch to AdamW optimizer
# - Add gradient clipping

# DAY 5: Test training
# - Run smoke test (10 epochs)
# - Verify improvements
# - Check TensorBoard metrics
```

---

## 📊 EXPECTED IMPROVEMENTS

### Quick Wins (Week 1-2):

| Change | Expected Gain | Difficulty |
|--------|--------------|------------|
| Advanced Augmentation | +5-8% | Easy |
| Focal Loss | +2-3% | Easy |
| GIoU Loss | +2-3% | Easy |
| LR Schedule | +2-3% | Easy |
| AdamW | +1-2% | Easy |
| **TOTAL** | **+12-19%** | **2 weeks** |

### Target:
```
Current: 26.3% mAP
After Week 2: 38-45% mAP (conservative)
Goal for Month 3: 50-60% mAP
```

---

## ✅ CHECKLIST GIAI ĐOẠN 1

### 🔴 HIGH PRIORITY (Week 1-2)

```
[ ] Run dataset audit
[ ] Install albumentations
[ ] Install tensorflow-addons
[ ] Create data/augmentation_advanced.py
[ ] Update loader_tf2.py with augmentation
[ ] Add focal_loss to losses_tf2.py
[ ] Add giou_loss to losses_tf2.py
[ ] Create WarmupCosineDecay schedule
[ ] Switch to AdamW optimizer
[ ] Test training (10 epochs)
```

### 🟡 MEDIUM PRIORITY (Week 3-4)

```
[ ] Upgrade backbone to EfficientNetB4
[ ] Implement BiFPN (optional)
[ ] Add dropout + batch norm to detector
[ ] Implement mixed precision training
[ ] Add early stopping with mAP metric
```

### 🟢 LOW PRIORITY (Optional)

```
[ ] Multi-scale training
[ ] Copy-paste augmentation
[ ] Test-time augmentation
[ ] Model distillation
```

---

## 🚀 GETTING STARTED

### Run these commands NOW:

```bash
# 1. Check current environment
cd d:\ComparisonDetector
python --version
python -c "import tensorflow as tf; print(tf.__version__)"

# 2. Install missing packages
pip install albumentations==1.3.1
pip install tensorflow-addons==0.22.0

# 3. Run dataset audit
python -c "
import tensorflow as tf
from collections import defaultdict

dataset = tf.data.TFRecordDataset('tfdata/tct/train.tfrecord')
count = 0
for _ in dataset:
    count += 1
    if count > 10:
        break

print(f'✓ Can read TFRecord: {count} samples')
"

# 4. Check GPU
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')
for gpu in gpus:
    print(f'  - {gpu}')
"
```

---

## 📞 NEXT STEPS

**SAU KHI CHẠY AUDIT:**

1. Share `dataset_audit_report.json` với tôi
2. Tôi sẽ phân tích chi tiết:
   - Class imbalance cụ thể
   - Data quality issues
   - Recommended fixes

3. Tôi sẽ tạo code cụ thể cho:
   - Augmentation pipeline phù hợp
   - Class balancing strategy
   - Training configuration

**Bạn sẵn sàng chạy audit chưa?** 🚀
