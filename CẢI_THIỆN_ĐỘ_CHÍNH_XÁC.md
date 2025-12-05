# 🎯 CÁCH CẢI THIỆN ĐỘ CHÍNH XÁC DỰ ÁN CELL DETECTION

**Dự án hiện tại:** ComparisonDetector (Faster R-CNN + ResNet50/101 + FPN)  
**Accuracy hiện tại:** ~27.91% (smoke test), ~50-70% (expected after full training)  
**Mục tiêu:** 85-95% mAP

---

## 📋 MỤC LỤC

1. [Phân tích nguyên nhân độ chính xác thấp](#1-phân-tích-nguyên-nhân)
2. [Cải thiện dữ liệu](#2-cải-thiện-dữ-liệu-data-centric-approach)
3. [Cải thiện model](#3-cải-thiện-model-model-centric-approach)
4. [Tối ưu hyperparameters](#4-tối-ưu-hyperparameters)
5. [Advanced techniques](#5-advanced-techniques)
6. [Ensemble methods](#6-ensemble-methods)
7. [Domain-specific improvements](#7-domain-specific-improvements-medical-imaging)
8. [Debugging & monitoring](#8-debugging--monitoring)

---

# 1. PHÂN TÍCH NGUYÊN NHÂN

## 1.1. Checklist chẩn đoán

```python
# Script: tools/diagnose_accuracy.py

import tensorflow as tf
import numpy as np
from utils.visualize import plot_predictions
from config import Config

cfg = Config()

def diagnose_model(model, test_dataset):
    """Chẩn đoán vấn đề của model"""
    
    print("=== DIAGNOSTIC REPORT ===\n")
    
    # 1. Check overfitting/underfitting
    train_loss, train_acc = model.evaluate(train_dataset)
    val_loss, val_acc = model.evaluate(val_dataset)
    
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    if train_acc > val_acc + 0.1:
        print("⚠️ OVERFITTING detected!")
        print("Solutions: Regularization, Dropout, Data augmentation")
    elif train_acc < 0.5:
        print("⚠️ UNDERFITTING detected!")
        print("Solutions: Larger model, More epochs, Better features")
    
    # 2. Class imbalance
    class_counts = count_classes(test_dataset)
    print(f"\nClass distribution: {class_counts}")
    
    imbalance_ratio = max(class_counts) / min(class_counts)
    if imbalance_ratio > 5:
        print(f"⚠️ CLASS IMBALANCE: {imbalance_ratio:.2f}x")
        print("Solutions: Focal loss, Class weights, Oversampling")
    
    # 3. Per-class accuracy
    per_class_acc = evaluate_per_class(model, test_dataset)
    print(f"\nPer-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"  Class {cfg.CLASS_NAMES[i]}: {acc:.4f}")
        if acc < 0.3:
            print(f"    ⚠️ Low accuracy! Check training samples")
    
    # 4. Common errors
    analyze_errors(model, test_dataset)
    
    # 5. Data quality
    check_data_quality(test_dataset)
    
    return {
        'overfitting': train_acc > val_acc + 0.1,
        'underfitting': train_acc < 0.5,
        'class_imbalance': imbalance_ratio > 5,
        'low_quality_data': False  # Implement check
    }

def analyze_errors(model, dataset):
    """Phân tích lỗi phổ biến"""
    
    print("\n=== ERROR ANALYSIS ===")
    
    false_positives = []
    false_negatives = []
    misclassifications = []
    
    for images, targets in dataset.take(100):
        boxes, labels, scores = model(images, training=False)
        
        # Compare with ground truth
        for i in range(len(images)):
            gt_boxes = targets['boxes'][i]
            gt_labels = targets['labels'][i]
            
            pred_boxes = boxes[i]
            pred_labels = labels[i]
            
            # Calculate IoU
            ious = calculate_iou_matrix(pred_boxes, gt_boxes)
            
            # Find matches
            for j, iou in enumerate(ious):
                if iou.max() < 0.5:
                    false_positives.append({
                        'box': pred_boxes[j],
                        'label': pred_labels[j],
                        'score': scores[i][j]
                    })
                elif pred_labels[j] != gt_labels[iou.argmax()]:
                    misclassifications.append({
                        'predicted': pred_labels[j],
                        'actual': gt_labels[iou.argmax()]
                    })
            
            # Check for missed detections
            for k, gt_box in enumerate(gt_boxes):
                if ious[:, k].max() < 0.5:
                    false_negatives.append({
                        'box': gt_box,
                        'label': gt_labels[k]
                    })
    
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    print(f"Misclassifications: {len(misclassifications)}")
    
    # Analyze patterns
    if len(misclassifications) > 0:
        confusion = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
        for mc in misclassifications:
            confusion[mc['actual'], mc['predicted']] += 1
        
        print("\nMost common misclassifications:")
        top_errors = np.argsort(confusion.flatten())[-5:]
        for idx in top_errors:
            i, j = np.unravel_index(idx, confusion.shape)
            if confusion[i, j] > 0:
                print(f"  {cfg.CLASS_NAMES[i]} → {cfg.CLASS_NAMES[j]}: {int(confusion[i, j])} times")

# Run diagnosis
diagnose_model(model, test_dataset)
```

---

# 2. CẢI THIỆN DỮ LIỆU (DATA-CENTRIC APPROACH)

## 2.1. Thu thập thêm dữ liệu

### ✅ **Priority actions:**

```
[ ] Thu thập thêm 500-1000 ảnh
[ ] Tập trung vào classes có accuracy thấp
[ ] Đảm bảo diversity (lighting, angle, zoom levels)
[ ] Add hard negatives (ảnh tương tự nhưng không phải target)
```

### 📊 **Data distribution target:**

```python
# Mục tiêu cân bằng classes
target_distribution = {
    'class_1': 1000 samples,
    'class_2': 1000 samples,
    'class_3': 1000 samples,
    # ...
    'background': 2000 samples  # Hard negatives
}

# Script: tools/balance_dataset.py
def balance_classes(dataset_dir, target_per_class=1000):
    """Balance dataset by oversampling minority classes"""
    
    class_counts = count_samples_per_class(dataset_dir)
    
    for class_name, count in class_counts.items():
        if count < target_per_class:
            shortage = target_per_class - count
            print(f"{class_name}: Need {shortage} more samples")
            
            # Oversample with augmentation
            augment_class(class_name, shortage)
```

---

## 2.2. Data Augmentation nâng cao

### 🔄 **Current augmentation:**
```python
# data/loader_tf2.py - HIỆN TẠI (CƠ BẢN)
- Flip horizontal
- Brightness adjustment
- Resize
```

### 🚀 **Advanced augmentation:**

```python
# data/augmentation_advanced.py

import albumentations as A
import cv2

def get_training_augmentation():
    """Advanced augmentation pipeline"""
    
    return A.Compose([
        # 1. Geometric transforms
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.RandomResizedCrop(
            height=640,
            width=640,
            scale=(0.8, 1.0),
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        
        # 2. Color transforms (quan trọng cho medical imaging)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        A.CLAHE(  # Contrast Limited Adaptive Histogram Equalization
            clip_limit=4.0,
            tile_grid_size=(8, 8),
            p=0.5
        ),
        
        # 3. Noise & Blur (giống real-world conditions)
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.2),
        
        # 4. Medical-specific
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        
        # 5. Cutout / Coarse Dropout (regularization)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.3
        ),
        
    ], bbox_params=A.BboxParams(
        format='pascal_voc',  # [x_min, y_min, x_max, y_max]
        label_fields=['labels'],
        min_visibility=0.3  # Drop boxes if <30% visible after augmentation
    ))

def augment_image(image, boxes, labels):
    """Apply augmentation"""
    
    augmentation = get_training_augmentation()
    
    augmented = augmentation(
        image=image,
        bboxes=boxes,
        labels=labels
    )
    
    return augmented['image'], augmented['bboxes'], augmented['labels']

# Cài đặt:
# pip install albumentations
```

### 📦 **Update loader:**

```python
# data/loader_tf2.py - THÊM VÀO

def augment_with_albumentations(image, boxes, labels):
    """Wrapper for Albumentations"""
    
    def aug_fn(img, boxes, labels):
        import albumentations as A
        from data.augmentation_advanced import augment_image
        
        # Convert to numpy
        img_np = img.numpy().astype(np.uint8)
        boxes_np = boxes.numpy()
        labels_np = labels.numpy()
        
        # Apply augmentation
        img_aug, boxes_aug, labels_aug = augment_image(
            img_np, boxes_np, labels_np
        )
        
        return img_aug, boxes_aug, labels_aug
    
    # Wrap in tf.py_function
    img_aug, boxes_aug, labels_aug = tf.py_function(
        aug_fn,
        [image, boxes, labels],
        [tf.uint8, tf.float32, tf.int32]
    )
    
    return img_aug, boxes_aug, labels_aug
```

---

## 2.3. Data quality improvement

### 🔍 **Quality checks:**

```python
# tools/check_data_quality.py

def check_annotation_quality(tfrecord_path):
    """Check for annotation issues"""
    
    issues = {
        'empty_boxes': [],
        'invalid_boxes': [],
        'tiny_boxes': [],
        'duplicate_boxes': [],
        'out_of_bounds': []
    }
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    for idx, record in enumerate(dataset):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        
        # Parse boxes
        boxes_and_labels = np.frombuffer(
            example.features.feature['gtboxes_and_label'].bytes_list.value[0],
            dtype=np.int32
        )
        
        num_boxes = len(boxes_and_labels) // 5
        boxes = boxes_and_labels[:num_boxes*4].reshape(-1, 4)
        
        # Check 1: Empty
        if len(boxes) == 0:
            issues['empty_boxes'].append(idx)
        
        # Check 2: Invalid (x1 >= x2 or y1 >= y2)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x1 >= x2 or y1 >= y2:
                issues['invalid_boxes'].append((idx, i, box))
        
        # Check 3: Tiny boxes (< 10x10)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w < 10 or h < 10:
                issues['tiny_boxes'].append((idx, i, box))
        
        # Check 4: Duplicates
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                iou = calculate_iou(boxes[i], boxes[j])
                if iou > 0.95:
                    issues['duplicate_boxes'].append((idx, i, j))
        
        # Check 5: Out of bounds
        height = example.features.feature['img_height'].int64_list.value[0]
        width = example.features.feature['img_width'].int64_list.value[0]
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                issues['out_of_bounds'].append((idx, i, box))
    
    # Report
    print("=== DATA QUALITY REPORT ===")
    for issue_type, items in issues.items():
        if items:
            print(f"⚠️ {issue_type}: {len(items)} cases")
    
    return issues

# Fix issues automatically
def fix_annotations(issues):
    """Auto-fix common issues"""
    # Implementation...
```

---

# 3. CẢI THIỆN MODEL (MODEL-CENTRIC APPROACH)

## 3.1. Backbone upgrade

### 🔄 **Current: ResNet50/101**
### 🚀 **Better options:**

```python
# models/backbone_upgraded.py

from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB4, EfficientNetB7,
    ResNet152V2, ResNeXt50, ResNeXt101,
    DenseNet121, DenseNet201
)

def build_backbone_v2(name='efficientnetb4', weights='imagenet'):
    """Upgraded backbone options"""
    
    backbones = {
        # EfficientNet (BEST cho medical imaging)
        'efficientnetb0': EfficientNetB0,
        'efficientnetb4': EfficientNetB4,  # Khuyến nghị
        'efficientnetb7': EfficientNetB7,  # Nếu có GPU mạnh
        
        # ResNet variants
        'resnet152v2': ResNet152V2,
        'resnext50': ResNeXt50,
        'resnext101': ResNeXt101,
        
        # DenseNet
        'densenet121': DenseNet121,
        'densenet201': DenseNet201,
    }
    
    if name not in backbones:
        raise ValueError(f"Unknown backbone: {name}")
    
    base_model = backbones[name](
        include_top=False,
        weights=weights,
        input_shape=(None, None, 3)
    )
    
    # Extract multi-scale features
    if 'efficientnet' in name:
        # EfficientNet layer names
        layer_names = [
            'block4a_expand_activation',  # Stride 8
            'block6a_expand_activation',  # Stride 16
            'top_activation'              # Stride 32
        ]
    elif 'resnext' in name or 'resnet' in name:
        layer_names = [
            'conv3_block4_out',
            'conv4_block6_out',
            'conv5_block3_out'
        ]
    elif 'densenet' in name:
        layer_names = [
            'pool3_pool',
            'pool4_pool',
            'relu'
        ]
    
    outputs = [base_model.get_layer(name).output for name in layer_names]
    
    return tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name=f'{name}_backbone'
    )

# Usage in config.py:
# BACKBONE = 'efficientnetb4'  # Thay vì 'resnet50'
```

### 📊 **Backbone comparison:**

| Backbone | Params | Speed | Accuracy | Medical Imaging |
|----------|--------|-------|----------|-----------------|
| ResNet50 | 25M | Fast | Good | ⭐⭐⭐ |
| ResNet101 | 44M | Medium | Better | ⭐⭐⭐ |
| EfficientNetB4 | 19M | Medium | **Best** | ⭐⭐⭐⭐⭐ |
| EfficientNetB7 | 66M | Slow | Excellent | ⭐⭐⭐⭐⭐ |
| ResNeXt101 | 88M | Slow | Excellent | ⭐⭐⭐⭐ |

**→ Khuyến nghị: EfficientNetB4 (best balance)**

---

## 3.2. FPN improvements

### 🔄 **Current: Basic FPN**
### 🚀 **Upgrade: BiFPN (Bidirectional FPN)**

```python
# models/bifpn.py

class BiFPN(tf.keras.layers.Layer):
    """
    Bidirectional Feature Pyramid Network
    
    Paper: EfficientDet (Google Brain, 2020)
    Improvement: +2-3% mAP vs regular FPN
    """
    
    def __init__(self, channels=256, num_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_layers = num_layers
        
        # Learnable weights for feature fusion
        self.w1 = tf.Variable(tf.ones(3), trainable=True)
        self.w2 = tf.Variable(tf.ones(3), trainable=True)
        
        # Convolutions
        self.convs = []
        for i in range(num_layers):
            self.convs.append([
                layers.Conv2D(channels, 1, name=f'bifpn_conv_{i}_lateral'),
                layers.Conv2D(channels, 3, padding='same', name=f'bifpn_conv_{i}_output'),
                layers.BatchNormalization(name=f'bifpn_bn_{i}'),
                layers.Activation('swish', name=f'bifpn_act_{i}')
            ])
    
    def call(self, features, training=False):
        """
        Args:
            features: [C3, C4, C5] from backbone
        Returns:
            pyramid: [P3, P4, P5] enhanced features
        """
        c3, c4, c5 = features
        
        # Initialize pyramid
        p3, p4, p5 = c3, c4, c5
        
        # Repeat BiFPN blocks
        for layer_idx in range(self.num_layers):
            lateral, output_conv, bn, activation = self.convs[layer_idx]
            
            # Top-down pathway
            p5_td = lateral(p5)
            p4_td = self.fuse([
                lateral(p4),
                tf.image.resize(p5_td, tf.shape(p4)[1:3])
            ], weights=self.w1)
            p3_td = self.fuse([
                lateral(p3),
                tf.image.resize(p4_td, tf.shape(p3)[1:3])
            ], weights=self.w2)
            
            # Bottom-up pathway
            p4_out = self.fuse([
                p4_td,
                tf.image.resize(p3_td, tf.shape(p4)[1:3], method='nearest'),
                lateral(p4)
            ], weights=self.w1)
            p5_out = self.fuse([
                p5_td,
                tf.image.resize(p4_out, tf.shape(p5)[1:3], method='nearest'),
                lateral(p5)
            ], weights=self.w2)
            
            # Apply convolution
            p3 = activation(bn(output_conv(p3_td), training=training))
            p4 = activation(bn(output_conv(p4_out), training=training))
            p5 = activation(bn(output_conv(p5_out), training=training))
        
        return [p3, p4, p5]
    
    def fuse(self, features, weights):
        """Weighted feature fusion"""
        # Fast normalized fusion
        weights = tf.nn.relu(weights)
        weights = weights / (tf.reduce_sum(weights) + 1e-4)
        
        fused = tf.add_n([w * f for w, f in zip(weights, features)])
        return fused

# Update config.py:
# USE_BIFPN = True
```

---

## 3.3. Attention mechanisms

### 🎯 **Add attention modules:**

```python
# models/attention.py

class CBAM(tf.keras.layers.Layer):
    """
    Convolutional Block Attention Module
    
    Paper: CBAM (ECCV 2018)
    Improvement: +1-2% accuracy
    """
    
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        
        # Channel attention
        self.shared_dense_1 = layers.Dense(channels // self.ratio, activation='relu')
        self.shared_dense_2 = layers.Dense(channels)
        
        # Spatial attention
        self.conv_spatial = layers.Conv2D(1, 7, padding='same', activation='sigmoid')
    
    def call(self, inputs, training=False):
        # Channel attention
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        
        avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
        max_out = self.shared_dense_2(self.shared_dense_1(max_pool))
        
        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        inputs = inputs * channel_attention
        
        # Spatial attention
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        
        spatial_attention = self.conv_spatial(concat)
        outputs = inputs * spatial_attention
        
        return outputs

# Add to FPN outputs:
# p3 = CBAM()(p3)
# p4 = CBAM()(p4)
# p5 = CBAM()(p5)
```

---

## 3.4. Loss function improvements

### 🔄 **Current: Standard cross-entropy + L1**
### 🚀 **Better: Focal Loss + GIoU Loss**

```python
# models/losses_v2.py

def focal_loss_v2(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Improved Focal Loss
    
    Paper: RetinaNet (ICCV 2017)
    Best for: Imbalanced classes (nhiều background)
    """
    
    # Clip predictions
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # Calculate focal term
    ce = -y_true * tf.math.log(y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal_weight = tf.pow(1 - y_pred, gamma)
    
    loss = alpha_t * focal_weight * ce
    
    return tf.reduce_sum(loss, axis=-1)

def giou_loss(y_true, y_pred):
    """
    Generalized IoU Loss
    
    Paper: GIoU (CVPR 2019)
    Better than: Smooth L1, IoU Loss
    Improvement: +2-3% mAP
    """
    
    # y_true, y_pred: [B, N, 4] format [x1, y1, x2, y2]
    
    # Intersection
    x1_inter = tf.maximum(y_true[..., 0], y_pred[..., 0])
    y1_inter = tf.maximum(y_true[..., 1], y_pred[..., 1])
    x2_inter = tf.minimum(y_true[..., 2], y_pred[..., 2])
    y2_inter = tf.minimum(y_true[..., 3], y_pred[..., 3])
    
    inter_area = tf.maximum(0., x2_inter - x1_inter) * \
                 tf.maximum(0., y2_inter - y1_inter)
    
    # Union
    true_area = (y_true[..., 2] - y_true[..., 0]) * \
                (y_true[..., 3] - y_true[..., 1])
    pred_area = (y_pred[..., 2] - y_pred[..., 0]) * \
                (y_pred[..., 3] - y_pred[..., 1])
    union_area = true_area + pred_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    # Smallest enclosing box
    x1_enclosing = tf.minimum(y_true[..., 0], y_pred[..., 0])
    y1_enclosing = tf.minimum(y_true[..., 1], y_pred[..., 1])
    x2_enclosing = tf.maximum(y_true[..., 2], y_pred[..., 2])
    y2_enclosing = tf.maximum(y_true[..., 3], y_pred[..., 3])
    
    enclosing_area = (x2_enclosing - x1_enclosing) * \
                     (y2_enclosing - y1_enclosing)
    
    # GIoU
    giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-7)
    
    # Loss = 1 - GIoU
    loss = 1. - giou
    
    return tf.reduce_mean(loss)

def combined_loss_v2(y_true, y_pred):
    """Combined detection loss with improvements"""
    
    # Classification: Focal Loss
    cls_loss = focal_loss_v2(
        y_true['labels'],
        y_pred['class_probs'],
        alpha=0.25,
        gamma=2.0
    )
    
    # Box regression: GIoU Loss
    box_loss = giou_loss(
        y_true['boxes'],
        y_pred['box_deltas']
    )
    
    # Objectness: Binary cross-entropy
    obj_loss = tf.keras.losses.binary_crossentropy(
        y_true['objectness'],
        y_pred['objectness']
    )
    
    # Weighted combination
    total_loss = cls_loss + 1.5 * box_loss + 0.5 * obj_loss
    
    return total_loss, cls_loss, box_loss, obj_loss
```

---

# 4. TỐI ƯU HYPERPARAMETERS

## 4.1. Learning rate schedule

### 🔄 **Current: Fixed LR = 1e-3**
### 🚀 **Better: Cosine Annealing + Warmup**

```python
# train_keras.py - UPDATE

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule với warmup và cosine decay
    
    Best practice for training stability
    """
    
    def __init__(
        self,
        initial_lr=1e-3,
        warmup_steps=500,
        total_steps=10000,
        min_lr=1e-6
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        # Warmup phase
        warmup_lr = self.initial_lr * step / self.warmup_steps
        
        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(
            np.pi * (step - self.warmup_steps) / decay_steps
        ))
        decayed_lr = (self.initial_lr - self.min_lr) * cosine_decay + self.min_lr
        
        # Choose based on step
        lr = tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: decayed_lr
        )
        
        return lr

# Usage:
total_steps = cfg.EPOCHS * steps_per_epoch

lr_schedule = WarmupCosineDecay(
    initial_lr=1e-3,
    warmup_steps=500,
    total_steps=total_steps,
    min_lr=1e-6
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

---

## 4.2. Optimizer upgrades

### 🔄 **Current: Adam**
### 🚀 **Better options:**

```python
# 1. AdamW (Adam with Weight Decay) - BEST
from tensorflow_addons.optimizers import AdamW

optimizer = AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-4,  # L2 regularization
    beta_1=0.9,
    beta_2=0.999
)

# 2. LAMB (Large Batch Optimizer)
from tensorflow_addons.optimizers import LAMB

optimizer = LAMB(
    learning_rate=lr_schedule,
    weight_decay_rate=0.01
)

# 3. Lookahead
from tensorflow_addons.optimizers import Lookahead

base_optimizer = AdamW(learning_rate=lr_schedule)
optimizer = Lookahead(base_optimizer, sync_period=6, slow_step_size=0.5)

# Install: pip install tensorflow-addons
```

---

## 4.3. Regularization

### 🛡️ **Add to model:**

```python
# models/detector_v2.py

class DetectorV2(tf.keras.Model):
    """Detector with regularization"""
    
    def __init__(self, num_classes, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        
        # ... existing code ...
        
        # Add Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Add L2 regularization to Dense layers
        self.classifier = layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name='classifier'
        )
        
        # Add Batch Normalization
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
    
    def call(self, inputs, training=False):
        # ... feature extraction ...
        
        # Apply dropout during training
        x = self.dropout1(x, training=training)
        x = self.bn1(x, training=training)
        
        # ... continue ...
        
        return outputs
```

---

## 4.4. Batch size & epochs

### 📊 **Optimal settings:**

```python
# config_v2.py - UPDATE

# Tùy theo GPU memory:
# GPU 8GB:  BATCH_SIZE = 4-8
# GPU 16GB: BATCH_SIZE = 8-16
# GPU 24GB: BATCH_SIZE = 16-32

BATCH_SIZE = 8  # Tăng từ 2 → 8 (nếu GPU cho phép)

# Epochs
EPOCHS = 100  # Tăng từ 20 → 100

# Early stopping
EARLY_STOPPING_PATIENCE = 15  # Stop nếu không improve sau 15 epochs

# Checkpoint
SAVE_BEST_ONLY = True
MONITOR_METRIC = 'val_mAP'  # Thay vì 'val_loss'
```

---

# 5. ADVANCED TECHNIQUES

## 5.1. Multi-scale training

```python
# data/multi_scale_loader.py

def multi_scale_training(image, targets, scales=[480, 512, 544, 576, 608, 640]):
    """
    Random scale training
    
    Improvement: +2-3% mAP
    """
    
    # Random chọn scale
    scale = tf.random.shuffle(scales)[0]
    
    # Resize
    image = tf.image.resize(image, [scale, scale])
    
    # Adjust boxes
    # ...
    
    return image, targets

# Add to data pipeline:
dataset = dataset.map(multi_scale_training)
```

---

## 5.2. Test-time augmentation (TTA)

```python
# scripts/inference_with_tta.py

def predict_with_tta(model, image, num_augmentations=5):
    """
    Test-time augmentation for better accuracy
    
    Improvement: +1-2% mAP (but slower)
    """
    
    predictions = []
    
    # Original
    boxes, labels, scores = model.predict(image)
    predictions.append((boxes, labels, scores))
    
    # Horizontal flip
    image_flip = tf.image.flip_left_right(image)
    boxes, labels, scores = model.predict(image_flip)
    boxes = flip_boxes_horizontal(boxes, image.shape[1])
    predictions.append((boxes, labels, scores))
    
    # Vertical flip
    image_flip = tf.image.flip_up_down(image)
    boxes, labels, scores = model.predict(image_flip)
    boxes = flip_boxes_vertical(boxes, image.shape[0])
    predictions.append((boxes, labels, scores))
    
    # Multi-scale
    for scale in [0.9, 1.1]:
        h, w = int(image.shape[0] * scale), int(image.shape[1] * scale)
        image_scaled = tf.image.resize(image, [h, w])
        boxes, labels, scores = model.predict(image_scaled)
        boxes = rescale_boxes(boxes, scale)
        predictions.append((boxes, labels, scores))
    
    # Merge predictions (Weighted Box Fusion)
    final_boxes, final_labels, final_scores = weighted_boxes_fusion(predictions)
    
    return final_boxes, final_labels, final_scores
```

---

## 5.3. Knowledge distillation

```python
# models/distillation.py

def knowledge_distillation(
    teacher_model,  # Large model (ResNet101)
    student_model,  # Small model (ResNet50)
    dataset,
    temperature=3.0,
    alpha=0.7
):
    """
    Transfer knowledge từ model lớn → model nhỏ
    
    Improvement: Student đạt 95% accuracy của Teacher
    """
    
    @tf.function
    def distillation_loss(images, targets):
        # Teacher predictions (soft labels)
        teacher_logits = teacher_model(images, training=False)
        
        # Student predictions
        student_logits = student_model(images, training=True)
        
        # Distillation loss (soft targets)
        soft_loss = tf.keras.losses.KLDivergence()(
            tf.nn.softmax(teacher_logits / temperature),
            tf.nn.softmax(student_logits / temperature)
        )
        
        # Hard loss (ground truth)
        hard_loss = tf.keras.losses.sparse_categorical_crossentropy(
            targets['labels'],
            student_logits
        )
        
        # Combined loss
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return loss
    
    # Train student
    # ...
```

---

# 6. ENSEMBLE METHODS

## 6.1. Model ensemble

```python
# scripts/ensemble.py

def ensemble_predictions(models, image, method='voting'):
    """
    Ensemble nhiều models
    
    Improvement: +3-5% mAP
    """
    
    all_predictions = []
    
    for model in models:
        boxes, labels, scores = model.predict(image)
        all_predictions.append({
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        })
    
    if method == 'voting':
        # Soft voting
        final_boxes, final_labels, final_scores = soft_voting(all_predictions)
    
    elif method == 'wbf':
        # Weighted Box Fusion (BEST)
        final_boxes, final_labels, final_scores = weighted_boxes_fusion(
            all_predictions,
            iou_threshold=0.5,
            skip_box_threshold=0.3
        )
    
    elif method == 'nms':
        # Non-Maximum Suppression ensemble
        final_boxes, final_labels, final_scores = nms_ensemble(
            all_predictions,
            iou_threshold=0.5
        )
    
    return final_boxes, final_labels, final_scores

# Train different models:
models = [
    load_model('resnet50_model.h5'),
    load_model('resnet101_model.h5'),
    load_model('efficientnetb4_model.h5'),
]

# Ensemble prediction
boxes, labels, scores = ensemble_predictions(models, image, method='wbf')
```

---

# 7. DOMAIN-SPECIFIC IMPROVEMENTS (MEDICAL IMAGING)

## 7.1. Preprocessing cho medical images

```python
# data/medical_preprocessing.py

import cv2
import numpy as np

def preprocess_medical_image(image):
    """
    Medical imaging specific preprocessing
    
    Improvement: +5-10% accuracy cho medical data
    """
    
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Tăng contrast cho vùng tối
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if len(image.shape) == 2:  # Grayscale
        image = clahe.apply(image)
    else:  # RGB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 2. Denoise
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # 3. Sharpening
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel)
    
    # 4. Normalization (Z-score)
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    image = (image - mean) / (std + 1e-7)
    
    return image
```

---

## 7.2. Transfer learning from medical domain

```python
# models/medical_transfer_learning.py

def build_medical_backbone():
    """
    Sử dụng pretrained weights from medical datasets
    
    Improvement: +10-15% vs ImageNet pretrained
    """
    
    # Option 1: CheXNet (Chest X-ray pretrained)
    # Download: https://stanfordmlgroup.github.io/projects/chexnet/
    
    # Option 2: Pretrain on larger medical dataset first
    # E.g., ImageNet → PatchCamelyon → Your dataset
    
    backbone = build_backbone('resnet50', weights=None)
    
    # Load medical pretrained weights
    medical_weights_path = 'pretrained/chexnet_weights.h5'
    backbone.load_weights(medical_weights_path, by_name=True, skip_mismatch=True)
    
    return backbone
```

---

## 7.3. Hard example mining

```python
# training/hard_example_mining.py

def online_hard_example_mining(loss, k=0.7):
    """
    Chỉ focus vào examples khó nhất
    
    Paper: OHEM (CVPR 2016)
    Improvement: +2-3% mAP
    """
    
    # Sort losses
    sorted_loss, indices = tf.nn.top_k(loss, k=int(k * tf.shape(loss)[0]))
    
    # Only backprop on top k% hardest examples
    hard_loss = tf.reduce_mean(sorted_loss)
    
    return hard_loss

# Usage in training loop:
loss = compute_loss(predictions, targets)
loss = online_hard_example_mining(loss, k=0.7)
```

---

# 8. DEBUGGING & MONITORING

## 8.1. Visualization tools

```python
# utils/visualize_v2.py

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_predictions_detailed(image, pred_boxes, pred_labels, pred_scores,
                                    gt_boxes=None, gt_labels=None,
                                    class_names=None, save_path=None):
    """
    Detailed visualization với ground truth comparison
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Predictions
    ax = axes[0]
    ax.imshow(image)
    ax.set_title("Predictions", fontsize=16)
    
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2-x1, y2-y1,
                         linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        text = f"{class_names[label]}: {score:.2f}"
        ax.text(x1, y1-5, text, color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Right: Ground Truth
    if gt_boxes is not None:
        ax = axes[1]
        ax.imshow(image)
        ax.set_title("Ground Truth", fontsize=16)
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            
            text = class_names[label]
            ax.text(x1, y1-5, text, color='green', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_training_curves(history, save_path='training_curves.png'):
    """Plot loss and accuracy curves"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    
    # Learning rate
    axes[1, 0].plot(history['lr'], label='Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].legend()
    
    # mAP
    axes[1, 1].plot(history['val_mAP'], label='Val mAP')
    axes[1, 1].set_title('mAP over Epochs')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
```

---

## 8.2. TensorBoard callbacks

```python
# train_keras.py - ADD

# Custom callback for advanced metrics
class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    """Extended TensorBoard with custom metrics"""
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Calculate mAP
        mAP = evaluate_map(self.model, self.validation_data)
        logs['val_mAP'] = mAP
        
        # Per-class accuracy
        per_class_acc = evaluate_per_class(self.model, self.validation_data)
        for i, acc in enumerate(per_class_acc):
            logs[f'val_acc_class_{i}'] = acc
        
        # Log to TensorBoard
        super().on_epoch_end(epoch, logs)
        
        # Visualize predictions
        if epoch % 5 == 0:
            self.visualize_predictions(epoch)
    
    def visualize_predictions(self, epoch):
        """Log sample predictions to TensorBoard"""
        
        # Get sample batch
        images, targets = next(iter(self.validation_data))
        
        # Predict
        boxes, labels, scores = self.model(images, training=False)
        
        # Visualize
        for i in range(min(4, len(images))):
            fig = visualize_predictions_detailed(
                images[i],
                boxes[i],
                labels[i],
                scores[i],
                gt_boxes=targets['boxes'][i],
                gt_labels=targets['labels'][i]
            )
            
            # Log to TensorBoard
            with self.writer.as_default():
                tf.summary.image(
                    f'predictions/sample_{i}',
                    plot_to_image(fig),
                    step=epoch
                )

# Usage:
callbacks = [
    CustomTensorBoard(log_dir=cfg.LOG_DIR, histogram_freq=1),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_mAP',
        patience=15,
        mode='max',
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cfg.MODEL_DIR, 'best_model.h5'),
        monitor='val_mAP',
        mode='max',
        save_best_only=True
    )
]

model.fit(train_ds, validation_data=val_ds, callbacks=callbacks)
```

---

# 🎯 PRIORITY ACTION PLAN

## Phase 1: Quick wins (1-2 tuần) → +10-15% accuracy

```
✅ 1. Data augmentation nâng cao (Albumentations)
✅ 2. Focal Loss + GIoU Loss
✅ 3. Learning rate schedule (Warmup + Cosine)
✅ 4. AdamW optimizer
✅ 5. Batch size tăng lên 8
```

## Phase 2: Model improvements (2-3 tuần) → +5-10% accuracy

```
✅ 6. Upgrade backbone → EfficientNetB4
✅ 7. BiFPN thay cho FPN
✅ 8. CBAM attention module
✅ 9. Dropout + L2 regularization
✅ 10. Multi-scale training
```

## Phase 3: Advanced (3-4 tuần) → +5-10% accuracy

```
✅ 11. Hard example mining
✅ 12. Test-time augmentation
✅ 13. Model ensemble (3 models)
✅ 14. Medical-specific preprocessing
✅ 15. Transfer learning from medical domain
```

---

# 📊 EXPECTED IMPROVEMENTS

| Technique | Expected Gain | Difficulty | Time |
|-----------|--------------|------------|------|
| Better augmentation | +5-8% | Easy | 1 day |
| Focal Loss + GIoU | +3-5% | Easy | 1 day |
| LR schedule | +2-3% | Easy | 1 day |
| EfficientNetB4 | +5-10% | Medium | 2 days |
| BiFPN | +2-3% | Medium | 2 days |
| CBAM | +1-2% | Easy | 1 day |
| Multi-scale | +2-3% | Medium | 1 day |
| Hard mining | +2-3% | Hard | 2 days |
| TTA | +1-2% | Easy | 1 day |
| Ensemble | +3-5% | Easy | 1 day |
| **TOTAL** | **+25-45%** | - | **2-3 weeks** |

---

# ✅ IMPLEMENTATION CHECKLIST

```bash
# Week 1: Data & Loss improvements
[ ] Cài Albumentations: pip install albumentations
[ ] Update data/loader_tf2.py với augmentation nâng cao
[ ] Update models/losses.py với Focal Loss + GIoU
[ ] Update train_keras.py với WarmupCosineDecay
[ ] Update optimizer → AdamW
[ ] Test training 5 epochs

# Week 2: Model architecture
[ ] Implement BiFPN trong models/bifpn.py
[ ] Implement CBAM trong models/attention.py
[ ] Update backbone → EfficientNetB4
[ ] Add dropout + regularization
[ ] Test training 10 epochs

# Week 3: Advanced techniques
[ ] Implement multi-scale training
[ ] Implement hard example mining
[ ] Implement TTA inference
[ ] Train ensemble models (3 variants)
[ ] Full training 100 epochs

# Week 4: Evaluation & fine-tuning
[ ] Evaluate on test set
[ ] Error analysis
[ ] Hyperparameter tuning
[ ] Final model selection
[ ] Documentation
```

---

# 📞 TROUBLESHOOTING

**Q: Accuracy vẫn thấp sau khi apply tất cả?**
```
A: Check theo thứ tự:
1. Data quality (annotations đúng chưa?)
2. Class imbalance (cân bằng chưa?)
3. Model converge chưa? (loss giảm chưa?)
4. Overfitting? (train acc >> val acc)
5. Learning rate quá cao/thấp?
```

**Q: GPU out of memory?**
```
A: Giảm theo thứ tự:
1. BATCH_SIZE từ 8 → 4 → 2
2. IMAGE_SIZE từ 640 → 512
3. Backbone từ B4 → B0
4. Mixed precision: tf.keras.mixed_precision
```

**Q: Training quá lâu?**
```
A: Tối ưu:
1. Use tf.data pipeline với prefetch
2. Mixed precision training
3. Multi-GPU nếu có
4. Reduce augmentation complexity
```

---

**Good luck! Hãy implement từng bước một và monitor results! 🚀**
