# 📚 Hướng Dẫn Viết Object Detection Project Tương Tự Comparison Detector

## Mục lục
1. [Kiến trúc dự án](#kiến-trúc-dự-án)
2. [Các bước xây dựng](#các-bước-xây-dựng)
3. [Code mẫu chi tiết](#code-mẫu-chi-tiết)
4. [Giải thích khái niệm](#giải-thích-khái-niệm)
5. [Tips & Tricks](#tips--tricks)

---

## 🏗️ Kiến Trúc Dự Án

### Cấu trúc folder được khuyến khích:

```
YourObjectDetectionProject/
├── config.py                    # Tất cả cấu hình
├── train.py                     # Script training
├── evaluate.py                  # Script đánh giá
├── predict.py                   # Script dự đoán
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Load dữ liệu
│   └── preprocessing.py        # Tiền xử lý ảnh
│
├── models/
│   ├── __init__.py
│   ├── backbone.py             # ResNet backbone
│   ├── fpn.py                  # Feature Pyramid Network
│   ├── rpn.py                  # Region Proposal Network
│   ├── roi_head.py             # RoI Head (Classification + Regression)
│   └── detector.py             # Model chính
│
├── utils/
│   ├── __init__.py
│   ├── box_utils.py            # Xử lý bounding box
│   ├── iou.py                  # Tính IoU
│   ├── nms.py                  # Non-Maximum Suppression
│   ├── metrics.py              # mAP, Recall...
│   └── visualization.py        # Vẽ boxes
│
├── losses/
│   ├── __init__.py
│   └── losses.py               # Tất cả loss functions
│
└── README.md
```

---

## 🚀 Các Bước Xây Dựng

### Bước 1: Config - Cấu hình tất cả thông số

**File: `config.py`**

```python
# config.py
import numpy as np
import tensorflow as tf
import math

class Config:
    # ==================== Data Configuration ====================
    NUM_CLASSES = 11 + 1  # 11 object classes + 1 background
    TARGET_IMAGE_SIZE = 1024  # Resize ảnh về 1024x1024
    PIXEL_MEANS = np.array([115.2, 118.8, 123.0])  # ImageNet means
    
    # ==================== Network Configuration ====================
    BACKBONE = 'resnet_v2_50'  # Hoặc resnet_v2_101, resnet_v1_50...
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]  # Stride của mỗi feature level
    FEATURE_PYRAMID_LEVELS = ['P2', 'P3', 'P4', 'P5', 'P6']  # FPN levels
    FEATURE_DIM = 256  # Số channels sau FPN
    
    # ==================== RPN Configuration ====================
    # Anchors
    ANCHOR_RATIOS = [0.5, 1, 2]  # Tỷ lệ aspect ratio
    ANCHOR_SCALES = [32, 64, 128, 256, 512]  # Base sizes mỗi level
    RPN_ANCHOR_STRIDE = 1
    
    # RPN Training
    RPN_MINIBATCH_SIZE = 256
    RPN_POSITIVE_IOU_THRESHOLD = 0.7
    RPN_NEGATIVE_IOU_THRESHOLD = 0.3
    RPN_POSITIVE_RATE = 0.5  # Số positive anchors trong minibatch
    
    # RPN Inference
    RPN_NMS_IOU_THRESHOLD = 0.7
    RPN_TOP_K_BEFORE_NMS = 6000
    RPN_TOP_K_AFTER_NMS = 2000  # Training
    RPN_TOP_K_AFTER_NMS_INFER = 1000  # Inference
    
    # RPN Bbox Encoding/Decoding
    RPN_BBOX_STD_DEV = [0.1, 0.1, 0.25, 0.27]
    
    # ==================== Fast R-CNN Configuration ====================
    # RoI Pooling
    ROI_POOL_SIZE = 7  # 7x7 pooled features
    
    # Fast R-CNN Training
    FAST_RCNN_MINIBATCH_SIZE = 200
    FAST_RCNN_POSITIVE_IOU_THRESHOLD = 0.5
    FAST_RCNN_POSITIVE_RATE = 0.33
    FAST_RCNN_NMS_IOU_THRESHOLD = 0.3
    FAST_RCNN_BBOX_STD_DEV = [0.13, 0.13, 0.27, 0.26]
    
    # Fast R-CNN Inference
    DETECTION_SCORE_THRESHOLD = 0.7
    DETECTION_MAX_INSTANCES = 200
    
    # ==================== Training Configuration ====================
    BATCH_SIZE = 2  # images per GPU
    NUM_GPUS = 2
    LEARNING_RATE = 0.001
    LEARNING_RATE_BOUNDARIES = [35, 50]  # Epochs
    LEARNING_RATE_VALUES = [0.001, 0.0001, 0.00001]
    WEIGHT_DECAY = 0.0001
    MOMENTUM = 0.9
    GRADIENT_CLIP = 5.0
    
    NUM_EPOCHS = 60
    STEPS_PER_EPOCH = 1000
    
    # ==================== Derived Configuration ====================
    def __init__(self):
        self.TOTAL_BATCH_SIZE = self.BATCH_SIZE * self.NUM_GPUS
        self.NUM_ANCHORS_PER_LOCATION = len(self.ANCHOR_RATIOS)
```

---

### Bước 2: Backbone Network - Trích xuất Features

**File: `models/backbone.py`**

```python
# models/backbone.py
import tensorflow as tf
import tensorflow.contrib.slim as slim

class ResNetBackbone:
    """
    ResNet backbone để trích xuất multi-level features
    Ví dụ: ResNet-50 có 4 main blocks
    """
    
    def __init__(self, config):
        self.config = config
        self.backbone_name = config.BACKBONE
    
    def __call__(self, images, is_training=True):
        """
        Args:
            images: [batch_size, height, width, 3]
            is_training: bool
            
        Returns:
            features_dict: {
                'C2': [batch, H/4, W/4, 256],
                'C3': [batch, H/8, W/8, 512],
                'C4': [batch, H/16, W/16, 1024],
                'C5': [batch, H/32, W/32, 2048]
            }
        """
        
        # Sử dụng Slim (TensorFlow-Slim)
        with slim.arg_scope(
            [slim.conv2d],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            weights_regularizer=slim.l2_regularizer(self.config.WEIGHT_DECAY)
        ):
            # ResNet-50 pre-trained on ImageNet
            if self.backbone_name == 'resnet_v2_50':
                logits, end_points = resnet_v2_50(
                    images,
                    is_training=is_training,
                    output_stride=32
                )
            else:
                raise ValueError(f"Unknown backbone: {self.backbone_name}")
        
        # Lấy intermediate features từ các layers
        features_dict = {
            'C2': end_points['resnet_v2_50/block1/unit_2/bottleneck_v2'],
            'C3': end_points['resnet_v2_50/block2/unit_3/bottleneck_v2'],
            'C4': end_points['resnet_v2_50/block3/unit_5/bottleneck_v2'],
            'C5': end_points['resnet_v2_50/block4/unit_3/bottleneck_v2']
        }
        
        return features_dict

# ResNet-50 implementation
def resnet_v2_50(inputs, is_training=True, output_stride=32):
    """
    Đây là simplified version. Trong thực tế, sử dụng:
    from nets import resnet_v2
    """
    # Khởi tạo ResNet từ checkpoint
    # resnet_v2.resnet_v2_50(inputs, is_training=is_training)
    pass
```

---

### Bước 3: Feature Pyramid Network - Tạo Multi-Scale Features

**File: `models/fpn.py`**

```python
# models/fpn.py
import tensorflow as tf
import tensorflow.contrib.slim as slim

class FPN:
    """
    Feature Pyramid Network
    Tạo các feature maps ở nhiều scale
    """
    
    def __init__(self, config):
        self.config = config
        self.feature_dim = config.FEATURE_DIM
    
    def __call__(self, backbone_features, is_training=True):
        """
        Args:
            backbone_features: dict {C2, C3, C4, C5}
            is_training: bool
            
        Returns:
            pyramid_features: dict {P2, P3, P4, P5, P6}
        """
        
        pyramid = {}
        
        with tf.variable_scope('fpn'):
            with slim.arg_scope(
                [slim.conv2d],
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(self.config.WEIGHT_DECAY)
            ):
                # ============ Top-down pathway ============
                # Bắt đầu từ C5 (highest level, lowest resolution)
                
                # 1. C5 -> P5 (1x1 conv)
                p5 = slim.conv2d(
                    backbone_features['C5'],
                    num_outputs=self.feature_dim,
                    kernel_size=[1, 1],
                    scope='C5_to_P5'
                )
                pyramid['P5'] = p5
                
                # 2. P5 + C4 -> P4 (upsample + add + 3x3 conv)
                p4_from_p5 = tf.image.resize_nearest_neighbor(
                    p5,
                    tf.shape(backbone_features['C4'])[1:3]
                )
                c4_reduced = slim.conv2d(
                    backbone_features['C4'],
                    num_outputs=self.feature_dim,
                    kernel_size=[1, 1],
                    scope='C4_to_P4_reduced'
                )
                p4 = p4_from_p5 + c4_reduced
                p4 = slim.conv2d(
                    p4,
                    num_outputs=self.feature_dim,
                    kernel_size=[3, 3],
                    scope='P4_refined'
                )
                pyramid['P4'] = p4
                
                # 3. Tương tự cho P3, P2
                # (Code lặp tương tự cho C3->P3, C2->P2)
                
                # ============ Forward pathway (downsampling) ============
                # P5 -> P6 (max pooling)
                p6 = slim.max_pool2d(
                    pyramid['P5'],
                    kernel_size=[2, 2],
                    stride=2,
                    scope='P6'
                )
                pyramid['P6'] = p6
        
        return pyramid  # {P2, P3, P4, P5, P6}
```

---

### Bước 4: Anchor Generation - Tạo các vùng đề xuất

**File: `utils/anchors.py`**

```python
# utils/anchors.py
import numpy as np
import tensorflow as tf

def generate_anchors(base_size, ratios, scales):
    """
    Tạo anchor templates cho một feature map
    
    Args:
        base_size: int (e.g., 16)
        ratios: list [0.5, 1, 2]
        scales: list [8, 16, 32]
    
    Returns:
        anchors: [num_anchors, 4] dạng [y1, x1, y2, x2]
    
    Ví dụ:
        base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]
        → 9 anchors (3 ratios × 3 scales)
    """
    
    base_area = base_size ** 2
    anchors = []
    
    for scale in scales:
        for ratio in ratios:
            # Tính width, height từ area và ratio
            area = base_area * (scale ** 2)
            w = np.sqrt(area / ratio)
            h = ratio * w
            
            # Tạo anchor [y1, x1, y2, x2]
            x1 = -w / 2
            y1 = -h / 2
            x2 = w / 2
            y2 = h / 2
            
            anchors.append([y1, x1, y2, x2])
    
    return np.array(anchors, dtype=np.float32)


def generate_pyramid_anchors(config, image_shape):
    """
    Tạo ALL anchors cho toàn bộ pyramid
    
    Args:
        config: Config object
        image_shape: [height, width]
    
    Returns:
        anchors: [total_anchors, 4]
    
    Ví dụ:
        Image 1024×1024
        → P2: 256×256×3 = 196,608 anchors
        → P3: 128×128×3 = 49,152 anchors
        → ...
        → Total: ~290k anchors
    """
    
    all_anchors = []
    
    for level_idx, level in enumerate(config.FEATURE_PYRAMID_LEVELS):
        stride = config.BACKBONE_STRIDES[level_idx]
        base_size = config.ANCHOR_SCALES[level_idx]
        
        # Feature map size cho level này
        feature_height = image_shape[0] // stride
        feature_width = image_shape[1] // stride
        
        # Template anchors cho level này
        template_anchors = generate_anchors(
            base_size,
            config.ANCHOR_RATIOS,
            [1.0]  # hoặc scales khác
        )
        
        # Shift anchors tới từng điểm trên feature map
        for y in range(feature_height):
            for x in range(feature_width):
                center_y = (y + 0.5) * stride
                center_x = (x + 0.5) * stride
                
                for anchor in template_anchors:
                    shifted_anchor = anchor + [center_y, center_x, center_y, center_x]
                    all_anchors.append(shifted_anchor)
    
    return np.array(all_anchors, dtype=np.float32)
```

---

### Bước 5: RPN - Region Proposal Network

**File: `models/rpn.py`**

```python
# models/rpn.py
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.anchors import generate_pyramid_anchors
from utils.box_utils import encode_boxes, decode_boxes, compute_iou

class RPN:
    """
    Region Proposal Network
    Input: Feature pyramid
    Output: Region proposals (hàng nghìn bounding boxes)
    """
    
    def __init__(self, config):
        self.config = config
    
    def build_rpn_net(self, feature_pyramid, is_training=True):
        """
        Xây dựng RPN network
        
        Args:
            feature_pyramid: dict {P2, P3, P4, P5, P6}
            is_training: bool
        
        Returns:
            rpn_class_logits: [batch, num_anchors, 2] (object/background)
            rpn_bbox_deltas: [batch, num_anchors, 4] (dx, dy, dw, dh)
        """
        
        class_logits_list = []
        bbox_deltas_list = []
        
        with tf.variable_scope('rpn_net'):
            with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(self.config.WEIGHT_DECAY)
            ):
                
                for level in self.config.FEATURE_PYRAMID_LEVELS:
                    feature = feature_pyramid[level]
                    
                    # Shared conv layer (3x3)
                    rpn_conv = slim.conv2d(
                        feature,
                        num_outputs=512,
                        kernel_size=[3, 3],
                        stride=1,
                        activation_fn=tf.nn.relu,
                        scope=f'{level}/rpn_conv',
                        reuse=tf.AUTO_REUSE
                    )
                    
                    # Classification branch (2 classes: object/background)
                    num_anchors = len(self.config.ANCHOR_RATIOS)
                    class_logits = slim.conv2d(
                        rpn_conv,
                        num_outputs=num_anchors * 2,  # 2 = binary classification
                        kernel_size=[1, 1],
                        stride=1,
                        activation_fn=None,
                        scope=f'{level}/rpn_class',
                        reuse=tf.AUTO_REUSE
                    )
                    
                    # Reshape: [batch, H, W, 2*k] -> [batch, H*W*k, 2]
                    batch_size = tf.shape(class_logits)[0]
                    class_logits = tf.reshape(
                        class_logits,
                        [batch_size, -1, 2]
                    )
                    class_logits_list.append(class_logits)
                    
                    # Bbox regression branch (4 coordinates)
                    bbox_deltas = slim.conv2d(
                        rpn_conv,
                        num_outputs=num_anchors * 4,
                        kernel_size=[1, 1],
                        stride=1,
                        activation_fn=None,
                        scope=f'{level}/rpn_bbox',
                        reuse=tf.AUTO_REUSE
                    )
                    
                    # Reshape: [batch, H, W, 4*k] -> [batch, H*W*k, 4]
                    bbox_deltas = tf.reshape(
                        bbox_deltas,
                        [batch_size, -1, 4]
                    )
                    bbox_deltas_list.append(bbox_deltas)
                
                # Concatenate từ tất cả levels
                rpn_class_logits = tf.concat(class_logits_list, axis=1)
                rpn_bbox_deltas = tf.concat(bbox_deltas_list, axis=1)
        
        return rpn_class_logits, rpn_bbox_deltas
    
    def generate_proposals(self, rpn_class_logits, rpn_bbox_deltas, 
                          anchors, image_shape, is_training=True):
        """
        Chuyển đổi RPN outputs thành proposals
        
        Args:
            rpn_class_logits: [batch, num_anchors, 2]
            rpn_bbox_deltas: [batch, num_anchors, 4]
            anchors: [num_anchors, 4]
            image_shape: [height, width]
            is_training: bool
        
        Returns:
            proposals: [batch, num_proposals, 4]
            scores: [batch, num_proposals]
        """
        
        # 1. Lấy class scores
        rpn_class_probs = tf.nn.softmax(rpn_class_logits)
        object_scores = rpn_class_probs[:, :, 1]  # Cột 1 = object class
        
        # 2. Decode bounding boxes từ deltas
        proposals = decode_boxes(
            encoded_boxes=rpn_bbox_deltas,
            reference_boxes=anchors,
            std_dev=self.config.RPN_BBOX_STD_DEV
        )
        
        # 3. Clip proposals vào image bounds
        proposals = tf.clip_by_value(
            proposals,
            [0, 0, 0, 0],
            [image_shape[0], image_shape[1], image_shape[0], image_shape[1]]
        )
        
        # 4. NMS + top-k
        num_proposals = (self.config.RPN_TOP_K_AFTER_NMS 
                        if is_training 
                        else self.config.RPN_TOP_K_AFTER_NMS_INFER)
        
        # Áp dụng NMS (Non-Maximum Suppression)
        keep_indices = tf.image.non_max_suppression(
            proposals,
            object_scores[0],  # Lấy batch đầu tiên
            max_output_size=num_proposals,
            iou_threshold=self.config.RPN_NMS_IOU_THRESHOLD,
            score_threshold=0.0
        )
        
        proposals = tf.gather(proposals[0], keep_indices)
        scores = tf.gather(object_scores[0], keep_indices)
        
        return proposals, scores
    
    def compute_rpn_loss(self, rpn_class_logits, rpn_bbox_deltas,
                         gt_boxes, gt_labels, anchors, config):
        """
        Tính RPN loss
        
        Loss = Classification Loss + Bbox Regression Loss
        """
        
        # Gán positive/negative anchors dựa trên IoU
        ious = compute_iou(anchors, gt_boxes)  # [num_anchors, num_gt]
        max_ious = tf.reduce_max(ious, axis=1)  # [num_anchors]
        gt_argmax_ious = tf.argmax(ious, axis=1)  # [num_anchors]
        
        # Positive anchors: IoU > threshold
        positive_mask = tf.greater_equal(
            max_ious,
            config.RPN_POSITIVE_IOU_THRESHOLD
        )
        
        # Negative anchors: IoU < threshold
        negative_mask = tf.less(
            max_ious,
            config.RPN_NEGATIVE_IOU_THRESHOLD
        )
        
        # Sampling: balanced ratio
        num_positive = tf.cast(
            config.RPN_MINIBATCH_SIZE * config.RPN_POSITIVE_RATE,
            tf.int32
        )
        
        # Encode ground truth boxes
        gt_boxes_for_anchors = tf.gather(gt_boxes, gt_argmax_ious)
        encoded_gt_boxes = encode_boxes(
            boxes=gt_boxes_for_anchors,
            reference_boxes=anchors,
            std_dev=config.RPN_BBOX_STD_DEV
        )
        
        # Classification loss
        cls_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.cast(
                tf.stack([~positive_mask, positive_mask], axis=1),
                tf.float32
            ),
            logits=rpn_class_logits
        )
        
        # Bbox regression loss (chỉ cho positive anchors)
        bbox_loss = tf.losses.huber_loss(
            labels=encoded_gt_boxes,
            predictions=rpn_bbox_deltas,
            weights=tf.cast(positive_mask, tf.float32)
        )
        
        total_loss = cls_loss + bbox_loss
        
        return total_loss, cls_loss, bbox_loss
```

---

### Bước 6: RoI Pooling - Trích xuất Features từ Proposals

**File: `models/roi_pooling.py`**

```python
# models/roi_pooling.py
import tensorflow as tf

def roi_pool(features, proposals, pool_size=7):
    """
    RoI Pooling: Lấy fixed-size features từ proposals
    
    Args:
        features: [batch, height, width, channels]
        proposals: [num_proposals, 4] trong format [y1, x1, y2, x2]
        pool_size: int (7x7)
    
    Returns:
        pooled_features: [num_proposals, pool_size, pool_size, channels]
    
    Hoạt động:
        - Chia proposal thành 7x7 regions
        - Max pool mỗi region
        → Fixed size output bất kể proposal size
    """
    
    # Sử dụng crop_and_resize từ TensorFlow
    # Cần convert [y1, x1, y2, x2] -> normalized [y1, x1, y2, x2]
    
    height = tf.shape(features)[1]
    width = tf.shape(features)[2]
    
    # Normalize proposals
    proposals_norm = tf.cast(proposals, tf.float32)
    proposals_norm = proposals_norm / tf.cast(
        [height, width, height, width],
        tf.float32
    )
    
    # Crop and resize
    pooled = tf.image.crop_and_resize(
        image=features,
        boxes=proposals_norm,
        box_indices=tf.zeros([tf.shape(proposals)[0]], dtype=tf.int32),
        crop_size=[pool_size, pool_size],
        method='bilinear'
    )
    
    return pooled  # [num_proposals, 7, 7, channels]
```

---

### Bước 7: Fast R-CNN Head - Phân loại & Điều chỉnh Box

**File: `models/roi_head.py`**

```python
# models/roi_head.py
import tensorflow as tf
import tensorflow.contrib.slim as slim

class FastRCNNHead:
    """
    Fast R-CNN head: phân loại proposals và tinh chỉnh bounding boxes
    """
    
    def __init__(self, config):
        self.config = config
    
    def build_head(self, pooled_features, num_classes):
        """
        Args:
            pooled_features: [num_proposals, 7, 7, 256]
            num_classes: int
        
        Returns:
            class_logits: [num_proposals, num_classes]
            bbox_deltas: [num_proposals, num_classes, 4]
        """
        
        with tf.variable_scope('fast_rcnn_head'):
            # Flatten
            flat = tf.layers.flatten(pooled_features)
            
            # FC layer 1
            fc1 = tf.layers.dense(
                flat,
                units=1024,
                activation=tf.nn.relu,
                kernel_regularizer=slim.l2_regularizer(self.config.WEIGHT_DECAY),
                name='fc1'
            )
            
            # FC layer 2
            fc2 = tf.layers.dense(
                fc1,
                units=1024,
                activation=tf.nn.relu,
                kernel_regularizer=slim.l2_regularizer(self.config.WEIGHT_DECAY),
                name='fc2'
            )
            
            # Classification branch
            class_logits = tf.layers.dense(
                fc2,
                units=num_classes,
                activation=None,
                name='class_logits'
            )
            
            # Bbox regression branch (4 coordinates × num_classes)
            bbox_deltas = tf.layers.dense(
                fc2,
                units=num_classes * 4,
                activation=None,
                name='bbox_deltas'
            )
            bbox_deltas = tf.reshape(
                bbox_deltas,
                [-1, num_classes, 4]
            )
        
        return class_logits, bbox_deltas
```

---

### Bước 8: Loss Functions

**File: `losses/losses.py`**

```python
# losses/losses.py
import tensorflow as tf

def smooth_l1_loss(y_true, y_pred):
    """
    Smooth L1 Loss cho bounding box regression
    
    Loss = 0.5 * x^2          if |x| < 1
           |x| - 0.5          otherwise
    
    Hybrid giữa L1 (robust) và L2 (smooth)
    """
    
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    
    loss = (less_than_one * 0.5 * diff ** 2 +
            (1 - less_than_one) * (diff - 0.5))
    
    return tf.reduce_mean(loss)


def rpn_loss(rpn_class_logits, rpn_bbox_deltas, 
             gt_boxes, gt_labels, anchors, config):
    """
    Total RPN loss = Classification loss + Bbox loss
    """
    
    # Classification loss (object vs background)
    # Cross-entropy loss
    cls_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=gt_labels,
        logits=rpn_class_logits
    )
    
    # Bbox regression loss
    bbox_loss = smooth_l1_loss(gt_boxes, rpn_bbox_deltas)
    
    # Weighted sum
    total_loss = cls_loss + bbox_loss
    
    return total_loss


def fast_rcnn_loss(class_logits, bbox_deltas,
                   gt_class_ids, gt_boxes, config):
    """
    Total Fast R-CNN loss
    """
    
    # Classification loss
    cls_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.one_hot(gt_class_ids, config.NUM_CLASSES),
        logits=class_logits
    )
    
    # Bbox regression loss (chỉ cho positive classes)
    # Weighted by number of classes
    bbox_loss = smooth_l1_loss(gt_boxes, bbox_deltas)
    
    # Usually Fast R-CNN loss có weight cao hơn
    total_loss = 5.0 * cls_loss + bbox_loss
    
    return total_loss
```

---

### Bước 9: Utils - Xử lý Bounding Boxes

**File: `utils/box_utils.py`**

```python
# utils/box_utils.py
import tensorflow as tf
import numpy as np

def compute_iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) giữa 2 sets boxes
    
    Args:
        boxes1: [N, 4] [y1, x1, y2, x2]
        boxes2: [M, 4] [y1, x1, y2, x2]
    
    Returns:
        iou_matrix: [N, M]
    
    Công thức:
        IoU = Intersection / Union
    
    Ví dụ:
        Box A: [0, 0, 10, 10] (10x10 square)
        Box B: [5, 5, 15, 15] (10x10 square)
        
        Intersection: [5, 5, 10, 10] = 5x5 = 25
        Union: 100 + 100 - 25 = 175
        IoU = 25 / 175 ≈ 0.14
    """
    
    # Tính diện tích
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Tính intersection
    y1_inter = tf.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
    x1_inter = tf.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
    y2_inter = tf.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
    x2_inter = tf.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)
    
    inter_area = tf.maximum(y2_inter - y1_inter, 0) * \
                 tf.maximum(x2_inter - x1_inter, 0)
    
    # Tính union
    union_area = area1[:, None] + area2[None, :] - inter_area
    
    # IoU
    iou = inter_area / tf.maximum(union_area, 1e-8)
    
    return iou


def encode_boxes(boxes, reference_boxes, std_dev):
    """
    Encode boxes thành deltas (để training)
    
    Công thức:
        dy = (by - ry) / h / std_dev[0]
        dx = (bx - rx) / w / std_dev[1]
        dh = log(h / rh) / std_dev[2]
        dw = log(w / rw) / std_dev[3]
    
    Tại sao encoding?
    - Normalize scale của regression targets
    - Network dễ học hơn
    """
    
    boxes = tf.cast(boxes, tf.float32)
    reference_boxes = tf.cast(reference_boxes, tf.float32)
    std_dev = tf.cast(std_dev, tf.float32)
    
    # Tính tọa độ tâm và chiều dài
    by = (boxes[:, 0] + boxes[:, 2]) / 2
    bx = (boxes[:, 1] + boxes[:, 3]) / 2
    bh = boxes[:, 2] - boxes[:, 0]
    bw = boxes[:, 3] - boxes[:, 1]
    
    ry = (reference_boxes[:, 0] + reference_boxes[:, 2]) / 2
    rx = (reference_boxes[:, 1] + reference_boxes[:, 3]) / 2
    rh = reference_boxes[:, 2] - reference_boxes[:, 0]
    rw = reference_boxes[:, 3] - reference_boxes[:, 1]
    
    # Compute deltas
    dy = (by - ry) / rh / std_dev[0]
    dx = (bx - rx) / rw / std_dev[1]
    dh = tf.math.log(bh / rh) / std_dev[2]
    dw = tf.math.log(bw / rw) / std_dev[3]
    
    deltas = tf.stack([dy, dx, dh, dw], axis=1)
    
    return deltas


def decode_boxes(encoded_boxes, reference_boxes, std_dev):
    """
    Decode deltas thành boxes (inference)
    Reverse của encode_boxes
    """
    
    encoded_boxes = tf.cast(encoded_boxes, tf.float32)
    reference_boxes = tf.cast(reference_boxes, tf.float32)
    std_dev = tf.cast(std_dev, tf.float32)
    
    dy = encoded_boxes[:, 0] * std_dev[0]
    dx = encoded_boxes[:, 1] * std_dev[1]
    dh = encoded_boxes[:, 2] * std_dev[2]
    dw = encoded_boxes[:, 3] * std_dev[3]
    
    ry = (reference_boxes[:, 0] + reference_boxes[:, 2]) / 2
    rx = (reference_boxes[:, 1] + reference_boxes[:, 3]) / 2
    rh = reference_boxes[:, 2] - reference_boxes[:, 0]
    rw = reference_boxes[:, 3] - reference_boxes[:, 1]
    
    by = dy * rh + ry
    bx = dx * rw + rx
    bh = tf.exp(dh) * rh
    bw = tf.exp(dw) * rw
    
    y1 = by - bh / 2
    x1 = bx - bw / 2
    y2 = by + bh / 2
    x2 = bx + bw / 2
    
    boxes = tf.stack([y1, x1, y2, x2], axis=1)
    
    return boxes


def non_max_suppression(boxes, scores, max_output_size=100, iou_threshold=0.5):
    """
    Non-Maximum Suppression (NMS)
    Loại bỏ overlapping boxes
    
    Thuật toán:
    1. Sort boxes by score
    2. Pick highest score box
    3. Remove all boxes with IoU > threshold
    4. Repeat until no boxes left
    
    Ví dụ:
        Boxes: A (score=0.9), B (score=0.8), C (score=0.7)
        If IoU(A,B) > threshold:
            Keep A, remove B
        Repeat...
    """
    
    selected_indices = tf.image.non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold
    )
    
    selected_boxes = tf.gather(boxes, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)
    
    return selected_boxes, selected_scores
```

---

## 📊 Giải Thích Khái Niệm

### 1. **Anchors - Những hình dạp ứng cử**

```
Tại mỗi điểm (x, y) trên feature map, ta có K anchor boxes
với các hình dạng khác nhau.

Ví dụ: Tại điểm (64, 64) trên P3:
  Anchor 1: aspect ratio 0.5 (wide)     [50, 40, 78, 88]
  Anchor 2: aspect ratio 1.0 (square)   [50, 50, 78, 78]
  Anchor 3: aspect ratio 2.0 (tall)     [50, 56, 78, 72]

RPN sẽ dự đoán:
  - Xác suất object: P1, P2, P3
  - Điều chỉnh box: Δ1, Δ2, Δ3
```

### 2. **RPN - Dự đoán Regions**

```
Input: Feature pyramid
Output: ~2000 region proposals (potential objects)

Process:
  1. Tạo 97k+ anchors trên feature maps
  2. RPN network dự đoán:
     - Object/Background score cho mỗi anchor
     - Bbox adjustment (dy, dx, dh, dw)
  3. Decode adjustments → new boxes
  4. NMS → loại bỏ duplicates
  5. Keep top 2000
```

### 3. **IoU - Intersection over Union**

```
Dùng để đo độ overlap giữa 2 boxes

        ┌────────────┐
        │ Predicted  │
        │  ┌──────┐  │
        │  │      │  │
    ┌───┤  │      │  │
    │   │  │      │  │
    │   └──┤ ────────┘
    │ Actual

IoU = Area_intersection / Area_union

Nếu IoU > 0.5 → considered a match (positive)
Nếu IoU < 0.3 → considered mismatch (negative)
```

### 4. **Comparison Detector - Phương pháp đặc biệt**

```
Thay vì học weights cho 11 fully-connected layers:

THÔNG THƯỜNG:
  Feature → FC layers (11 outputs) → Softmax

COMPARISON DETECTOR:
  1. Load reference images (3 per class)
  2. Extract features từ reference
  3. Average → Get prototype per class (11 prototypes)
  
  4. Feature từ proposal
  5. Compute similarity với 11 prototypes
  6. Class = argmax(similarity)
  
LỢI ÍCH:
  ✓ Giảm parameters (chỉ prototypes + 1 bg conv)
  ✓ Better transfer learning
  ✓ Dễ thêm class mới
```

---

## 🎯 Tips & Tricks

### 1. **Data Preprocessing**

```python
# Luôn normalize ảnh bằng ImageNet means
image = image.astype(np.float32)
image = image - PIXEL_MEANS  # [115.2, 118.8, 123.0]

# Không nên chia cho 255 hoặc std dev
# Vì backbone đã pre-trained với normalization này
```

### 2. **Anchors Generation**

```python
# Anchor scales phải cover toàn bộ object sizes
# Ví dụ objects từ 32×32 đến 512×512
ANCHOR_SCALES = [32, 64, 128, 256, 512]

# Aspect ratios phải cover hình dạng objects
# Ví dụ cells dài vs tròn
ANCHOR_RATIOS = [0.5, 1, 2]
```

### 3. **Balanced Sampling**

```python
# Vấn đề: Background anchors >> object anchors
# Solution: Hard negative mining

# During RPN training:
# - Sample 50% positive anchors
# - Sample 50% negative anchors (hardest examples)

# Không làm này → model chỉ học detect background
```

### 4. **Learning Rate Schedule**

```python
# Start with higher LR, reduce when plateauing
boundaries = [35, 50]  # Epoch numbers
values = [0.001, 0.0001, 0.00001]

# Epoch 0-35: LR = 0.001
# Epoch 35-50: LR = 0.0001
# Epoch 50-60: LR = 0.00001
```

### 5. **Loss Weighting**

```python
# Different losses có scales khác nhau
# Cần weight để balance

# Fast R-CNN loss ít hơn RPN loss
# → Nhân với factor: 5.0 * cls_loss + bbox_loss

# Thử với values khác nhau (5, 10, 20)
```

### 6. **Debugging Tips**

```python
# 1. Visualize anchors trên image
#    → Kiểm tra có cover toàn bộ objects không

# 2. Check RPN proposals
#    → Có tạo bounding boxes tốt không?

# 3. Check ground truth assignments
#    → Positive anchors có match với GT boxes không?

# 4. Monitor individual losses
#    → Cái nào không giảm? Có problem nào không?

# 5. Start with small data (10-20 images)
#    → Nếu overfit là tốt, model có học không?
```

### 7. **Common Bugs**

```python
# BUG 1: Coordinate format inconsistency
# Sử dụng [y1, x1, y2, x2] consistently!
# Đừng mix [x1, y1, x2, y2]

# BUG 2: NMS dengan sai threshold
# IoU = 0.7 quá cao → loại bỏ valid objects
# Thử 0.5 hoặc 0.3

# BUG 3: Không normalize proposals
# Crop_and_resize cần normalized coordinates [0-1]
# Normalize trước: boxes / [H, W, H, W]

# BUG 4: Forget to clip boxes
# Proposals ra ngoài image bounds
# Clip: boxes = clip(boxes, 0, [H, W, H, W])

# BUG 5: Mismatch batch size
# RPN output: [batch, num_anchors, 4]
# Proposals: [1, num_proposals, 4]
# Cần squeeze/expand dimensions đúng
```

---

## 📝 Training Script Skeleton

```python
# train.py - File chính

import tensorflow as tf
from config import Config
from models.detector import FasterRCNN
from data.dataset import load_dataset
from losses.losses import rpn_loss, fast_rcnn_loss

def main():
    # 1. Load config
    config = Config()
    
    # 2. Load data
    train_dataset = load_dataset('train', config)
    val_dataset = load_dataset('val', config)
    
    # 3. Build model
    model = FasterRCNN(config)
    
    # 4. Setup optimizer
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=config.LEARNING_RATE,
        momentum=config.MOMENTUM
    )
    
    # 5. Training loop
    for epoch in range(config.NUM_EPOCHS):
        for batch_idx, (images, gt_boxes, gt_labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # Forward pass
                outputs = model(images, training=True)
                
                # Compute losses
                total_loss = compute_loss(
                    outputs, gt_boxes, gt_labels, config
                )
            
            # Backward pass
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss}")
        
        # Validation
        val_loss = validate(model, val_dataset, config)
        print(f"Validation Loss: {val_loss}")
        
        # Save checkpoint
        model.save_weights(f'checkpoint_epoch_{epoch}.ckpt')

if __name__ == '__main__':
    main()
```

---

## 🔗 Tài liệu Tham Khảo

- **Faster R-CNN**: https://arxiv.org/abs/1506.01497
- **Feature Pyramid Networks**: https://arxiv.org/abs/1612.03144
- **Comparison Networks**: (paper của dự án này)

---

## 🎓 Bài Tập Thực Hành

1. **Implement anchors generation** → Visualize trên image
2. **Implement IoU computation** → Test với known values
3. **Implement RPN network** → Output class logits
4. **Implement RPN loss** → Training một RPN đơn giản
5. **Implement FPN** → Visualize multi-scale features
6. **Implement Fast R-CNN** → Full object detection pipeline

---

**Good luck với project của bạn! 🚀**

Nếu có câu hỏi, hãy hỏi từng phần để tôi giải thích chi tiết hơn.
