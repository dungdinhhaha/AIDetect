# 🎓 HƯỚNG DẪN LÀM DỰ ÁN OBJECT DETECTION CHO Y TẾ

**Dự án mẫu:** Phát hiện tế bào ung thư cổ tử cung (ComparisonDetector)  
**Thời gian:** 3-6 tháng (part-time)  
**Level:** Intermediate → Advanced

---

## 📋 MỤC LỤC

1. [Chuẩn bị & Lên kế hoạch](#bước-1-chuẩn-bị--lên-kế-hoạch)
2. [Thu thập dữ liệu](#bước-2-thu-thập-dữ-liệu)
3. [Gán nhãn dữ liệu](#bước-3-gán-nhãn-dữ-liệu)
4. [Setup môi trường](#bước-4-setup-môi-trường)
5. [Tạo data pipeline](#bước-5-tạo-data-pipeline)
6. [Xây dựng model](#bước-6-xây-dựng-model)
7. [Training](#bước-7-training)
8. [Evaluation](#bước-8-evaluation)
9. [Deployment](#bước-9-deployment)
10. [Tối ưu hóa](#bước-10-tối-ưu-hóa)

---

# BƯỚC 1: CHUẨN BỊ & LÊN KẾ HOẠCH

## 1.1. Chọn bài toán cụ thể

### ✅ **Các ý tưởng dự án tương tự:**

#### **A. Y tế (Medical Imaging):**
```
1. Phát hiện tế bào máu bất thường (Blood Cell Detection)
   - Dataset: BCCD Dataset, Kaggle
   - Classes: WBC, RBC, Platelets, abnormal cells
   - Độ khó: ⭐⭐⭐

2. Phát hiện khối u phổi (Lung Nodule Detection)
   - Dataset: LUNA16, NIH Chest X-rays
   - Classes: Nodule, Mass, Normal
   - Độ khó: ⭐⭐⭐⭐

3. Phát hiện võng mạc bệnh tiểu đường (Diabetic Retinopathy)
   - Dataset: Kaggle DR Detection
   - Classes: 5 levels (No DR → Proliferative DR)
   - Độ khó: ⭐⭐⭐⭐

4. Phân loại tế bào da ung thư (Skin Cancer Classification)
   - Dataset: ISIC Archive
   - Classes: Melanoma, Nevus, Seborrheic Keratosis...
   - Độ khó: ⭐⭐⭐
```

#### **B. Nông nghiệp:**
```
1. Phát hiện sâu bệnh trên lá cây
2. Đếm hoa quả trên cây (Apple, Orange counting)
3. Phân loại bệnh cây trồng
```

#### **C. Công nghiệp:**
```
1. Phát hiện lỗi sản phẩm (Defect detection)
2. Đếm linh kiện điện tử
3. Kiểm tra chất lượng hàn
```

### 📝 **Template chọn bài toán:**

```
[Tên dự án]: _______________________
[Mục tiêu]: Phát hiện/Phân loại ______ trong ảnh _______
[Input]: Ảnh kích thước ______ x ______
[Output]: 
  - Bounding boxes: [x1, y1, x2, y2]
  - Labels: [Class 1, Class 2, ...]
  - Confidence: 0.0 - 1.0
[Số classes]: ______ (+ 1 background)
[Dataset size]: ______ ảnh (tối thiểu 500-1000)
```

---

## 1.2. Nghiên cứu papers liên quan

### 📚 **Checklist nghiên cứu:**

```bash
[ ] Đọc 3-5 papers về bài toán tương tự
[ ] Tìm baseline model (YOLOv5, Faster R-CNN, RetinaNet)
[ ] Xem code implementation trên GitHub
[ ] Đọc discussion trên Kaggle (nếu có competition)
[ ] Tìm pretrained models
```

### 🔍 **Websites tìm papers:**
```
- paperswithcode.com
- arxiv.org
- Google Scholar
- Kaggle Notebooks (Code + Discussion)
```

---

## 1.3. Lên timeline

### 📅 **Timeline mẫu (6 tháng):**

```
Tháng 1: Thu thập + Gán nhãn dữ liệu (50%)
Tháng 2: Gán nhãn tiếp (50%) + Setup môi trường
Tháng 3: Data pipeline + Baseline model
Tháng 4: Training + Debugging
Tháng 5: Tối ưu model + Evaluation
Tháng 6: Deployment + Viết báo cáo
```

---

# BƯỚC 2: THU THẬP DỮ LIỆU

## 2.1. Tìm dataset public

### 🌐 **Nguồn dataset:**

```
1. Kaggle Datasets
   - https://www.kaggle.com/datasets
   - Search: "[your problem] detection dataset"

2. Papers with Code Datasets
   - https://paperswithcode.com/datasets
   - Filter by task: Object Detection

3. Medical-specific:
   - NIH Clinical Center: https://nihcc.app.box.com/v/ChestXray-NIHCC
   - Grand Challenges: https://grand-challenge.org/
   - ISBI Challenges: https://biomedicalimaging.org/

4. Roboflow Universe
   - https://universe.roboflow.com/
   - Pre-annotated datasets

5. GitHub Awesome Lists
   - "awesome-medical-imaging"
   - "awesome-object-detection"
```

### ✅ **Checklist đánh giá dataset:**

```
[ ] Đủ số lượng: Tối thiểu 500-1000 ảnh
[ ] Chất lượng tốt: Không blur, đủ sáng
[ ] Có annotations: Bounding boxes + labels
[ ] License cho phép sử dụng
[ ] Train/Test split có sẵn
[ ] Balanced classes (không lệch quá nhiều)
```

---

## 2.2. Thu thập dữ liệu riêng (nếu cần)

### 📸 **Hướng dẫn chụp ảnh:**

```python
# Quy tắc chụp ảnh y tế:
1. Resolution: Tối thiểu 1024x1024 pixels
2. Lighting: Đồng đều, không có shadow
3. Focus: Rõ nét, không blur
4. Background: Sạch, đơn giản
5. Angle: Nhất quán, thẳng góc
6. Số lượng: 
   - Mỗi class: Tối thiểu 100 ảnh
   - Total: 500-2000 ảnh

# Tool:
- Smartphone camera (12MP+)
- Microscope camera (cho medical)
- Scanner (cho slide)
```

### 📂 **Cấu trúc thư mục:**

```
raw_data/
├── class_1/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── class_2/
│   ├── img_001.jpg
│   └── ...
└── metadata.csv
```

---

# BƯỚC 3: GÁN NHÃN DỮ LIỆU

## 3.1. Chọn công cụ annotation

### 🛠️ **Công cụ miễn phí:**

| Tool | Pros | Cons | Link |
|------|------|------|------|
| **LabelImg** | Đơn giản, offline | Chậm cho nhiều ảnh | GitHub |
| **CVAT** | Web-based, team | Cần setup server | cvat.org |
| **Roboflow** | Auto-suggest, cloud | Free có giới hạn | roboflow.com |
| **VGG Image Annotator** | Lightweight | UI cũ | vgg.ox.ac.uk |
| **LabelMe** | Polygon support | Cần Python | GitHub |

### 💡 **Khuyến nghị:**
```
Solo project → LabelImg hoặc Roboflow
Team project → CVAT
Medical imaging → QuPath (cho WSI)
```

---

## 3.2. Hướng dẫn gán nhãn

### 📐 **Quy tắc vẽ bounding box:**

```
1. Box CHU bọc toàn bộ đối tượng
2. Không bỏ sót phần nào (tay, chân, viền...)
3. Không vẽ quá rộng (chừa khoảng trống)
4. Nhất quán về kích thước padding
5. Với đối tượng bị che khuất: Vẫn vẽ phần nhìn thấy

Ví dụ:
❌ Box quá nhỏ: [x: 10, y: 10, w: 30, h: 30] → thiếu viền
✅ Box vừa đủ: [x: 5, y: 5, w: 40, h: 40] → bao hết object
❌ Box quá lớn: [x: 0, y: 0, w: 60, h: 60] → nhiều background
```

### 🎯 **Quality control:**

```python
# Checklist review:
[ ] Mỗi object có đúng 1 box
[ ] Không overlap giữa các class khác nhau
[ ] Label chính xác 100%
[ ] Tọa độ nằm trong ảnh
[ ] Box không quá nhỏ (< 10x10 pixels)
[ ] Random check 10% annotations
```

---

## 3.3. Convert annotations sang COCO/PASCAL VOC format

### 📦 **COCO Format (Khuyến nghị):**

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img_001.jpg",
      "width": 640,
      "height": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "cell_type_1"},
    {"id": 2, "name": "cell_type_2"}
  ]
}
```

### 🔄 **Script convert:**

```python
# convert_to_coco.py
import json
import os
from PIL import Image

def labelimg_to_coco(labelimg_folder, output_json):
    """Convert LabelImg XML to COCO JSON"""
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    categories = ["background", "class_1", "class_2"]  # Thay đổi
    for i, cat in enumerate(categories):
        coco["categories"].append({
            "id": i,
            "name": cat
        })
    
    ann_id = 1
    for img_id, xml_file in enumerate(os.listdir(labelimg_folder)):
        if not xml_file.endswith('.xml'):
            continue
        
        # Parse XML (dùng xml.etree.ElementTree)
        # ... (xem full code trong project)
        
        coco["images"].append({
            "id": img_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })
        
        for box in boxes:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": box['category_id'],
                "bbox": [box['x'], box['y'], box['w'], box['h']],
                "area": box['w'] * box['h'],
                "iscrowd": 0
            })
            ann_id += 1
    
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)

# Usage:
labelimg_to_coco("annotations/", "coco_annotations.json")
```

---

# BƯỚC 4: SETUP MÔI TRƯỜNG

## 4.1. Cài đặt Python và dependencies

### 🐍 **Python environment:**

```bash
# 1. Cài Anaconda hoặc Miniconda
# Download: https://www.anaconda.com/

# 2. Tạo virtual environment
conda create -n cell_detection python=3.10
conda activate cell_detection

# 3. Cài TensorFlow (GPU)
pip install tensorflow==2.19.0
pip install tensorflow-gpu  # Nếu có NVIDIA GPU

# 4. Cài libraries
pip install opencv-python-headless
pip install pillow
pip install matplotlib
pip install scikit-image
pip install scipy
pip install tqdm
pip install pycocotools

# 5. Kiểm tra
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

---

## 4.2. Cấu trúc project

### 📁 **Template structure:**

```
my_detection_project/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
│
├── data/
│   ├── raw/                    # Ảnh gốc
│   ├── annotations/            # XML/JSON annotations
│   ├── tfrecords/             # TFRecord files
│   │   ├── train.tfrecord
│   │   └── test.tfrecord
│   └── splits/
│       ├── train.txt
│       └── test.txt
│
├── models/
│   ├── __init__.py
│   ├── backbone.py            # ResNet, VGG...
│   ├── fpn.py                 # Feature Pyramid
│   ├── rpn.py                 # Region Proposal Network
│   ├── detector.py            # Main model
│   └── losses.py              # Loss functions
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # Load TFRecord
│   ├── augmentation.py        # Data augmentation
│   ├── box_utils.py           # IoU, NMS...
│   └── visualize.py           # Vẽ boxes
│
├── scripts/
│   ├── prepare_data.py        # Convert to TFRecord
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation
│   └── inference.py           # Prediction
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_test.ipynb
│   └── 03_results_analysis.ipynb
│
├── outputs/
│   ├── models/                # Saved models
│   ├── logs/                  # TensorBoard logs
│   └── predictions/           # Inference results
│
└── deployment/
    ├── api/
    │   ├── app.py             # FastAPI
    │   └── Dockerfile
    └── web/
        └── index.html         # Demo UI
```

---

## 4.3. Config file template

### ⚙️ **config.py:**

```python
import os

class Config:
    # Project
    PROJECT_NAME = "my_cell_detection"
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    TFRECORD_DIR = os.path.join(DATA_DIR, "tfrecords")
    MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
    LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")
    
    # Dataset
    NUM_CLASSES = 3  # Số classes + background
    CLASS_NAMES = ["background", "class_1", "class_2"]
    IMAGE_SIZE = (640, 640)  # (height, width)
    TRAIN_SPLIT = 0.8  # 80% train, 20% test
    
    # Model
    BACKBONE = "resnet50"  # resnet50, resnet101, efficientnet
    BACKBONE_WEIGHTS = "imagenet"
    FPN_CHANNELS = 256
    
    # Training
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # Anchors
    ANCHOR_SCALES = [32, 64, 128, 256, 512]
    ANCHOR_RATIOS = [0.5, 1.0, 2.0]
    
    # Inference
    CONFIDENCE_THRESHOLD = 0.5
    NMS_IOU_THRESHOLD = 0.3
    MAX_DETECTIONS = 100
    
    # Augmentation
    USE_AUGMENTATION = True
    FLIP_HORIZONTAL = True
    FLIP_VERTICAL = False
    ROTATION_RANGE = 10  # degrees
    BRIGHTNESS_RANGE = 0.2
    
    # GPU
    USE_GPU = True
    MIXED_PRECISION = True  # Faster training

# Usage:
cfg = Config()
```

---

# BƯỚC 5: TẠO DATA PIPELINE

## 5.1. Convert to TFRecord

### 📦 **Script: `prepare_data.py`**

```python
import tensorflow as tf
import json
import cv2
import numpy as np
from config import Config

cfg = Config()

def create_tfrecord_from_coco(coco_json, image_dir, output_file):
    """
    Convert COCO annotations to TFRecord
    
    Args:
        coco_json: Path to COCO JSON file
        image_dir: Directory containing images
        output_file: Output TFRecord file
    """
    
    # Load COCO
    with open(coco_json) as f:
        coco = json.load(f)
    
    # Create category mapping
    cat_id_to_label = {cat['id']: i for i, cat in enumerate(coco['categories'])}
    
    # Group annotations by image
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Write TFRecord
    with tf.io.TFRecordWriter(output_file) as writer:
        for img_info in coco['images']:
            img_id = img_info['id']
            img_path = os.path.join(image_dir, img_info['file_name'])
            
            # Read image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_bytes = img.tobytes()
            
            # Get annotations
            anns = img_to_anns.get(img_id, [])
            
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.extend([x, y, x+w, y+h])
                labels.append(cat_id_to_label[ann['category_id']])
            
            # Create TFRecord example
            feature = {
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[0]])),
                'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[1]])),
                'gtboxes_and_label': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[np.array(boxes + labels, dtype=np.int32).tobytes()]
                    )
                ),
                'img_name': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img_info['file_name'].encode()])
                )
            }
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            
    print(f"✓ Created {output_file}")

# Usage:
create_tfrecord_from_coco(
    "data/annotations/train.json",
    "data/raw/images/",
    "data/tfrecords/train.tfrecord"
)
```

---

## 5.2. Data loader

### 🔄 **Script: `utils/data_loader.py`**

```python
import tensorflow as tf
from config import Config

cfg = Config()

def parse_tfrecord(example_proto):
    """Parse TFRecord example"""
    
    features = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'img_height': tf.io.FixedLenFeature([], tf.int64),
        'img_width': tf.io.FixedLenFeature([], tf.int64),
        'gtboxes_and_label': tf.io.FixedLenFeature([], tf.string),
        'img_name': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, features)
    
    # Decode image
    height = tf.cast(parsed['img_height'], tf.int32)
    width = tf.cast(parsed['img_width'], tf.int32)
    
    image = tf.io.decode_raw(parsed['img'], tf.uint8)
    image = tf.reshape(image, [height, width, 3])
    
    # Decode boxes and labels
    gtboxes_and_label = tf.io.decode_raw(parsed['gtboxes_and_label'], tf.int32)
    num_boxes = tf.shape(gtboxes_and_label)[0] // 5
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [num_boxes, 5])
    
    boxes = tf.cast(gtboxes_and_label[:, :4], tf.float32)
    labels = gtboxes_and_label[:, 4]
    
    return image, {'boxes': boxes, 'labels': labels}

def preprocess(image, targets, image_size=(640, 640)):
    """Preprocess image and targets"""
    
    # Resize image
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    # Normalize boxes to [0, 1]
    # TODO: Implement box normalization
    
    return image, targets

def augment(image, targets):
    """Data augmentation"""
    
    if cfg.USE_AUGMENTATION:
        # Random flip
        if cfg.FLIP_HORIZONTAL:
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                # TODO: Flip boxes coordinates
        
        # Random brightness
        image = tf.image.random_brightness(image, cfg.BRIGHTNESS_RANGE)
        
        # Random rotation (advanced)
        # TODO: Implement rotation with box transformation
    
    return image, targets

def build_dataset(tfrecord_paths, batch_size=4, shuffle=True, augment=False):
    """Build training/validation dataset"""
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Load TFRecord
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    
    # Parse
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    
    # Augment
    if augment:
        dataset = dataset.map(
            lambda img, tgt: augment(img, tgt),
            num_parallel_calls=AUTOTUNE
        )
    
    # Preprocess
    dataset = dataset.map(
        lambda img, tgt: preprocess(img, tgt, cfg.IMAGE_SIZE),
        num_parallel_calls=AUTOTUNE
    )
    
    # Shuffle and batch
    if shuffle:
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset
```

---

# BƯỚC 6: XÂY DỰNG MODEL

## 6.1. Backbone (Feature Extractor)

### 🧠 **Script: `models/backbone.py`**

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, ResNet101

def build_backbone(name='resnet50', weights='imagenet'):
    """
    Build feature extraction backbone
    
    Args:
        name: 'resnet50', 'resnet101', 'efficientnetb0'...
        weights: 'imagenet' or None
    
    Returns:
        Keras Model với multi-scale outputs
    """
    
    if name == 'resnet50':
        base_model = ResNet50(
            include_top=False,
            weights=weights,
            input_shape=(None, None, 3)
        )
    elif name == 'resnet101':
        base_model = ResNet101(
            include_top=False,
            weights=weights,
            input_shape=(None, None, 3)
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
    
    # Extract multi-scale features
    # C3, C4, C5 (stride 8, 16, 32)
    
    layer_names = {
        'resnet50': ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'],
        'resnet101': ['conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out']
    }
    
    outputs = [base_model.get_layer(name).output for name in layer_names[name]]
    
    backbone = Model(inputs=base_model.input, outputs=outputs, name=f'{name}_backbone')
    
    return backbone
```

---

## 6.2. Feature Pyramid Network (FPN)

### 🔺 **Script: `models/fpn.py`**

```python
class FPN(tf.keras.layers.Layer):
    """Feature Pyramid Network"""
    
    def __init__(self, channels=256, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        
        # Lateral convolutions (1x1)
        self.lateral_c5 = layers.Conv2D(channels, 1, name='lateral_c5')
        self.lateral_c4 = layers.Conv2D(channels, 1, name='lateral_c4')
        self.lateral_c3 = layers.Conv2D(channels, 1, name='lateral_c3')
        
        # Output convolutions (3x3)
        self.output_p5 = layers.Conv2D(channels, 3, padding='same', name='output_p5')
        self.output_p4 = layers.Conv2D(channels, 3, padding='same', name='output_p4')
        self.output_p3 = layers.Conv2D(channels, 3, padding='same', name='output_p3')
    
    def call(self, features, training=False):
        """
        Args:
            features: [C3, C4, C5] từ backbone
        Returns:
            pyramid: [P3, P4, P5] feature maps
        """
        c3, c4, c5 = features
        
        # Top-down pathway
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + tf.image.resize(p5, tf.shape(c4)[1:3])
        p3 = self.lateral_c3(c3) + tf.image.resize(p4, tf.shape(c3)[1:3])
        
        # Refine with 3x3 conv
        p5 = self.output_p5(p5)
        p4 = self.output_p4(p4)
        p3 = self.output_p3(p3)
        
        return [p3, p4, p5]
```

---

## 6.3. Region Proposal Network (RPN)

### 🎯 **Script: `models/rpn.py`**

```python
class RPN(tf.keras.layers.Layer):
    """Region Proposal Network"""
    
    def __init__(self, channels=256, num_anchors=9, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_anchors = num_anchors
        
        # Shared conv
        self.conv = layers.Conv2D(channels, 3, padding='same', activation='relu')
        
        # Objectness (có object hay không)
        self.objectness = layers.Conv2D(num_anchors, 1, name='rpn_objectness')
        
        # Bounding box regression
        self.bbox_reg = layers.Conv2D(num_anchors * 4, 1, name='rpn_bbox')
    
    def call(self, pyramid, training=False):
        """
        Args:
            pyramid: [P3, P4, P5]
        Returns:
            objectness_logits: List of [B, H, W, num_anchors]
            bbox_deltas: List of [B, H, W, num_anchors*4]
        """
        objectness_logits = []
        bbox_deltas = []
        
        for p in pyramid:
            # Shared conv
            x = self.conv(p)
            
            # Predictions
            obj = self.objectness(x)
            bbox = self.bbox_reg(x)
            
            objectness_logits.append(obj)
            bbox_deltas.append(bbox)
        
        return objectness_logits, bbox_deltas
```

---

## 6.4. Main Detector

### 🎯 **Script: `models/detector.py`**

```python
from models.backbone import build_backbone
from models.fpn import FPN
from models.rpn import RPN
from utils.box_utils import nms

class Detector(tf.keras.Model):
    """Complete object detection model"""
    
    def __init__(self, num_classes, backbone_name='resnet50', **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        
        # Components
        self.backbone = build_backbone(backbone_name)
        self.fpn = FPN(channels=256)
        self.rpn = RPN(channels=256, num_anchors=9)
        
        # Fast R-CNN head (simplified)
        self.roi_align = layers.Lambda(lambda x: x)  # TODO: Implement
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        """
        Args:
            inputs: [B, H, W, 3] images
        Returns:
            boxes: [N, 4] detected boxes
            labels: [N] class labels
            scores: [N] confidence scores
        """
        # Feature extraction
        c3, c4, c5 = self.backbone(inputs)
        
        # FPN
        pyramid = self.fpn([c3, c4, c5], training=training)
        
        # RPN
        objectness, bbox_deltas = self.rpn(pyramid, training=training)
        
        # Generate proposals (simplified)
        # TODO: Implement anchor generation and proposal selection
        
        # For now, return dummy outputs
        boxes = tf.zeros((1, 4))
        labels = tf.zeros((1,), dtype=tf.int32)
        scores = tf.zeros((1,))
        
        return boxes, labels, scores
    
    def predict_on_image(self, image):
        """Inference on single image"""
        # Preprocess
        img = tf.expand_dims(image, 0)
        
        # Predict
        boxes, labels, scores = self(img, training=False)
        
        return boxes.numpy(), labels.numpy(), scores.numpy()
```

---

# BƯỚC 7: TRAINING

## 7.1. Loss functions

### 📉 **Script: `models/losses.py`**

```python
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for imbalanced classification
    
    Args:
        y_true: Ground truth labels [B, N]
        y_pred: Predicted probabilities [B, N, num_classes]
        alpha: Balancing factor
        gamma: Focusing parameter
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    
    loss = weight * cross_entropy
    return tf.reduce_sum(loss, axis=-1)

def smooth_l1_loss(y_true, y_pred, sigma=3.0):
    """
    Smooth L1 loss for bounding box regression
    
    Args:
        y_true: Ground truth boxes [B, N, 4]
        y_pred: Predicted boxes [B, N, 4]
        sigma: Smoothing parameter
    """
    diff = tf.abs(y_true - y_pred)
    
    less_than_one = tf.cast(tf.less(diff, 1.0 / sigma ** 2), tf.float32)
    
    smooth_l1 = (less_than_one * 0.5 * sigma ** 2 * tf.pow(diff, 2)) + \
                ((1 - less_than_one) * (diff - 0.5 / sigma ** 2))
    
    return tf.reduce_mean(smooth_l1)

def detection_loss(y_true, y_pred):
    """
    Combined detection loss
    
    Returns:
        total_loss, cls_loss, box_loss
    """
    # Classification loss
    cls_loss = focal_loss(
        y_true['labels'],
        y_pred['class_probs']
    )
    
    # Box regression loss
    box_loss = smooth_l1_loss(
        y_true['boxes'],
        y_pred['box_deltas']
    )
    
    # Total loss
    total_loss = cls_loss + box_loss
    
    return total_loss, cls_loss, box_loss
```

---

## 7.2. Training script

### 🏋️ **Script: `train.py`**

```python
import tensorflow as tf
from tensorflow.keras import optimizers
from config import Config
from utils.data_loader import build_dataset
from models.detector import Detector
from models.losses import detection_loss

cfg = Config()

def train():
    """Main training function"""
    
    # Setup
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    
    # Build dataset
    train_ds = build_dataset(
        [os.path.join(cfg.TFRECORD_DIR, 'train.tfrecord')],
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        augment=True
    )
    
    val_ds = build_dataset(
        [os.path.join(cfg.TFRECORD_DIR, 'test.tfrecord')],
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    # Build model
    model = Detector(
        num_classes=cfg.NUM_CLASSES,
        backbone_name=cfg.BACKBONE
    )
    
    # Optimizer
    optimizer = optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
    
    # Metrics
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    
    # TensorBoard
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(cfg.LOG_DIR, 'train')
    )
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(cfg.LOG_DIR, 'val')
    )
    
    # Training loop
    for epoch in range(cfg.EPOCHS):
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS}")
        
        # Train
        train_loss_metric.reset_states()
        for step, (images, targets) in enumerate(train_ds):
            loss = train_step(model, images, targets, optimizer)
            train_loss_metric.update_state(loss)
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")
        
        # Validation
        val_loss_metric.reset_states()
        for images, targets in val_ds:
            loss = val_step(model, images, targets)
            val_loss_metric.update_state(loss)
        
        # Log
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss_metric.result(), step=epoch)
        
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss_metric.result(), step=epoch)
        
        print(f"Train Loss: {train_loss_metric.result():.4f}")
        print(f"Val Loss: {val_loss_metric.result():.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            model.save_weights(
                os.path.join(cfg.MODEL_DIR, f'model_epoch_{epoch+1}.h5')
            )
    
    # Save final model
    model.save(os.path.join(cfg.MODEL_DIR, 'final_model.keras'))
    print("\n✓ Training completed!")

@tf.function
def train_step(model, images, targets, optimizer):
    """Single training step"""
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss, cls_loss, box_loss = detection_loss(targets, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

@tf.function
def val_step(model, images, targets):
    """Single validation step"""
    predictions = model(images, training=False)
    loss, _, _ = detection_loss(targets, predictions)
    return loss

if __name__ == '__main__':
    train()
```

---

# BƯỚC 8: EVALUATION

## 8.1. Metrics

### 📊 **Script: `evaluate.py`**

```python
from utils.box_utils import calculate_iou

def calculate_ap(precision, recall):
    """Calculate Average Precision"""
    # 11-point interpolation
    ap = 0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.
    return ap

def evaluate_model(model, test_dataset, iou_threshold=0.5):
    """
    Evaluate model on test set
    
    Returns:
        mAP: Mean Average Precision
        per_class_ap: AP for each class
    """
    
    all_predictions = []
    all_ground_truths = []
    
    # Collect predictions
    for images, targets in test_dataset:
        boxes, labels, scores = model(images, training=False)
        
        all_predictions.append({
            'boxes': boxes.numpy(),
            'labels': labels.numpy(),
            'scores': scores.numpy()
        })
        
        all_ground_truths.append({
            'boxes': targets['boxes'].numpy(),
            'labels': targets['labels'].numpy()
        })
    
    # Calculate AP per class
    aps = []
    for class_id in range(1, cfg.NUM_CLASSES):  # Skip background
        precision, recall = calculate_precision_recall(
            all_predictions,
            all_ground_truths,
            class_id,
            iou_threshold
        )
        
        ap = calculate_ap(precision, recall)
        aps.append(ap)
        
        print(f"Class {cfg.CLASS_NAMES[class_id]}: AP = {ap:.4f}")
    
    mAP = np.mean(aps)
    print(f"\nmAP@{iou_threshold}: {mAP:.4f}")
    
    return mAP, aps

# Run evaluation
model = tf.keras.models.load_model('outputs/models/final_model.keras')
test_ds = build_dataset(['data/tfrecords/test.tfrecord'], batch_size=1)

mAP, aps = evaluate_model(model, test_ds)
```

---

# BƯỚC 9: DEPLOYMENT

## 9.1. FastAPI Service

### 🚀 **Script: `deployment/api/app.py`**

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Cell Detection API")

# Load model
MODEL_PATH = "../../outputs/models/final_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

@app.post("/detect")
async def detect_cells(file: UploadFile = File(...)):
    """Detect cells in uploaded image"""
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert('RGB')
        image = image.resize((640, 640))
        
        # Preprocess
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        # Predict
        boxes, labels, scores = model.predict(img_array)
        
        # Format response
        detections = []
        for i in range(len(boxes)):
            if scores[i] > 0.5:
                detections.append({
                    "box": boxes[i].tolist(),
                    "label": int(labels[i]),
                    "confidence": float(scores[i])
                })
        
        return {
            "status": "success",
            "total_detections": len(detections),
            "detections": detections
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
def health():
    return {"status": "healthy"}

# Run: uvicorn app:app --reload
```

---

## 9.2. Dockerfile

### 🐳 **deployment/api/Dockerfile:**

```dockerfile
FROM tensorflow/tensorflow:2.19.0-gpu

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

# BƯỚC 10: TỐI ƯU HÓA

## 10.1. Model optimization

### ⚡ **Techniques:**

```python
# 1. Model pruning
import tensorflow_model_optimization as tfmot

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    **pruning_params
)

# 2. Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 3. TensorRT (NVIDIA GPU)
from tensorflow.python.compiler.tensorrt import trt_convert as trt

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='saved_model/',
    precision_mode='FP16'
)
converter.convert()
converter.save('tensorrt_model/')
```

---

# 📚 CHECKLIST HOÀN THÀNH DỰ ÁN

## Giai đoạn 1: Data (Tuần 1-8)
```
[ ] Thu thập 500-2000 ảnh
[ ] Gán nhãn 100% ảnh
[ ] Split train/val/test (70/15/15)
[ ] Convert sang TFRecord
[ ] Data augmentation ready
```

## Giai đoạn 2: Model (Tuần 9-12)
```
[ ] Backbone implementation
[ ] FPN implementation
[ ] RPN implementation
[ ] Detector integration
[ ] Loss functions
```

## Giai đoạn 3: Training (Tuần 13-16)
```
[ ] Train baseline (ResNet50)
[ ] Monitor TensorBoard
[ ] Checkpoint best model
[ ] Hyperparameter tuning
[ ] Train final model
```

## Giai đoạn 4: Evaluation (Tuần 17-18)
```
[ ] mAP calculation
[ ] Confusion matrix
[ ] Error analysis
[ ] Compare với baseline papers
```

## Giai đoạn 5: Deployment (Tuần 19-20)
```
[ ] FastAPI service
[ ] Docker containerization
[ ] API documentation
[ ] Demo frontend
```

## Giai đoạn 6: Documentation (Tuần 21-24)
```
[ ] README.md
[ ] Code comments
[ ] API docs
[ ] Technical report/Paper
[ ] Presentation slides
```

---

# 🎯 TIPS THÀNH CÔNG

## DO:
✅ Bắt đầu với dataset nhỏ (100 ảnh) để test pipeline  
✅ Visualize mọi thứ (data, predictions, errors)  
✅ Version control với Git từ đầu  
✅ Log experiments (MLflow, Weights & Biases)  
✅ Đọc papers related works  
✅ Tham gia communities (Reddit, Discord)  

## DON'T:
❌ Train ngay trên full dataset mà chưa test  
❌ Bỏ qua data quality check  
❌ Hardcode paths và parameters  
❌ Train quá nhiều epochs mà không monitor  
❌ Bỏ qua validation set  
❌ Copy code mà không hiểu  

---

# 📞 LIÊN HỆ & HỎI ĐÁP

Nếu gặp khó khăn ở bước nào, hãy:

1. **Google error message** đầu tiên
2. **Check GitHub Issues** của libraries liên quan
3. **Hỏi trên Stack Overflow** với tag `tensorflow`, `object-detection`
4. **Discord communities:**
   - TensorFlow Discord
   - PyTorch Discord
   - r/MachineLearning
5. **Paper authors:** Email nếu cần clarification

---

# 🎓 KẾT LUẬN

Bạn vừa có một roadmap đầy đủ để tự làm dự án Object Detection từ A-Z!

**Nhớ:**
- Làm từng bước một, đừng vội
- Test và debug liên tục
- Document code ngay từ đầu
- Học từ errors (chúng là thầy tốt nhất!)

**Chúc bạn thành công! 🚀**

---

**Author:** Based on ComparisonDetector project  
**GitHub:** https://github.com/dungdinhhaha/AIDetect  
**Date:** December 2025
