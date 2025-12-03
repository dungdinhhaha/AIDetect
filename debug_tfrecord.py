"""Debug TFRecord data"""
import tensorflow as tf
import numpy as np

tfrecord_path = '/content/drive/MyDrive/content/data/tct/train.tfrecord'

# Read first record
raw_dataset = tf.data.TFRecordDataset([tfrecord_path])

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    # Get image bytes
    img_bytes = example.features.feature['img'].bytes_list.value[0]
    print(f"Image bytes length: {len(img_bytes)}")
    print(f"First 20 bytes (hex): {img_bytes[:20].hex()}")
    
    # Check image format
    if img_bytes[:2] == b'\xff\xd8':
        print("Format: JPEG")
    elif img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        print("Format: PNG")
    elif img_bytes[:2] == b'BM':
        print("Format: BMP")
    else:
        print("Format: UNKNOWN - may be raw bytes")
        print(f"Image dimensions from TFRecord:")
        h = example.features.feature['img_height'].int64_list.value[0]
        w = example.features.feature['img_width'].int64_list.value[0]
        print(f"  Height: {h}, Width: {w}")
        print(f"  Expected size for RGB: {h*w*3}")
        print(f"  Actual size: {len(img_bytes)}")
    
    # Check boxes
    gtboxes = example.features.feature['gtboxes_and_label'].bytes_list.value[0]
    print(f"\nBoxes bytes length: {len(gtboxes)}")
    
    # Try decode as int32
    try:
        boxes_array = np.frombuffer(gtboxes, dtype=np.int32)
        print(f"Decoded as int32: {boxes_array}")
        print(f"Shape: {boxes_array.shape}")
    except:
        pass
    
    # Try decode as float32
    try:
        boxes_array = np.frombuffer(gtboxes, dtype=np.float32)
        print(f"Decoded as float32: {boxes_array}")
    except:
        pass
