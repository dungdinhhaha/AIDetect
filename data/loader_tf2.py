import tensorflow as tf
from typing import Tuple
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE


def parse_example(example_proto: tf.Tensor) -> Tuple[tf.Tensor, dict]:
    # Updated parser for your TFRecord format
    features = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'img_height': tf.io.FixedLenFeature([], tf.int64),
        'img_width': tf.io.FixedLenFeature([], tf.int64),
        'gtboxes_and_label': tf.io.FixedLenFeature([], tf.string),
        'img_name': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    
    # Decode image
    image = tf.image.decode_jpeg(parsed['img'], channels=3)
    
    # Decode boxes and labels (stored as serialized bytes)
    # Format: [x1, y1, x2, y2, label, x1, y1, x2, y2, label, ...]
    gtboxes_and_label = tf.io.decode_raw(parsed['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])  # Nx5: [x1, y1, x2, y2, label]
    
    # Extract boxes [x1, y1, x2, y2] and convert to [y1, x1, y2, x2] format
    boxes = tf.cast(gtboxes_and_label[:, :4], tf.float32)
    # Swap from [x1, y1, x2, y2] to [y1, x1, y2, x2]
    boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
    
    # Extract labels
    labels = gtboxes_and_label[:, 4]
    
    target = {'boxes': boxes, 'labels': labels}
    return image, target


def preprocess(image: tf.Tensor, target: dict, image_size: Tuple[int, int]) -> Tuple[tf.Tensor, dict]:
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    return image, target


def build_dataset(tfrecord_paths, image_size=(640, 640), batch_size=2, shuffle=1000):
    ds = tf.data.TFRecordDataset(tfrecord_paths, num_parallel_reads=AUTOTUNE)
    ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img, tgt: preprocess(img, tgt, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(shuffle).batch(batch_size).prefetch(AUTOTUNE)
    return ds
