import tensorflow as tf
from typing import Tuple

AUTOTUNE = tf.data.AUTOTUNE


def parse_example(example_proto: tf.Tensor) -> Tuple[tf.Tensor, dict]:
    # Placeholder TFRecord parser schema; adjust to your actual TFRecord
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)
    # Convert sparse to dense
    labels = tf.sparse.to_dense(parsed['image/object/class/label'])
    xmin = tf.sparse.to_dense(parsed['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(parsed['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(parsed['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(parsed['image/object/bbox/ymax'])
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
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
