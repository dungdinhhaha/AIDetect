import tensorflow as tf
from tensorflow.keras import layers

class RPN(tf.keras.Model):
    def __init__(self, channels=256, num_anchors=9, name='rpn', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(channels, 3, padding='same', activation='relu')
        self.obj_logits = layers.Conv2D(num_anchors, 1, padding='same')
        self.bbox_deltas = layers.Conv2D(num_anchors * 4, 1, padding='same')

    def call(self, features, training=False):
        obj_outputs = []
        bbox_outputs = []
        for f in features:
            h = self.conv(f)
            obj_outputs.append(self.obj_logits(h))
            bbox_outputs.append(self.bbox_deltas(h))
        return obj_outputs, bbox_outputs
