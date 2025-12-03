import tensorflow as tf
from tensorflow.keras import layers

class FastRCNNHead(tf.keras.Model):
    def __init__(self, num_classes, fc_units=1024, name='fast_rcnn_head', **kwargs):
        super().__init__(name=name, **kwargs)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(fc_units, activation='relu')
        self.fc2 = layers.Dense(fc_units, activation='relu')
        self.cls_logits = layers.Dense(num_classes)
        self.bbox_deltas = layers.Dense(num_classes * 4)

    def call(self, pooled_features, training=False):
        x = self.flatten(pooled_features)
        x = self.fc1(x)
        x = self.fc2(x)
        cls = self.cls_logits(x)
        bbox = self.bbox_deltas(x)
        return cls, bbox


# Alias for backward compatibility
FastRCNN = FastRCNNHead
