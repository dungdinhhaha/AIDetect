import tensorflow as tf
from tensorflow.keras import layers

class FPN(tf.keras.Model):
    def __init__(self, channels=256, name='fpn', **kwargs):
        super().__init__(name=name, **kwargs)
        self.lateral_c3 = layers.Conv2D(channels, 1, padding='same')
        self.lateral_c4 = layers.Conv2D(channels, 1, padding='same')
        self.lateral_c5 = layers.Conv2D(channels, 1, padding='same')
        self.smooth_p3 = layers.Conv2D(channels, 3, padding='same')
        self.smooth_p4 = layers.Conv2D(channels, 3, padding='same')
        self.smooth_p5 = layers.Conv2D(channels, 3, padding='same')

    def call(self, inputs, training=False):
        c3, c4, c5 = inputs
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + tf.image.resize(p5, tf.shape(c4)[1:3])
        p3 = self.lateral_c3(c3) + tf.image.resize(p4, tf.shape(c3)[1:3])
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)
        return [p3, p4, p5]


def build_fpn(channels=256):
    """Build Feature Pyramid Network."""
    return FPN(channels=channels)
