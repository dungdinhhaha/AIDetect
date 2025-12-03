import tensorflow as tf

class ROIAlign(tf.keras.layers.Layer):
    def __init__(self, pool_size=7, name='roi_align', **kwargs):
        super().__init__(name=name, **kwargs)
        self.pool_size = pool_size

    def call(self, feature_map, boxes, box_indices):
        # boxes normalized [ymin, xmin, ymax, xmax]
        pooled = tf.image.crop_and_resize(feature_map, boxes, box_indices,
                                          crop_size=(self.pool_size, self.pool_size))
        return pooled
