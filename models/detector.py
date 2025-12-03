import tensorflow as tf
from models.backbone_keras import build_backbone
from models.fpn import FPN
from models.rpn import RPN
from utils.box_utils_tf2 import nms

class Detector(tf.keras.Model):
    def __init__(self, backbone_name='resnet50', backbone_weights='imagenet',
                 channels=256, num_anchors=9, num_classes=12, name='detector', **kwargs):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.backbone = build_backbone(backbone_name, backbone_weights)
        self.fpn = FPN(channels)
        self.rpn = RPN(channels, num_anchors)

    def call(self, inputs, training=False):
        c3, c4, c5 = self.backbone(inputs)
        pyramid = self.fpn([c3, c4, c5], training=training)
        obj_logits, bbox_deltas = self.rpn(pyramid, training=training)
        # For stub inference: pick the highest-resolution map and flatten
        obj = obj_logits[0]
        scores = tf.reshape(tf.nn.sigmoid(obj), [tf.shape(obj)[0], -1])
        # Dummy boxes: use center points grid (placeholder)
        h, w = tf.shape(obj)[1], tf.shape(obj)[2]
        yy = tf.linspace(0.0, 1.0, h)
        xx = tf.linspace(0.0, 1.0, w)
        Y, X = tf.meshgrid(yy, xx)
        boxes = tf.stack([Y-0.05, X-0.05, Y+0.05, X+0.05], axis=-1)
        boxes = tf.clip_by_value(boxes, 0.0, 1.0)
        boxes = tf.reshape(boxes, [-1, 4])
        # NMS per image (batch=1 for simplicity)
        sel_boxes, sel_scores = nms(boxes, scores[0], max_output=100, iou_thresh=0.5)
        return sel_boxes, sel_scores


# Alias for backward compatibility
ComparisonDetector = Detector
