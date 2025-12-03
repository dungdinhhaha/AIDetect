import tensorflow as tf

def iou(boxes1, boxes2):
    # boxes: [N,4] in [ymin,xmin,ymax,xmax]
    y1 = tf.maximum(boxes1[:,0:1], boxes2[:,0])
    x1 = tf.maximum(boxes1[:,1:2], boxes2[:,1])
    y2 = tf.minimum(boxes1[:,2:3], boxes2[:,2])
    x2 = tf.minimum(boxes1[:,3:4], boxes2[:,3])
    inter = tf.maximum(0.0, y2 - y1) * tf.maximum(0.0, x2 - x1)
    area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
    area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
    union = area1[:,None] + area2[None,:] - inter
    return inter / tf.maximum(union, 1e-6)


def nms(boxes, scores, max_output=100, iou_thresh=0.5):
    idx = tf.image.non_max_suppression(boxes, scores, max_output, iou_threshold=iou_thresh)
    return tf.gather(boxes, idx), tf.gather(scores, idx)
