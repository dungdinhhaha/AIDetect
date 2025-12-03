import tensorflow as tf

# Simple losses as placeholders; refine with anchors/targets later

def rpn_objectness_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))


def rpn_bbox_loss(pred_deltas, target_deltas, weights=None):
    diff = pred_deltas - target_deltas
    loss = tf.keras.losses.Huber()(target_deltas, pred_deltas)
    if weights is not None:
        loss = tf.reduce_mean(loss * weights)
    else:
        loss = tf.reduce_mean(loss)
    return loss


def rcnn_cls_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def rcnn_bbox_loss(pred_deltas, target_deltas):
    return tf.reduce_mean(tf.keras.losses.Huber()(target_deltas, pred_deltas))
