import tensorflow as tf
from tensorflow.keras import layers


def build_backbone(name: str = "resnet50", weights: str = "imagenet"):
    if name == "resnet101":
        base = tf.keras.applications.ResNet101(include_top=False, weights=weights)
    else:
        base = tf.keras.applications.ResNet50(include_top=False, weights=weights)

    c3 = base.get_layer('conv3_block4_out').output
    c4 = base.get_layer('conv4_block6_out').output if name == "resnet50" else base.get_layer('conv4_block23_out').output
    c5 = base.get_layer('conv5_block3_out').output

    return tf.keras.Model(inputs=base.input, outputs=[c3, c4, c5], name=f"{name}_backbone")
