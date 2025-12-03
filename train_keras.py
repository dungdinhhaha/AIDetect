import os
import tensorflow as tf
from tensorflow.keras import optimizers
from configs.config_v2 import ConfigV2
from data.loader_tf2 import build_dataset
from models.backbone_keras import build_backbone


def main():
    cfg = ConfigV2()
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    strategy = tf.distribute.MirroredStrategy() if cfg.USE_DISTRIBUTE else tf.distribute.get_strategy()
    with strategy.scope():
        backbone = build_backbone(cfg.BACKBONE, cfg.BACKBONE_WEIGHTS)
        # Simple head: global pooling + dense for smoke; replace with FPN/RPN/RCNN later
        inputs = backbone.input
        features = backbone(inputs)[-1]
        x = tf.keras.layers.GlobalAveragePooling2D()(features)
        outputs = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='comparison_detector_v2')

        opt = optimizers.SGD(learning_rate=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Dataset placeholders: update TFRecord paths
    tfrecord_paths = tf.io.gfile.glob(os.path.join(cfg.DATA_DIR, '*.tfrecord'))
    if not tfrecord_paths:
        print('Warning: No TFRecords found in', cfg.DATA_DIR)
        print('Running a dummy dataset for smoke test...')
        dummy_images = tf.random.uniform((cfg.BATCH_SIZE, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], 3))
        dummy_labels = tf.random.uniform((cfg.BATCH_SIZE,), minval=0, maxval=cfg.NUM_CLASSES, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels)).batch(cfg.BATCH_SIZE)
    else:
        ds = build_dataset(tfrecord_paths, image_size=cfg.IMAGE_SIZE, batch_size=cfg.BATCH_SIZE)
        # Map targets to labels placeholder (for the simple head smoke)
        ds = ds.map(lambda img, tgt: (img, tf.zeros((), dtype=tf.int32)))

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cfg.CHECKPOINT_DIR, 'ckpt_{epoch:02d}.weights.h5'),
        save_weights_only=True,
        save_freq='epoch'
    )
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=cfg.LOG_DIR)

    model.fit(ds, epochs=1, steps_per_epoch=10, callbacks=[ckpt_cb, tb_cb])
    model.save(os.path.join(cfg.MODEL_DIR, 'model.keras'))
    print('✓ Training smoke run completed')


if __name__ == '__main__':
    main()
