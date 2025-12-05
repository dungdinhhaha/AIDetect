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
        steps_per_epoch = 10  # Dummy dataset
    else:
        # Build dataset with .repeat() to avoid running out of data
        ds = build_dataset(tfrecord_paths, image_size=cfg.IMAGE_SIZE, batch_size=cfg.BATCH_SIZE, shuffle=1000)
        
        # Use all valid labels from all cells in each image
        def extract_all_labels(img, tgt):
            # tgt: {boxes: [B, 100, 4], labels: [B, 100], valid: [B, 100]}
            labels = tgt['labels']  # [B, 100]
            valid = tgt['valid']    # [B, 100]
            
            # Filter to only valid labels (where valid == 1)
            # For simplicity, we'll use the first valid label per image
            # but mark this as needing proper multi-label support
            batch_size = tf.shape(labels)[0]
            first_valid_labels = []
            
            for i in range(cfg.BATCH_SIZE):
                valid_mask = valid[i] > 0
                valid_labels = tf.boolean_mask(labels[i], valid_mask)
                # Use first valid label if any exist, else use 0
                first_label = tf.cond(
                    tf.size(valid_labels) > 0,
                    lambda: valid_labels[0],
                    lambda: tf.constant(0, dtype=labels.dtype)
                )
                first_valid_labels.append(first_label)
            
            first_valid_labels = tf.stack(first_valid_labels)
            return img, first_valid_labels
        
        ds = ds.map(extract_all_labels)
        
        # Calculate actual steps per epoch based on dataset size
        # Estimate: 2.3GB / ~500KB per sample ≈ 4600 images per TFRecord
        num_tfrecords = len(tfrecord_paths)
        estimated_samples = num_tfrecords * 4600  # Rough estimate
        steps_per_epoch = estimated_samples // cfg.BATCH_SIZE
        
        print(f'📊 Dataset info:')
        print(f'  - TFRecord files: {num_tfrecords}')
        print(f'  - Estimated samples: {estimated_samples}')
        print(f'  - Batch size: {cfg.BATCH_SIZE}')
        print(f'  - Steps per epoch: {steps_per_epoch}')
        
        # Add .repeat() to cycle through data indefinitely
        ds = ds.repeat()

    # Callbacks
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cfg.CHECKPOINT_DIR, 'ckpt_{epoch:02d}.weights.h5'),
        save_weights_only=True,
        save_freq='epoch'
    )
    
    best_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cfg.MODEL_DIR, 'best_model.h5'),
        save_best_only=True,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        verbose=1
    )
    
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=cfg.LOG_DIR, histogram_freq=1)
    
    # Learning rate scheduler (optional but recommended)
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # Compute class weights to handle imbalance
    # Assuming relatively equal distribution across classes
    class_weights = {i: 1.0 for i in range(cfg.NUM_CLASSES)}
    
    print(f'\n🚀 Starting training...')
    print(f'  Epochs: {cfg.EPOCHS}')
    print(f'  Steps per epoch: {steps_per_epoch}')
    print(f'  Total steps: {cfg.EPOCHS * steps_per_epoch}')
    print(f'  Class weights: {class_weights}')
    
    history = model.fit(
        ds, 
        epochs=cfg.EPOCHS, 
        steps_per_epoch=steps_per_epoch, 
        callbacks=[ckpt_cb, best_ckpt_cb, tb_cb, reduce_lr_cb],
        class_weight=class_weights
    )
    
    # Save final model
    final_model_path = os.path.join(cfg.MODEL_DIR, 'final_model.keras')
    model.save(final_model_path)
    print(f'\n✅ Training completed!')
    print(f'  Final model saved to: {final_model_path}')
    print(f'  Best model saved to: {os.path.join(cfg.MODEL_DIR, "best_model.h5")}')
    print(f'  TensorBoard logs: {cfg.LOG_DIR}')


if __name__ == '__main__':
    main()
