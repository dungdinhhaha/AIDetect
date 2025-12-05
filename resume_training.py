"""
Resume training from checkpoint
"""
import os
import tensorflow as tf
from tensorflow.keras import optimizers
from configs.config_v2 import ConfigV2
from data.loader_tf2 import build_dataset
from models.backbone_keras import build_backbone


def main():
    cfg = ConfigV2()
    
    # Find latest checkpoint
    checkpoint_files = tf.io.gfile.glob(os.path.join(cfg.CHECKPOINT_DIR, 'ckpt_*.weights.h5'))
    
    if not checkpoint_files:
        print('❌ No checkpoint found in', cfg.CHECKPOINT_DIR)
        print('Please run train_keras.py first')
        return
    
    # Get latest checkpoint
    latest_checkpoint = sorted(checkpoint_files)[-1]
    # Extract epoch number from filename: ckpt_07.weights.h5 -> 7
    epoch_num = int(os.path.basename(latest_checkpoint).split('_')[1].split('.')[0])
    
    print(f'📂 Resuming from checkpoint:')
    print(f'  File: {latest_checkpoint}')
    print(f'  Epoch: {epoch_num}')
    
    strategy = tf.distribute.MirroredStrategy() if cfg.USE_DISTRIBUTE else tf.distribute.get_strategy()
    with strategy.scope():
        # Rebuild model
        backbone = build_backbone(cfg.BACKBONE, cfg.BACKBONE_WEIGHTS)
        inputs = backbone.input
        features = backbone(inputs)[-1]
        x = tf.keras.layers.GlobalAveragePooling2D()(features)
        outputs = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='comparison_detector_v2')
        
        # Load weights
        print(f'⏳ Loading weights...')
        model.load_weights(latest_checkpoint)
        print(f'✅ Weights loaded!')
        
        opt = optimizers.SGD(learning_rate=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Build dataset
    tfrecord_paths = tf.io.gfile.glob(os.path.join(cfg.DATA_DIR, '*.tfrecord'))
    if not tfrecord_paths:
        print('❌ No TFRecords found in', cfg.DATA_DIR)
        return
    
    ds = build_dataset(tfrecord_paths, image_size=cfg.IMAGE_SIZE, batch_size=cfg.BATCH_SIZE, shuffle=1000)
    
    def extract_label(img, tgt):
        labels = tgt['labels']
        first_labels = labels[:, 0]
        return img, first_labels
    
    ds = ds.map(extract_label)
    
    # Calculate steps
    num_tfrecords = len(tfrecord_paths)
    estimated_samples = num_tfrecords * 4600
    steps_per_epoch = estimated_samples // cfg.BATCH_SIZE
    
    print(f'\n📊 Dataset info:')
    print(f'  - TFRecord files: {num_tfrecords}')
    print(f'  - Estimated samples: {estimated_samples}')
    print(f'  - Steps per epoch: {steps_per_epoch}')
    
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
    
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Resume from next epoch
    initial_epoch = epoch_num
    remaining_epochs = cfg.EPOCHS - initial_epoch
    
    print(f'\n🚀 Resuming training...')
    print(f'  Initial epoch: {initial_epoch}')
    print(f'  Target epochs: {cfg.EPOCHS}')
    print(f'  Remaining: {remaining_epochs} epochs')
    print(f'  Steps per epoch: {steps_per_epoch}')
    
    history = model.fit(
        ds,
        initial_epoch=initial_epoch,
        epochs=cfg.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[ckpt_cb, best_ckpt_cb, tb_cb, reduce_lr_cb]
    )
    
    # Save final model
    final_model_path = os.path.join(cfg.MODEL_DIR, 'final_model.keras')
    model.save(final_model_path)
    
    print(f'\n✅ Training completed!')
    print(f'  Final model: {final_model_path}')
    print(f'  Best model: {os.path.join(cfg.MODEL_DIR, "best_model.h5")}')


if __name__ == '__main__':
    main()
