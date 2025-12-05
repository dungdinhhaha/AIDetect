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
        # Simple head: global pooling + dense for classification
        inputs = backbone.input
        features = backbone(inputs)[-1]
        x = tf.keras.layers.GlobalAveragePooling2D()(features)
        outputs = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cell_classifier_v2')

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
        
        # Extract individual cells from each image using box coordinates
        def extract_cells_from_boxes(img, tgt):
            # tgt: {boxes: [B, 100, 4], labels: [B, 100], valid: [B, 100]}
            # This function extracts ALL valid cells from the full image
            # and prepares them as separate training samples
            
            boxes = tgt['boxes']      # [B, 100, 4] in [y1, x1, y2, x2] format
            labels = tgt['labels']    # [B, 100]
            valid = tgt['valid']      # [B, 100]
            
            batch_size = tf.shape(img)[0]
            img_height = tf.shape(img)[1]
            img_width = tf.shape(img)[2]
            
            cell_images = []
            cell_labels = []
            
            # For each image in batch
            for b in range(batch_size):
                img_b = img[b]  # [H, W, 3]
                boxes_b = boxes[b]  # [100, 4]
                labels_b = labels[b]  # [100]
                valid_b = valid[b]  # [100]
                
                # Extract all valid cells from this image
                for i in range(100):
                    if valid_b[i] > 0:  # Only valid boxes
                        y1, x1, y2, x2 = boxes_b[i]
                        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
                        
                        # Clip to image bounds
                        y1 = tf.maximum(y1, 0)
                        x1 = tf.maximum(x1, 0)
                        y2 = tf.minimum(y2, img_height)
                        x2 = tf.minimum(x2, img_width)
                        
                        # Skip if invalid box
                        if y2 <= y1 or x2 <= x1:
                            continue
                        
                        # Crop cell from image
                        cell = tf.image.crop_to_bounding_box(
                            img_b, 
                            y1, x1, 
                            y2 - y1, x2 - x1
                        )
                        
                        # Resize to model input size
                        cell = tf.image.resize(cell, cfg.IMAGE_SIZE)
                        
                        cell_images.append(cell)
                        cell_labels.append(labels_b[i])
            
            # If we have cells, convert to tensors and return
            if len(cell_images) > 0:
                cell_images = tf.stack(cell_images)
                cell_labels = tf.stack(cell_labels)
                return cell_images, cell_labels
            else:
                # Return empty tensors if no valid cells
                return tf.zeros((0, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], 3)), tf.zeros((0,), dtype=tf.int32)
        
        # Apply the cell extraction function
        ds = ds.map(extract_cells_from_boxes, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Filter out empty batches
        ds = ds.filter(lambda x, y: tf.shape(x)[0] > 0)
        
        # Unbatch to get individual cells, then rebatch
        ds = ds.unbatch()
        ds = ds.batch(cfg.BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        # Calculate actual steps per epoch based on dataset size
        # Estimate: 2.3GB / ~500KB per sample ≈ 4600 images * ~5 cells per image ≈ 23000 cells
        num_tfrecords = len(tfrecord_paths)
        estimated_samples = num_tfrecords * 4600  # Rough estimate of images
        estimated_cells = estimated_samples * 5  # Average 5 cells per image
        steps_per_epoch = max(1, estimated_cells // cfg.BATCH_SIZE)
        
        print(f'📊 Dataset info:')
        print(f'  - TFRecord files: {num_tfrecords}')
        print(f'  - Estimated images: {estimated_samples}')
        print(f'  - Estimated cells (assuming ~5 per image): {estimated_cells}')
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
        filepath=os.path.join(cfg.MODEL_DIR, 'best_model_multicell.h5'),
        save_best_only=True,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        verbose=1
    )
    
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=cfg.LOG_DIR, histogram_freq=1)
    
    # Learning rate scheduler
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # Compute class weights to handle imbalance
    class_weights = {i: 1.0 for i in range(cfg.NUM_CLASSES)}
    
    print(f'\n🚀 Starting multi-cell training...')
    print(f'  Epochs: {cfg.EPOCHS}')
    print(f'  Steps per epoch: {steps_per_epoch}')
    print(f'  Total steps: {cfg.EPOCHS * steps_per_epoch}')
    print(f'  Class weights: {class_weights}')
    print(f'\n  Training on INDIVIDUAL CELLS extracted from boxes!')
    print(f'  → Each cell is classified independently')
    print()
    
    history = model.fit(
        ds, 
        epochs=cfg.EPOCHS, 
        steps_per_epoch=steps_per_epoch, 
        callbacks=[ckpt_cb, best_ckpt_cb, tb_cb, reduce_lr_cb],
        class_weight=class_weights
    )
    
    # Save final model
    final_model_path = os.path.join(cfg.MODEL_DIR, 'final_model_multicell.keras')
    model.save(final_model_path)
    print(f'\n✅ Multi-cell training completed!')
    print(f'  Final model saved to: {final_model_path}')
    print(f'  Best model saved to: {os.path.join(cfg.MODEL_DIR, "best_model_multicell.h5")}')
    print(f'  TensorBoard logs: {cfg.LOG_DIR}')
    print(f'\n💡 Usage:')
    print(f'  - Load best_model_multicell.h5')
    print(f'  - For each input image, extract cells using box coordinates')
    print(f'  - Classify each cell → get per-cell predictions!')


if __name__ == '__main__':
    main()
