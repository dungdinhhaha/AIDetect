class ConfigV2:
    # Paths (Colab defaults)
    DATA_DIR = "/content/data/tct"
    MODEL_DIR = "/content/drive/MyDrive/comparison_detector_models_v2"
    LOG_DIR = f"{MODEL_DIR}/logs"
    CHECKPOINT_DIR = f"{MODEL_DIR}/checkpoints"

    # Training hyperparameters
    BATCH_SIZE = 2
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_BOUNDARIES = [30000, 50000]

    # Dataset
    NUM_CLASSES = 12  # 11 + background
    IMAGE_SIZE = (640, 640)

    # Backbone
    BACKBONE = "resnet50"  # or "resnet101"
    BACKBONE_WEIGHTS = "imagenet"

    # FPN / RPN / ROI
    ROI_POOL_SIZE = 7
    ANCHOR_SCALES = [32, 64, 128, 256, 512]
    ANCHOR_RATIOS = [0.5, 1.0, 2.0]

    # Distribution
    USE_DISTRIBUTE = True

    # Save
    SAVE_EVERY_N_STEPS = 500
