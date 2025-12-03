import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from models.backbone_keras import build_backbone
from configs.config_v2 import ConfigV2

print('TF version:', tf.__version__)

cfg = ConfigV2()
backbone = build_backbone(cfg.BACKBONE, cfg.BACKBONE_WEIGHTS)

dummy = tf.random.uniform((1, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], 3))
outputs = backbone(dummy)
print('Backbone outputs shapes:', [o.shape for o in outputs])
print('✓ Smoke TF2 passed')
