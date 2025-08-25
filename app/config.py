# sunsets/model/config.py
"""
Configuration for sunset_ml app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

# This file lives in: sunsets/model/config.py
# BASE_DIR points to the "sunsets" folder
BASE_DIR: Path = Path(__file__).resolve().parent.parent  # -> sunsets/

# Paths
JSON_PATH: Path = BASE_DIR / "sunset_images_grouped.json"
IMAGE_DIR: Path = BASE_DIR / "img" / "ob_hotel"
MODEL_DIR: Path = BASE_DIR / "model"
MODEL_PATH: Path = MODEL_DIR / "sunset_dual_resnet18_best.pth"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Model / training config
NUM_CLASSES: int = 5
IMAGE_SIZE: int = 64
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 10
LEARNING_RATE: float = 1e-4
VAL_SPLIT: float = 0.2
RANDOM_SEED: int = 42

# Normalization (ImageNet)
IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)
