# sunsets/model/config.py
"""
Configuration for sunset_ml app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

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


# ── Weather / METAR config ─────────────────────────────────────────
# Local timezone that your JSON datetimes are in (image timestamps).
LOCAL_TIMEZONE: str = "America/Los_Angeles"

# Primary airport / station near your camera location (change if needed).
# Examples: KSAN (San Diego), KSFO (San Francisco), KLAX (Los Angeles)
WEATHER_STATION_ID: str = "KSAN"

# FAA/NOAA ADDS dataserver endpoint
WEATHER_SOURCE_URL: str = "https://aviationweather.gov/dataserver_current"

# How far around the target '-2h' time to search for METARs
WEATHER_LOOKBACK_HOURS: int = 3
WEATHER_LOOKAHEAD_HOURS: int = 1

# HTTP behavior
WEATHER_REQUEST_TIMEOUT_SEC: float = 10.0
WEATHER_MAX_RETRIES: int = 3

# If the nearest obs is AFTER the target time by more than this (minutes),
# you *could* skip it. We keep it anyway in the module, but expose the knob.
WEATHER_MIN_OBS_AGE_MIN: int = 90