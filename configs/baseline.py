"""Baseline configuration for the Biohub Cell Tracking project.

The constants in this file are intentionally plain Python values so the
configuration works in local scripts, Kaggle notebooks, and final offline runs
without requiring a separate config parser.
"""

from pathlib import Path


# Kaggle competition input root used during development.
DEFAULT_DATA_ROOT = Path(
    "/kaggle/input/competitions/biohub-cell-tracking-during-development"
)

TRAIN_DIR_NAME = "train"
TEST_DIR_NAME = "test"

IMAGE_ARRAY_KEY = "0"

# Axis order throughout the project is Z, Y, X for spatial coordinates.
VOXEL_SIZE_ZYX_UM = (1.625, 0.40625, 0.40625)

# First-version detector/tracker choices fixed from the global GT analysis.
TEMPORAL_OFFSETS = (-1, 0, 1)
CROP_SIZE_ZYX = (32, 128, 128)
TRACK_GATE_UM = 10.0
TARGET_SIGMA_UM = 1.5

# Dataset split. Split is by whole sample name, never by individual nodes.
VAL_FRACTION = 0.20
RANDOM_SEED = 42

# Positive-centered crops avoid the all-background failure mode on sparse labels.
POSITIVE_CROP_PROB = 0.70

# V1 keeps division handling out of the model and measures it in analysis only.
ENABLE_DIVISION_HEAD = False
