"""
config.py — Central configuration for Chest X-ray Multi-Class Diagnostic System
Updated: CLAHE, 320x320, Focal Loss, Mixup, TorchXRayVision, medical augmentation
"""

import os
import torch
from typing import List, Tuple

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = "data/processed"
TRAIN_DIR   = os.path.join(DATA_ROOT, "train")
VAL_DIR     = os.path.join(DATA_ROOT, "val")
TEST_DIR    = os.path.join(DATA_ROOT, "test")
CKPT_DIR    = "checkpoints"
LOG_DIR     = "logs"
RESULTS_DIR = "results"

for _d in [CKPT_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Classes ───────────────────────────────────────────────────────────────────
CLASSES: List[str] = [
    "normal",
    "pneumonia",
    "pneumothorax",
    "tuberculosis",
    "cardiomegaly",
    "effusion",
]
NUM_CLASSES  = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ─── Image Settings ────────────────────────────────────────────────────────────
IMAGE_SIZE: Tuple[int, int] = (320, 320)   # ↑ from 224 — captures finer details
NUM_CHANNELS = 3

# ─── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME       = "densenet121-res224-all"   # TorchXRayVision pretrained weights
USE_TORCHXRAY    = True                        # False → fallback to ImageNet DenseNet121
DROPOUT_RATE     = 0.4

# ─── Training ──────────────────────────────────────────────────────────────────
SEED             = 42
BATCH_SIZE       = 32
NUM_EPOCHS       = 60
LEARNING_RATE    = 2e-4       # slightly lower — domain-pretrained backbone is closer
WEIGHT_DECAY     = 1e-4
GRADIENT_CLIP    = 1.0

# Cosine Annealing
T_MAX            = 60
ETA_MIN          = 1e-6

# Warm-up phase — freeze backbone for first N epochs
WARMUP_EPOCHS    = 5          # longer warm-up: TorchXRayVision features need gentle start

# Early stopping
PATIENCE         = 10
MIN_DELTA        = 1e-4

# DataLoader
NUM_WORKERS      = 4
PIN_MEMORY       = True

# ─── Augmentation (Albumentations medical pipeline) ────────────────────────────
AUGMENT_TRAIN        = True
CLAHE_CLIP_LIMIT     = 2.0
CLAHE_TILE_GRID      = (8, 8)
H_FLIP_PROB          = 0.5
ROTATION_LIMIT       = 10          # ±10° — anatomically safe
SHIFT_LIMIT          = 0.05
SCALE_LIMIT          = 0.10
ELASTIC_PROB         = 0.4         # elastic / grid distortion probability
BRIGHTNESS_LIMIT     = 0.15
CONTRAST_LIMIT       = 0.15
GAUSS_BLUR_PROB      = 0.2
GAUSS_NOISE_PROB     = 0.3

# ─── Normalisation ─────────────────────────────────────────────────────────────
# TorchXRayVision models expect [-1024, 1024] normalised range internally,
# but we apply standard ImageNet stats after converting to 3-ch float tensor
# so the custom head trains stably.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# ─── Loss ──────────────────────────────────────────────────────────────────────
USE_FOCAL_LOSS   = True
FOCAL_GAMMA      = 2.0
LABEL_SMOOTHING  = 0.10

# Confusion-guided class weights (tune after first confusion matrix inspection)
# Order: normal, pneumonia, pneumothorax, tuberculosis, cardiomegaly, effusion
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS     = [1.0, 1.5, 2.0, 1.3, 1.0, 1.4]

# ─── Mixup ─────────────────────────────────────────────────────────────────────
USE_MIXUP    = True
MIXUP_ALPHA  = 0.3

# ─── Multi-Scan Inference ──────────────────────────────────────────────────────
NUM_SCANS            = 6
SCAN_DROPOUT_ACTIVE  = True
CONFIDENCE_THRESHOLD = 0.35
SECONDARY_THRESHOLD  = 0.25
AGGREGATION_METHOD   = "mean"

# ─── Post-Diagnosis Layer ──────────────────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "critical" : 0.80,
    "high"     : 0.60,
    "moderate" : 0.40,
    "low"      : 0.20,
}
DEFAULT_REGION = "MA"

# ─── Checkpointing ─────────────────────────────────────────────────────────────
BEST_CKPT  = os.path.join(CKPT_DIR, "best_model.pth")
LAST_CKPT  = os.path.join(CKPT_DIR, "last_model.pth")
SAVE_EVERY = 5

# ─── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")