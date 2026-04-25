"""
dataset.py — Dataset loading, CLAHE preprocessing, and Albumentations medical
             augmentation pipeline.

Changes from v1:
  • CLAHE applied as first step (boosts local anatomical contrast)
  • Albumentations replaces torchvision transforms for training
  • Resolution upgraded to 320×320
  • Stochastic inference transform also upgraded
  • Full graceful degradation on bad images
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms   # still used for val/test deterministic path

import config

logger = logging.getLogger(__name__)


# ─── CLAHE Helper ─────────────────────────────────────────────────────────────

class CLAHE:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Enhances local contrast in X-rays without over-amplifying noise.
    Applied to grayscale numpy arrays.
    """
    def __init__(
        self,
        clip_limit: float = config.CLAHE_CLIP_LIMIT,
        tile_grid: tuple  = config.CLAHE_TILE_GRID,
    ) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def __call__(self, img_gray: np.ndarray) -> np.ndarray:
        """Input: H×W uint8 grayscale. Output: H×W uint8 CLAHE-enhanced."""
        return self._clahe.apply(img_gray)


_clahe = CLAHE()   # module-level singleton — no overhead per sample


def pil_to_gray_np(img: Image.Image) -> np.ndarray:
    """PIL Image (any mode) → H×W uint8 numpy grayscale array."""
    return np.array(img.convert("L"), dtype=np.uint8)


def gray_np_to_3ch_np(arr: np.ndarray) -> np.ndarray:
    """H×W uint8 → H×W×3 uint8 (grayscale replicated to 3 channels)."""
    return np.stack([arr, arr, arr], axis=-1)


# ─── Albumentations Pipelines ─────────────────────────────────────────────────

def _norm_and_totensor() -> list:
    """Shared final steps for all splits."""
    return [
        A.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ToTensorV2(),
    ]


def get_train_transform() -> A.Compose:
    """
    Full medical augmentation pipeline for training.
    All operations are anatomically valid for chest X-rays.
    """
    return A.Compose([
        # ── Geometry ──────────────────────────────────────────────────────────
        A.HorizontalFlip(p=config.H_FLIP_PROB),
        A.ShiftScaleRotate(
            shift_limit  = config.SHIFT_LIMIT,
            scale_limit  = config.SCALE_LIMIT,
            rotate_limit = config.ROTATION_LIMIT,
            border_mode  = cv2.BORDER_CONSTANT,
            value        = 0,
            p            = 0.70,
        ),
        # Elastic/grid distortion — simulates anatomical variation
        A.OneOf([
            A.ElasticTransform(
                alpha       = 30,
                sigma       = 5,
                alpha_affine= 5,
                border_mode = cv2.BORDER_CONSTANT,
                p           = 1.0,
            ),
            A.GridDistortion(
                num_steps    = 5,
                distort_limit= 0.10,
                border_mode  = cv2.BORDER_CONSTANT,
                p            = 1.0,
            ),
        ], p=config.ELASTIC_PROB),

        # ── Intensity ─────────────────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit = config.BRIGHTNESS_LIMIT,
            contrast_limit   = config.CONTRAST_LIMIT,
            p                = 0.60,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=config.GAUSS_BLUR_PROB),
        A.GaussNoise(var_limit=(5.0, 20.0), p=config.GAUSS_NOISE_PROB),

        # ── Resize (after augmentation) ───────────────────────────────────────
        A.Resize(*config.IMAGE_SIZE),

        # ── Normalise + Tensor ────────────────────────────────────────────────
        *_norm_and_totensor(),
    ])


def get_val_transform() -> A.Compose:
    """Deterministic pipeline for val/test."""
    return A.Compose([
        A.Resize(*config.IMAGE_SIZE),
        *_norm_and_totensor(),
    ])


def get_stochastic_inference_transform() -> A.Compose:
    """
    Light stochastic transform used during multi-scan inference.
    Less aggressive than training — just enough to diversify per-scan predictions.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.30),
        A.ShiftScaleRotate(
            shift_limit  = 0.03,
            scale_limit  = 0.05,
            rotate_limit = 5,
            border_mode  = cv2.BORDER_CONSTANT,
            value        = 0,
            p            = 0.50,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.05,
            contrast_limit  =0.05,
            p=0.40,
        ),
        A.Resize(*config.IMAGE_SIZE),
        *_norm_and_totensor(),
    ])


def get_deterministic_inference_transform() -> A.Compose:
    """Clean anchor transform — used for the first scan pass."""
    return A.Compose([
        A.Resize(*config.IMAGE_SIZE),
        *_norm_and_totensor(),
    ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    Folder-based chest X-ray dataset with CLAHE preprocessing.

    Pipeline per sample:
        1. Load image as PIL → convert to grayscale numpy (H×W)
        2. Apply CLAHE (local contrast enhancement)
        3. Replicate to 3 channels (H×W×3)
        4. Apply Albumentations transform (augment / normalise / tensorise)

    Args:
        root:      Root directory containing class subdirectories.
        split:     "train" | "val" | "test"  (controls augmentation).
        transform: Custom Albumentations Compose (overrides split-default).
    """

    def __init__(
        self,
        root     : str,
        split    : str = "train",
        transform: Optional[A.Compose] = None,
    ) -> None:
        super().__init__()
        self.root      = Path(root)
        self.split     = split
        self.transform = transform or (
            get_train_transform() if split == "train" else get_val_transform()
        )
        self.samples   = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        for class_name in config.CLASSES:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                logger.warning(f"[{self.split}] Missing class folder: {class_dir}")
                continue
            idx = config.CLASS_TO_IDX[class_name]
            for p in sorted(class_dir.glob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    samples.append((p, idx))
        logger.info(f"[{self.split}] {len(samples)} samples | {config.NUM_CLASSES} classes")
        return samples

    def _load_image(self, path: Path) -> np.ndarray:
        """Load → grayscale → CLAHE → 3-channel numpy (H×W×3 uint8)."""
        try:
            pil = Image.open(path)
        except Exception as e:
            logger.error(f"Cannot open {path}: {e} — using blank image")
            pil = Image.new("L", (256, 256), 0)

        gray = pil_to_gray_np(pil)            # H×W uint8
        gray = _clahe(gray)                    # CLAHE enhancement
        rgb  = gray_np_to_3ch_np(gray)         # H×W×3 uint8
        return rgb

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = self._load_image(path)                    # numpy H×W×3
        if self.transform:
            img = self.transform(image=img)["image"]    # albumentations API
        return img, label

    def class_distribution(self) -> dict:
        dist = {c: 0 for c in config.CLASSES}
        for _, label in self.samples:
            dist[config.IDX_TO_CLASS[label]] += 1
        return dist


# ─── DataLoader Factory ────────────────────────────────────────────────────────

def build_dataloaders(
    batch_size  : int  = config.BATCH_SIZE,
    num_workers : int  = config.NUM_WORKERS,
    pin_memory  : bool = config.PIN_MEMORY,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = ChestXrayDataset(config.TRAIN_DIR, split="train")
    val_ds   = ChestXrayDataset(config.VAL_DIR,   split="val")
    test_ds  = ChestXrayDataset(config.TEST_DIR,  split="test")

    # Log class distributions
    for ds in [train_ds, val_ds, test_ds]:
        logger.info(f"[{ds.split}] Distribution: {ds.class_distribution()}")

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )
    return train_loader, val_loader, test_loader