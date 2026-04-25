"""
train.py — Full training pipeline v2.

v2 Changes vs v1:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  1. FocalLoss  — down-weights easy/noisy samples, focuses on hard cases  │
  │  2. Mixup      — soft label mixing, robust to NIH label noise             │
  │  3. Class weights — confusion-guided upweighting of clinically critical   │
  │     and visually confusable classes (pneumothorax, pneumonia, effusion)   │
  │  4. Longer warm-up (5 epochs) — TorchXRayVision backbone needs gentle LR │
  │  5. LR reset after warm-up at 30% of base LR                             │
  │  6. Per-epoch per-class F1 logged to TensorBoard                         │
  └──────────────────────────────────────────────────────────────────────────┘

Usage:
    python train.py
    python train.py --resume checkpoints/last_model.pth
    python train.py --no-mixup --no-focal    # ablation
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import build_dataloaders
from model import build_model

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOG_DIR, "train_v2.log")),
    ],
)
logger = logging.getLogger(__name__)


# ─── Reproducibility ──────────────────────────────────────────────────────────

def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss = -(1-pt)^γ · log(pt)

    • Down-weights well-classified / easy samples
    • Forces the model to focus training signal on hard, ambiguous cases
    • Critical for noisy-label datasets (NIH annotations have ~15-20% noise)

    Args:
        gamma:           Focusing parameter (2.0 is standard).
        label_smoothing: Applied before focal weighting.
        weight:          Per-class tensor for confusion-guided weighting.
    """

    def __init__(
        self,
        gamma           : float               = config.FOCAL_GAMMA,
        label_smoothing : float               = config.LABEL_SMOOTHING,
        weight          : torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)   # None or (C,) tensor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw model outputs
            targets: (B,)   integer class indices

        Returns:
            Scalar mean focal loss.
        """
        # Cross-entropy with label smoothing — reduction='none' for per-sample
        ce = F.cross_entropy(
            logits,
            targets,
            weight          = self.weight,
            label_smoothing = self.label_smoothing,
            reduction       = "none",
        )                           # (B,)

        # Focal weighting: (1 - p_t)^γ
        pt   = torch.exp(-ce)       # probability of correct class
        loss = (1.0 - pt) ** self.gamma * ce

        return loss.mean()


class MixupCrossEntropyLoss(nn.Module):
    """
    Standard CrossEntropy adapted for Mixup's two-label batches.
    Used when USE_MIXUP=True to handle (labels_a, labels_b, lambda) triplets.
    """

    def __init__(
        self,
        label_smoothing : float               = config.LABEL_SMOOTHING,
        weight          : torch.Tensor | None = None,
        gamma           : float               = config.FOCAL_GAMMA,
        use_focal       : bool                = config.USE_FOCAL_LOSS,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.use_focal       = use_focal
        self.gamma           = gamma
        self.register_buffer("weight", weight)

    def _single_ce(self, logits, targets):
        return F.cross_entropy(
            logits, targets,
            weight          = self.weight,
            label_smoothing = self.label_smoothing,
            reduction       = "none",
        )

    def forward(
        self,
        logits  : torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam     : float,
    ) -> torch.Tensor:
        ce_a = self._single_ce(logits, targets_a)
        ce_b = self._single_ce(logits, targets_b)
        ce   = lam * ce_a + (1.0 - lam) * ce_b            # (B,)

        if self.use_focal:
            pt   = torch.exp(-ce)
            ce   = (1.0 - pt) ** self.gamma * ce

        return ce.mean()


# ─── Mixup ────────────────────────────────────────────────────────────────────

def mixup_batch(
    images  : torch.Tensor,
    labels  : torch.Tensor,
    alpha   : float = config.MIXUP_ALPHA,
) -> tuple:
    """
    Beta(α, α) interpolation of image pairs within the batch.

    Returns:
        (mixed_images, labels_a, labels_b, lambda)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = config.PATIENCE, min_delta: float = config.MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            logger.info(f"[early_stop] No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ─── One Epoch ────────────────────────────────────────────────────────────────

def run_epoch(
    model       : nn.Module,
    loader,
    criterion   : nn.Module,
    mixup_crit  : nn.Module | None = None,
    optimiser               = None,
    phase       : str       = "train",
    use_mixup   : bool      = False,
) -> dict:
    """
    Run one full epoch (train or eval).

    Returns:
        dict: loss, macro_f1, per_class_f1, report
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []
    n_batches = len(loader)

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            # ── Mixup (training only) ──────────────────────────────────────────
            if is_train and use_mixup and mixup_crit is not None:
                mixed, labels_a, labels_b, lam = mixup_batch(images, labels)
                logits = model(mixed)
                loss   = mixup_crit(logits, labels_a, labels_b, lam)
            else:
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimiser.step()

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            running_loss += loss.item()

            if is_train and (batch_idx + 1) % max(1, n_batches // 5) == 0:
                logger.info(f"  [{batch_idx+1}/{n_batches}] loss={loss.item():.4f}")

    avg_loss  = running_loss / n_batches
    macro_f1  = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None,     zero_division=0)
    report    = classification_report(
        all_labels, all_preds,
        target_names=config.CLASSES,
        zero_division=0,
    )

    return {
        "loss"         : avg_loss,
        "macro_f1"     : macro_f1,
        "per_class_f1" : per_class.tolist(),
        "report"       : report,
    }


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model, optimiser, scheduler, epoch: int, metrics: dict, path: str) -> None:
    torch.save({
        "epoch"           : epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimiser.state_dict(),
        "sched_state_dict": scheduler.state_dict(),
        "metrics"         : metrics,
    }, path)
    logger.info(f"[ckpt] Saved → {path}")


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(
    resume_from   : str  | None = None,
    use_mixup     : bool        = config.USE_MIXUP,
    use_focal     : bool        = config.USE_FOCAL_LOSS,
    use_weights   : bool        = config.USE_CLASS_WEIGHTS,
) -> None:
    seed_everything()

    logger.info("=" * 70)
    logger.info("Chest X-ray Diagnostic System v2 — Training")
    logger.info(f"Device     : {config.DEVICE}")
    logger.info(f"Epochs     : {config.NUM_EPOCHS}  |  Batch: {config.BATCH_SIZE}")
    logger.info(f"Resolution : {config.IMAGE_SIZE}")
    logger.info(f"Backbone   : {'TorchXRayVision DenseNet121' if config.USE_TORCHXRAY else 'ImageNet DenseNet121'}")
    logger.info(f"Focal Loss : {use_focal}  |  Mixup: {use_mixup}  |  Class Weights: {use_weights}")
    logger.info("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model()

    # ── Class Weights ─────────────────────────────────────────────────────────
    class_weights = None
    if use_weights:
        class_weights = torch.tensor(
            config.CLASS_WEIGHTS, dtype=torch.float, device=config.DEVICE
        )
        logger.info(f"[train] Class weights: {dict(zip(config.CLASSES, config.CLASS_WEIGHTS))}")

    # ── Loss Functions ────────────────────────────────────────────────────────
    if use_focal:
        criterion  = FocalLoss(
            gamma           = config.FOCAL_GAMMA,
            label_smoothing = config.LABEL_SMOOTHING,
            weight          = class_weights,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight          = class_weights,
            label_smoothing = config.LABEL_SMOOTHING,
        )

    mixup_crit = MixupCrossEntropyLoss(
        label_smoothing = config.LABEL_SMOOTHING,
        weight          = class_weights,
        gamma           = config.FOCAL_GAMMA,
        use_focal       = use_focal,
    ) if use_mixup else None

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max   = config.T_MAX,
        eta_min = config.ETA_MIN,
    )

    # ── Optional Resume ───────────────────────────────────────────────────────
    start_epoch = 1
    best_val_f1 = 0.0

    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optim_state_dict"])
        scheduler.load_state_dict(ckpt["sched_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["metrics"].get("val_f1", 0.0)
        logger.info(f"[resume] Epoch {start_epoch} | best_val_F1={best_val_f1:.4f}")

    # ── TensorBoard + Early Stopping ──────────────────────────────────────────
    writer     = SummaryWriter(log_dir=config.LOG_DIR)
    early_stop = EarlyStopping()

    # ─────────────────────────────────────────────────────────────────────────
    backbone_unfrozen = False

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        t0 = time.time()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"\n{'─'*65}")
        logger.info(f"Epoch {epoch}/{config.NUM_EPOCHS}  |  LR={current_lr:.2e}")

        # ── Backbone warm-up schedule ─────────────────────────────────────────
        if epoch <= config.WARMUP_EPOCHS:
            if not backbone_unfrozen:
                model.freeze_backbone()
                logger.info("[warm-up] Backbone frozen — training head only.")
        elif not backbone_unfrozen:
            model.unfreeze_backbone()
            backbone_unfrozen = True
            # Re-add backbone params to optimiser with lower LR
            optimiser = torch.optim.AdamW([
                {"params": model.features.parameters(),    "lr": config.LEARNING_RATE * 0.1},
                {"params": model.classifier.parameters(),  "lr": config.LEARNING_RATE * 0.3},
            ], weight_decay=config.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max   = config.T_MAX - epoch,
                eta_min = config.ETA_MIN,
            )
            logger.info(
                f"[warm-up] Backbone unfrozen at epoch {epoch}. "
                f"Backbone LR={config.LEARNING_RATE * 0.1:.1e}, "
                f"Head LR={config.LEARNING_RATE * 0.3:.1e}"
            )

        # ── Mixup is disabled in last 10% of epochs ───────────────────────────
        # Allows the model to "sharpen" on clean boundaries near convergence
        effective_mixup = use_mixup and (epoch < int(config.NUM_EPOCHS * 0.90))

        # ── Train ─────────────────────────────────────────────────────────────
        train_m = run_epoch(
            model, train_loader, criterion, mixup_crit, optimiser,
            phase="train", use_mixup=effective_mixup,
        )
        logger.info(
            f"[TRAIN] loss={train_m['loss']:.4f}  macro_F1={train_m['macro_f1']:.4f}"
            + (f"  [mixup={'ON' if effective_mixup else 'OFF'}]" if use_mixup else "")
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_m = run_epoch(model, val_loader, criterion, phase="val")
        logger.info(
            f"[VAL]   loss={val_m['loss']:.4f}  macro_F1={val_m['macro_f1']:.4f}"
        )
        logger.info(f"\n{val_m['report']}")

        # ── Per-class F1 breakdown ─────────────────────────────────────────────
        for i, cls in enumerate(config.CLASSES):
            logger.info(
                f"  {cls:15s}  train_F1={train_m['per_class_f1'][i]:.4f}  "
                f"val_F1={val_m['per_class_f1'][i]:.4f}"
            )

        # ── TensorBoard ───────────────────────────────────────────────────────
        writer.add_scalars("Loss",     {"train": train_m["loss"],     "val": val_m["loss"]},     epoch)
        writer.add_scalars("Macro_F1", {"train": train_m["macro_f1"], "val": val_m["macro_f1"]}, epoch)
        writer.add_scalar("LR/head",   current_lr, epoch)
        for i, cls in enumerate(config.CLASSES):
            writer.add_scalars(
                f"F1/{cls}",
                {"train": train_m["per_class_f1"][i], "val": val_m["per_class_f1"][i]},
                epoch,
            )

        # ── Best model ────────────────────────────────────────────────────────
        val_f1 = val_m["macro_f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                model, optimiser, scheduler, epoch,
                {"val_f1": best_val_f1, "val_loss": val_m["loss"]},
                config.BEST_CKPT,
            )
            logger.info(f"★ New best val macro_F1 = {best_val_f1:.4f}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % config.SAVE_EVERY == 0:
            dated = os.path.join(config.CKPT_DIR, f"ckpt_epoch_{epoch:03d}.pth")
            save_checkpoint(
                model, optimiser, scheduler, epoch,
                {"val_f1": val_f1, "val_loss": val_m["loss"]},
                dated,
            )

        # ── LR Scheduler step ─────────────────────────────────────────────────
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(f"Epoch time: {elapsed:.1f}s  |  best_val_F1 so far={best_val_f1:.4f}")

        # ── Early Stopping ────────────────────────────────────────────────────
        if early_stop.step(val_f1):
            logger.info(f"[early_stop] Triggered at epoch {epoch}. Best F1={best_val_f1:.4f}")
            break

    # ── Final checkpoint ──────────────────────────────────────────────────────
    save_checkpoint(
        model, optimiser, scheduler, epoch,
        {"val_f1": val_f1, "val_loss": val_m["loss"]},
        config.LAST_CKPT,
    )
    writer.close()
    logger.info(f"\n{'='*70}")
    logger.info(f"Training complete. Best Val Macro F1 = {best_val_f1:.4f}")
    logger.info(f"Best model: {config.BEST_CKPT}")
    logger.info(f"{'='*70}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chest X-ray Classifier v2")
    parser.add_argument("--resume",     type=str,  default=None,  help="Checkpoint to resume from")
    parser.add_argument("--no-mixup",   action="store_true",      help="Disable Mixup augmentation")
    parser.add_argument("--no-focal",   action="store_true",      help="Use standard CE instead of Focal Loss")
    parser.add_argument("--no-weights", action="store_true",      help="Disable class weighting")
    args = parser.parse_args()

    train(
        resume_from = args.resume,
        use_mixup   = not args.no_mixup,
        use_focal   = not args.no_focal,
        use_weights = not args.no_weights,
    )