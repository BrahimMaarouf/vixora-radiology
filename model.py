"""
model.py — DenseNet121 backbone with domain-specific pretraining.

v2 Changes:
  • Primary backbone: TorchXRayVision DenseNet121 pretrained on chest X-ray
    datasets (CheXpert + NIH + MIMIC-CXR + PadChest combined weights).
    This replaces ImageNet pretraining and is the single biggest quality lever.
  • Fallback: torchvision DenseNet121 (ImageNet) if torchxrayvision not installed.
  • Custom head: BN → Dropout(0.4) → Linear(1024→512) → GELU → BN → Dropout(0.2)
    → Linear(512→6).  Same structure as v1 — only the backbone weights change.
  • enable_mc_dropout(), freeze_backbone(), unfreeze_backbone() preserved.

Installation:
    pip install torchxrayvision
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config

logger = logging.getLogger(__name__)


# ─── Backbone Loader ──────────────────────────────────────────────────────────

def _load_torchxray_backbone() -> tuple[nn.Module, int]:
    """
    Load DenseNet121 pretrained on all major chest X-ray datasets via
    TorchXRayVision.  Returns (features_module, in_features).

    Weight key: "densenet121-res224-all"
      Trained on: NIH ChestX-ray14 + CheXpert + MIMIC-CXR + PadChest
    """
    import torchxrayvision as xrv   # noqa: PLC0415

    model_xrv = xrv.models.DenseNet(weights=config.MODEL_NAME)
    # xrv DenseNet has .features (same as torchvision) and .classifier (14-class)
    features    = model_xrv.features
    in_features = 1024    # DenseNet121 fixed output channels

    logger.info(
        "[model] Backbone: TorchXRayVision DenseNet121 "
        f"({config.MODEL_NAME}) — pretrained on chest X-ray datasets."
    )
    return features, in_features


def _load_imagenet_backbone() -> tuple[nn.Module, int]:
    """Fallback: torchvision DenseNet121 with ImageNet weights."""
    backbone    = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    features    = backbone.features
    in_features = backbone.classifier.in_features   # 1024
    logger.warning(
        "[model] TorchXRayVision not found — using ImageNet DenseNet121. "
        "Install with: pip install torchxrayvision"
    )
    return features, in_features


# ─── Classifier ───────────────────────────────────────────────────────────────

class ChestXrayClassifier(nn.Module):
    """
    DenseNet121 (domain-pretrained) + 6-class medical classifier head.

    Args:
        num_classes:  Number of output classes.
        dropout_rate: Dropout probability in the head.
        use_torchxray: If True (and torchxrayvision is installed), load X-ray
                       pretrained backbone. Falls back to ImageNet automatically.
    """

    def __init__(
        self,
        num_classes   : int   = config.NUM_CLASSES,
        dropout_rate  : float = config.DROPOUT_RATE,
        use_torchxray : bool  = config.USE_TORCHXRAY,
    ) -> None:
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        if use_torchxray:
            try:
                self.features, in_features = _load_torchxray_backbone()
            except (ImportError, Exception) as e:
                logger.warning(f"[model] TorchXRayVision load failed ({e}) — using ImageNet.")
                self.features, in_features = _load_imagenet_backbone()
        else:
            self.features, in_features = _load_imagenet_backbone()

        # ── Classifier Head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes),
        )

        # ── Head weight init ──────────────────────────────────────────────────
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — normalised float tensor.

        Returns:
            logits: (B, num_classes)
        """
        feats   = self.features(x)                                      # (B, 1024, h, w)
        out     = F.relu(feats, inplace=True)
        out     = F.adaptive_avg_pool2d(out, (1, 1))                    # (B, 1024, 1, 1)
        out     = torch.flatten(out, 1)                                  # (B, 1024)
        logits  = self.classifier(out)                                   # (B, C)
        return logits

    # ── MC Dropout ────────────────────────────────────────────────────────────

    def enable_mc_dropout(self) -> None:
        """Set all Dropout layers to train mode for Monte-Carlo inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    # ── Backbone freeze / unfreeze ────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ─── Factory Functions ────────────────────────────────────────────────────────

def build_model(
    num_classes   : int   = config.NUM_CLASSES,
    dropout_rate  : float = config.DROPOUT_RATE,
    use_torchxray : bool  = config.USE_TORCHXRAY,
) -> ChestXrayClassifier:
    """Build model and move to configured device."""
    model  = ChestXrayClassifier(num_classes, dropout_rate, use_torchxray)
    model  = model.to(config.DEVICE)
    params = model.count_parameters()
    logger.info(
        f"[model] Total params: {params['total']:,} | "
        f"Trainable: {params['trainable']:,} | "
        f"Device: {config.DEVICE}"
    )
    return model


def load_model(checkpoint_path: str, eval_mode: bool = True) -> ChestXrayClassifier:
    """
    Load a saved model from checkpoint.

    Args:
        checkpoint_path: Path to .pth file saved by train.py.
        eval_mode:       Call model.eval() after loading (default True).

    Returns:
        ChestXrayClassifier on config.DEVICE.
    """
    model = build_model(pretrained=False) if not config.USE_TORCHXRAY else build_model()
    ckpt  = torch.load(checkpoint_path, map_location=config.DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    if eval_mode:
        model.eval()
    logger.info(f"[model] Loaded checkpoint: {checkpoint_path}")
    return model