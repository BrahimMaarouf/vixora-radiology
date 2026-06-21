"""
ChestX AI — Inference module
============================
Loads the ResNet-50 model trained in PFA.ipynb and runs:
  - 4-class prediction  (COVID · Normal · Pneumonia · Tuberculosis)
  - Grad-CAM heatmap on the last conv layer (layer4[-1].conv3)

The model architecture, class order, transforms, and Grad-CAM target layer
are taken verbatim from the training notebook so the checkpoint loads
without surprises and predictions match what you saw at training time.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# =============================================================================
# Constants — must match the training notebook exactly
# =============================================================================
# torchvision.datasets.ImageFolder sorts class folders alphabetically, so the
# checkpoint's output logits are in this exact order. DO NOT REORDER.
CLASSES: List[str] = ["COVID", "Normal", "Pneumonia", "Tuberculosis"]
NUM_CLASSES = len(CLASSES)

# ImageNet normalization stats (from notebook cell 12)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Eval transform — matches `eval_tf` in the notebook
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Per-class display metadata used by the UI
CLASS_META: Dict[str, Dict] = {
    "COVID":        {"icd": "U07.1",  "color": "#0f7e8a",
                     "description": "COVID-19, virus identified"},
    "Normal":       {"icd": "Z00.00", "color": "#0f9d58",
                     "description": "No acute cardiopulmonary findings"},
    "Pneumonia":    {"icd": "J18.9",  "color": "#dc2626",
                     "description": "Pneumonia, unspecified organism"},
    "Tuberculosis": {"icd": "A15.0",  "color": "#f59e0b",
                     "description": "Respiratory tuberculosis"},
}

# Severity assigned to the *winning* class (drives the diagnosis card color)
SEVERITY_BY_CLASS: Dict[str, str] = {
    "Normal":       "ok",
    "Pneumonia":    "alert",
    "COVID":        "alert",
    "Tuberculosis": "warn",
}


# =============================================================================
# Prediction dataclass — contract the UI renders against
# =============================================================================
@dataclass
class Prediction:
    label: str
    icd: str
    description: str
    confidence: float           # 0..1
    severity: str               # "alert" | "warn" | "ok"
    differentials: List[Dict]   # [{label, icd, prob, color}]
    findings: List[Dict]        # [{label, value, score, status}]


# =============================================================================
# Model — same architecture as build_model(num_classes) in the notebook
# =============================================================================
def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-50 with the custom head from the notebook (cell 15).
    pretrained=False because we're loading our own weights right after."""
    # weights=None avoids downloading ImageNet weights — we override them anyway
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


# =============================================================================
# Lazy singleton — load once, reuse across reruns (Streamlit calls top-level
# code on every interaction, so we cache the heavy model on the module).
# =============================================================================
_MODEL: nn.Module | None = None
_DEVICE: torch.device | None = None


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def load_model(weights_path: str | Path) -> nn.Module:
    """Load the trained checkpoint into a fresh model. Returns eval-mode model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {weights_path}\n"
            f"Expected location: <project>/model/best_model_final.pth"
        )

    device = get_device()
    model = build_model(NUM_CLASSES)
    state = torch.load(str(weights_path), map_location=device)

    # The notebook saves the raw state_dict, but be defensive about common wrappers
    if isinstance(state, dict) and "state_dict" in state and not any(
        k.startswith(("conv1", "layer", "fc")) for k in state.keys()
    ):
        state = state["state_dict"]

    # Strip a "module." prefix if the model was trained with DataParallel
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(device).eval()
    _MODEL = model
    return model


# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def predict_probs(model: nn.Module, image: Image.Image) -> np.ndarray:
    """Return softmax probabilities over the 4 classes, in CLASSES order."""
    device = get_device()
    x = EVAL_TF(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def run_inference(model: nn.Module, image: Image.Image) -> Prediction:
    """Full prediction including differentials and pseudo-findings."""
    probs = predict_probs(model, image)
    top_idx = int(np.argmax(probs))
    top_label = CLASSES[top_idx]
    top_conf  = float(probs[top_idx])

    # Differentials: all classes, sorted by probability desc
    order = np.argsort(-probs)
    differentials = [
        {
            "label": CLASSES[i],
            "icd":   CLASS_META[CLASSES[i]]["icd"],
            "prob":  float(probs[i]),
            "color": CLASS_META[CLASSES[i]]["color"],
        }
        for i in order
    ]

    # Findings strip — these are heuristic UI hints derived from the same
    # 4-class probabilities, not separate detectors. Keeps the layout populated
    # without overclaiming a finer-grained capability than the model has.
    findings = _derive_findings(probs)

    meta = CLASS_META[top_label]
    return Prediction(
        label=top_label,
        icd=meta["icd"],
        description=meta["description"],
        confidence=top_conf,
        severity=SEVERITY_BY_CLASS[top_label],
        differentials=differentials,
        findings=findings,
    )


def _derive_findings(probs: np.ndarray) -> List[Dict]:
    """Translate class probabilities into clinical-sounding findings tiles.
    These are illustrative, not separate predictions."""
    p = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}

    def status(score: float, alert_at=0.6, warn_at=0.3) -> str:
        return "alert" if score >= alert_at else "warn" if score >= warn_at else "ok"

    consolidation_score = p["Pneumonia"] + 0.5 * p["COVID"]
    consolidation_score = min(consolidation_score, 1.0)
    ground_glass_score  = p["COVID"]
    cavitation_score    = p["Tuberculosis"]
    normal_score        = p["Normal"]

    return [
        {"label": "Consolidation",     "value": _pct_label(consolidation_score),
         "score": consolidation_score, "status": status(consolidation_score)},
        {"label": "Ground-glass",      "value": _pct_label(ground_glass_score),
         "score": ground_glass_score,  "status": status(ground_glass_score)},
        {"label": "Cavitation / TB",   "value": _pct_label(cavitation_score),
         "score": cavitation_score,    "status": status(cavitation_score)},
        {"label": "Clear lung fields", "value": _pct_label(normal_score, positive=True),
         "score": normal_score,        "status": "ok" if normal_score >= 0.6 else "warn"},
    ]


def _pct_label(s: float, positive: bool = False) -> str:
    pct = int(round(s * 100))
    if positive:
        return f"Likely {pct}%" if pct >= 50 else f"Reduced ({pct}%)"
    if pct >= 60: return f"Likely {pct}%"
    if pct >= 30: return f"Possible {pct}%"
    return f"Unlikely ({pct}%)"


# =============================================================================
# Grad-CAM — same target layer as notebook cell 35
# =============================================================================
class GradCAM:
    """Grad-CAM hooked on layer4[-1].conv3, matching the notebook."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients:   torch.Tensor | None = None
        target_layer = model.layer4[-1].conv3
        # full_backward_hook is the modern replacement for backward_hook;
        # behavior is the same for our use case
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, image: Image.Image, class_idx: int | None = None
                 ) -> Tuple[np.ndarray, int]:
        device = get_device()
        x = EVAL_TF(image.convert("RGB")).unsqueeze(0).to(device)

        self.model.eval()
        # Enable grads even though parameters might be frozen
        for p in self.model.parameters():
            p.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot)

        # weights = global-avg-pool of gradients per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        return cam, class_idx


def make_gradcam_overlay(model: nn.Module, image: Image.Image,
                         alpha: float = 0.45) -> Image.Image:
    """Build the heatmap and composite it on top of the original radiograph.
    Returned image is 224×224 RGB, ready for the UI."""
    gc = GradCAM(model)
    cam, _ = gc.generate(image)

    base = image.convert("RGB").resize((224, 224))
    base_np = np.array(base).astype(np.float32) / 255.0

    # jet colormap on the cam → 0..255 RGB
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = (1 - alpha) * base_np + alpha * heatmap_rgb
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


# =============================================================================
# Convenience for the UI
# =============================================================================
def analyze(image: Image.Image, weights_path: str | Path
            ) -> Tuple[Prediction, Image.Image]:
    """One-call entry point: returns (Prediction, gradcam_overlay)."""
    model = load_model(weights_path)
    pred = run_inference(model, image)
    overlay = make_gradcam_overlay(model, image)
    return pred, overlay
