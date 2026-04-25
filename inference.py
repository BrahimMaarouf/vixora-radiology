"""
inference.py — Multi-Scan Inference Engine

Core Innovation:
  Instead of a single forward pass, each image is evaluated N times with:
    • Stochastic dropout (MC Dropout) active in the model
    • Slight geometric / photometric perturbations per scan
  The N probability vectors are then aggregated (mean / max / vote) to produce:
    • A robust primary diagnosis (highest confidence condition)
    • All secondary conditions above SECONDARY_THRESHOLD
    • Per-class confidence scores + uncertainty estimates (std dev)

This approach reduces:
  • Prediction instability from noisy labels
  • False positives from borderline images
  • Overconfidence on ambiguous findings
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from dataset import get_inference_transform, get_stochastic_transform
from model import ChestXrayClassifier, load_model

logger = logging.getLogger(__name__)


# ─── Result Dataclass ─────────────────────────────────────────────────────────

class ScanResult:
    """
    Structured output of one multi-scan inference call.

    Attributes:
        primary_class:    The most-likely condition name.
        primary_conf:     Mean confidence for the primary class [0, 1].
        detections:       All flagged conditions with their confidence scores.
        possible_flags:   Conditions above SECONDARY but below PRIMARY threshold.
        confidence_map:   Full {class: mean_confidence} mapping.
        uncertainty_map:  Per-class std deviation across N scans.
        raw_probs:        (N, C) array of per-scan softmax probabilities.
        inference_ms:     Wall-clock time for the full multi-scan run.
    """

    def __init__(
        self,
        confidence_map  : Dict[str, float],
        uncertainty_map : Dict[str, float],
        raw_probs       : np.ndarray,
        inference_ms    : float,
    ) -> None:
        self.confidence_map  = confidence_map
        self.uncertainty_map = uncertainty_map
        self.raw_probs       = raw_probs
        self.inference_ms    = inference_ms

        # ── Derive detections ─────────────────────────────────────────────────
        self.detections: Dict[str, float] = {
            cls: conf
            for cls, conf in confidence_map.items()
            if conf >= config.CONFIDENCE_THRESHOLD
        }
        self.possible_flags: Dict[str, float] = {
            cls: conf
            for cls, conf in confidence_map.items()
            if config.SECONDARY_THRESHOLD <= conf < config.CONFIDENCE_THRESHOLD
        }

        # Primary = highest confidence class
        sorted_conf       = sorted(confidence_map.items(), key=lambda x: x[1], reverse=True)
        self.primary_class = sorted_conf[0][0]
        self.primary_conf  = sorted_conf[0][1]

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        det = ", ".join(f"{k}={v:.3f}" for k, v in self.detections.items()) or "none"
        return (
            f"ScanResult(primary='{self.primary_class}' @ {self.primary_conf:.3f}, "
            f"detections=[{det}], time={self.inference_ms:.0f}ms)"
        )

    def to_dict(self) -> dict:
        return {
            "primary_class"  : self.primary_class,
            "primary_conf"   : round(self.primary_conf, 4),
            "detections"     : {k: round(v, 4) for k, v in self.detections.items()},
            "possible_flags" : {k: round(v, 4) for k, v in self.possible_flags.items()},
            "confidence_map" : {k: round(v, 4) for k, v in self.confidence_map.items()},
            "uncertainty_map": {k: round(v, 4) for k, v in self.uncertainty_map.items()},
            "inference_ms"   : round(self.inference_ms, 1),
        }


# ─── Inference Engine ─────────────────────────────────────────────────────────

class MultiScanInferenceEngine:
    """
    Wraps a loaded ChestXrayClassifier and performs multi-scan inference.

    Args:
        model:        A loaded (and moved to device) ChestXrayClassifier.
        num_scans:    Number of stochastic forward passes (default: config.NUM_SCANS).
        aggregation:  "mean" | "max" | "vote"  (default: config.AGGREGATION_METHOD).
    """

    def __init__(
        self,
        model       : ChestXrayClassifier,
        num_scans   : int = config.NUM_SCANS,
        aggregation : str = config.AGGREGATION_METHOD,
    ) -> None:
        self.model       = model
        self.num_scans   = num_scans
        self.aggregation = aggregation

        # Transforms
        self._det_tf = get_inference_transform()    # deterministic (1st pass)
        self._sto_tf = get_stochastic_transform()   # stochastic (remaining passes)

    # ── Image Loading ─────────────────────────────────────────────────────────

    def _load_pil(self, source: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("L")
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("L")

    # ── Single Forward Pass ───────────────────────────────────────────────────

    @torch.no_grad()
    def _single_pass(self, tensor: torch.Tensor) -> np.ndarray:
        """Return softmax probabilities for one tensor of shape (1,3,H,W)."""
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]   # (C,)
        return probs

    # ── Multi-Scan Core ───────────────────────────────────────────────────────

    def predict(
        self,
        image        : Union[str, Path, Image.Image],
        return_raw   : bool = False,
    ) -> ScanResult:
        """
        Run N-scan stochastic inference on a single image.

        Args:
            image:       File path, Path object, or PIL Image.
            return_raw:  If True, include raw (N, C) prob array in result.

        Returns:
            ScanResult
        """
        t_start = time.perf_counter()

        pil_img  = self._load_pil(image)

        # Enable MC Dropout
        self.model.eval()
        if config.SCAN_DROPOUT_ACTIVE:
            self.model.enable_mc_dropout()

        all_probs: List[np.ndarray] = []

        for scan_idx in range(self.num_scans):
            # First pass: deterministic transform (clean anchor)
            tf     = self._det_tf if scan_idx == 0 else self._sto_tf
            tensor = tf(pil_img).unsqueeze(0).to(config.DEVICE)
            probs  = self._single_pass(tensor)
            all_probs.append(probs)

        raw_matrix = np.stack(all_probs, axis=0)   # (N, C)

        # ── Aggregation ───────────────────────────────────────────────────────
        if self.aggregation == "mean":
            agg_probs = raw_matrix.mean(axis=0)
        elif self.aggregation == "max":
            agg_probs = raw_matrix.max(axis=0)
        elif self.aggregation == "vote":
            votes     = raw_matrix.argmax(axis=1)
            vote_cnt  = np.bincount(votes, minlength=config.NUM_CLASSES) / self.num_scans
            agg_probs = vote_cnt
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        uncertainty = raw_matrix.std(axis=0)        # (C,)

        confidence_map  = {config.IDX_TO_CLASS[i]: float(agg_probs[i])  for i in range(config.NUM_CLASSES)}
        uncertainty_map = {config.IDX_TO_CLASS[i]: float(uncertainty[i]) for i in range(config.NUM_CLASSES)}

        inference_ms = (time.perf_counter() - t_start) * 1000
        result = ScanResult(confidence_map, uncertainty_map, raw_matrix if return_raw else np.array([]), inference_ms)
        logger.debug(result)
        return result

    # ── Batch Inference ───────────────────────────────────────────────────────

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
    ) -> List[ScanResult]:
        """Run multi-scan inference on a list of images."""
        results = []
        for i, img in enumerate(images):
            logger.info(f"[batch] Scanning {i+1}/{len(images)} ...")
            results.append(self.predict(img))
        return results


# ─── Convenience Function ─────────────────────────────────────────────────────

def build_engine(
    checkpoint_path : str  = config.BEST_CKPT,
    num_scans       : int  = config.NUM_SCANS,
    aggregation     : str  = config.AGGREGATION_METHOD,
) -> MultiScanInferenceEngine:
    """
    Convenience factory: load model from checkpoint and wrap in engine.

    Args:
        checkpoint_path: Path to saved .pth checkpoint.
        num_scans:       Number of stochastic forward passes.
        aggregation:     Aggregation strategy.

    Returns:
        Ready-to-use MultiScanInferenceEngine.
    """
    model  = load_model(checkpoint_path)
    engine = MultiScanInferenceEngine(model, num_scans=num_scans, aggregation=aggregation)
    logger.info(f"[engine] Ready | num_scans={num_scans} | aggregation={aggregation}")
    return engine


# ─── CLI Quick Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run multi-scan inference on a chest X-ray image")
    parser.add_argument("image",      type=str,                     help="Path to input X-ray image")
    parser.add_argument("--ckpt",     type=str, default=config.BEST_CKPT)
    parser.add_argument("--scans",    type=int, default=config.NUM_SCANS)
    parser.add_argument("--agg",      type=str, default=config.AGGREGATION_METHOD)
    parser.add_argument("--json",     action="store_true",          help="Output result as JSON")
    args = parser.parse_args()

    engine = build_engine(args.ckpt, args.scans, args.agg)
    result = engine.predict(args.image)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result)
        print("\nConfidence Map:")
        for cls, conf in sorted(result.confidence_map.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(conf * 30)
            print(f"  {cls:15s} {conf:.4f}  {bar}")
        if result.detections:
            print(f"\n⚠ Detected: {list(result.detections.keys())}")
        if result.possible_flags:
            print(f"? Possible:  {list(result.possible_flags.keys())}")
