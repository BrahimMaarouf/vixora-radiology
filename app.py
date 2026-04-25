"""
app.py — Gradio-based web interface for the Chest X-ray Diagnostic System.

Launch:
    python app.py

Optional flags:
    --ckpt   : path to model checkpoint   (default: config.BEST_CKPT)
    --scans  : number of inference scans  (default: config.NUM_SCANS)
    --port   : server port                (default: 7860)
    --share  : create a public Gradio URL
"""

import argparse
import json
import logging
import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from diagnosis import DiagnosisEngine, Severity
from inference import MultiScanInferenceEngine, build_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Lazy-load engine (avoids loading model until first request) ───────────────

_engine: MultiScanInferenceEngine = None
_diag_engine: DiagnosisEngine     = None


def get_engine(ckpt_path: str, num_scans: int) -> MultiScanInferenceEngine:
    global _engine
    if _engine is None:
        _engine = build_engine(ckpt_path, num_scans)
    return _engine


def get_diag_engine(region: str = config.DEFAULT_REGION) -> DiagnosisEngine:
    global _diag_engine
    if _diag_engine is None:
        _diag_engine = DiagnosisEngine(region=region)
    return _diag_engine


# ─── Confidence Bar Chart ─────────────────────────────────────────────────────

CONDITION_COLORS = {
    "normal"       : "#2ecc71",
    "pneumonia"    : "#e74c3c",
    "pneumothorax" : "#8e44ad",
    "tuberculosis" : "#d35400",
    "cardiomegaly" : "#2980b9",
    "effusion"     : "#16a085",
}


def plot_confidence_bars(confidence_map: dict, uncertainty_map: dict) -> plt.Figure:
    classes = config.CLASSES
    confs   = [confidence_map.get(c, 0.0) for c in classes]
    uncerts = [uncertainty_map.get(c, 0.0) for c in classes]
    colors  = [CONDITION_COLORS.get(c, "#7f8c8d") for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(
        classes, confs,
        color=colors, alpha=0.85,
        xerr=uncerts, error_kw={"ecolor": "white", "capsize": 4, "linewidth": 1.2},
    )
    # Threshold line
    ax.axvline(x=config.CONFIDENCE_THRESHOLD, color="white", linestyle="--", linewidth=1, alpha=0.5, label=f"Primary threshold ({config.CONFIDENCE_THRESHOLD})")
    ax.axvline(x=config.SECONDARY_THRESHOLD,  color="#f1c40f",linestyle=":",  linewidth=1, alpha=0.5, label=f"Secondary threshold ({config.SECONDARY_THRESHOLD})")

    # Value labels
    for bar, conf, unc in zip(bars, confs, uncerts):
        ax.text(
            min(conf + unc + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{conf:.3f}", va="center", ha="left", color="white", fontsize=8,
        )

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Confidence", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2c3e50")
    ax.legend(fontsize=7, facecolor="#16213e", edgecolor="#2c3e50", labelcolor="white")
    ax.set_title("Multi-Scan Confidence Distribution", color="white", fontsize=11, pad=8)

    plt.tight_layout()
    return fig


# ─── Inference Handler ────────────────────────────────────────────────────────

def run_inference(image, region: str, ckpt_path: str, num_scans: int):
    """
    Main Gradio handler.

    Args:
        image:     PIL Image from Gradio upload.
        region:    ISO country code for medication guidance.
        ckpt_path: Checkpoint file path.
        num_scans: Number of inference scans.

    Returns:
        Tuple of (report_md, chart_fig, raw_json)
    """
    if image is None:
        return "⚠ Please upload a chest X-ray image.", None, "{}"

    try:
        # ── Inference ─────────────────────────────────────────────────────────
        engine      = get_engine(ckpt_path, num_scans)
        scan_result = engine.predict(image, return_raw=False)

        # ── Diagnosis ─────────────────────────────────────────────────────────
        diag_engine = get_diag_engine(region)
        report      = diag_engine.generate(scan_result)

        # ── Confidence chart ──────────────────────────────────────────────────
        fig = plot_confidence_bars(scan_result.confidence_map, scan_result.uncertainty_map)

        # ── Markdown report ───────────────────────────────────────────────────
        sev       = report.severity
        det_table = "\n".join(
            f"| {cond} | {conf:.3f} |"
            for cond, conf in sorted(report.detected_conditions.items(), key=lambda x: x[1], reverse=True)
        )

        poss_section = ""
        if scan_result.possible_flags:
            poss_table = "\n".join(
                f"| {cond} | {conf:.3f} |"
                for cond, conf in scan_result.possible_flags.items()
            )
            poss_section = f"""
### 🔍 Possible Findings (Weak Signal)
| Condition | Confidence |
|-----------|-----------|
{poss_table}
"""

        interp_md = "\n\n".join(f"> {i}" for i in report.interpretations)

        fu_md = "\n".join(f"- {a}" for a in report.follow_up_actions) or "- Routine follow-up"

        # Medication table
        if report.medications:
            med_rows = "\n".join(
                f"| {m['name']} | {m['condition']} | {'✅ OTC' if m['otc'] else '🔒 Rx'} | {m['note']} |"
                for m in report.medications
            )
            med_section = f"""
### 💊 Medication Guidance *(Advisory Only — Region: {region.upper()})*
| Medication | Condition | Type | Note |
|------------|-----------|------|------|
{med_rows}
"""
        else:
            med_section = "### 💊 No Pharmacological Intervention Required"

        unc_highest = max(scan_result.uncertainty_map, key=scan_result.uncertainty_map.get)
        unc_val     = scan_result.uncertainty_map[unc_highest]
        unc_note    = f"Highest uncertainty: **{unc_highest}** (σ={unc_val:.3f})"

        md = f"""
# {sev.emoji} Chest X-ray Analysis Report

## Patient State: **{sev.value}** (Score: {report.severity_score:.2f})

---

### 🎯 Primary Finding: `{report.primary_condition.upper()}`

### ⚠ Detected Conditions
| Condition | Confidence |
|-----------|-----------|
{det_table}
{poss_section}

---

### 🔬 Clinical Interpretation
{interp_md}

---

### 📋 Recommended Follow-Up Actions
{fu_md}

---

{med_section}

---

### 📊 Scan Statistics
- Inference scans: **{num_scans}**
- Inference time: **{scan_result.inference_ms:.0f} ms**
- Aggregation: **{config.AGGREGATION_METHOD}**
- {unc_note}

---

### ⚠ Disclaimer
{report.disclaimer}
"""

        # ── Raw JSON ──────────────────────────────────────────────────────────
        raw = {
            "scan_result" : scan_result.to_dict(),
            "diagnosis"   : report.to_dict(),
        }

        return md, fig, json.dumps(raw, indent=2)

    except FileNotFoundError as e:
        return f"❌ Model checkpoint not found: {e}", None, "{}"
    except Exception as e:
        logger.exception("Inference error")
        return f"❌ Error during inference: {e}", None, "{}"


# ─── Gradio Interface ─────────────────────────────────────────────────────────

def build_interface(ckpt_path: str, num_scans: int) -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""

    # Pre-warm model if checkpoint exists
    if Path(ckpt_path).exists():
        logger.info("[app] Pre-warming model...")
        get_engine(ckpt_path, num_scans)
    else:
        logger.warning(f"[app] Checkpoint not found: {ckpt_path}. Model will be loaded on first request.")

    css = """
    .report-box { font-family: 'Courier New', monospace; }
    .gr-button-primary { background: #2980b9 !important; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="Chest X-ray Diagnostic System",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=css,
    ) as demo:

        gr.Markdown("""
        # 🫁 Chest X-ray Multi-Class Diagnostic System
        ### Robust Multi-Scan Inference · DenseNet121 · 6-Class Classification

        Upload a chest X-ray (PA or AP view) to receive AI-assisted analysis.
        Conditions detected: **Normal · Pneumonia · Pneumothorax · Tuberculosis · Cardiomegaly · Effusion**

        > ⚠ **This is a research/screening tool. Always consult a qualified clinician.**
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Chest X-ray (PNG / JPG)",
                    image_mode="L",
                    height=320,
                )
                with gr.Accordion("⚙ Advanced Settings", open=False):
                    region_input = gr.Dropdown(
                        choices=["MA", "DEFAULT", "EG", "TN", "DZ", "SA", "GB", "FR", "US"],
                        value=config.DEFAULT_REGION,
                        label="Region (for medication guidance)",
                    )
                    ckpt_input = gr.Textbox(
                        value=ckpt_path,
                        label="Model Checkpoint Path",
                    )
                    scans_input = gr.Slider(
                        minimum=1, maximum=12, step=1,
                        value=num_scans,
                        label="Number of Inference Scans",
                    )

                submit_btn = gr.Button("🔍 Analyse X-ray", variant="primary", size="lg")
                gr.Markdown("*Processing typically takes 2–8 seconds depending on scan count.*")

                gr.Examples(
                    examples=[],   # Populate with sample images if available
                    inputs=[image_input],
                    label="Example Images",
                )

            with gr.Column(scale=2):
                report_output = gr.Markdown(label="Diagnosis Report", elem_classes=["report-box"])
                chart_output  = gr.Plot(label="Confidence Distribution")
                json_output   = gr.Code(
                    label="Raw JSON Output",
                    language="json",
                    visible=True,
                )

        submit_btn.click(
            fn=lambda img, reg, ckpt, scans: run_inference(img, reg, ckpt, scans),
            inputs=[image_input, region_input, ckpt_input, scans_input],
            outputs=[report_output, chart_output, json_output],
        )

        gr.Markdown("""
        ---
        **Model**: DenseNet121 pretrained on ImageNet, fine-tuned on curated chest X-ray datasets.
        **Multi-Scan Inference**: Each image is evaluated N times with stochastic dropout + slight augmentation.
        Predictions are aggregated for robustness. Uncertainty (σ) per class is reported.
        """)

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Chest X-ray Diagnostic App")
    parser.add_argument("--ckpt",  type=str,  default=config.BEST_CKPT, help="Model checkpoint path")
    parser.add_argument("--scans", type=int,  default=config.NUM_SCANS,  help="Number of inference scans")
    parser.add_argument("--port",  type=int,  default=7860,              help="Server port")
    parser.add_argument("--share", action="store_true",                  help="Create public Gradio URL")
    args = parser.parse_args()

    demo = build_interface(args.ckpt, args.scans)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
