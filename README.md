# 🫁 Chest X-ray Multi-Class Diagnostic System
### Robust Multi-Scan Inference · DenseNet121 · 6-Class Classification

---

## Overview

A production-grade deep learning system for chest X-ray classification that detects **6 thoracic conditions** with robust, uncertainty-aware predictions via a novel **multi-scan inference** strategy. Model output feeds into a clinical interpretation and medication guidance layer.

**Detectable Conditions:**
| Class | Description |
|-------|-------------|
| `normal` | No pathological findings |
| `pneumonia` | Bacterial / viral / atypical pneumonia |
| `pneumothorax` | Air in pleural space (emergency) |
| `tuberculosis` | Pulmonary TB patterns |
| `cardiomegaly` | Enlarged cardiac silhouette |
| `effusion` | Pleural fluid accumulation |

---

## Project Structure

```
xray_system/
├── config.py        # All hyperparameters and paths
├── dataset.py       # Dataset class + DataLoader factory
├── model.py         # DenseNet121 + custom classifier head
├── train.py         # Full training loop with early stopping
├── evaluate.py      # Test-set evaluation + confusion matrix
├── inference.py     # Multi-scan inference engine (core innovation)
├── diagnosis.py     # Post-diagnosis: severity, interpretation, meds
├── app.py           # Gradio web application
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone / download the project
cd xray_system

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure your dataset matches:
#    data/processed/{train,val,test}/{normal,pneumonia,pneumothorax,tuberculosis,cardiomegaly,effusion}/
```

---

## Training

```bash
# Standard training
python train.py

# Resume from checkpoint
python train.py --resume checkpoints/last_model.pth
```

Training features:
- **Backbone warm-up**: DenseNet121 backbone is frozen for the first 3 epochs, then unfrozen for full fine-tuning.
- **Cosine LR annealing** from 3×10⁻⁴ → 1×10⁻⁶ over 50 epochs.
- **Early stopping** on validation macro F1 (patience=8 epochs).
- **Label smoothing** (ε=0.1) for robustness to noisy annotations.
- TensorBoard logs saved to `logs/`. View with: `tensorboard --logdir logs`

---

## Evaluation

```bash
python evaluate.py --ckpt checkpoints/best_model.pth
```

Outputs:
- Classification report (precision / recall / F1 per class)
- Macro and weighted F1
- Per-class ROC-AUC (one-vs-rest)
- Confusion matrix PNG → `results/confusion_matrix.png`
- JSON summary → `results/test_results.json`

---

## Inference

### CLI — Single Image
```bash
python inference.py path/to/xray.png

# With custom settings
python inference.py path/to/xray.png --scans 8 --agg mean --json
```

### Python API
```python
from inference import build_engine

engine = build_engine("checkpoints/best_model.pth", num_scans=6)
result = engine.predict("xray.png")

print(result.primary_class)       # e.g. "pneumonia"
print(result.confidence_map)      # {class: confidence}
print(result.uncertainty_map)     # {class: std_dev across scans}
print(result.detections)          # conditions above threshold
```

---

## Web Application

```bash
# Local
python app.py

# Public URL (Gradio share)
python app.py --share --scans 6 --port 7860
```

Open `http://localhost:7860` in your browser.

---

## Multi-Scan Inference — How It Works

```
Input Image
    │
    ├─── Scan 1: Deterministic transform → Forward pass
    ├─── Scan 2: Stochastic transform + MC Dropout → Forward pass
    ├─── Scan 3: Stochastic transform + MC Dropout → Forward pass
    │    ...
    └─── Scan N: Stochastic transform + MC Dropout → Forward pass

    ─────────────────────────────────────────────────
    Aggregate N probability vectors (mean / max / vote)
    ─────────────────────────────────────────────────
    │
    ├── Mean confidence per class
    ├── Std deviation per class (uncertainty estimate)
    ├── Primary detection (highest mean confidence)
    └── Secondary detections (above threshold)
```

**Why multi-scan?**
- Reduces prediction instability from noisy labels
- Surfaces uncertainty explicitly (σ per class)
- Catches borderline multi-condition cases
- Makes the system robust to adversarial image variations

---

## Post-Diagnosis Layer

```python
from diagnosis import DiagnosisEngine

engine = DiagnosisEngine(region="MA")   # Morocco
report = engine.generate(scan_result)

print(report.severity)            # Severity.HIGH
print(report.interpretations)     # Clinical descriptions
print(report.follow_up_actions)   # Recommended steps
print(report.medications)         # Advisory medications
```

Severity levels: `Normal → Low → Moderate → High → Critical`

**Pneumothorax** always triggers a `CRITICAL` override due to lethality risk.

---

## Configuration

All key parameters live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_SCANS` | 6 | Stochastic inference passes |
| `CONFIDENCE_THRESHOLD` | 0.35 | Primary detection cutoff |
| `SECONDARY_THRESHOLD` | 0.25 | Possible finding cutoff |
| `LEARNING_RATE` | 3e-4 | AdamW LR |
| `LABEL_SMOOTHING` | 0.1 | CE smoothing ε |
| `PATIENCE` | 8 | Early stopping epochs |
| `DEFAULT_REGION` | "MA" | Medication region |

---

## Disclaimer

> ⚠ This system is an **AI-assisted research and screening tool**.
> It does **not** constitute a medical diagnosis, prescription, or treatment plan.
> All outputs must be reviewed by a **qualified medical professional** before any clinical decisions are made.
> The medication guidance is **advisory only** and is not a substitute for clinical judgment.
