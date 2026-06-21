# ChestX AI — `web_ui`

The real Streamlit application. Same visual design as `web_guide/`, but wired to your
trained ResNet-50 (`model/best_model_final.pth`).

## Project layout this expects

```
C:\Users\bdaar\Desktop\pfa\
├── model\
│   └── best_model_final.pth        ← your trained checkpoint
├── web_guide\                       ← reference design (unchanged)
└── web_ui\                          ← THIS folder
    ├── app.py                       Streamlit app
    ├── inference.py                 model + transforms + Grad-CAM
    ├── requirements.txt
    └── README.md
```

The app finds the checkpoint by going one directory up from `web_ui/` and looking
in `model/`. To use a different path, set the env var `CHESTX_WEIGHTS`.

## One-time setup

Open **PowerShell** in `C:\Users\bdaar\Desktop\pfa` and run:

```powershell
# 1. Create a venv (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r web_ui\requirements.txt
```

If `pip install torch` is slow or you want a specific CUDA build, follow the
selector at https://pytorch.org/get-started/locally/ (e.g. for CUDA 12.1:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).

## Run

```powershell
streamlit run web_ui\app.py
```

Your browser opens at http://localhost:8501. The first inference takes a few
seconds (model load + Grad-CAM warm-up); subsequent inferences are fast.

## What the app does

1. **Loads** `model/best_model_final.pth` into a ResNet-50 with the custom head
   from your notebook (`Linear(2048→512) → BN → ReLU → Dropout(0.5) → Linear(512→4)`).
2. **Predicts** softmax probabilities over `[COVID, Normal, Pneumonia, Tuberculosis]`
   — the exact class order from `ImageFolder`, which is what your checkpoint was
   trained on.
3. **Generates a Grad-CAM** heatmap from `layer4[-1].conv3` (same target layer
   you used in the notebook) and overlays it on the original radiograph.
4. **Renders** the result in the clinical workspace UI: diagnosis card,
   differential probabilities, findings strip, and model card.

## Troubleshooting

**"Model weights not loaded" banner at the top**
- The app couldn't find or load `model/best_model_final.pth`. Check the path
  shown in the banner. If your `.pth` lives elsewhere, set
  `$env:CHESTX_WEIGHTS = "C:\full\path\to\best_model_final.pth"` before
  `streamlit run`.

**`Missing key(s) in state_dict` / `Unexpected key(s)`**
- This means the architecture in `inference.py:build_model()` doesn't match
  what you trained. The current build matches your notebook exactly. If you
  later change the head, change it in both places.

**`Error(s) in loading state_dict ... size mismatch for fc.4`**
- Number of classes changed. `inference.py` has `NUM_CLASSES = 4` to match
  `["COVID", "Normal", "Pneumonia", "Tuberculosis"]`. Update both if you
  retrain with different classes.

**CUDA out of memory / no GPU**
- The app auto-falls back to CPU. Inference will take ~1–3 seconds per image
  on CPU vs ~50 ms on GPU. No code changes needed.

**Predictions look wrong / inverted**
- Most likely the class order is off. Verify with one image of each class.
  The notebook used `datasets.ImageFolder` which sorts alphabetically →
  `['COVID', 'Normal', 'Pneumonia', 'Tuberculosis']`. If you trained with a
  different order, edit `CLASSES` in `inference.py`.

## Wiring notes (for future you)

- **`inference.py`** is the only file that touches PyTorch. The UI never
  imports `torch` directly except for the GPU/CPU label in the model card.
- **`Prediction` dataclass** is the contract between inference and UI. Add
  fields here if you want to surface new info in the workspace.
- **PDF report generation** is currently a placeholder. Drop `reportlab` into
  `requirements.txt` and generate real bytes in the `st.download_button` call.
