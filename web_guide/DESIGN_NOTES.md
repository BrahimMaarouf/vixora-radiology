# ChestX AI — Design Direction

## 1. Visual language

| Token | Value | Role |
|---|---|---|
| `--blue-950` | `#061a33` | Top nav background |
| `--blue-700` | `#0b5394` | Primary brand / CTAs |
| `--blue-500` | `#2576d1` | Focus, accents |
| `--blue-50`  | `#eef4fb` | Tinted surfaces |
| `--green-600`| `#0f9d58` | Normal / healthy status |
| `--amber-500`| `#f59e0b` | Warning / equivocal |
| `--red-600`  | `#dc2626` | Critical finding |
| `--ink-900`  | `#0f1729` | Primary text, footer |
| `--ink-500`  | `#64748b` | Secondary text |
| `--ink-200`  | `#e2e8f0` | Card borders |
| `--bg`       | `#f4f7fb` | App background |

**Type**: IBM Plex Sans (UI), IBM Plex Mono (codes, IDs, model
metadata, confidence numbers). Plex reads as clinical/technical
without being cold. Base 13 px, banner names 14 px, diagnosis 22 px.

**Density**: clinical — radiologists work in dense PACS workspaces
all day. Generous *gutters* between zones, tight *spacing inside*
each zone. 8 / 12 / 16 / 24 px rhythm.

**Shape**: 4 px radii for inputs, 8 px for cards, 12 px for hero
diagnosis card. Light shadows only — `0 1px 2px` and `0 4px 12px`.
No glass / gradients except status-color hero header.

## 2. Layout

```
┌────────────────────────────────────────────────────────────────┐
│  TOP BAR  (logo · nav · env chip · user)                  56px │
├────────────────────────────────────────────────────────────────┤
│  PATIENT BANNER  (name · MRN · DOB · study · priority)    52px │
├──────────────┬──────────────────────────────┬──────────────────┤
│              │  TOOLBAR (pan / zoom / W·L)  │                  │
│  SIDEBAR     │                              │   AI ANALYSIS    │
│  ‒ Patient   │   ORIGINAL  │  GRAD-CAM     │   ‒ Diagnosis    │
│  ‒ History   │   X-ray     │  overlay      │   ‒ Differential │
│  ‒ Upload    │                              │   ‒ Model perf   │
│  ‒ Re-run    │  FINDINGS strip (4 tiles)    │   ‒ Actions      │
├──────────────┴──────────────────────────────┴──────────────────┤
│  FOOTER  (disclaimer · compliance badges)                 36px │
└────────────────────────────────────────────────────────────────┘
```

## 3. Component decisions

- **Diagnosis card** is the only color-saturated element on the
  page. Header band carries the status color; body is white. This
  makes the call easy to spot from across the room without making
  the rest of the UI shout.
- **Two-up viewer** uses black film backgrounds with monospace
  technical corners (MRN, kVp/mAs, model layer). Mirrors how DICOM
  viewers annotate film.
- **Grad-CAM heatmap legend** is fixed bottom-right with a labeled
  gradient so radiologists know what the colors mean — a Grad-CAM
  without a legend is just a pretty picture.
- **Differential** rows always show all four classes with bars, not
  a single winner. Sums to ~100. Trains the user to read the
  uncertainty.
- **Findings strip** sits *between* the images and the AI analysis
  card — it bridges the visual and the verbal.

## 4. Trust & safety patterns

1. **Persistent disclaimer** in the sidebar AND the footer. Two
   touchpoints, can't be missed.
2. **"Decision-support only"** language verbatim — avoids implying
   diagnostic authority.
3. **Confirm / Flag-and-refer** twin buttons make the human-in-the-
   loop step explicit. No "auto-publish."
4. **Model card** exposes AUROC, sensitivity, specificity, version,
   and inference time directly in the workflow — not buried in
   settings. Lets radiologists calibrate trust per case.
5. **Threshold annotation** under the confidence number ("Threshold
   ≥ 80%") shows where the institutional cutoff lives.
6. **Production / HIPAA chip** in the top nav confirms the
   deployment environment at a glance.
7. **STAT priority** is colored red in the patient banner so urgent
   cases can't get lost.
8. **PHI** is shown only when actually needed; otherwise IDs are
   monospaced and quiet, never bolded.

## 5. Accessibility

- AA contrast everywhere (verified for body text on white, mono
  data on `--ink-50`, white text on `--blue-700` and `--red-600`).
- Focus rings: 3 px `rgba(37,118,209,.20)` halo on all inputs and
  buttons.
- Status is never color-only — the diagnosis card pairs color with
  the level chip ("HIGH"), iconography, and the confidence number.
- Form labels are explicit (no placeholder-only inputs).
- Heatmap has a labeled gradient legend; the bounding box is
  outlined white-on-dark and includes the class + score in text.

## 6. Wiring the placeholders to your real model

In `app.py`, replace these two functions:

- `run_inference(image) -> Prediction` — call your classifier, fill
  in `label`, `icd`, `confidence`, `severity` (`"ok"|"warn"|"alert"`),
  `differentials`, and `findings`.
- `make_gradcam(image) -> PIL.Image` — return your Grad-CAM
  composite. The viewer hands it straight to the `<img>` tag.

The `Prediction` dataclass is the contract the UI renders against;
keep the field names the same and everything else just works.
