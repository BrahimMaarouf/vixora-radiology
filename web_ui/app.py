"""
ChestX AI — Clinical Workspace (real model)
===========================================
Same UI as web_guide/app.py, wired to the real ResNet-50 trained in
PFA.ipynb. Loads best_model_final.pth from ../model/ at startup.

Run from the project root (C:\\Users\\bdaar\\Desktop\\pfa):
    streamlit run web_ui/app.py

Or from inside web_ui/:
    streamlit run app.py
"""
from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import streamlit as st
from PIL import Image

from inference import (
    CLASSES,
    analyze,
    load_model,
    Prediction,
)

# =============================================================================
# 0.  Resolve the model checkpoint path
#     Default: <project_root>/model/best_model_final.pth
#     Override with env var CHESTX_WEIGHTS if you keep the file elsewhere.
# =============================================================================
import os

_HERE = Path(__file__).resolve().parent           # .../pfa/web_ui
_PROJECT_ROOT = _HERE.parent                       # .../pfa
DEFAULT_WEIGHTS = _PROJECT_ROOT / "model" / "best_model_final.pth"
WEIGHTS_PATH = Path(os.environ.get("CHESTX_WEIGHTS", str(DEFAULT_WEIGHTS)))


# =============================================================================
# 1.  PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="ChestX AI · Clinical Workspace",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "ChestX AI — AI-assisted chest X-ray triage. "
                 "Decision support only. Not a substitute for clinical judgment.",
    },
)


# =============================================================================
# 2.  DESIGN TOKENS + GLOBAL CSS  (copied verbatim from web_guide)
# =============================================================================
CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --blue-950:#061a33; --blue-900:#0a2540; --blue-800:#0b3b76;
  --blue-700:#0b5394; --blue-600:#1565c0; --blue-500:#2576d1;
  --blue-100:#d6e6f7; --blue-50:#eef4fb;
  --green-700:#0a7a44; --green-600:#0f9d58; --green-500:#22c55e; --green-50:#e6f4ea;
  --amber-600:#b45309; --amber-500:#f59e0b; --amber-50:#fef5e7;
  --red-700:#b91c1c; --red-600:#dc2626; --red-50:#fef2f2;
  --ink-900:#0f1729; --ink-800:#1e293b; --ink-700:#334155; --ink-600:#475569;
  --ink-500:#64748b; --ink-400:#94a3b8; --ink-300:#cbd5e1; --ink-200:#e2e8f0;
  --ink-100:#eef2f7; --ink-50:#f6f8fb;
  --surface:#ffffff; --bg:#f4f7fb;
  --r-sm:4px; --r-md:8px; --r-lg:12px;
  --shadow-1:0 1px 2px rgba(15,23,41,.05);
  --shadow-2:0 4px 12px rgba(15,23,41,.07);
  --font-sans:"IBM Plex Sans",system-ui,sans-serif;
  --font-mono:"IBM Plex Mono",ui-monospace,monospace;
}

html, body, [class*="css"] { font-family: var(--font-sans) !important; color: var(--ink-800); }
header[data-testid="stHeader"] { display:none; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section.main > div { padding: 0 !important; }
[data-testid="stAppViewContainer"] { background: var(--bg); }

section[data-testid="stSidebar"] {
  background: #fff;
  border-right: 1px solid var(--ink-200);
  width: 340px !important;
}
section[data-testid="stSidebar"] > div { padding-top: 8px; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stNumberInput input {
  background: #fff !important;
  border: 1px solid var(--ink-200) !important;
  border-radius: var(--r-sm) !important;
  font-size: 12.5px !important;
  color: var(--ink-800) !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] label p {
  font-size: 11px !important;
  font-weight: 500 !important;
  color: var(--ink-600) !important;
  margin-bottom: 2px !important;
}

.stButton > button, .stDownloadButton > button {
  background: var(--blue-700);
  color: #fff;
  border: 1px solid var(--blue-700);
  border-radius: 7px;
  font-weight: 600;
  font-size: 13px;
  padding: 8px 16px;
  transition: background .12s;
  box-shadow: none;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--blue-800); color:#fff; }
.stButton > button:focus { box-shadow: 0 0 0 3px rgba(37,118,209,.20) !important; }

.btn-secondary .stButton > button {
  background: #fff; color: var(--ink-700); border-color: var(--ink-200);
}
.btn-secondary .stButton > button:hover { background: var(--ink-50); color: var(--ink-900); }

[data-testid="stFileUploader"] {
  border: 1.5px dashed var(--ink-300);
  background: var(--ink-50);
  border-radius: var(--r-md);
  padding: 4px;
}
[data-testid="stFileUploader"] section { background: transparent !important; }
[data-testid="stFileUploader"] small { color: var(--ink-500); font-size: 11px; }

.topbar {
  background: var(--blue-950);
  color: #fff;
  display: flex; align-items: stretch;
  padding: 0 24px;
  height: 56px;
}
.topbar .brand { display: flex; align-items: center; gap: 10px; padding-right: 28px;
  border-right: 1px solid rgba(255,255,255,.08); }
.brand-mark { width:30px; height:30px; border-radius:7px;
  background: linear-gradient(135deg,#2576d1 0%,#0b5394 100%);
  display:grid; place-items:center;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.18), 0 0 0 1px rgba(0,0,0,.3); }
.brand-name { font-weight:700; font-size:15px; letter-spacing:.2px; line-height:1; }
.brand-tag { font-size:10px; color:#93b6dc; letter-spacing:.18em;
  text-transform:uppercase; margin-top:3px; font-family:var(--font-mono); }
.nav { display:flex; align-items:stretch; margin-left:4px; }
.nav a { display:flex; align-items:center; padding:0 16px; color:#c8d6e6;
  text-decoration:none; font-size:13px; font-weight:500;
  border-bottom: 2px solid transparent; }
.nav a.active { color:#fff; border-bottom-color: var(--blue-500); }
.nav a:hover:not(.active) { color:#fff; }
.nav .badge { background: var(--blue-600); color:#fff; font-size:10px; font-weight:600;
  padding:2px 6px; border-radius:10px; margin-left:6px; font-family: var(--font-mono); }
.top-right { margin-left:auto; display:flex; align-items:center; gap:14px; }
.env-chip { display:inline-flex; align-items:center; gap:6px;
  background: rgba(15,157,88,.14); color:#6be39e;
  border: 1px solid rgba(15,157,88,.4);
  padding:4px 9px; border-radius:999px;
  font-size:10px; font-family: var(--font-mono);
  font-weight:600; letter-spacing:.12em; text-transform:uppercase; }
.env-chip .dot { width:6px; height:6px; background: var(--green-500);
  border-radius:50%; box-shadow: 0 0 6px var(--green-500); }
.avatar { width:32px; height:32px; border-radius:50%;
  background: linear-gradient(135deg,#2576d1,#0f7e8a); color:#fff;
  display:grid; place-items:center; font-size:11px; font-weight:600;
  border: 1px solid rgba(255,255,255,.2); }
.user-block { display:flex; align-items:center; gap:10px; padding-left:12px;
  border-left:1px solid rgba(255,255,255,.08); color:#c8d6e6; }
.user-block .name { font-size:12px; font-weight:600; color:#fff; line-height:1.1; }
.user-block .role { font-size:10px; color:#93b6dc; font-family: var(--font-mono); letter-spacing:.08em; }

.patient-banner { background: var(--surface); border-bottom:1px solid var(--ink-200);
  display:flex; align-items:center; padding: 12px 24px; gap: 28px; }
.pb-avatar { width:36px; height:36px; border-radius:8px; background:var(--blue-50);
  color: var(--blue-700); display:grid; place-items:center; font-weight:600; font-size:13px;
  border: 1px solid var(--blue-100); }
.pb-name { font-weight:600; font-size:14px; color: var(--ink-900); line-height:1.1; }
.pb-sub { display:flex; gap:12px; font-family: var(--font-mono);
  font-size:11px; color: var(--ink-500); margin-top:3px; }
.pb-sub .sep { color: var(--ink-300); }
.pb-meta { display:flex; gap:28px; }
.pb-meta-item .k { font-size:10px; font-family: var(--font-mono);
  text-transform:uppercase; letter-spacing:.1em; color: var(--ink-500); margin-bottom:2px; }
.pb-meta-item .v { font-size:12.5px; font-weight:500; color: var(--ink-800); }
.pb-meta-item .v.stat { color: var(--red-600); font-weight:600; }

.card { background: #fff; border: 1px solid var(--ink-200);
  border-radius: var(--r-lg); box-shadow: var(--shadow-1); padding: 0; margin-bottom: 14px; }
.card-h { padding: 12px 16px; border-bottom: 1px solid var(--ink-200);
  display:flex; align-items:center; }
.card-h .t { font-size:11px; font-family: var(--font-mono);
  text-transform:uppercase; letter-spacing:.14em; font-weight:600; color: var(--ink-600); }
.card-b { padding: 16px; }

.dx { border-radius: var(--r-lg); overflow:hidden; box-shadow: var(--shadow-2); margin-bottom: 14px; }
.dx-h { padding: 10px 16px; color:#fff; display:flex; align-items:center; gap:8px; }
.dx-h .label { font-size:10.5px; text-transform:uppercase; letter-spacing:.16em;
  font-family: var(--font-mono); font-weight:600; }
.dx-h .level { margin-left:auto; font-size:10px; background: rgba(255,255,255,.18);
  border: 1px solid rgba(255,255,255,.25); padding:2px 7px; border-radius:999px;
  font-family: var(--font-mono); letter-spacing:.14em; text-transform:uppercase; font-weight:600; }
.dx-h .pulse { width:8px; height:8px; border-radius:50%; background:#fff;
  animation: dxpulse 1.6s infinite; }
@keyframes dxpulse {
  0%{box-shadow:0 0 0 0 rgba(255,255,255,.55);}
  70%{box-shadow:0 0 0 8px rgba(255,255,255,0);}
  100%{box-shadow:0 0 0 0 rgba(255,255,255,0);}
}
.dx-b { background:#fff; padding: 18px 18px 16px; }
.dx-name { font-size:22px; font-weight:700; color: var(--ink-900);
  letter-spacing:-.01em; margin-bottom:4px; }
.dx-code { font-family: var(--font-mono); font-size:11px; color: var(--ink-500); margin-bottom:14px; }
.dx-code b { color: var(--ink-700); font-weight:600; }
.dx-conf-row { display:flex; align-items:baseline; gap:8px; margin-bottom:4px; }
.dx-conf-num { font-family: var(--font-mono); font-size:28px; font-weight:600;
  letter-spacing:-.02em; line-height:1; }
.dx-conf-lbl { font-size:11px; color: var(--ink-500); text-transform:uppercase;
  font-family: var(--font-mono); letter-spacing:.1em; }
.dx-bar { height:6px; background: var(--ink-100); border-radius:3px; overflow:hidden; }
.dx-bar > i { display:block; height:100%; }
.dx-meta { display:flex; justify-content: space-between; font-family: var(--font-mono);
  font-size:10px; color: var(--ink-500); margin-top:5px; }

.dx.alert { border: 1px solid var(--red-600); }
.dx.alert .dx-h { background: linear-gradient(180deg,var(--red-600),#a8160e); }
.dx.alert .dx-conf-num { color: var(--red-600); }
.dx.alert .dx-bar > i { background: linear-gradient(90deg,#f59e0b,#dc2626); }

.dx.warn { border: 1px solid var(--amber-500); }
.dx.warn .dx-h { background: linear-gradient(180deg,#d97706,#92400e); }
.dx.warn .dx-conf-num { color: var(--amber-600); }
.dx.warn .dx-bar > i { background: linear-gradient(90deg,#facc15,#f59e0b); }

.dx.ok { border: 1px solid var(--green-600); }
.dx.ok .dx-h { background: linear-gradient(180deg,var(--green-600),var(--green-700)); }
.dx.ok .dx-conf-num { color: var(--green-700); }
.dx.ok .dx-bar > i { background: linear-gradient(90deg,#86efac,#0f9d58); }

.diff { padding: 4px 16px 12px; }
.diff-row { display:grid; grid-template-columns: 14px 1fr 56px;
  align-items:center; column-gap: 10px; padding: 7px 0; }
.diff-row .sw { width:10px; height:10px; border-radius:2px; }
.diff-row .nm { font-size:12.5px; color: var(--ink-800); font-weight:500; }
.diff-row .icd { font-family: var(--font-mono); font-size:10px;
  color: var(--ink-500); margin-left:6px; }
.diff-row .mt { height:4px; background: var(--ink-100); border-radius:2px;
  overflow:hidden; margin-top:4px; }
.diff-row .mt > i { display:block; height:100%; }
.diff-row .pc { font-family: var(--font-mono); font-size:12px;
  color: var(--ink-700); text-align:right; font-weight:500; }

.xr-frame { position:relative; background:#0a0f1c; border:1px solid #000;
  border-radius: var(--r-md); overflow:hidden; box-shadow: var(--shadow-2); }
.xr-title { position:absolute; top:10px; left:12px;
  font-family: var(--font-mono); font-size:10.5px; color:#9bb1d4;
  letter-spacing:.1em; text-transform:uppercase; z-index:2; }
.xr-title b { color:#fff; font-weight:600; margin-right:6px; }
.xr-frame img { width:100%; display:block; }

.findings { display:grid; grid-template-columns: repeat(4, 1fr);
  background:#fff; border:1px solid var(--ink-200); border-radius: var(--r-md);
  box-shadow: var(--shadow-1); }
.finding { padding: 12px 14px; border-right:1px solid var(--ink-200);
  display:flex; flex-direction:column; gap:4px; }
.finding:last-child { border-right:none; }
.finding .fk { font-size:10.5px; font-family: var(--font-mono);
  text-transform:uppercase; letter-spacing:.1em; color: var(--ink-500); }
.finding .fv { font-weight:600; color: var(--ink-900); font-size:13px; }
.finding .fbar { height:4px; background: var(--ink-100); border-radius:2px;
  position:relative; overflow:hidden; margin-top:2px; }
.finding .fbar > i { position:absolute; left:0; top:0; bottom:0; border-radius:2px; }
.finding.alert .fbar > i { background: var(--red-600); }
.finding.warn  .fbar > i { background: var(--amber-500); }
.finding.ok    .fbar > i { background: var(--green-600); }

.callout { padding:10px 12px; background: var(--amber-50); border:1px solid #f5d99a;
  border-radius: var(--r-sm); color:#6b3d04; font-size:11.5px;
  display:flex; gap:9px; margin: 8px 0 14px; }
.callout b { color:#5b2f00; }

.footer { background: var(--ink-900); color: #cbd5e1;
  display:flex; align-items:center; padding: 10px 24px; gap:18px;
  font-size:11px; font-family: var(--font-mono); margin-top: 14px; }
.footer .sep { color: var(--ink-600); }
.footer .right { margin-left:auto; display:flex; gap:14px; align-items:center; }
.footer .pill { display:inline-flex; align-items:center; gap:6px; color:#93b6dc; }
.footer .pill .dot { width:6px; height:6px; border-radius:50%;
  background: var(--green-500); box-shadow: 0 0 4px var(--green-500); }

.main-pad { padding: 16px 24px 0; }

.section-h { font-size:11px; font-family: var(--font-mono);
  text-transform:uppercase; letter-spacing:.14em; color: var(--ink-600);
  font-weight:600; margin: 4px 0 8px; }

.model-error { background: var(--red-50); border: 1px solid #fecaca;
  color: var(--red-700); padding: 12px 14px; border-radius: var(--r-md);
  font-size:12.5px; margin: 12px 0; }
.model-error b { color: var(--red-700); }
.model-error code { font-family: var(--font-mono); font-size: 11.5px;
  background: rgba(220,38,38,.08); padding: 1px 5px; border-radius: 3px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# 3.  TOP BAR  (static)
# =============================================================================
TOPBAR_HTML = """
<div class="topbar">
  <div class="brand">
    <div class="brand-mark">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M12 3v6.5" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
        <path d="M9 9.5C7 10 5 12 5 15.5C5 18 6 20 8 20C9.5 20 10 18.5 10 16.5V11"
              stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M15 9.5C17 10 19 12 19 15.5C19 18 18 20 16 20C14.5 20 14 18.5 14 16.5V11"
              stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <div>
      <div class="brand-name">ChestX AI</div>
      <div class="brand-tag">Radiology Suite · v1.0</div>
    </div>
  </div>
  <nav class="nav">
    <a class="active">Workspace</a>
    <a>Worklist <span class="badge">14</span></a>
    <a>Patients</a>
    <a>Reports</a>
    <a>Analytics</a>
  </nav>
  <div class="top-right">
    <span class="env-chip"><span class="dot"></span> ResNet-50 · Local</span>
    <div class="user-block">
      <div class="avatar">SR</div>
      <div>
        <div class="name">Dr. S. Rahman</div>
        <div class="role">RADIOLOGIST · MD</div>
      </div>
    </div>
  </div>
</div>
"""
st.markdown(TOPBAR_HTML, unsafe_allow_html=True)


# =============================================================================
# 4.  Pre-flight: try to load the model now so we can show a clear error if
#     the .pth file is missing, rather than failing only on the first upload.
# =============================================================================
MODEL_LOADED = False
MODEL_LOAD_ERROR: str | None = None

@st.cache_resource(show_spinner="Loading ResNet-50 weights…")
def _cached_load(path_str: str):
    return load_model(path_str)

try:
    _cached_load(str(WEIGHTS_PATH))
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOAD_ERROR = str(e)


# =============================================================================
# 5.  Patient banner renderer  (visual identical to web_guide)
# =============================================================================
def render_patient_banner(name: str, age: int, sex: str, mrn: str,
                          accession: str, study: str, priority: str = "Routine"):
    initials = "".join([p[0] for p in name.split()][:2]).upper() or "PT"
    dob_year = datetime.now().year - int(age) if age else "—"
    stat_cls = "v stat" if priority.upper() == "STAT" else "v"
    st.markdown(f"""
    <div class="patient-banner">
      <div style="display:flex;align-items:center;gap:12px;">
        <div class="pb-avatar">{initials}</div>
        <div>
          <div class="pb-name">{name}, {age} {sex[0] if sex else ''}</div>
          <div class="pb-sub">
            <span>MRN {mrn}</span><span class="sep">·</span>
            <span>DOB ~{dob_year}</span><span class="sep">·</span>
            <span>Accession {accession}</span>
          </div>
        </div>
      </div>
      <div class="pb-meta">
        <div class="pb-meta-item"><div class="k">Study</div><div class="v">{study}</div></div>
        <div class="pb-meta-item"><div class="k">Acquired</div>
            <div class="v">{datetime.now().strftime("%b %d, %Y · %H:%M")}</div></div>
        <div class="pb-meta-item"><div class="k">Priority</div>
            <div class="{stat_cls}">● {priority}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 6.  SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(
        '<div class="section-h" style="padding:8px 4px 0;">① Patient &amp; Study</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    pid       = c1.text_input("Patient ID", value="008-2241-77")
    accession = c2.text_input("Accession", value="A-2026-0517")
    name      = st.text_input("Full name", value="Eleanor Marsh")
    c3, c4 = st.columns(2)
    age = c3.number_input("Age", min_value=0, max_value=120, value=64, step=1)
    sex = c4.selectbox("Sex", ["Female", "Male", "Other"])
    study    = st.selectbox("Modality / view",
                            ["CXR · PA + Lateral", "CXR · AP supine", "CXR · PA only"])
    priority = st.selectbox("Priority", ["Routine", "STAT", "Urgent"])

    st.markdown(
        '<div class="section-h" style="padding:8px 4px 0;">② Clinical history</div>',
        unsafe_allow_html=True,
    )
    indication = st.text_area(
        "Indication",
        value=("Productive cough × 8 days, fever 38.7°C, "
               "pleuritic chest pain. Recent travel to South Asia."),
        height=80,
    )
    risk_factors = st.multiselect(
        "Risk factors",
        ["Smoker","Recent travel","Immunocompromised","COPD","Diabetes","Pregnancy"],
        default=["Smoker","Recent travel"],
    )

    st.markdown(
        '<div class="section-h" style="padding:8px 4px 0;">③ Image input</div>',
        unsafe_allow_html=True,
    )
    upload = st.file_uploader(
        "Drop a chest X-ray  ·  PNG / JPG  ·  max 50 MB",
        type=["png","jpg","jpeg"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    run_btn = st.button("⚡  Run analysis", use_container_width=True, type="primary")


# =============================================================================
# 7.  PATIENT BANNER
# =============================================================================
render_patient_banner(name, age, sex, pid, accession, study, priority)


# =============================================================================
# 8.  Main grid
# =============================================================================
st.markdown('<div class="main-pad">', unsafe_allow_html=True)

# Surface model-load error prominently if needed
if not MODEL_LOADED:
    st.markdown(f"""
    <div class="model-error">
      <div>
        <b>Model weights not loaded.</b> The app is running, but inference is disabled until the
        checkpoint can be found.<br>
        Expected at: <code>{WEIGHTS_PATH}</code><br>
        <span style="font-size:11.5px;">{MODEL_LOAD_ERROR}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

left_col, right_col = st.columns([2, 1], gap="medium")


# ---- helpers ----------------------------------------------------------------
def img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def render_xray_pair(original: Image.Image, gradcam: Image.Image):
    o_b64 = img_to_b64(original)
    g_b64 = img_to_b64(gradcam)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
      <div class="xr-frame">
        <div class="xr-title"><b>Original</b> · PA</div>
        <img src="data:image/png;base64,{o_b64}" alt="original chest x-ray"/>
      </div>
      <div class="xr-frame">
        <div class="xr-title"><b>Grad-CAM</b> · activation overlay</div>
        <img src="data:image/png;base64,{g_b64}" alt="grad-cam overlay"/>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_findings(findings: List[Dict]):
    tiles = ""
    for f in findings:
        pct = int(f["score"] * 100)
        tiles += f"""
        <div class="finding {f['status']}">
          <div class="fk">{f['label']}</div>
          <div class="fv">{f['value']}</div>
          <div class="fbar"><i style="width:{pct}%"></i></div>
        </div>"""
    st.markdown(
        f'<div class="findings" style="margin-top:14px;">{tiles}</div>',
        unsafe_allow_html=True,
    )


def render_diagnosis(pred: Prediction):
    pct = pred.confidence * 100
    level = "HIGH" if pct >= 80 else "MODERATE" if pct >= 50 else "LOW"
    st.markdown(f"""
    <div class="dx {pred.severity}">
      <div class="dx-h">
        <span class="pulse"></span>
        <span class="label">Primary prediction</span>
        <span class="level">{level}</span>
      </div>
      <div class="dx-b">
        <div class="dx-name">{pred.label}</div>
        <div class="dx-code"><b>ICD-10 {pred.icd}</b> · {pred.description}</div>
        <div class="dx-conf-row">
          <div class="dx-conf-num">{pct:.1f}%</div>
          <div class="dx-conf-lbl">Confidence</div>
        </div>
        <div class="dx-bar"><i style="width:{pct:.1f}%"></i></div>
        <div class="dx-meta">
          <span>Threshold ≥ 80%</span>
          <span>Softmax · 4 classes</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_differential(pred: Prediction):
    rows = ""
    for d in pred.differentials:
        p = d["prob"] * 100
        rows += f"""
        <div class="diff-row">
          <span class="sw" style="background:{d['color']};"></span>
          <div>
            <div class="nm">{d['label']}<span class="icd">{d['icd']}</span></div>
            <div class="mt"><i style="width:{p}%;background:{d['color']};"></i></div>
          </div>
          <div class="pc">{p:.1f}%</div>
        </div>"""
    st.markdown(f"""
    <div class="card">
      <div class="card-h"><div class="t">Differential probabilities</div></div>
      <div class="diff">{rows}</div>
    </div>
    """, unsafe_allow_html=True)


def render_model_card(inference_ms: float | None):
    import torch  # local import keeps the top of file lighter for cold starts
    device_label = "GPU · CUDA" if torch.cuda.is_available() else "CPU"
    infer_str = f"{inference_ms/1000:.2f} s · {device_label}" if inference_ms else f"— · {device_label}"
    st.markdown(f"""
    <div class="card">
      <div class="card-h"><div class="t">Model &amp; performance</div></div>
      <div class="card-b" style="display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;
           font-family:var(--font-mono);font-size:12px;color:var(--ink-800);">
        <div><div style="font-size:10px;color:var(--ink-500);letter-spacing:.1em;text-transform:uppercase;">Model</div>resnet-50 · custom head</div>
        <div><div style="font-size:10px;color:var(--ink-500);letter-spacing:.1em;text-transform:uppercase;">Classes</div>{len(CLASSES)} · {', '.join(CLASSES)}</div>
        <div><div style="font-size:10px;color:var(--ink-500);letter-spacing:.1em;text-transform:uppercase;">Input</div>224 × 224 · RGB</div>
        <div><div style="font-size:10px;color:var(--ink-500);letter-spacing:.1em;text-transform:uppercase;">Normalize</div>ImageNet mean/std</div>
        <div><div style="font-size:10px;color:var(--ink-500);letter-spacing:.1em;text-transform:uppercase;">Grad-CAM</div>layer4[-1].conv3</div>
        <div><div style="font-size:10px;color:var(--ink-500);letter-spacing:.1em;text-transform:uppercase;">Inference</div>{infer_str}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---- Open the image ---------------------------------------------------------
original: Image.Image | None = None
if upload is not None:
    try:
        original = Image.open(upload).convert("RGB")
    except Exception:
        st.error("Could not decode image. Supported: PNG, JPG.")
        original = None


# =============================================================================
# 9.  LEFT — Viewer + findings
# =============================================================================
prediction: Prediction | None = None
inference_ms: float | None = None

with left_col:
    if original is None:
        st.markdown("""
        <div class="card">
          <div class="card-h"><div class="t">Study viewer</div></div>
          <div class="card-b" style="text-align:center;padding:60px 16px;color:var(--ink-500);">
            <div style="font-size:14px;font-weight:600;color:var(--ink-700);margin-bottom:6px;">
              No image loaded
            </div>
            <div style="font-size:12px;">
              Upload a chest X-ray in the sidebar to begin analysis.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    elif not MODEL_LOADED:
        # Image is uploaded but model isn't ready — show the image, skip inference.
        b64 = img_to_b64(original.resize((512, 512)))
        st.markdown(f"""
        <div class="xr-frame">
          <div class="xr-title"><b>Original</b> · uploaded</div>
          <img src="data:image/png;base64,{b64}" alt="uploaded x-ray"/>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Re-run when the button is pressed OR when there's no cached result yet
        upload_id = upload.name + str(upload.size)
        cache_key = "result_" + upload_id
        if run_btn or cache_key not in st.session_state:
            with st.spinner("Running inference…"):
                t0 = time.perf_counter()
                pred, overlay = analyze(original, WEIGHTS_PATH)
                inference_ms = (time.perf_counter() - t0) * 1000
                st.session_state[cache_key] = (pred, overlay, inference_ms)
        prediction, gradcam, inference_ms = st.session_state[cache_key]
        render_xray_pair(original, gradcam)
        render_findings(prediction.findings)


# =============================================================================
# 10.  RIGHT — AI analysis
# =============================================================================
with right_col:
    st.markdown('<div class="section-h">AI Analysis</div>', unsafe_allow_html=True)

    if original is None or prediction is None:
        st.markdown("""
        <div class="card">
          <div class="card-b" style="color:var(--ink-500);font-size:12.5px;text-align:center;padding:40px 16px;">
            Awaiting image…
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        render_diagnosis(prediction)
        render_differential(prediction)
        render_model_card(inference_ms)

        st.download_button(
            label="📄  Generate radiology report (PDF)",
            data=b"PDF placeholder - wire up reportlab to enable",
            file_name=f"ChestXAI_{pid}_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        cA, cB = st.columns(2)
        with cA:
            st.button("✓  Confirm", use_container_width=True)
        with cB:
            st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
            st.button("⚑  Flag & refer", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout" style="margin-top:14px;">
      <div>
        <b>Decision-support only.</b> Results must be reviewed and signed by a
        licensed radiologist before clinical use. Not a substitute for clinical judgment.
      </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True)  # /main-pad


# =============================================================================
# 11.  FOOTER
# =============================================================================
status_pill = ('<span class="pill"><span class="dot"></span> Model online</span>'
               if MODEL_LOADED else
               '<span class="pill" style="color:#fca5a5;">'
               '<span class="dot" style="background:#dc2626;box-shadow:0 0 4px #dc2626;"></span>'
               ' Model offline</span>')

st.markdown(f"""
<div class="footer">
  {status_pill}
  <span class="sep">·</span>
  <span>For investigational decision-support use. Not a substitute for clinical judgment.</span>
  <div class="right">
    <span>Local deployment</span><span class="sep">·</span>
    <span>ResNet-50 · 4-class</span><span class="sep">·</span>
    <span>© 2026 ChestX AI</span>
  </div>
</div>
""", unsafe_allow_html=True)
