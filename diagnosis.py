"""
diagnosis.py — Post-Diagnosis Interpretation Layer

This module translates raw model predictions (ScanResult) into:
  1. Patient state severity estimation (Critical / High / Moderate / Low / Normal)
  2. Plain-language clinical interpretation of detected conditions
  3. Region-aware medication suggestions (advisory only, not prescriptive)
  4. Urgency-ranked action recommendations

⚠ DISCLAIMER: This system is an AI-assisted screening tool.
  It does NOT replace professional medical evaluation, diagnosis, or treatment.
  All outputs must be reviewed by a qualified clinician before any clinical action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)


# ─── Severity Enum ────────────────────────────────────────────────────────────

class Severity(str, Enum):
    NORMAL   = "Normal"
    LOW      = "Low"
    MODERATE = "Moderate"
    HIGH     = "High"
    CRITICAL = "Critical"

    @property
    def color(self) -> str:
        return {
            "Normal"  : "#2ecc71",
            "Low"     : "#f1c40f",
            "Moderate": "#e67e22",
            "High"    : "#e74c3c",
            "Critical": "#8e44ad",
        }[self.value]

    @property
    def emoji(self) -> str:
        return {
            "Normal"  : "✅",
            "Low"     : "🟡",
            "Moderate": "🟠",
            "High"    : "🔴",
            "Critical": "🚨",
        }[self.value]


# ─── Condition Knowledge Base ──────────────────────────────────────────────────

CONDITION_INFO: Dict[str, dict] = {
    "normal": {
        "description": (
            "No significant pathology detected. The lung fields appear clear, "
            "cardiac silhouette is within normal limits, and no pleural abnormality is identified."
        ),
        "severity_weight": 0.0,
        "urgency": "routine",
        "follow_up": "Routine clinical follow-up as clinically indicated.",
    },
    "pneumonia": {
        "description": (
            "Consolidative opacification consistent with pneumonia is present. "
            "Bacterial pneumonia most commonly presents as lobar or segmental consolidation, "
            "while viral and atypical organisms often produce bilateral interstitial infiltrates."
        ),
        "severity_weight": 0.7,
        "urgency": "urgent",
        "follow_up": "Prompt clinical assessment, sputum culture, and antibiotic therapy initiation recommended.",
    },
    "pneumothorax": {
        "description": (
            "Pneumothorax — air in the pleural space — is identified. "
            "A tension pneumothorax is a life-threatening emergency. "
            "Clinical correlation with respiratory status and haemodynamic stability is critical."
        ),
        "severity_weight": 0.95,
        "urgency": "emergency",
        "follow_up": (
            "EMERGENCY: Immediate clinical assessment. "
            "Large or tension pneumothorax requires urgent needle decompression or chest drain insertion."
        ),
    },
    "tuberculosis": {
        "description": (
            "Radiographic features consistent with pulmonary tuberculosis are present, "
            "including upper lobe infiltrates, cavitation, or miliary pattern. "
            "Confirmation requires sputum smear/culture and/or molecular testing (GeneXpert)."
        ),
        "severity_weight": 0.75,
        "urgency": "urgent",
        "follow_up": (
            "Isolation precautions, sputum AFB smear and GeneXpert, chest physician referral, "
            "and notification per local public health regulations."
        ),
    },
    "cardiomegaly": {
        "description": (
            "Cardiomegaly (cardiothoracic ratio >0.5 on PA view) suggests cardiac enlargement. "
            "Causes include heart failure, valvular disease, cardiomyopathy, or pericardial effusion. "
            "Echocardiography is the next recommended investigation."
        ),
        "severity_weight": 0.55,
        "urgency": "semi-urgent",
        "follow_up": "Cardiology referral, echocardiogram, BNP/NT-proBNP, ECG.",
    },
    "effusion": {
        "description": (
            "Pleural effusion (fluid in the pleural space) is identified, indicated by blunting of "
            "costophrenic angles or a meniscus sign. Causes are broad — cardiac failure, parapneumonic, "
            "malignancy, and hypoalbuminaemia being the most common."
        ),
        "severity_weight": 0.60,
        "urgency": "semi-urgent",
        "follow_up": (
            "Thoracocentesis for diagnostic and/or therapeutic purposes. "
            "Serum/fluid LDH and protein (Light's criteria). Underlying cause investigation."
        ),
    },
}


# ─── Medication Guidance Knowledge Base ───────────────────────────────────────
# Medications listed are internationally common generic agents.
# Availability annotations reflect general availability in North African / Middle-Eastern markets (MA).
# This is NON-PRESCRIPTIVE guidance only. Dosing not included intentionally.

MEDICATION_GUIDANCE: Dict[str, Dict[str, list]] = {
    "pneumonia": {
        "MA": [
            {"name": "Amoxicillin-Clavulanate",  "note": "First-line CAP — widely available",   "otc": False},
            {"name": "Azithromycin",              "note": "Atypical coverage / macrolide option", "otc": False},
            {"name": "Paracetamol (Acetaminophen)","note": "Fever / analgesia — OTC",             "otc": True},
            {"name": "Ibuprofen",                 "note": "Anti-inflammatory — OTC",             "otc": True},
            {"name": "Oral Rehydration Salts",    "note": "Hydration support — OTC",             "otc": True},
        ],
        "DEFAULT": [
            {"name": "Amoxicillin-Clavulanate",  "note": "Broad-spectrum first-line",           "otc": False},
            {"name": "Azithromycin",             "note": "Atypical / macrolide",                "otc": False},
            {"name": "Paracetamol",              "note": "Symptom management",                  "otc": True},
        ],
    },
    "pneumothorax": {
        "MA": [
            {"name": "Oxygen therapy (high-flow)", "note": "Accelerates air resorption",         "otc": False},
            {"name": "Analgesics (IV/IM)",          "note": "Post-procedural pain management",  "otc": False},
        ],
        "DEFAULT": [
            {"name": "High-flow O₂",               "note": "Essential — hasten resorption",    "otc": False},
        ],
    },
    "tuberculosis": {
        "MA": [
            {"name": "Isoniazid (H)",     "note": "DOTS regimen — national TB programme",       "otc": False},
            {"name": "Rifampicin (R)",    "note": "DOTS regimen",                               "otc": False},
            {"name": "Pyrazinamide (Z)", "note": "Intensive phase",                             "otc": False},
            {"name": "Ethambutol (E)",   "note": "Intensive phase",                             "otc": False},
            {"name": "Pyridoxine (B6)",  "note": "Prevent isoniazid neuropathy — often OTC",   "otc": True},
        ],
        "DEFAULT": [
            {"name": "Standard HRZE regimen", "note": "Per local TB control programme",         "otc": False},
        ],
    },
    "cardiomegaly": {
        "MA": [
            {"name": "Furosemide",           "note": "Loop diuretic — if HF suspected",         "otc": False},
            {"name": "ACE Inhibitor (e.g. Enalapril)", "note": "Heart failure / hypertension",  "otc": False},
            {"name": "Beta-blocker (e.g. Bisoprolol)", "note": "Rate control / HF",             "otc": False},
            {"name": "Spironolactone",       "note": "Aldosterone antagonist in HF",            "otc": False},
        ],
        "DEFAULT": [
            {"name": "Diuretic therapy",     "note": "Symptom relief — requires prescription", "otc": False},
            {"name": "ACE Inhibitor",        "note": "Cardioprotective — requires prescription","otc": False},
        ],
    },
    "effusion": {
        "MA": [
            {"name": "Furosemide",      "note": "If cardiac cause — prescription required",      "otc": False},
            {"name": "Antibiotics",     "note": "If parapneumonic — culture-guided",             "otc": False},
            {"name": "Ibuprofen",       "note": "If inflammatory / post-viral — OTC",           "otc": True},
        ],
        "DEFAULT": [
            {"name": "Treat underlying cause", "note": "Specific therapy determined by aetiology", "otc": False},
        ],
    },
    "normal": {
        "MA"      : [],
        "DEFAULT" : [],
    },
}


# ─── Result Dataclass ─────────────────────────────────────────────────────────

@dataclass
class DiagnosisReport:
    severity            : Severity
    severity_score      : float
    primary_condition   : str
    detected_conditions : Dict[str, float]
    interpretations     : List[str]
    follow_up_actions   : List[str]
    medications         : List[dict]
    disclaimer          : str = field(default=(
        "⚠ This is an AI-assisted screening report. "
        "It must be reviewed by a qualified medical professional before any clinical decisions are made. "
        "It does not constitute a medical diagnosis, prescription, or treatment plan."
    ))

    def to_dict(self) -> dict:
        return {
            "severity"           : self.severity.value,
            "severity_score"     : round(self.severity_score, 3),
            "primary_condition"  : self.primary_condition,
            "detected_conditions": self.detected_conditions,
            "interpretations"    : self.interpretations,
            "follow_up_actions"  : self.follow_up_actions,
            "medications"        : self.medications,
            "disclaimer"         : self.disclaimer,
        }


# ─── Diagnosis Engine ─────────────────────────────────────────────────────────

class DiagnosisEngine:
    """
    Converts a ScanResult into a structured DiagnosisReport.

    Args:
        region: ISO country code for medication guidance (default: config.DEFAULT_REGION).
    """

    def __init__(self, region: str = config.DEFAULT_REGION) -> None:
        self.region = region.upper()

    # ── Severity ──────────────────────────────────────────────────────────────

    def _compute_severity(self, detected: Dict[str, float]) -> Tuple[Severity, float]:
        """
        Weighted severity score from all detected conditions and their confidence.

        Pneumothorax dominates due to potential lethality.
        Multiple co-detections compound severity.
        """
        if not detected or list(detected.keys()) == ["normal"]:
            return Severity.NORMAL, 0.0

        score = 0.0
        for cond, conf in detected.items():
            info  = CONDITION_INFO.get(cond, {})
            weight= info.get("severity_weight", 0.3)
            score  += weight * conf

        # Clamp to [0, 1]
        score = min(score, 1.0)

        # Emergency override
        if "pneumothorax" in detected and detected["pneumothorax"] >= 0.5:
            return Severity.CRITICAL, max(score, 0.95)

        # Map score to enum
        if score >= config.SEVERITY_THRESHOLDS["critical"]:
            sev = Severity.CRITICAL
        elif score >= config.SEVERITY_THRESHOLDS["high"]:
            sev = Severity.HIGH
        elif score >= config.SEVERITY_THRESHOLDS["moderate"]:
            sev = Severity.MODERATE
        elif score >= config.SEVERITY_THRESHOLDS["low"]:
            sev = Severity.LOW
        else:
            sev = Severity.NORMAL

        return sev, score

    # ── Interpretations ───────────────────────────────────────────────────────

    def _build_interpretations(self, detected: Dict[str, float]) -> List[str]:
        interps = []
        for cond in sorted(detected, key=lambda c: detected[c], reverse=True):
            info = CONDITION_INFO.get(cond, {})
            desc = info.get("description", f"Finding consistent with {cond}.")
            interps.append(f"[{cond.upper()}] {desc}")
        return interps

    # ── Follow-Up ─────────────────────────────────────────────────────────────

    def _build_follow_up(self, detected: Dict[str, float]) -> List[str]:
        seen = set()
        actions = []
        for cond in sorted(detected, key=lambda c: detected[c], reverse=True):
            info = CONDITION_INFO.get(cond, {})
            fu   = info.get("follow_up", "")
            if fu and fu not in seen:
                seen.add(fu)
                actions.append(fu)
        return actions

    # ── Medications ───────────────────────────────────────────────────────────

    def _build_medications(self, detected: Dict[str, float]) -> List[dict]:
        meds_seen = set()
        meds = []
        for cond in sorted(detected, key=lambda c: detected[c], reverse=True):
            region_meds = (
                MEDICATION_GUIDANCE.get(cond, {}).get(self.region)
                or MEDICATION_GUIDANCE.get(cond, {}).get("DEFAULT", [])
            )
            for med in region_meds:
                key = med["name"].lower()
                if key not in meds_seen:
                    meds_seen.add(key)
                    meds.append({**med, "condition": cond})
        return meds

    # ── Main Generate ─────────────────────────────────────────────────────────

    def generate(self, scan_result) -> DiagnosisReport:
        """
        Generate a full DiagnosisReport from a ScanResult.

        Args:
            scan_result: An inference.ScanResult instance.

        Returns:
            DiagnosisReport
        """
        detected = dict(scan_result.detections)

        # If nothing crosses the threshold, fall back to primary prediction
        if not detected:
            detected = {scan_result.primary_class: scan_result.primary_conf}

        severity, score = self._compute_severity(detected)
        interpretations = self._build_interpretations(detected)
        follow_up       = self._build_follow_up(detected)
        medications     = self._build_medications(detected)
        primary         = scan_result.primary_class

        report = DiagnosisReport(
            severity            = severity,
            severity_score      = score,
            primary_condition   = primary,
            detected_conditions = {k: round(v, 4) for k, v in detected.items()},
            interpretations     = interpretations,
            follow_up_actions   = follow_up,
            medications         = medications,
        )
        logger.info(f"[diagnosis] {severity.emoji} {severity.value} | primary={primary} | score={score:.3f}")
        return report
