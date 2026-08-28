"""
SIF-Potential Classifier.

Supports two modes:
  1. Zero-shot (no labeled data) via facebook/bart-large-mnli
  2. Keyword/embedding ensemble (fast, lightweight — default for prototype)

Each report gets:
  - sif_score    : float [0, 1]
  - sif_potential: bool
  - confidence   : "HIGH" | "MEDIUM" | "LOW"
  - top_signals  : list of contributing phrases
  - explanation  : short human-readable reason string
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple

from utils.config import (
    SIF_SCORE_THRESHOLD,
    HIGH_CONFIDENCE_CUTOFF,
    LOW_CONFIDENCE_CUTOFF,
    WEIGHT_LLM,
    WEIGHT_KEYWORD,
    HIGH_ENERGY_KEYWORDS,
    BARRIER_FAILURE_SIGNALS,
    NEAR_FATAL_SIGNALS,
)

# ── Zero-shot pipeline (lazy-loaded) ──────────────────────────────────────────
_zs_pipeline = None

SIF_HYPOTHESIS = "This safety report describes a potential Serious Injury or Fatality precursor with high-energy hazard or critical barrier failure."
NON_SIF_HYPOTHESIS = "This safety report describes a minor observation or low-severity housekeeping issue."


def _get_zero_shot_pipeline():
    global _zs_pipeline
    if _zs_pipeline is None:
        from transformers import pipeline
        from utils.config import ZERO_SHOT_MODEL
        try:
            _zs_pipeline = pipeline(
                "zero-shot-classification",
                model=ZERO_SHOT_MODEL,
                device=-1,  # CPU
            )
        except Exception:
            _zs_pipeline = "unavailable"
    return _zs_pipeline


# ── Precompiled Signal Patterns ──────────────────────────────────────────────
_ENERGY_PATTERNS = [re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE) for k in HIGH_ENERGY_KEYWORDS]
_BARRIER_PATTERNS = [re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE) for k in BARRIER_FAILURE_SIGNALS]
_FATAL_PATTERNS = [re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE) for k in NEAR_FATAL_SIGNALS]
_FATALITY_BOOST_RE = re.compile(
    r"\b(fatal|fatality|death|killed|died|life threatening|serious injury|critical|hospitalised|hospitalized)\b",
    re.IGNORECASE
)

# ── Keyword / Signal-Based Scoring ────────────────────────────────────────────
def _keyword_sif_score(text_lower: str) -> Tuple[float, List[str]]:
    """
    Score based on energy hazards, barrier failures, and near-fatal signals.
    Returns (score, matched_signals).
    """
    matched = []

    # High-energy hazard hits
    energy_hits = [k for k, p in zip(HIGH_ENERGY_KEYWORDS, _ENERGY_PATTERNS) if p.search(text_lower)]
    energy_score = min(len(energy_hits) / 2.0, 1.0) * 0.45
    matched.extend(energy_hits[:3])

    # Barrier failure hits
    barrier_hits = [k for k, p in zip(BARRIER_FAILURE_SIGNALS, _BARRIER_PATTERNS) if p.search(text_lower)]
    barrier_score = min(len(barrier_hits) / 1.0, 1.0) * 0.35
    matched.extend(barrier_hits[:3])

    # Near-fatal / consequence hits
    fatal_hits = [k for k, p in zip(NEAR_FATAL_SIGNALS, _FATAL_PATTERNS) if p.search(text_lower)]
    fatal_score = min(len(fatal_hits) / 1.0, 1.0) * 0.30
    matched.extend(fatal_hits[:2])

    total_score = energy_score + barrier_score + fatal_score

    # Synergy bonus for energy hazard + barrier failure / near fatal combination
    if (energy_hits and barrier_hits) or (barrier_hits and fatal_hits) or (energy_hits and fatal_hits):
        total_score += 0.15

    return min(total_score, 1.0), list(dict.fromkeys(matched))[:6]


def _llm_sif_score(text: str) -> float:
    """
    Use zero-shot classification to get P(SIF-potential).
    Falls back to 0.5 if model unavailable.
    """
    pipe = _get_zero_shot_pipeline()
    if pipe == "unavailable" or not isinstance(pipe, object) or isinstance(pipe, str):
        return 0.5  # neutral fallback

    try:
        result = pipe(
            text[:512],
            candidate_labels=["SIF-potential serious injury fatality precursor", "minor observation low severity"],
            hypothesis_template="{}.",
        )
        # Get the probability for the SIF label
        labels = result["labels"]
        scores = result["scores"]
        sif_idx = next((i for i, l in enumerate(labels) if "SIF" in l or "serious" in l.lower()), 0)
        return float(scores[sif_idx])
    except Exception:
        return 0.5


# ── Main Classification Function ───────────────────────────────────────────────
def classify_report(
    clean_text: str,
    use_llm: bool = False,
) -> Dict:
    """
    Classify a single report for SIF potential.

    Args:
        clean_text : preprocessed narrative text
        use_llm    : if True, uses zero-shot LLM (slower but more accurate)

    Returns dict with keys: sif_score, sif_potential, confidence, top_signals, explanation
    """
    text_lower = clean_text.lower()

    # Keyword score (always computed)
    kw_score, top_signals = _keyword_sif_score(text_lower)

    if use_llm:
        llm_score  = _llm_sif_score(clean_text)
        sif_score  = WEIGHT_LLM * llm_score + WEIGHT_KEYWORD * kw_score
    else:
        # Fast heuristic boost: explicit fatality/serious injury language
        boost = 0.15 if _FATALITY_BOOST_RE.search(text_lower) else 0.0
        sif_score = min(kw_score + boost, 1.0)

    sif_score     = round(sif_score, 4)
    sif_potential = sif_score >= SIF_SCORE_THRESHOLD

    if sif_score >= HIGH_CONFIDENCE_CUTOFF:
        confidence = "HIGH"
    elif sif_score >= LOW_CONFIDENCE_CUTOFF:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Generate human-readable explanation
    explanation = _build_explanation(sif_potential, top_signals, sif_score)

    return {
        "sif_score":     sif_score,
        "sif_potential": sif_potential,
        "confidence":    confidence,
        "top_signals":   top_signals,
        "explanation":   explanation,
    }


def _build_explanation(sif_potential: bool, signals: List[str], score: float) -> str:
    """Build a short explanation string for the classification."""
    if not signals:
        if sif_potential:
            return "Classified SIF-potential based on contextual risk indicators."
        else:
            return "No high-energy hazard or barrier failure signals detected."

    top = ", ".join(f'"{s}"' for s in signals[:3])
    if sif_potential:
        return f"SIF-potential flagged: detected high-risk signals — {top}. Score: {score:.2f}."
    else:
        return f"Non-SIF: signals {top} present but below SIF threshold. Score: {score:.2f}."


def batch_classify(
    texts: List[str],
    use_llm: bool = False,
) -> List[Dict]:
    """
    High-speed batch classification for lists of texts.
    """
    results = []
    if use_llm:
        for text in texts:
            results.append(classify_report(text, use_llm=True))
    else:
        for text in texts:
            results.append(classify_report(text, use_llm=False))
    return results

