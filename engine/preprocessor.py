"""
Text preprocessing pipeline:
  - Oil-field abbreviation expansion
  - spaCy NER for entity extraction (equipment, location, activity)
  - Sentence embeddings via sentence-transformers
"""

import re
import string
from typing import Dict, List, Tuple

from utils.config import OIL_FIELD_ABBREVIATIONS, HIGH_ENERGY_KEYWORDS, BARRIER_FAILURE_SIGNALS, NEAR_FATAL_SIGNALS


# ── Lazy model loading (avoid import-time cost) ────────────────────────────────
_nlp = None
_embedder = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
        except ImportError:
            _nlp = None
    return _nlp


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        from utils.config import EMBEDDING_MODEL
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ── Step 1: Abbreviation Expansion ─────────────────────────────────────────────
def expand_abbreviations(text: str) -> str:
    """Replace known oil-field abbreviations with full forms."""
    words = text.split()
    expanded = []
    for word in words:
        clean = word.lower().strip(string.punctuation)
        if clean in OIL_FIELD_ABBREVIATIONS:
            expanded.append(OIL_FIELD_ABBREVIATIONS[clean])
        else:
            expanded.append(word)
    return " ".join(expanded)


# ── Step 2: Text Normalization ─────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, remove special chars."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-\.\,\/\%]", " ", text)
    return text.strip()


# ── Step 3: Entity Extraction ──────────────────────────────────────────────────
def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities using spaCy.
    Falls back to keyword-based extraction if spaCy model unavailable.
    """
    nlp = _get_nlp()
    entities: Dict[str, List[str]] = {
        "equipment": [],
        "location":  [],
        "activity":  [],
        "hazard":    [],
    }

    # Keyword-based extraction (always runs)
    equipment_kw = [
        "pump", "compressor", "valve", "crane", "ladder", "scaffold",
        "vessel", "tank", "generator", "motor", "panel", "cable",
        "flange", "pipeline", "wellhead", "bop", "separator", "exchanger",
    ]
    location_kw = [
        "wellhead", "platform", "tank farm", "compressor station",
        "workshop", "control room", "laydown", "yard", "shed",
        "road", "access track", "roof", "mezzanine", "pit", "sump",
    ]
    activity_kw = [
        "welding", "grinding", "lifting", "entering", "driving",
        "climbing", "removing", "cutting", "inspection", "maintenance",
        "commissioning", "testing", "sampling", "painting", "drilling",
    ]

    text_lower = text.lower()
    entities["equipment"] = [k for k in equipment_kw if k in text_lower]
    entities["location"]  = [k for k in location_kw  if k in text_lower]
    entities["activity"]  = [k for k in activity_kw  if k in text_lower]
    entities["hazard"]    = [k for k in HIGH_ENERGY_KEYWORDS if k in text_lower]

    # spaCy augmentation
    if nlp:
        doc = nlp(text[:1000])  # limit for speed
        for ent in doc.ents:
            if ent.label_ in ("FAC", "LOC", "GPE"):
                entities["location"].append(ent.text.lower())
            elif ent.label_ in ("ORG", "PRODUCT"):
                entities["equipment"].append(ent.text.lower())

    # Deduplicate
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))

    return entities


# ── Step 4: Signal Scoring ─────────────────────────────────────────────────────
def score_signals(text: str) -> Dict[str, float]:
    """
    Score the text for keyword signal categories.
    Returns normalised scores (0-1) for energy hazard, barrier failure, near-fatal.
    """
    text_lower = text.lower()

    def _hit_ratio(keywords: List[str]) -> float:
        hits = sum(1 for k in keywords if k in text_lower)
        return min(hits / max(len(keywords) * 0.15, 1), 1.0)

    return {
        "energy_hazard":   _hit_ratio(HIGH_ENERGY_KEYWORDS),
        "barrier_failure": _hit_ratio(BARRIER_FAILURE_SIGNALS),
        "near_fatal":      _hit_ratio(NEAR_FATAL_SIGNALS),
    }


# ── Step 5: Full Preprocessing Pipeline ───────────────────────────────────────
def preprocess(raw_text: str) -> Dict:
    """
    Full preprocessing pipeline.
    Returns a dict with cleaned text, entities, signal scores, and embedding.
    """
    text_expanded   = expand_abbreviations(raw_text)
    text_normalized = normalize_text(text_expanded)
    entities        = extract_entities(text_expanded)
    signals         = score_signals(text_normalized)

    return {
        "raw_text":        raw_text,
        "clean_text":      text_normalized,
        "expanded_text":   text_expanded,
        "entities":        entities,
        "signal_scores":   signals,
    }


def get_embedding(text: str):
    """Return sentence embedding for a given text."""
    embedder = _get_embedder()
    return embedder.encode(text, convert_to_numpy=True)


def get_embeddings_batch(texts: List[str], batch_size: int = 64, show_progress: bool = True):
    """Return sentence embeddings for a list of texts."""
    embedder = _get_embedder()
    return embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
