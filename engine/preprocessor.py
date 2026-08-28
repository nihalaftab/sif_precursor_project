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
                _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "textcat"])
            except Exception:
                try:
                    from spacy.cli import download
                    download("en_core_web_sm")
                    _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "textcat"])
                except Exception:
                    _nlp = None
        except Exception:
            _nlp = None
    return _nlp



def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            from utils.config import EMBEDDING_MODEL
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
        except Exception:
            _embedder = "fallback"
    return _embedder


# ── Precompiled Regex Patterns for High Performance ─────────────────────────
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(OIL_FIELD_ABBREVIATIONS.keys(), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)

def _abbrev_replace(match):
    key = match.group(0).lower()
    return OIL_FIELD_ABBREVIATIONS.get(key, match.group(0))

# Precompiled keyword patterns
_EQUIPMENT_KW = [
    "pump", "compressor", "valve", "crane", "ladder", "scaffold",
    "vessel", "tank", "generator", "motor", "panel", "cable",
    "flange", "pipeline", "wellhead", "bop", "separator", "exchanger",
]
_LOCATION_KW = [
    "wellhead", "platform", "tank farm", "compressor station",
    "workshop", "control room", "laydown", "yard", "shed",
    "road", "access track", "roof", "mezzanine", "pit", "sump",
]
_ACTIVITY_KW = [
    "welding", "grinding", "lifting", "entering", "driving",
    "climbing", "removing", "cutting", "inspection", "maintenance",
    "commissioning", "testing", "sampling", "painting", "drilling",
]

_EQUIPMENT_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _EQUIPMENT_KW) + r")\b", re.IGNORECASE)
_LOCATION_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _LOCATION_KW) + r")\b", re.IGNORECASE)
_ACTIVITY_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _ACTIVITY_KW) + r")\b", re.IGNORECASE)
_ENERGY_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in HIGH_ENERGY_KEYWORDS) + r")\b", re.IGNORECASE)


# ── Step 1: Abbreviation Expansion ─────────────────────────────────────────────
def expand_abbreviations(text: str) -> str:
    """Replace known oil-field abbreviations with full forms (high-speed regex)."""
    return _ABBREV_PATTERN.sub(_abbrev_replace, text)


# ── Step 2: Text Normalization ─────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, remove special chars."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-\.\,\/\%]", " ", text)
    return text.strip()


# ── Step 3: Entity Extraction ──────────────────────────────────────────────────
def extract_entities(text: str, doc=None, use_spacy: bool = True) -> Dict[str, List[str]]:
    """
    Extract named entities using fast regex with optional spaCy augmentation.
    """
    entities: Dict[str, List[str]] = {
        "equipment": list(dict.fromkeys(m.lower() for m in _EQUIPMENT_RE.findall(text))),
        "location":  list(dict.fromkeys(m.lower() for m in _LOCATION_RE.findall(text))),
        "activity":  list(dict.fromkeys(m.lower() for m in _ACTIVITY_RE.findall(text))),
        "hazard":    list(dict.fromkeys(m.lower() for m in _ENERGY_RE.findall(text))),
    }

    # spaCy augmentation if doc is passed or when use_spacy is True for single text
    if doc is not None:
        for ent in doc.ents:
            if ent.label_ in ("FAC", "LOC", "GPE"):
                entities["location"].append(ent.text.lower())
            elif ent.label_ in ("ORG", "PRODUCT"):
                entities["equipment"].append(ent.text.lower())
        for key in ("location", "equipment"):
            entities[key] = list(dict.fromkeys(entities[key]))
    elif use_spacy:
        nlp = _get_nlp()
        if nlp:
            try:
                spacy_doc = nlp(text[:1000])
                for ent in spacy_doc.ents:
                    if ent.label_ in ("FAC", "LOC", "GPE"):
                        entities["location"].append(ent.text.lower())
                    elif ent.label_ in ("ORG", "PRODUCT"):
                        entities["equipment"].append(ent.text.lower())
                for key in ("location", "equipment"):
                    entities[key] = list(dict.fromkeys(entities[key]))
            except Exception:
                pass

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
    Full preprocessing pipeline for a single report.
    """
    text_expanded   = expand_abbreviations(raw_text)
    text_normalized = normalize_text(text_expanded)
    entities        = extract_entities(text_expanded, use_spacy=True)
    signals         = score_signals(text_normalized)

    return {
        "raw_text":        raw_text,
        "clean_text":      text_normalized,
        "expanded_text":   text_expanded,
        "entities":        entities,
        "signal_scores":   signals,
    }


def preprocess_batch(raw_texts: List[str], use_spacy: bool = False) -> List[Dict]:
    """
    High-speed batched preprocessing for hundreds or thousands of reports.
    """
    results = []
    expanded_list = [expand_abbreviations(t) for t in raw_texts]
    normalized_list = [normalize_text(t) for t in expanded_list]

    docs = [None] * len(raw_texts)
    if use_spacy:
        nlp = _get_nlp()
        if nlp:
            try:
                # Use nlp.pipe with only ner for high throughput
                docs = list(nlp.pipe(
                    [t[:1000] for t in expanded_list],
                    batch_size=128,
                    disable=["tagger", "parser", "lemmatizer"]
                ))
            except Exception:
                docs = [None] * len(raw_texts)

    for i in range(len(raw_texts)):
        entities = extract_entities(expanded_list[i], doc=docs[i], use_spacy=use_spacy)
        signals = score_signals(normalized_list[i])
        results.append({
            "raw_text":        raw_texts[i],
            "clean_text":      normalized_list[i],
            "expanded_text":   expanded_list[i],
            "entities":        entities,
            "signal_scores":   signals,
        })
    return results



import numpy as np

def get_embedding(text: str):
    """Return sentence embedding for a given text."""
    embedder = _get_embedder()
    if embedder != "fallback":
        try:
            return embedder.encode(text, convert_to_numpy=True)
        except Exception:
            pass
    res = get_embeddings_batch([text], show_progress=False)
    return res[0]


def get_embeddings_batch(texts: List[str], batch_size: int = 128, show_progress: bool = False, fast_mode: bool = False):
    """Return sentence embeddings for a list of texts (high throughput)."""
    if not fast_mode:
        embedder = _get_embedder()
        if embedder != "fallback":
            try:
                return embedder.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                )
            except Exception:
                pass
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        vec = TfidfVectorizer(max_features=384, stop_words="english")
        X = vec.fit_transform(texts).toarray()
        if X.shape[1] < 384:
            pad = np.zeros((X.shape[0], 384 - X.shape[1]))
            X = np.hstack([X, pad])
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms
    except Exception:
        return np.zeros((len(texts), 384))


