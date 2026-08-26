"""
IOGP Life-Saving Rule Tagger.
Two-stage approach:
  1. Keyword matching (fast, high recall)
  2. Semantic similarity via sentence embeddings (high precision)
Returns primary and secondary LSR tags with confidence scores.
"""

from typing import Dict, List, Tuple
import numpy as np

from utils.config import LIFE_SAVING_RULES, LSR_SIMILARITY_THRESHOLD

# Cache for rule embeddings
_rule_embeddings: Dict[str, np.ndarray] = {}


def _get_rule_embeddings() -> Dict[str, np.ndarray]:
    """Lazily compute and cache LSR description embeddings."""
    global _rule_embeddings
    if not _rule_embeddings:
        from engine.preprocessor import get_embedding
        for rule_name, rule_data in LIFE_SAVING_RULES.items():
            _rule_embeddings[rule_name] = get_embedding(rule_data["description"])
    return _rule_embeddings


def _keyword_score(text_lower: str, rule_name: str) -> float:
    """Count keyword hits for a rule, normalised to [0, 1]."""
    keywords = LIFE_SAVING_RULES[rule_name]["keywords"]
    hits = sum(1 for kw in keywords if kw in text_lower)
    return min(hits / max(len(keywords) * 0.10, 1), 1.0)


def _semantic_score(text_embedding: np.ndarray, rule_name: str) -> float:
    """Cosine similarity between text embedding and rule description embedding."""
    rule_emb = _get_rule_embeddings()[rule_name]
    # Cosine similarity
    num = np.dot(text_embedding, rule_emb)
    den = (np.linalg.norm(text_embedding) * np.linalg.norm(rule_emb)) + 1e-10
    return float(num / den)


def tag_report(
    clean_text: str,
    text_embedding: np.ndarray = None,
) -> Dict:
    """
    Tag a report with IOGP Life-Saving Rules.
    
    Returns:
        {
          "primary_rule":   str,
          "primary_score":  float,
          "all_rules":      [(rule_name, combined_score), ...],
          "tagged_rules":   [str, ...]   # rules above threshold
        }
    """
    text_lower = clean_text.lower()

    # Compute embeddings if not supplied
    if text_embedding is None:
        from engine.preprocessor import get_embedding
        text_embedding = get_embedding(clean_text)

    scores: Dict[str, float] = {}
    for rule_name in LIFE_SAVING_RULES:
        kw_score  = _keyword_score(text_lower, rule_name)
        sem_score = _semantic_score(text_embedding, rule_name)
        # Ensemble: 50% keyword + 50% semantic
        combined  = 0.50 * kw_score + 0.50 * max(sem_score, 0.0)
        scores[rule_name] = round(combined, 4)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    tagged = [rule for rule, score in ranked if score >= LSR_SIMILARITY_THRESHOLD]

    primary_rule  = ranked[0][0] if ranked else "Unknown"
    primary_score = ranked[0][1] if ranked else 0.0

    return {
        "primary_rule":  primary_rule,
        "primary_score": primary_score,
        "all_rules":     ranked,
        "tagged_rules":  tagged if tagged else [primary_rule],
    }


def batch_tag(
    texts: List[str],
    embeddings: np.ndarray = None,
) -> List[Dict]:
    """
    Tag multiple reports. Accepts pre-computed embeddings for efficiency.
    """
    if embeddings is None:
        from engine.preprocessor import get_embeddings_batch
        embeddings = get_embeddings_batch(texts)

    results = []
    for text, emb in zip(texts, embeddings):
        results.append(tag_report(text, text_embedding=emb))
    return results
