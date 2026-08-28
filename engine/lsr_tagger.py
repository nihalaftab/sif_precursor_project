import re
from typing import Dict, List, Tuple
import numpy as np

from utils.config import LIFE_SAVING_RULES, LSR_SIMILARITY_THRESHOLD

# Cache for rule embeddings
_rule_embeddings: Dict[str, np.ndarray] = {}
_rule_names = list(LIFE_SAVING_RULES.keys())
_rule_matrix: np.ndarray = None  # Normalized matrix shape (9, 384)

# Precompile keyword patterns for each rule
_RULE_PATTERNS = {
    rule_name: re.compile(
        r"\b(" + "|".join(re.escape(k) for k in rule_data["keywords"]) + r")\b",
        re.IGNORECASE
    )
    for rule_name, rule_data in LIFE_SAVING_RULES.items()
}


_rule_embeddings_fast: Dict[str, np.ndarray] = {}
_rule_matrix_fast: np.ndarray = None

def _get_rule_embeddings(fast_mode: bool = False) -> Dict[str, np.ndarray]:
    """Lazily compute and cache LSR description embeddings (in a single batch)."""
    global _rule_embeddings, _rule_matrix, _rule_embeddings_fast, _rule_matrix_fast
    if fast_mode:
        if not _rule_embeddings_fast:
            from engine.preprocessor import get_embeddings_batch
            descs = [LIFE_SAVING_RULES[rule_name]["description"] for rule_name in _rule_names]
            mat = get_embeddings_batch(descs, show_progress=False, fast_mode=True)
            for i, rule_name in enumerate(_rule_names):
                _rule_embeddings_fast[rule_name] = mat[i]
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            _rule_matrix_fast = mat / norms
        return _rule_embeddings_fast

    if not _rule_embeddings:
        from engine.preprocessor import get_embeddings_batch
        descs = [LIFE_SAVING_RULES[rule_name]["description"] for rule_name in _rule_names]
        mat = get_embeddings_batch(descs, show_progress=False, fast_mode=False)
        for i, rule_name in enumerate(_rule_names):
            _rule_embeddings[rule_name] = mat[i]
        
        # Build normalized matrix
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _rule_matrix = mat / norms
    return _rule_embeddings


def _get_rule_matrix(fast_mode: bool = False) -> np.ndarray:
    global _rule_matrix, _rule_matrix_fast
    if fast_mode:
        if _rule_matrix_fast is None:
            _get_rule_embeddings(fast_mode=True)
        return _rule_matrix_fast

    if _rule_matrix is None:
        _get_rule_embeddings(fast_mode=False)
    return _rule_matrix



def _keyword_score(text_lower: str, rule_name: str) -> float:
    """Count keyword hits for a rule, normalised to [0, 1]."""
    pat = _RULE_PATTERNS.get(rule_name)
    if pat:
        hits = len(pat.findall(text_lower))
    else:
        keywords = LIFE_SAVING_RULES[rule_name]["keywords"]
        hits = sum(1 for kw in keywords if kw in text_lower)
    return min(hits / max(len(LIFE_SAVING_RULES[rule_name]["keywords"]) * 0.10, 1), 1.0)


def _semantic_score(text_embedding: np.ndarray, rule_name: str) -> float:
    """Cosine similarity between text embedding and rule description embedding."""
    rule_emb = _get_rule_embeddings()[rule_name]
    num = np.dot(text_embedding, rule_emb)
    den = (np.linalg.norm(text_embedding) * np.linalg.norm(rule_emb)) + 1e-10
    return float(num / den)


def tag_report(
    clean_text: str,
    text_embedding: np.ndarray = None,
) -> Dict:
    """
    Tag a report with IOGP Life-Saving Rules.
    """
    text_lower = clean_text.lower()

    if text_embedding is None:
        from engine.preprocessor import get_embedding
        text_embedding = get_embedding(clean_text)

    scores: Dict[str, float] = {}
    for rule_name in _rule_names:
        kw_score  = _keyword_score(text_lower, rule_name)
        sem_score = _semantic_score(text_embedding, rule_name)
        combined  = 0.50 * kw_score + 0.50 * max(sem_score, 0.0)
        scores[rule_name] = round(combined, 4)

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
    fast_mode: bool = False,
) -> List[Dict]:
    """
    High-speed vectorized batch tagging using single matrix multiplication.
    """
    if embeddings is None:
        from engine.preprocessor import get_embeddings_batch
        embeddings = get_embeddings_batch(texts, fast_mode=fast_mode)

    # Normalized text embeddings matrix
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norms[emb_norms == 0] = 1.0
    norm_embeddings = embeddings / emb_norms

    # Rule matrix shape (9, 384)
    rule_mat = _get_rule_matrix(fast_mode=fast_mode)  # shape (9, dim)


    # Matrix multiplication: shape (N, 9)
    cos_sim_matrix = np.dot(norm_embeddings, rule_mat.T)
    cos_sim_matrix = np.maximum(cos_sim_matrix, 0.0)

    results = []
    for i, text in enumerate(texts):
        text_lower = text.lower()
        scores = {}
        for j, rule_name in enumerate(_rule_names):
            kw_score = _keyword_score(text_lower, rule_name)
            sem_score = float(cos_sim_matrix[i, j])
            combined = 0.50 * kw_score + 0.50 * sem_score
            scores[rule_name] = round(combined, 4)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        tagged = [rule for rule, score in ranked if score >= LSR_SIMILARITY_THRESHOLD]
        primary_rule = ranked[0][0] if ranked else "Unknown"
        primary_score = ranked[0][1] if ranked else 0.0

        results.append({
            "primary_rule":  primary_rule,
            "primary_score": primary_score,
            "all_rules":     ranked,
            "tagged_rules":  tagged if tagged else [primary_rule],
        })

    return results

