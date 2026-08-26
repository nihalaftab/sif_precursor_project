"""
Precursor Pattern Miner using BERTopic / sklearn clustering.
Surfaces recurring activity + location + barrier-failure triads
from SIF-flagged reports.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def mine_patterns(
    narratives: List[str],
    embeddings: Optional[np.ndarray] = None,
    n_topics: int = 10,
) -> Dict:
    """
    Mine recurring precursor patterns from SIF-flagged report narratives.

    Falls back to TF-IDF + KMeans clustering if BERTopic not available.

    Returns:
        {
          "topics": [(topic_id, label, keywords, count), ...],
          "doc_topics": [topic_id per narrative],
          "method": "bertopic" | "tfidf_kmeans"
        }
    """
    if len(narratives) < 5:
        return {"topics": [], "doc_topics": [], "method": "insufficient_data"}

    # Try BERTopic first
    try:
        return _mine_bertopic(narratives, embeddings, n_topics)
    except Exception:
        pass

    # Fallback: TF-IDF + KMeans
    return _mine_tfidf_kmeans(narratives, n_topics)


def _mine_bertopic(narratives: List[str], embeddings, n_topics: int) -> Dict:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    n_clusters = min(n_topics, max(2, len(narratives) // 3))

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        nr_topics=n_clusters,
        calculate_probabilities=False,
        verbose=False,
        min_topic_size=2,
    )

    if embeddings is not None:
        topics, _ = topic_model.fit_transform(narratives, embeddings)
    else:
        topics, _ = topic_model.fit_transform(narratives)

    topic_info = topic_model.get_topic_info()
    topics_out = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        words_scores = topic_model.get_topic(tid)
        keywords = [w for w, _ in words_scores[:8]] if words_scores else []
        label = _label_from_keywords(keywords)
        count = int(row["Count"])
        topics_out.append({
            "topic_id": int(tid),
            "label": label,
            "keywords": keywords,
            "count": count,
        })

    topics_out.sort(key=lambda x: x["count"], reverse=True)
    return {"topics": topics_out[:n_topics], "doc_topics": topics, "method": "bertopic"}


def _mine_tfidf_kmeans(narratives: List[str], n_topics: int) -> Dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    n_clusters = min(n_topics, max(2, len(narratives) // 3))

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=500,
        min_df=1,
    )
    X = vec.fit_transform(narratives)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    feature_names = vec.get_feature_names_out()
    topics_out = []
    for cluster_id in range(n_clusters):
        center = km.cluster_centers_[cluster_id]
        top_indices = center.argsort()[::-1][:8]
        keywords = [feature_names[i] for i in top_indices]
        count = int((labels == cluster_id).sum())
        label = _label_from_keywords(keywords)
        topics_out.append({
            "topic_id": cluster_id,
            "label": label,
            "keywords": keywords,
            "count": count,
        })

    topics_out.sort(key=lambda x: x["count"], reverse=True)
    return {"topics": topics_out, "doc_topics": labels.tolist(), "method": "tfidf_kmeans"}


def _label_from_keywords(keywords: List[str]) -> str:
    """Generate a human-readable label from top keywords."""
    # Map common keyword groups to safety labels
    mappings = [
        (["confined", "space", "tank", "h2s", "atmospheric"],      "Confined Space Entry"),
        (["lockout", "tagout", "loto", "isolation", "energy"],      "Energy Isolation Failure"),
        (["welding", "hot", "work", "spark", "fire"],               "Hot Work Control"),
        (["fall", "height", "scaffold", "ladder", "harness"],       "Fall from Height"),
        (["crane", "lifting", "sling", "rigging", "load"],          "Unsafe Lifting Operation"),
        (["permit", "ptw", "authorization", "work", "expired"],     "Work Authorisation Gap"),
        (["driving", "vehicle", "speed", "seatbelt", "collision"],  "Road Safety Violation"),
        (["bypass", "override", "interlock", "defeat", "guard"],    "Safety Control Bypass"),
        (["line", "fire", "dropped", "struck", "object"],           "Line of Fire Exposure"),
        (["pressure", "electrical", "shock", "energized", "arc"],   "High-Energy Release"),
    ]
    kw_lower = [k.lower() for k in keywords]
    for triggers, label in mappings:
        if any(t in kw_lower for t in triggers):
            return label
    # Generic label from top 3 keywords
    return " | ".join(keywords[:3]).title()


def summarise_patterns(patterns_result: Dict) -> pd.DataFrame:
    """Convert pattern mining result to a DataFrame for display."""
    topics = patterns_result.get("topics", [])
    if not topics:
        return pd.DataFrame(columns=["Rank", "Precursor Pattern", "Keywords", "Report Count"])
    rows = []
    for i, t in enumerate(topics, 1):
        rows.append({
            "Rank": i,
            "Precursor Pattern": t["label"],
            "Keywords": ", ".join(t["keywords"][:6]),
            "Report Count": t["count"],
        })
    return pd.DataFrame(rows)
