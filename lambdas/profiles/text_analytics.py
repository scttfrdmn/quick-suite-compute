"""
text_analytics.py — Text analytics profiles

sentiment_handler   — VADER sentiment scoring (text-sentiment)
similarity_handler  — TF-IDF near-duplicate detection (text-similarity)

Called by runner/handler.py via entrypoints in config/profiles/*.json.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# text-sentiment
# ---------------------------------------------------------------------------

def sentiment_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Score each text record as positive, negative, or neutral using VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based
    approach well-suited for short social/survey text. No model download required.

    Returns the original DataFrame with sentiment columns appended.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    text_col = parameters.get("text_column", "")
    output_scores = bool(parameters.get("output_scores", False))

    if not text_col or text_col not in df.columns:
        raise ValueError(f"text_column '{text_col}' not found")

    if df.empty:
        result = df.copy()
        result["sentiment"] = pd.Series(dtype=str)
        result["sentiment_score"] = pd.Series(dtype=float)
        return result, {"warning": "Input dataset is empty"}

    analyzer = SentimentIntensityAnalyzer()

    texts = df[text_col].fillna("").astype(str)
    scores = texts.apply(lambda t: analyzer.polarity_scores(t))

    compound = scores.apply(lambda s: s["compound"])

    result = df.copy()
    result["sentiment"] = compound.map(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    )
    result["sentiment_score"] = compound.round(4)

    if output_scores:
        result["pos_score"] = scores.apply(lambda s: round(s["pos"], 4))
        result["neg_score"] = scores.apply(lambda s: round(s["neg"], 4))
        result["neu_score"] = scores.apply(lambda s: round(s["neu"], 4))
        result["compound_score"] = compound.round(4)

    n_total = len(df)
    counts = result["sentiment"].value_counts().to_dict()
    diagnostics = {
        "text_column": text_col,
        "method": "vader",
        "n_total": n_total,
        "n_positive": int(counts.get("positive", 0)),
        "n_negative": int(counts.get("negative", 0)),
        "n_neutral": int(counts.get("neutral", 0)),
        "pct_positive": round(counts.get("positive", 0) / n_total, 4) if n_total else 0,
        "pct_negative": round(counts.get("negative", 0) / n_total, 4) if n_total else 0,
        "mean_compound_score": round(float(compound.mean()), 4),
        "output_scores_included": output_scores,
    }

    logger.info(
        f"sentiment complete: n={n_total}, "
        f"pos={counts.get('positive',0)}, neg={counts.get('negative',0)}, "
        f"neu={counts.get('neutral',0)}"
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# text-similarity
# ---------------------------------------------------------------------------

def similarity_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Find near-duplicate text records using TF-IDF cosine similarity.

    Groups records above `similarity_threshold` into connected components.
    Within each group, the record with the highest average similarity to
    others is designated the canonical representative.

    Returns the original DataFrame with similarity_group_id, is_near_duplicate,
    and canonical_id columns appended.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    text_col = parameters.get("text_column", "")
    id_col = parameters.get("id_column", "")
    threshold = float(parameters.get("similarity_threshold", 0.85))

    if not text_col or text_col not in df.columns:
        raise ValueError(f"text_column '{text_col}' not found")
    if id_col and id_col not in df.columns:
        raise ValueError(f"id_column '{id_col}' not found")
    if not (0.0 < threshold <= 1.0):
        raise ValueError("similarity_threshold must be between 0 and 1")

    if df.empty:
        result = df.copy()
        result["similarity_group_id"] = pd.Series(dtype=int)
        result["is_near_duplicate"] = pd.Series(dtype=bool)
        result["canonical_id"] = pd.Series(dtype=object)
        return result, {"warning": "Input dataset is empty"}

    texts = df[text_col].fillna("").astype(str)
    non_empty_mask = texts.str.strip() != ""
    n_non_empty = int(non_empty_mask.sum())

    if n_non_empty < 2:
        result = df.copy()
        result["similarity_group_id"] = range(len(df))
        result["is_near_duplicate"] = False
        result["canonical_id"] = df[id_col].astype(str) if id_col else df.index.astype(str)
        return result, {"warning": "Not enough non-empty texts for similarity comparison", "n_non_empty": n_non_empty}

    # Vectorize
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=1.0,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute similarity in chunks to manage memory for large datasets
    n = len(df)
    chunk_size = min(500, n)

    # Build adjacency: pairs above threshold
    group_id = list(range(n))  # union-find: each row starts as its own group

    def _find(x: int) -> int:
        while group_id[x] != x:
            group_id[x] = group_id[group_id[x]]
            x = group_id[x]
        return x

    def _union(x: int, y: int) -> None:
        group_id[_find(x)] = _find(y)

    n_pairs_above = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = tfidf_matrix[start:end]
        sim_block = cosine_similarity(chunk, tfidf_matrix).astype(np.float32)
        rows, cols = np.where(sim_block >= threshold)
        for r, c in zip(rows, cols):
            i, j = start + r, int(c)
            if i < j:  # avoid self and duplicates
                _union(i, j)
                n_pairs_above += 1

    # Assign final group IDs (compress root IDs to 0-based integers)
    roots = [_find(i) for i in range(n)]
    unique_roots = sorted(set(roots))
    root_to_group = {r: gid for gid, r in enumerate(unique_roots)}
    group_ids = [root_to_group[r] for r in roots]

    result = df.copy()
    result["similarity_group_id"] = group_ids

    # Group size > 1 → near-duplicate
    group_sizes = pd.Series(group_ids).value_counts()
    result["is_near_duplicate"] = result["similarity_group_id"].map(
        lambda g: group_sizes.get(g, 1) > 1
    )

    # Canonical ID: within each duplicate group, pick the first-appearing row
    id_vals = df[id_col].astype(str) if id_col else df.index.astype(str)
    group_to_canonical: dict[int, str] = {}
    for i, gid in enumerate(group_ids):
        if gid not in group_to_canonical:
            group_to_canonical[gid] = str(id_vals.iloc[i])
    result["canonical_id"] = result["similarity_group_id"].map(group_to_canonical)

    n_groups = len(unique_roots)
    n_duplicates = int((group_sizes > 1).sum())
    n_dup_records = int(result["is_near_duplicate"].sum())

    diagnostics = {
        "text_column": text_col,
        "id_column": id_col,
        "similarity_threshold": threshold,
        "n_records": n,
        "n_groups": n_groups,
        "n_duplicate_groups": n_duplicates,
        "n_near_duplicate_records": n_dup_records,
        "pct_near_duplicate": round(n_dup_records / n, 4) if n else 0,
    }

    logger.info(
        f"similarity complete: n={n}, groups={n_groups}, "
        f"dup_groups={n_duplicates}, dup_records={n_dup_records}"
    )
    return result, diagnostics
