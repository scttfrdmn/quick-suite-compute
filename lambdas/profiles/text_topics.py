"""
text_topics.py — Text Topic Modeling profile

Discovers latent topics in text data using scikit-learn LDA or NMF.
Appends dominant_topic, topic_probability, and topic_label columns.

Called by runner/handler.py via the entrypoint "text_topics.topics_handler".
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logger = logging.getLogger(__name__)

_TOP_WORDS_PER_TOPIC = 10


def topics_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Run topic modeling on a text column.

    Returns (result_df, diagnostics). result_df has dominant_topic,
    topic_probability, and topic_label appended.
    """
    text_col = parameters.get("text_column", "")
    num_topics = int(parameters.get("num_topics", 8))
    if not (2 <= num_topics <= 50):
        raise ValueError(f"num_topics must be between 2 and 50 (got {num_topics})")
    method = parameters.get("method", "lda").lower()
    min_df = float(parameters.get("min_doc_frequency", 0.02))
    max_df = float(parameters.get("max_doc_frequency", 0.95))
    if not (0 < min_df < max_df <= 1):
        raise ValueError(
            f"min_doc_frequency ({min_df}) must be < max_doc_frequency ({max_df}), "
            f"both in range (0, 1]"
        )
    if min_df < 0.01:
        raise ValueError(
            f"min_doc_frequency ({min_df}) must be >= 0.01 to prevent "
            f"unbounded vocabulary growth"
        )

    if df.empty:
        cols = ["dominant_topic", "topic_probability", "topic_label"]
        return df.assign(**{c: None for c in cols}), {
            "warning": "Input dataset is empty (extract step stub)"
        }

    if not text_col:
        raise ValueError("Parameter 'text_column' is required")
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in dataset")

    texts = df[text_col].fillna("").astype(str)
    non_empty = (texts.str.strip() != "").sum()
    if non_empty < num_topics * 2:
        raise ValueError(
            f"Too few non-empty documents ({non_empty}) for {num_topics} topics"
        )

    if method == "nmf":
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words="english")
        X = vectorizer.fit_transform(texts)
        model = NMF(n_components=num_topics, random_state=42, max_iter=500)
    else:
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words="english")
        X = vectorizer.fit_transform(texts)
        model = LatentDirichletAllocation(
            n_components=num_topics, random_state=42, max_iter=50
        )

    doc_topic = model.fit_transform(X)

    feature_names = vectorizer.get_feature_names_out()

    # Build topic labels from top words
    topic_labels = []
    topic_terms = []
    for topic_idx, topic_weights in enumerate(model.components_):
        top_indices = topic_weights.argsort()[-_TOP_WORDS_PER_TOPIC:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        label = f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}"
        topic_labels.append(label)
        topic_terms.append({
            "topic_id": topic_idx,
            "label": label,
            "top_words": top_words,
            "weights": [float(topic_weights[i]) for i in top_indices],
        })

    dominant_topics = np.argmax(doc_topic, axis=1)
    topic_probs = doc_topic[np.arange(len(doc_topic)), dominant_topics]

    result = df.copy()
    result["dominant_topic"] = dominant_topics.astype(int)
    result["topic_probability"] = topic_probs.astype(float)
    result["topic_label"] = [topic_labels[t] for t in dominant_topics]

    diagnostics = {
        "text_column": text_col,
        "num_topics": num_topics,
        "method": method,
        "min_doc_frequency": min_df,
        "max_doc_frequency": max_df,
        "vocabulary_size": len(feature_names),
        "documents_processed": int(non_empty),
        "topic_terms": topic_terms,
    }

    logger.info(f"Topic modeling complete: method={method}, topics={num_topics}, vocab={len(feature_names)}")
    return result, diagnostics
