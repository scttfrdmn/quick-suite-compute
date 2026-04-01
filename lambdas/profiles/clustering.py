"""
clustering.py — K-Means Clustering profile

Segments records into k groups using scikit-learn KMeans.
Appends cluster_id and cluster_distance columns to the input DataFrame.

Called by runner/handler.py via the entrypoint "clustering.kmeans_handler".
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def kmeans_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Run K-Means clustering on numeric feature columns.

    Returns (result_df, diagnostics) where result_df is the input
    DataFrame with cluster_id and cluster_distance columns appended.
    """
    k = int(parameters.get("k", 5))
    feature_cols = parameters.get("features") or []
    standardize = bool(parameters.get("standardize", True))

    if df.empty:
        return df.assign(cluster_id=None, cluster_distance=None), {
            "warning": "Input dataset is empty (extract step stub)"
        }

    # Validate feature columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found in dataset: {missing}")

    # Auto-select numeric columns if none specified
    if not feature_cols:
        feature_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(feature_cols) < 2:
        raise ValueError(
            f"K-Means requires at least 2 numeric feature columns; found {len(feature_cols)}"
        )

    X = df[feature_cols].copy()

    # Drop rows with any NaN in features
    nan_mask = X.isna().any(axis=1)
    nan_count = int(nan_mask.sum())
    X_clean = X[~nan_mask].copy()

    if len(X_clean) < k:
        raise ValueError(
            f"Not enough complete rows ({len(X_clean)}) to form {k} clusters"
        )

    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
    else:
        X_scaled = X_clean.values

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    # Compute distance to assigned cluster centroid
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[labels], axis=1)

    result = df.copy()
    result["cluster_id"] = np.nan
    result["cluster_distance"] = np.nan
    result.loc[~nan_mask, "cluster_id"] = labels.astype(int)
    result.loc[~nan_mask, "cluster_distance"] = distances

    # Cluster summary for diagnostics
    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    diagnostics = {
        "k": k,
        "feature_columns": feature_cols,
        "standardized": standardize,
        "inertia": float(kmeans.inertia_),
        "n_iter": int(kmeans.n_iter_),
        "rows_used": len(X_clean),
        "rows_with_nan_excluded": nan_count,
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
    }

    logger.info(f"K-Means complete: k={k}, inertia={kmeans.inertia_:.2f}, rows={len(X_clean)}")
    return result, diagnostics
