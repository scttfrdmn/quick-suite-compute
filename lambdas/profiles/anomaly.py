"""
anomaly.py — Anomaly Detection (Isolation Forest) profile

Flags anomalous records using scikit-learn IsolationForest.
Appends is_anomaly (boolean) and anomaly_score (float) columns.

Called by runner/handler.py via the entrypoint "anomaly.isolation_forest_handler".
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def isolation_forest_handler(
    df: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, dict]:
    """
    Detect anomalies using Isolation Forest.

    Returns (result_df, diagnostics). result_df has is_anomaly and
    anomaly_score appended (NaN for rows excluded due to missing values).
    """
    feature_cols = parameters.get("features") or []
    contamination = float(parameters.get("contamination", 0.05))
    return_scores = bool(parameters.get("return_scores", True))

    if df.empty:
        return df.assign(is_anomaly=None, anomaly_score=None), {
            "warning": "Input dataset is empty (extract step stub)"
        }

    MAX_ROWS = 100_000
    if len(df) > MAX_ROWS:
        raise ValueError(
            f"Anomaly detection is limited to {MAX_ROWS} rows in Lambda mode "
            f"(received {len(df)}). Filter to a sample or use a smaller time window."
        )

    # Auto-select numeric columns if none specified
    if not feature_cols:
        feature_cols = list(df.select_dtypes(include=[np.number]).columns)
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in dataset: {missing}")

    if not feature_cols:
        raise ValueError("No numeric feature columns available for anomaly detection")

    X = df[feature_cols].copy()
    nan_mask = X.isna().any(axis=1)
    X_clean = X[~nan_mask]
    nan_count = int(nan_mask.sum())

    if len(X_clean) < 10:
        raise ValueError(f"Too few complete rows ({len(X_clean)}) for anomaly detection")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    clf = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    predictions = clf.fit_predict(X_scaled)  # 1 = normal, -1 = anomaly
    scores = clf.score_samples(X_scaled)  # lower = more anomalous

    is_anomaly = (predictions == -1)
    anomaly_count = int(is_anomaly.sum())

    result = df.copy()
    result["is_anomaly"] = np.nan
    result["anomaly_score"] = np.nan

    result.loc[~nan_mask, "is_anomaly"] = is_anomaly.astype(bool)
    if return_scores:
        result.loc[~nan_mask, "anomaly_score"] = scores.astype(float)
    else:
        result.drop(columns=["anomaly_score"], inplace=True)

    diagnostics = {
        "feature_columns": feature_cols,
        "contamination": contamination,
        "rows_used": len(X_clean),
        "rows_with_nan_excluded": nan_count,
        "anomalies_detected": anomaly_count,
        "anomaly_rate": float(anomaly_count / len(X_clean)) if len(X_clean) > 0 else 0.0,
    }

    logger.info(
        f"Isolation Forest complete: rows={len(X_clean)}, "
        f"anomalies={anomaly_count}, rate={diagnostics['anomaly_rate']:.3f}"
    )
    return result, diagnostics
