"""
correlations.py — Correlation & Feature Importance profile

Computes pairwise correlation matrix and optionally ranks features
against a target column. Uses pandas and scipy.

Called by runner/handler.py via the entrypoint "correlations.correlations_handler".
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def correlations_handler(
    df: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, dict]:
    """
    Compute correlation matrix and feature importance ranking.

    Returns (result_df, diagnostics). result_df is a long-form correlation
    table with columns: feature_a, feature_b, correlation, p_value, rank.
    """
    feature_cols = parameters.get("features") or []
    target_col = (parameters.get("target") or "").strip() or None
    method = parameters.get("method", "pearson")
    top_n = int(parameters.get("top_n", 20))

    if df.empty:
        cols = ["feature_a", "feature_b", "correlation", "p_value", "rank"]
        return pd.DataFrame(columns=cols), {
            "warning": "Input dataset is empty (extract step stub)"
        }

    # Auto-select numeric columns if none specified
    if not feature_cols:
        feature_cols = list(df.select_dtypes(include=[np.number]).columns)
        if target_col and target_col in feature_cols:
            feature_cols.remove(target_col)
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")

    if len(feature_cols) < 2:
        raise ValueError("Need at least 2 feature columns for correlation analysis")

    clean_df = df[feature_cols + ([target_col] if target_col else [])].dropna()

    if len(clean_df) < 10:
        raise ValueError(f"Too few complete rows ({len(clean_df)}) for correlation analysis")

    # Compute pairwise correlations
    rows = []
    if method == "mutual_info":
        from sklearn.feature_selection import mutual_info_regression
        # Use mutual info relative to target if provided, else pairwise
        ref_col = target_col if target_col else feature_cols[0]
        y = clean_df[ref_col]
        X = clean_df[[c for c in feature_cols if c != ref_col]]
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            raise ValueError(
                f"mutual_info method requires numeric features; "
                f"non-numeric columns: {non_numeric}"
            )
        mi_scores = mutual_info_regression(X, y, random_state=42)
        for col, score in zip(X.columns, mi_scores):
            rows.append({
                "feature_a": col,
                "feature_b": ref_col,
                "correlation": float(score),
                "p_value": None,
            })
    else:
        corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr
        if target_col and target_col in clean_df.columns:
            # Features vs target only
            for col in feature_cols:
                if col == target_col:
                    continue
                try:
                    r, p = corr_func(clean_df[col], clean_df[target_col])
                    rows.append({
                        "feature_a": col,
                        "feature_b": target_col,
                        "correlation": float(r),
                        "p_value": float(p),
                    })
                except Exception as exc:
                    logger.warning(f"Skipping correlation {col!r} vs {target_col!r}: {exc}")
        else:
            # All pairwise
            for i, a in enumerate(feature_cols):
                for b in feature_cols[i + 1:]:
                    try:
                        r, p = corr_func(clean_df[a], clean_df[b])
                        rows.append({
                            "feature_a": a,
                            "feature_b": b,
                            "correlation": float(r),
                            "p_value": float(p),
                        })
                    except Exception as exc:
                        logger.warning(f"Skipping correlation {a!r} vs {b!r}: {exc}")

    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "p_value", "rank"])
    else:
        result = result.reindex(columns=["feature_a", "feature_b", "correlation", "p_value"])
        result["abs_corr"] = result["correlation"].abs()
        result = result.sort_values("abs_corr", ascending=False).head(top_n)
        result["rank"] = range(1, len(result) + 1)
        result = result.drop(columns=["abs_corr"]).reset_index(drop=True)

    diagnostics = {
        "feature_columns": feature_cols,
        "target_column": target_col,
        "method": method,
        "rows_used": len(clean_df),
        "pairs_computed": len(result),
        "top_n": top_n,
    }

    logger.info(f"Correlation analysis complete: method={method}, pairs={len(result)}")
    return result, diagnostics
