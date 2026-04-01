"""
regression.py — GLM Regression profile

Fits a linear or logistic regression using statsmodels GLM.
Appends predicted_value and prediction_probability columns to the input.
Returns diagnostics including coefficients, p-values, and model fit stats.

Called by runner/handler.py via the entrypoint "regression.glm_handler".
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def glm_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Fit a GLM (linear or logistic regression).

    Returns (result_df, diagnostics). result_df has predicted_value and
    prediction_probability appended. Coefficients and diagnostics are in
    the diagnostics dict.
    """
    import statsmodels.api as sm

    target_col = parameters.get("target", "")
    feature_cols = parameters.get("features") or []
    model_type = parameters.get("model_type", "logistic")
    if model_type not in ("linear", "logistic"):
        raise ValueError(
            f"model_type must be 'linear' or 'logistic' (got '{model_type}')"
        )
    include_interactions = bool(parameters.get("include_interactions", False))

    if df.empty:
        return df.assign(predicted_value=None, prediction_probability=None), {
            "warning": "Input dataset is empty (extract step stub)"
        }

    if not target_col:
        raise ValueError("Parameter 'target' is required")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    if not feature_cols:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != target_col]
    if not feature_cols:
        raise ValueError("No numeric feature columns available")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")

    # Drop rows with NaN in target or features
    cols = [target_col] + feature_cols
    clean_df = df[cols].dropna()
    dropped = len(df) - len(clean_df)

    if len(clean_df) < 10:
        raise ValueError(f"Too few complete rows ({len(clean_df)}) for regression")

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    # Optional interaction terms
    if include_interactions and len(feature_cols) > 5:
        logger.warning(
            f"Skipping interaction terms: {len(feature_cols)} features exceeds 5-feature limit"
        )
    if include_interactions and len(feature_cols) <= 5:
        from itertools import combinations
        for a, b in combinations(feature_cols, 2):
            col_name = f"{a}_x_{b}"
            X = X.copy()
            X[col_name] = clean_df[a] * clean_df[b]

    X_const = sm.add_constant(X, has_constant="add")

    if model_type == "logistic":
        family = sm.families.Binomial()
    else:
        family = sm.families.Gaussian()

    model = sm.GLM(y, X_const, family=family)
    result_model = model.fit()

    # Predictions
    predictions = result_model.predict(X_const)

    result = df.copy()
    result["predicted_value"] = np.nan
    result["prediction_probability"] = np.nan
    clean_idx = clean_df.index
    result.loc[clean_idx, "predicted_value"] = predictions.values
    if model_type == "logistic":
        result.loc[clean_idx, "prediction_probability"] = predictions.values
    else:
        result.loc[clean_idx, "prediction_probability"] = np.nan

    # Coefficients table
    coef_df = pd.DataFrame({
        "variable": result_model.params.index,
        "coefficient": result_model.params.values,
        "std_error": result_model.bse.values,
        "z_score": result_model.tvalues.values,
        "p_value": result_model.pvalues.values,
    })

    # Diagnostics
    diagnostics: dict = {
        "model_type": model_type,
        "target_column": target_col,
        "feature_columns": feature_cols,
        "rows_used": len(clean_df),
        "rows_with_nan_excluded": dropped,
        "interactions_applied": include_interactions and len(feature_cols) <= 5,
        "interactions_skipped": include_interactions and len(feature_cols) > 5,
        "aic": float(result_model.aic),
        "bic": float(result_model.bic),
        "coefficients": coef_df.to_dict(orient="records"),
    }

    if model_type == "linear":
        ss_res = float(np.sum((y - predictions) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        diagnostics["r_squared"] = 1 - ss_res / ss_tot if ss_tot > 0 else None
        diagnostics["pseudo_r_squared"] = None
    else:
        diagnostics["pseudo_r_squared"] = float(result_model.pseudo_rsquared())
        # ROC-AUC if binary
        try:
            from sklearn.metrics import roc_auc_score
            diagnostics["auc"] = float(roc_auc_score(y, predictions))
        except Exception:
            pass

    logger.info(f"GLM complete: model_type={model_type}, rows={len(clean_df)}")
    return result, diagnostics
