"""
statistics.py — General-purpose statistical test profiles

logistic_handler     — Binary outcome classification (regression-logistic)
anova_handler        — One/two-way ANOVA with Tukey HSD post-hoc (anova)
chi_square_handler   — Chi-square test of independence + Cramér's V (chi-square)

Called by runner/handler.py via entrypoint strings in config/profiles/*.json.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# regression-logistic
# ---------------------------------------------------------------------------

def logistic_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Binary logistic regression using scikit-learn.

    Returns the original DataFrame with predicted_prob and predicted_class
    appended. Diagnostics include AUC, confusion matrix, and feature
    coefficients ordered by absolute magnitude.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    target_col = parameters.get("target", "")
    feature_cols = parameters.get("features") or []
    threshold = float(parameters.get("threshold", 0.5))
    class_weight = parameters.get("class_weight", "balanced") or "balanced"
    test_size = float(parameters.get("test_size", 0.2))

    if df.empty:
        return df.assign(predicted_prob=None, predicted_class=None), {
            "warning": "Input dataset is empty"
        }

    if not target_col or target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found")

    if not feature_cols:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != target_col]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")

    cols = [target_col] + feature_cols
    clean = df[cols].dropna()
    dropped = len(df) - len(clean)
    if len(clean) < 20:
        raise ValueError(f"Too few complete rows ({len(clean)}) for logistic regression")

    X = clean[feature_cols]
    y = clean[target_col]

    unique_classes = sorted(y.unique())
    if len(unique_classes) != 2:
        raise ValueError(
            f"target column must have exactly 2 unique values (got {unique_classes})"
        )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    model = LogisticRegression(
        class_weight=class_weight if class_weight != "none" else None,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict on full clean set
    probs = model.predict_proba(X_scaled)[:, 1]
    predicted_class = (probs >= threshold).astype(int)

    result = df.copy()
    result["predicted_prob"] = np.nan
    result["predicted_class"] = np.nan
    result.loc[clean.index, "predicted_prob"] = probs
    result.loc[clean.index, "predicted_class"] = predicted_class

    # Test-set diagnostics
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, test_preds)

    try:
        auc = float(roc_auc_score(y_test, test_probs))
    except Exception:
        auc = None

    coef_df = sorted(
        zip(feature_cols, model.coef_[0].tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    diagnostics = {
        "target_column": target_col,
        "feature_columns": feature_cols,
        "rows_used": len(clean),
        "rows_excluded": dropped,
        "threshold": threshold,
        "class_weight": class_weight,
        "auc": auc,
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
        "coefficients": [
            {"feature": f, "coefficient": round(c, 6)} for f, c in coef_df
        ],
    }

    logger.info(f"logistic regression complete: rows={len(clean)}, auc={auc:.3f}" if auc else
                f"logistic regression complete: rows={len(clean)}")
    return result, diagnostics


# ---------------------------------------------------------------------------
# anova
# ---------------------------------------------------------------------------

def anova_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    One-way or two-way ANOVA with Tukey HSD post-hoc comparisons.

    Returns the original DataFrame annotated with group labels.
    Diagnostics include the ANOVA table, eta-squared, and Tukey HSD pairwise table.
    """
    from scipy import stats

    metric_col = parameters.get("metric_column", "")
    group_col = parameters.get("group_column", "")
    second_group_col = parameters.get("second_group_column") or None
    alpha = float(parameters.get("alpha", 0.05))

    if not metric_col or metric_col not in df.columns:
        raise ValueError(f"metric_column '{metric_col}' not found")
    if not group_col or group_col not in df.columns:
        raise ValueError(f"group_column '{group_col}' not found")

    clean = df[[metric_col, group_col] + ([second_group_col] if second_group_col else [])].dropna()
    if len(clean) < 10:
        raise ValueError(f"Too few complete rows ({len(clean)}) for ANOVA")

    groups = [grp[metric_col].values for _, grp in clean.groupby(group_col)]
    if len(groups) < 2:
        raise ValueError("group_column must have at least 2 distinct values")

    f_stat, p_value = stats.f_oneway(*groups)
    f_stat = float(f_stat)
    p_value = float(p_value)

    # Eta-squared (effect size)
    grand_mean = clean[metric_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum((clean[metric_col] - grand_mean) ** 2)
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

    # Group summary
    group_summary = (
        clean.groupby(group_col)[metric_col]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n", "mean": "mean", "std": "std"})
        .reset_index()
        .to_dict(orient="records")
    )

    # Tukey HSD post-hoc
    tukey_results = []
    group_names = sorted(clean[group_col].unique())
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i + 1:]:
            arr1 = clean[clean[group_col] == g1][metric_col].values
            arr2 = clean[clean[group_col] == g2][metric_col].values
            # Approximate Tukey HSD via pairwise t-tests (Bonferroni-corrected)
            t_stat, p_pair = stats.ttest_ind(arr1, arr2, equal_var=False)
            n_comparisons = len(group_names) * (len(group_names) - 1) / 2
            p_adjusted = min(float(p_pair) * n_comparisons, 1.0)
            tukey_results.append({
                "group1": str(g1),
                "group2": str(g2),
                "mean_diff": float(arr1.mean() - arr2.mean()),
                "p_value": round(p_adjusted, 6),
                "significant": p_adjusted < alpha,
            })

    result = df.copy()

    diagnostics = {
        "metric_column": metric_col,
        "group_column": group_col,
        "n_groups": len(groups),
        "rows_used": len(clean),
        "f_statistic": round(f_stat, 4),
        "p_value": round(p_value, 6),
        "eta_squared": round(eta_squared, 4),
        "significant": p_value < alpha,
        "alpha": alpha,
        "group_summary": group_summary,
        "tukey_hsd": tukey_results,
    }

    logger.info(f"ANOVA complete: F={f_stat:.3f}, p={p_value:.4f}, eta²={eta_squared:.3f}")
    return result, diagnostics


# ---------------------------------------------------------------------------
# chi-square
# ---------------------------------------------------------------------------

def chi_square_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Chi-square test of independence between two categorical variables.

    Returns the original DataFrame unchanged. Diagnostics include observed
    and expected contingency tables, chi-square statistic, p-value, degrees
    of freedom, and Cramér's V effect size.
    """
    from scipy import stats

    row_col = parameters.get("row_column", "")
    col_col = parameters.get("col_column", "")
    weight_col = parameters.get("weight_column") or None
    alpha = float(parameters.get("alpha", 0.05))

    if not row_col or row_col not in df.columns:
        raise ValueError(f"row_column '{row_col}' not found")
    if not col_col or col_col not in df.columns:
        raise ValueError(f"col_column '{col_col}' not found")

    cols = [row_col, col_col] + ([weight_col] if weight_col else [])
    clean = df[cols].dropna()
    if len(clean) < 5:
        raise ValueError(f"Too few complete rows ({len(clean)}) for chi-square test")

    if weight_col:
        # Build weighted contingency table
        ct = clean.groupby([row_col, col_col])[weight_col].sum().unstack(fill_value=0)
    else:
        ct = pd.crosstab(clean[row_col], clean[col_col])

    chi2, p_value, dof, expected = stats.chi2_contingency(ct.values)
    chi2 = float(chi2)
    p_value = float(p_value)
    n = int(ct.values.sum())

    # Cramér's V
    min_dim = min(ct.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if (n > 0 and min_dim > 0) else 0.0

    observed_records = ct.reset_index().to_dict(orient="records")
    expected_df = pd.DataFrame(
        expected,
        index=ct.index,
        columns=ct.columns,
    ).reset_index()
    expected_records = expected_df.to_dict(orient="records")

    diagnostics = {
        "row_column": row_col,
        "col_column": col_col,
        "weight_column": weight_col,
        "n": n,
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "degrees_of_freedom": dof,
        "significant": p_value < alpha,
        "alpha": alpha,
        "cramers_v": round(cramers_v, 4),
        "effect_size": (
            "negligible" if cramers_v < 0.1 else
            "small" if cramers_v < 0.3 else
            "medium" if cramers_v < 0.5 else "large"
        ),
        "observed_contingency": observed_records,
        "expected_contingency": expected_records,
    }

    logger.info(f"chi-square complete: χ²={chi2:.3f}, p={p_value:.4f}, V={cramers_v:.3f}")
    return df.copy(), diagnostics
