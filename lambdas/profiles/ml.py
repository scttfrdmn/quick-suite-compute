"""
ml.py — Machine learning classification/regression profile

random_forest_handler  — Random Forest classifier or regressor (classification-random-forest)

Called by runner/handler.py via entrypoint "ml.random_forest_handler".
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def random_forest_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Random Forest classifier (binary/multiclass) or regressor (continuous target).

    Task is auto-detected: if target has ≤20 unique values and is integer or
    object dtype, classify; otherwise regress. Can be overridden with task param.

    Returns the original DataFrame with prediction columns appended plus a
    diagnostics dict containing feature importances and evaluation metrics.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    target_col = parameters.get("target", "")
    feature_cols = parameters.get("features") or []
    n_estimators = int(parameters.get("n_estimators", 100))
    max_depth = parameters.get("max_depth")
    if max_depth is not None:
        max_depth = int(max_depth)
    class_weight = parameters.get("class_weight", "balanced") or "balanced"
    test_size = float(parameters.get("test_size", 0.2))
    task = parameters.get("task")  # "classify" | "regress" | None (auto)

    if df.empty:
        return df.assign(predicted_class=None, predicted_prob=None), {
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
        raise ValueError(f"Too few complete rows ({len(clean)}) for Random Forest")

    X = clean[feature_cols]
    y_raw = clean[target_col]

    # Auto-detect task
    n_unique = y_raw.nunique()
    if task is None:
        is_classification = (
            y_raw.dtype == object or
            (y_raw.dtype in (int, "int64", "int32") and n_unique <= 20)
        )
        task = "classify" if is_classification else "regress"

    result = df.copy()

    if task == "classify":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        classes = le.classes_.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight if class_weight != "none" else None,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X)
        preds = model.classes_[np.argmax(probs, axis=1)]
        pred_labels = le.inverse_transform(preds)

        result["predicted_class"] = np.nan
        result.loc[clean.index, "predicted_class"] = pred_labels

        # Per-class probability columns
        for i, cls in enumerate(classes):
            col_name = f"predicted_prob_{cls}"
            result[col_name] = np.nan
            result.loc[clean.index, col_name] = probs[:, i]

        # Evaluation on test set
        test_preds = model.predict(X_test)
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = float(accuracy_score(y_test, test_preds))
        report = classification_report(
            y_test, test_preds, target_names=[str(c) for c in classes],
            output_dict=True, zero_division=0
        )

        diagnostics: dict = {
            "task": "classify",
            "target_column": target_col,
            "feature_columns": feature_cols,
            "classes": classes,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "rows_used": len(clean),
            "rows_excluded": dropped,
            "accuracy": round(accuracy, 4),
            "classification_report": {
                k: {mk: round(mv, 4) if isinstance(mv, float) else mv
                    for mk, mv in v.items()}
                if isinstance(v, dict) else round(v, 4)
                for k, v in report.items()
            },
            "feature_importances": sorted(
                [{"feature": f, "importance": round(float(imp), 6)}
                 for f, imp in zip(feature_cols, model.feature_importances_)],
                key=lambda x: x["importance"],
                reverse=True,
            ),
        }

        logger.info(
            f"random-forest classify complete: classes={classes}, "
            f"accuracy={accuracy:.3f}, rows={len(clean)}"
        )

    else:  # regress
        from sklearn.metrics import mean_absolute_error, r2_score

        y = y_raw.astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X)
        result["predicted_value"] = np.nan
        result.loc[clean.index, "predicted_value"] = predictions

        test_preds = model.predict(X_test)
        r2 = float(r2_score(y_test, test_preds))
        mae = float(mean_absolute_error(y_test, test_preds))

        diagnostics = {
            "task": "regress",
            "target_column": target_col,
            "feature_columns": feature_cols,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "rows_used": len(clean),
            "rows_excluded": dropped,
            "r_squared": round(r2, 4),
            "mean_absolute_error": round(mae, 4),
            "feature_importances": sorted(
                [{"feature": f, "importance": round(float(imp), 6)}
                 for f, imp in zip(feature_cols, model.feature_importances_)],
                key=lambda x: x["importance"],
                reverse=True,
            ),
        }

        logger.info(
            f"random-forest regress complete: r²={r2:.3f}, mae={mae:.3f}, rows={len(clean)}"
        )

    return result, diagnostics
