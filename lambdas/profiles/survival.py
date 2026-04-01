"""
survival.py — Survival Analysis (Kaplan-Meier) profile

Performs Kaplan-Meier survival analysis using the lifelines library.
Outputs survival curves per group plus log-rank test and median survival.

Called by runner/handler.py via the entrypoint "survival.kaplan_meier_handler".
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def kaplan_meier_handler(
    df: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, dict]:
    """
    Compute Kaplan-Meier survival curves, stratified by a group column.

    Returns (result_df, diagnostics). result_df is a long-form survival
    table with: group, timeline, survival_probability, survival_lower,
    survival_upper, at_risk, events.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    duration_col = parameters.get("duration_column", "")
    event_col = parameters.get("event_column", "")
    group_col = parameters.get("group_column", "")
    confidence_level = float(parameters.get("confidence_level", 0.95))
    if not (0 < confidence_level < 1):
        raise ValueError(
            f"confidence_level must be between 0 and 1 exclusive (got {confidence_level})"
        )

    if df.empty:
        cols = [
            "group", "timeline", "survival_probability",
            "survival_lower", "survival_upper", "at_risk", "events"
        ]
        return pd.DataFrame(columns=cols), {
            "warning": "Input dataset is empty (extract step stub)"
        }

    for col, name in [(duration_col, "duration_column"),
                      (event_col, "event_column"),
                      (group_col, "group_column")]:
        if not col:
            raise ValueError(f"Parameter '{name}' is required")
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    work = df[[duration_col, event_col, group_col]].copy()
    work.columns = ["duration", "event", "group"]
    work["duration"] = pd.to_numeric(work["duration"], errors="coerce")
    work["event"] = pd.to_numeric(work["event"], errors="coerce").fillna(0).astype(int)
    work = work.dropna(subset=["duration"]).copy()
    work = work[work["duration"] >= 0]

    if len(work) < 10:
        raise ValueError(f"Too few valid rows ({len(work)}) for survival analysis")

    alpha = 1 - confidence_level
    groups = sorted(work["group"].dropna().unique())

    result_rows = []
    median_survival = {}
    skipped_groups = []

    for group in groups:
        g_data = work[work["group"] == group]
        if len(g_data) < 5:
            logger.warning(f"Skipping group '{group}': only {len(g_data)} members (minimum 5)")
            skipped_groups.append(str(group))
            continue

        kmf = KaplanMeierFitter(alpha=alpha)
        kmf.fit(g_data["duration"], event_observed=g_data["event"], label=str(group))

        timeline = kmf.survival_function_.index.values
        sf = kmf.survival_function_[str(group)].values
        ci = kmf.confidence_interval_

        lower_col = ci.columns[0]
        upper_col = ci.columns[1]
        lower = ci[lower_col].values
        upper = ci[upper_col].values

        # Event table
        event_table = kmf.event_table
        at_risk_vals = event_table["at_risk"].reindex(timeline).ffill().fillna(0)
        event_vals = event_table["observed"].reindex(timeline).fillna(0)

        for i, t in enumerate(timeline):
            result_rows.append({
                "group": str(group),
                "timeline": float(t),
                "survival_probability": float(sf[i]),
                "survival_lower": float(lower[i]),
                "survival_upper": float(upper[i]),
                "at_risk": int(at_risk_vals.iloc[i]) if i < len(at_risk_vals) else 0,
                "events": int(event_vals.iloc[i]) if i < len(event_vals) else 0,
            })

        median_survival[str(group)] = float(kmf.median_survival_time_)

    result = pd.DataFrame(result_rows)

    # Log-rank test (if 2+ groups)
    logrank_result = {}
    if len(groups) >= 2:
        try:
            lr = multivariate_logrank_test(
                work["duration"], work["group"], event_observed=work["event"]
            )
            logrank_result = {
                "test_statistic": float(lr.test_statistic),
                "p_value": float(lr.p_value),
                "significant": bool(lr.p_value < alpha),
            }
        except Exception as exc:
            logger.warning(f"Log-rank test failed: {exc}")

    diagnostics = {
        "duration_column": duration_col,
        "event_column": event_col,
        "group_column": group_col,
        "confidence_level": confidence_level,
        "groups": [str(g) for g in groups],
        "groups_skipped_too_small": skipped_groups,
        "total_rows": len(work),
        "median_survival_by_group": median_survival,
        "log_rank_test": logrank_result,
    }

    logger.info(
        f"Kaplan-Meier complete: groups={len(groups)}, rows={len(work)}, "
        f"p_value={logrank_result.get('p_value', 'N/A')}"
    )
    return result, diagnostics
