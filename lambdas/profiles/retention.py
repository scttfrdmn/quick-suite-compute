"""
retention.py — Cohort Retention Analysis profile

Computes semester-by-semester retention matrices from enrollment records.
Outputs a long-form cohort × period retention table.

Called by runner/handler.py via the entrypoint "retention.cohort_handler".
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def cohort_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Compute cohort retention rates.

    Returns (result_df, diagnostics). result_df is a long-form table with
    cohort, period, enrolled_count, retention_rate, attrition_rate columns.
    """
    student_id_col = parameters.get("student_id_column", "")
    cohort_col = parameters.get("cohort_column", "")
    event_date_col = parameters.get("event_date_column", "")
    period_grain = parameters.get("period_grain", "semester")

    if df.empty:
        cols = ["cohort", "period", "enrolled_count", "retention_rate", "attrition_rate"]
        return pd.DataFrame(columns=cols), {
            "warning": "Input dataset is empty (extract step stub)"
        }

    for col, name in [(student_id_col, "student_id_column"),
                      (cohort_col, "cohort_column"),
                      (event_date_col, "event_date_column")]:
        if not col:
            raise ValueError(f"Parameter '{name}' is required")
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    work = df[[student_id_col, cohort_col, event_date_col]].copy()
    work.columns = ["student_id", "cohort", "event_date"]
    work["event_date"] = pd.to_datetime(work["event_date"])
    work = work.dropna(subset=["student_id", "cohort", "event_date"])

    # Assign period labels
    if period_grain == "semester":
        # Two periods per year: spring (Jan-Jul) and fall (Aug-Dec)
        work["period"] = work["event_date"].apply(
            lambda d: f"{d.year}-Spring" if d.month < 8 else f"{d.year}-Fall"
        )
    else:
        work["period"] = work["event_date"].dt.year.astype(str)

    # Cohort entry period = first period each student appears
    entry = work.groupby("student_id")["period"].min().rename("entry_period")
    work = work.join(entry, on="student_id")

    # For students whose cohort column disagrees with first observed period,
    # trust the explicit cohort column
    work["cohort_label"] = work["cohort"]

    # Count enrolled students per cohort × period
    enrolled = (
        work.groupby(["cohort_label", "period"])["student_id"]
        .nunique()
        .reset_index()
    )
    enrolled.columns = ["cohort", "period", "enrolled_count"]

    # Cohort initial size (first period)
    first_period = (
        enrolled.groupby("cohort")
        .apply(lambda g: g.loc[g["period"].idxmin(), "enrolled_count"])
        .rename("initial_count")
        .reset_index()
    )
    enrolled = enrolled.merge(first_period, on="cohort")
    enrolled["retention_rate"] = (
        enrolled["enrolled_count"] / enrolled["initial_count"]
    ).clip(0, 1)
    enrolled["attrition_rate"] = 1 - enrolled["retention_rate"]

    result = enrolled[
        ["cohort", "period", "enrolled_count", "retention_rate", "attrition_rate"]
    ].sort_values(["cohort", "period"]).reset_index(drop=True)

    cohort_count = result["cohort"].nunique()
    diagnostics = {
        "student_id_column": student_id_col,
        "cohort_column": cohort_col,
        "event_date_column": event_date_col,
        "period_grain": period_grain,
        "cohort_count": cohort_count,
        "period_count": result["period"].nunique(),
        "total_students": int(work["student_id"].nunique()),
    }

    logger.info(f"Cohort retention complete: {cohort_count} cohorts, grain={period_grain}")
    return result, diagnostics
