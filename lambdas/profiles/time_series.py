"""
time_series.py — Time series analysis profiles

change_detection_handler  — PELT structural break detection (change-detection)
decompose_handler          — STL trend/seasonal/residual decomposition (seasonality-decompose)

Called by runner/handler.py via entrypoints in config/profiles/*.json.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# change-detection
# ---------------------------------------------------------------------------

def change_detection_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Detect structural breaks (change points) in a time series using the
    PELT (Pruned Exact Linear Time) algorithm via the ruptures library.

    Handles an optional group_column to run detection independently per group.
    Returns the original DataFrame with change_point (bool) and segment_id (int)
    columns appended.
    """
    import ruptures as rpt

    date_col = parameters.get("date_column", "")
    value_col = parameters.get("value_column", "")
    group_col = parameters.get("group_column")
    min_size = int(parameters.get("min_segment_length", 3))
    model = parameters.get("model", "rbf")
    penalty = parameters.get("penalty")
    if penalty is not None:
        penalty = float(penalty)

    if not date_col or date_col not in df.columns:
        raise ValueError(f"date_column '{date_col}' not found")
    if not value_col or value_col not in df.columns:
        raise ValueError(f"value_column '{value_col}' not found")
    if group_col and group_col not in df.columns:
        raise ValueError(f"group_column '{group_col}' not found")
    if model not in ("rbf", "l2", "l1"):
        raise ValueError(f"model must be 'rbf', 'l2', or 'l1' (got '{model}')")

    if df.empty:
        result = df.copy()
        result["change_point"] = pd.Series(dtype=bool)
        result["segment_id"] = pd.Series(dtype=int)
        return result, {"warning": "Input dataset is empty"}

    result = df.copy()
    result["change_point"] = False
    result["segment_id"] = 0

    def _detect(sub: pd.DataFrame) -> tuple[list[int], int]:
        """Run PELT on a sorted sub-series. Returns (breakpoint_positions, n_breakpoints)."""
        sorted_sub = sub.sort_values(date_col)
        signal = sorted_sub[value_col].fillna(method="ffill").fillna(method="bfill").values
        if len(signal) < 2 * min_size:
            return [], 0
        signal_2d = signal.reshape(-1, 1)
        # Auto-select penalty via BIC approximation if not provided
        pen = penalty if penalty is not None else np.log(len(signal)) * signal.var()
        pen = max(pen, 1e-6)
        try:
            algo = rpt.Pelt(model=model, min_size=min_size).fit(signal_2d)
            breakpoints = algo.predict(pen=pen)
            # breakpoints are 1-indexed end positions of segments; last is always len(signal)
            breakpoints = [bp for bp in breakpoints if bp < len(signal)]
        except Exception as exc:
            logger.warning(f"ruptures PELT failed: {exc}")
            return [], 0
        return breakpoints, len(breakpoints)

    all_change_points: list[dict] = []

    if group_col:
        groups = result.groupby(group_col)
        for group_name, group_df in groups:
            breakpoints, n_bp = _detect(group_df)
            if breakpoints:
                sorted_idx = group_df.sort_values(date_col).index.tolist()
                for bp in breakpoints:
                    if bp < len(sorted_idx):
                        result.loc[sorted_idx[bp], "change_point"] = True
                # Assign segment IDs within this group
                segment = 0
                for idx in sorted_idx:
                    result.loc[idx, "segment_id"] = segment
                    if result.loc[idx, "change_point"]:
                        segment += 1
                for bp in breakpoints:
                    date_val = group_df.sort_values(date_col)[date_col].iloc[bp] if bp < len(group_df) else None
                    all_change_points.append({
                        "group": str(group_name),
                        "position": bp,
                        "date": str(date_val) if date_val is not None else None,
                    })
    else:
        breakpoints, n_bp = _detect(result)
        sorted_idx = result.sort_values(date_col).index.tolist()
        for bp in breakpoints:
            if bp < len(sorted_idx):
                result.loc[sorted_idx[bp], "change_point"] = True
        segment = 0
        for idx in sorted_idx:
            result.loc[idx, "segment_id"] = segment
            if result.loc[idx, "change_point"]:
                segment += 1
        sorted_dates = result.sort_values(date_col)[date_col]
        for bp in breakpoints:
            date_val = sorted_dates.iloc[bp] if bp < len(sorted_dates) else None
            all_change_points.append({
                "position": bp,
                "date": str(date_val) if date_val is not None else None,
            })

    n_total_breaks = int(result["change_point"].sum())
    diagnostics = {
        "date_column": date_col,
        "value_column": value_col,
        "group_column": group_col,
        "model": model,
        "min_segment_length": min_size,
        "penalty": penalty,
        "n_change_points": n_total_breaks,
        "change_points": all_change_points,
    }

    logger.info(
        f"change-detection complete: n_change_points={n_total_breaks}, "
        f"model={model}, groups={group_col or 'none'}"
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# seasonality-decompose
# ---------------------------------------------------------------------------

def decompose_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Decompose a time series into trend, seasonal, and residual components.

    Uses STL (Seasonal and Trend decomposition using Loess) for additive
    decomposition, which is robust to outliers. For multiplicative decomposition
    uses statsmodels classical_decompose.

    Handles an optional group_column to decompose each group independently.
    Returns the original DataFrame with trend, seasonal, residual columns appended.
    """
    from statsmodels.tsa.seasonal import STL, seasonal_decompose

    date_col = parameters.get("date_column", "")
    value_col = parameters.get("value_column", "")
    group_col = parameters.get("group_column")
    period = parameters.get("period")
    if period is not None:
        period = int(period)
    model = parameters.get("model", "additive")

    if not date_col or date_col not in df.columns:
        raise ValueError(f"date_column '{date_col}' not found")
    if not value_col or value_col not in df.columns:
        raise ValueError(f"value_column '{value_col}' not found")
    if group_col and group_col not in df.columns:
        raise ValueError(f"group_column '{group_col}' not found")
    if model not in ("additive", "multiplicative"):
        raise ValueError("model must be 'additive' or 'multiplicative'")

    if df.empty:
        result = df.copy()
        for col in ("trend", "seasonal", "residual"):
            result[col] = pd.Series(dtype=float)
        return result, {"warning": "Input dataset is empty"}

    result = df.copy()
    for col in ("trend", "seasonal", "residual"):
        result[col] = np.nan

    def _infer_period(dates: pd.Series) -> int:
        """Estimate seasonal period from median inter-observation gap."""
        sorted_dates = pd.to_datetime(dates).sort_values()
        deltas = sorted_dates.diff().dropna()
        if deltas.empty:
            return 12
        median_days = deltas.dt.total_seconds().median() / 86400
        if median_days <= 1.5:
            return 7      # daily → weekly seasonality
        elif median_days <= 8:
            return 52     # weekly → annual
        elif median_days <= 35:
            return 12     # monthly → annual
        elif median_days <= 100:
            return 4      # quarterly → annual
        else:
            return 2      # yearly → no meaningful seasonality

    def _decompose_series(sub: pd.DataFrame) -> pd.DataFrame | None:
        """Decompose one series; returns a df with trend/seasonal/residual indexed like sub."""
        sorted_sub = sub.sort_values(date_col).copy()
        values = sorted_sub[value_col].astype(float)
        n = len(values)

        p = period if period is not None else _infer_period(sorted_sub[date_col])

        if n < 2 * p:
            logger.warning(
                f"Too few observations ({n}) for period {p}; skipping decomposition"
            )
            return None

        values_filled = values.interpolate(limit_direction="both").fillna(method="ffill").fillna(method="bfill")

        try:
            if model == "additive":
                decomp = STL(values_filled, period=p, robust=True).fit()
            else:
                # statsmodels classical decomposition for multiplicative
                decomp = seasonal_decompose(values_filled, model="multiplicative", period=p)
        except Exception as exc:
            logger.warning(f"Decomposition failed: {exc}")
            return None

        out = pd.DataFrame({
            "trend": decomp.trend,
            "seasonal": decomp.seasonal,
            "residual": decomp.resid,
        }, index=sorted_sub.index)
        return out

    inferred_periods: list[int] = []

    if group_col:
        for group_name, group_df in result.groupby(group_col):
            decomp_df = _decompose_series(group_df)
            if decomp_df is not None:
                for col in ("trend", "seasonal", "residual"):
                    result.loc[decomp_df.index, col] = decomp_df[col]
    else:
        decomp_df = _decompose_series(result)
        if decomp_df is not None:
            for col in ("trend", "seasonal", "residual"):
                result.loc[decomp_df.index, col] = decomp_df[col]
            inferred_period = period if period is not None else _infer_period(result[date_col])
            inferred_periods.append(inferred_period)

    n_decomposed = int((~result["trend"].isna()).sum())
    diagnostics = {
        "date_column": date_col,
        "value_column": value_col,
        "group_column": group_col,
        "model": model,
        "period": period,
        "inferred_period": inferred_periods[0] if inferred_periods else None,
        "n_rows_decomposed": n_decomposed,
        "n_rows_skipped": len(df) - n_decomposed,
    }

    logger.info(
        f"decompose complete: model={model}, period={period or 'inferred'}, "
        f"n_decomposed={n_decomposed}"
    )
    return result, diagnostics
