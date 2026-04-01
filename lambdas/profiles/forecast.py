"""
forecast.py — Time Series Forecast (Prophet) profile

Projects a time series forward using Facebook Prophet.
Output DataFrame has Prophet's standard columns: ds, yhat, yhat_lower,
yhat_upper, trend, yearly (seasonal component).

Called by runner/handler.py via the entrypoint "forecast.prophet_handler".
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def prophet_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Fit Prophet and forecast forward.

    Returns (result_df, diagnostics). result_df contains both historical
    fitted values (if include_history=True) and future forecast rows.
    """
    from prophet import Prophet

    date_col = parameters.get("date_column", "")
    value_col = parameters.get("value_column", "")
    forecast_periods = int(parameters.get("forecast_periods", 12))
    if not (1 <= forecast_periods <= 1_000):
        raise ValueError(
            f"forecast_periods must be between 1 and 1000 (got {forecast_periods})"
        )
    seasonality_mode = parameters.get("seasonality_mode", "additive")
    include_history = bool(parameters.get("include_history", True))

    if df.empty:
        cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "yearly"]
        return pd.DataFrame(columns=cols), {
            "warning": "Input dataset is empty (extract step stub)"
        }

    if not date_col:
        raise ValueError("Parameter 'date_column' is required")
    if not value_col:
        raise ValueError("Parameter 'value_column' is required")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataset")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataset")

    # Prepare Prophet input
    prophet_df = df[[date_col, value_col]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df = prophet_df.dropna()
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    if len(prophet_df) < 10:
        raise ValueError(
            f"Prophet requires at least 10 observations; got {len(prophet_df)}"
        )

    # Infer frequency
    freq = _infer_freq(prophet_df["ds"])

    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality=False,
    )
    model.fit(prophet_df)

    # Make future dataframe
    future = model.make_future_dataframe(periods=forecast_periods, freq=freq)
    forecast = model.predict(future)

    if not include_history:
        forecast = forecast[forecast["ds"] > prophet_df["ds"].max()].copy()

    # Select output columns
    output_cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
    if "yearly" in forecast.columns:
        output_cols.append("yearly")
    result = forecast[output_cols].reset_index(drop=True)

    # Mark historical vs forecast rows
    cutoff = prophet_df["ds"].max()
    result["is_forecast"] = result["ds"] > cutoff

    diagnostics = {
        "date_column": date_col,
        "value_column": value_col,
        "forecast_periods": forecast_periods,
        "seasonality_mode": seasonality_mode,
        "inferred_frequency": freq,
        "training_rows": len(prophet_df),
        "forecast_start": result[result["is_forecast"]]["ds"].min().isoformat()
        if result["is_forecast"].any() else None,
        "forecast_end": result["ds"].max().isoformat(),
    }

    logger.info(
        f"Prophet forecast complete: periods={forecast_periods}, "
        f"freq={freq}, training_rows={len(prophet_df)}"
    )
    return result, diagnostics


def _infer_freq(dates: pd.Series) -> str:
    """Infer the dominant time series frequency."""
    if len(dates) < 2:
        return "MS"  # default to monthly
    deltas = dates.diff().dropna()
    median_days = deltas.dt.days.median()
    if pd.isna(median_days) or median_days <= 0:
        raise ValueError(
            "Cannot infer forecast frequency: all timestamps are identical or "
            "have zero time delta. Provide data with distinct dates."
        )
    if median_days <= 1.5:
        return "D"
    elif median_days <= 8:
        return "W"
    elif median_days <= 35:
        return "MS"
    elif median_days <= 100:
        return "QS"
    else:
        return "YS"
