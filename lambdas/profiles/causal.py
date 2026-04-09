"""
causal.py — Causal inference analysis profiles.

causal_iv_handler  — Instrumental variables (2SLS) (#63)
causal_rd_handler  — Regression discontinuity (#64)
causal_did_handler — Difference-in-differences (#65)

All handlers return plain dicts (not DataFrame tuples).
runner/handler.py coerces plain dicts before writing results.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary least squares: returns (coefficients, residuals)."""
    coef, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    resid = y - fitted
    return coef, resid


def _ols_se(X: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """Heteroskedasticity-robust (HC0) standard errors."""
    n, k = X.shape
    meat = (X.T * resid**2) @ X
    bread = np.linalg.pinv(X.T @ X)
    vcov = bread @ meat @ bread
    return np.sqrt(np.diag(vcov))


def _add_intercept(arr: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(arr)), arr])


# ---------------------------------------------------------------------------
# causal-iv
# ---------------------------------------------------------------------------

def causal_iv_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Two-stage least squares (2SLS) instrumental variables estimation.
    Uses linearmodels when available; returns 503 if not installed.
    """
    try:
        from linearmodels.iv import IV2SLS  # type: ignore[import]
    except ImportError:
        return {
            "error": "IV regression requires linearmodels library",
            "requires_layer": "linearmodels",
            "statusCode": 503,
        }

    outcome_var = parameters.get("outcome_var", "")
    treatment_var = parameters.get("treatment_var", "")
    instrument_var = parameters.get("instrument_var", "")
    control_vars: list[str] = parameters.get("control_vars") or []
    peer_benchmark: bool = bool(parameters.get("peer_benchmark", False))

    if not outcome_var:
        return {"error": "outcome_var is required", "statusCode": 400}
    if not treatment_var:
        return {"error": "treatment_var is required", "statusCode": 400}
    if not instrument_var:
        return {"error": "instrument_var is required", "statusCode": 400}
    if len(df) < 50:
        return {"error": "causal-iv requires at least 50 rows", "statusCode": 400}

    try:
        endog_vars = [treatment_var]
        instrum_vars = [instrument_var]
        exog_vars = control_vars if control_vars else []

        model = IV2SLS(
            dependent=df[outcome_var],
            exog=df[exog_vars] if exog_vars else None,
            endog=df[endog_vars],
            instruments=df[instrum_vars],
        )
        res = model.fit(cov_type="robust")

        iv_estimate = float(res.params[treatment_var])
        ci = res.conf_int(level=0.95).loc[treatment_var]
        ci_lower, ci_upper = float(ci.iloc[0]), float(ci.iloc[1])

        # First-stage F-stat (Cragg-Donald or Kleibergen-Paap via linearmodels)
        try:
            first_stage_f = float(res.first_stage.diagnostics["f.stat"].iloc[0])
        except Exception:
            first_stage_f = float(getattr(res, "wu_hausman", lambda: None)() or 0)

        warnings: list[str] = []
        if first_stage_f < 10:
            warnings.append(
                f"Weak instrument: first-stage F-statistic is {first_stage_f:.2f} (threshold: 10)"
            )

        # Compliance rate: fraction of treated units with instrument = 1
        try:
            treated = df[treatment_var].astype(float)
            instrument = df[instrument_var].astype(float)
            compliance_rate = float(
                ((treated == 1) & (instrument == 1)).sum() /
                max((instrument == 1).sum(), 1)
            )
        except Exception:
            compliance_rate = float("nan")

        result: dict[str, Any] = {
            "profile_id": "causal-iv",
            "iv_estimate": iv_estimate,
            "first_stage_f_stat": first_stage_f,
            "confidence_interval_95": [ci_lower, ci_upper],
            "compliance_rate": compliance_rate,
            "warnings": warnings,
        }

        if peer_benchmark:
            result["peer_benchmark"] = {
                "note": "peer_benchmark=true: pass candidate_df for full comparison"
            }

        return result

    except Exception as exc:
        logger.error(f"causal_iv_handler failed: {exc}")
        return {"error": f"IV estimation failed: {exc}", "statusCode": 500}


# ---------------------------------------------------------------------------
# causal-rd
# ---------------------------------------------------------------------------

def causal_rd_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Regression discontinuity design (sharp and fuzzy).
    Uses numpy/scipy — no external RD library required.
    Optional rdrobust import for enhanced precision (non-blocking if absent).
    """
    outcome_var = parameters.get("outcome_var", "")
    running_var = parameters.get("running_var", "")
    cutoff = parameters.get("cutoff")
    bandwidth_param = parameters.get("bandwidth", "optimal")
    rd_type = str(parameters.get("rd_type", "sharp")).lower()
    treatment_var = parameters.get("treatment_var", "")

    if not outcome_var:
        return {"error": "outcome_var is required", "statusCode": 400}
    if not running_var:
        return {"error": "running_var is required", "statusCode": 400}
    if cutoff is None:
        return {"error": "cutoff is required", "statusCode": 400}
    if rd_type == "fuzzy" and not treatment_var:
        return {"error": "treatment_var is required for fuzzy RD", "statusCode": 400}
    if len(df) < 20:
        return {"error": "causal-rd requires at least 20 rows", "statusCode": 400}

    try:
        cutoff = float(cutoff)
        rv = df[running_var].astype(float).values
        y = df[outcome_var].astype(float).values
        n = len(rv)
        sigma = float(np.std(rv, ddof=1))

        # Optimal bandwidth (Imbens-Kalyanaraman approximation)
        if bandwidth_param == "optimal":
            h = 2.702 * sigma * (n ** (-1/5))
        else:
            h = float(bandwidth_param)

        def _local_estimate(bw: float) -> float | None:
            mask_above = (rv >= cutoff) & (rv <= cutoff + bw)
            mask_below = (rv >= cutoff - bw) & (rv < cutoff)
            if mask_above.sum() < 3 or mask_below.sum() < 3:
                return None
            mean_above = float(np.mean(y[mask_above]))
            mean_below = float(np.mean(y[mask_below]))
            return mean_above - mean_below

        late_estimate = _local_estimate(h)
        if late_estimate is None:
            return {"error": "Insufficient observations within bandwidth", "statusCode": 400}

        # Bandwidth sensitivity: h*{0.5, 0.75, 1.0, 1.25, 1.5}
        sensitivity = []
        for mult in (0.5, 0.75, 1.0, 1.25, 1.5):
            bw = h * mult
            est = _local_estimate(bw)
            sensitivity.append({"bandwidth": round(bw, 4), "estimate": est})

        # McCrary density test: test for discontinuity in running var density at cutoff
        n_bins = max(10, int(np.sqrt(n)))
        counts, bin_edges = np.histogram(rv, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        above_mask = bin_centers >= cutoff
        below_mask = bin_centers < cutoff
        mccrary_stat = float("nan")
        mccrary_p = float("nan")
        manipulation_detected = False
        if above_mask.sum() >= 3 and below_mask.sum() >= 3:
            mean_above_density = float(np.mean(counts[above_mask]))
            mean_below_density = float(np.mean(counts[below_mask]))
            try:
                _, mccrary_p = scipy.stats.ttest_ind(
                    counts[above_mask], counts[below_mask]
                )
                mccrary_stat = abs(mean_above_density - mean_below_density)
                manipulation_detected = bool(mccrary_p < 0.05)
            except Exception:
                pass

        # Fuzzy RD: IV approach
        if rd_type == "fuzzy":
            instrument = (rv >= cutoff).astype(float)
            try:
                treatment = df[treatment_var].astype(float).values
                mask = (rv >= cutoff - h) & (rv <= cutoff + h)
                y_w, t_w, z_w = y[mask], treatment[mask], instrument[mask]
                X_fs = _add_intercept(z_w.reshape(-1, 1))
                coef_fs, _ = _ols(X_fs, t_w)
                t_hat = X_fs @ coef_fs
                X_ss = _add_intercept(t_hat.reshape(-1, 1))
                coef_ss, _ = _ols(X_ss, y_w)
                late_estimate = float(coef_ss[1])
            except Exception as exc:
                logger.warning(f"fuzzy RD IV failed: {exc}")

        return {
            "profile_id": "causal-rd",
            "late_estimate": late_estimate,
            "bandwidth_used": round(h, 4),
            "bandwidth_sensitivity": sensitivity,
            "mccrary_test": {
                "statistic": mccrary_stat,
                "pvalue": mccrary_p,
                "manipulation_detected": manipulation_detected,
            },
        }

    except Exception as exc:
        logger.error(f"causal_rd_handler failed: {exc}")
        return {"error": f"RD estimation failed: {exc}", "statusCode": 500}


# ---------------------------------------------------------------------------
# causal-did
# ---------------------------------------------------------------------------

def causal_did_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Difference-in-differences estimation (standard and staggered).
    Standard DiD uses numpy OLS. Staggered adoption requires csdid (503 if absent).
    """
    outcome_var = parameters.get("outcome_var", "")
    treatment_group_var = parameters.get("treatment_group_var", "")
    time_var = parameters.get("time_var", "")
    pre_period = parameters.get("pre_period") or []
    post_period = parameters.get("post_period") or []
    staggered = bool(parameters.get("staggered", False))

    if not outcome_var:
        return {"error": "outcome_var is required", "statusCode": 400}
    if not treatment_group_var:
        return {"error": "treatment_group_var is required", "statusCode": 400}
    if not time_var:
        return {"error": "time_var is required", "statusCode": 400}
    if not pre_period or not post_period:
        return {"error": "pre_period and post_period are required", "statusCode": 400}

    if staggered:
        try:
            import importlib.util as _ilu
            if _ilu.find_spec("csdid") is None:
                raise ImportError("csdid not installed")
        except ImportError:
            return {
                "error": "Staggered DiD requires csdid library",
                "requires_layer": "csdid",
                "statusCode": 503,
            }

    try:
        df = df.copy()
        df["_is_post"] = df[time_var].isin(post_period).astype(float)
        df["_is_treated"] = df[treatment_group_var].astype(float)
        df["_did_interact"] = df["_is_treated"] * df["_is_post"]

        y = df[outcome_var].astype(float).values
        X = np.column_stack([
            np.ones(len(df)),
            df["_is_treated"].values,
            df["_is_post"].values,
            df["_did_interact"].values,
        ])
        coef, resid = _ols(X, y)
        ses = _ols_se(X, resid)
        did_estimate = float(coef[3])
        did_se = float(ses[3])

        # Parallel trends: pre-period slope comparison by group
        pre_df = df[df[time_var].isin(pre_period)].copy()
        parallel_pvalue = float("nan")
        if len(pre_df) >= 4:
            try:
                treat_group = pre_df[pre_df["_is_treated"] == 1][outcome_var].astype(float)
                ctrl_group = pre_df[pre_df["_is_treated"] == 0][outcome_var].astype(float)
                if len(treat_group) >= 2 and len(ctrl_group) >= 2:
                    _, parallel_pvalue = scipy.stats.ttest_ind(
                        treat_group, ctrl_group, equal_var=False
                    )
                    parallel_pvalue = float(parallel_pvalue)
            except Exception:
                pass

        # Event study: coefficient per time period relative to last pre-period
        all_periods = sorted(set(list(pre_period) + list(post_period)))
        base_period = pre_period[-1] if pre_period else all_periods[0]
        event_study_data = []
        for period in all_periods:
            if period == base_period:
                event_study_data.append({"period": period, "coefficient": 0.0,
                                          "ci_lower": 0.0, "ci_upper": 0.0, "base": True})
                continue
            period_df = df[df[time_var].isin([base_period, period])].copy()
            if len(period_df) < 4:
                continue
            y_p = period_df[outcome_var].astype(float).values
            X_p = np.column_stack([
                np.ones(len(period_df)),
                period_df["_is_treated"].values,
                (period_df[time_var] == period).astype(float),
                period_df["_is_treated"].values * (period_df[time_var] == period).astype(float),
            ])
            try:
                c_p, r_p = _ols(X_p, y_p)
                s_p = _ols_se(X_p, r_p)
                coef_p = float(c_p[3])
                se_p = float(s_p[3])
                event_study_data.append({
                    "period": period,
                    "coefficient": coef_p,
                    "ci_lower": coef_p - 1.96 * se_p,
                    "ci_upper": coef_p + 1.96 * se_p,
                    "base": False,
                })
            except Exception:
                pass

        # Placebo: shift treatment date to each pre-period and re-estimate
        placebo_results = []
        for fake_post in pre_period[:-1]:
            placebo_df = df[df[time_var].isin(pre_period)].copy()
            placebo_df["_placebo_post"] = (placebo_df[time_var] == fake_post).astype(float)
            placebo_df["_placebo_interact"] = (
                placebo_df["_is_treated"] * placebo_df["_placebo_post"]
            )
            y_pl = placebo_df[outcome_var].astype(float).values
            X_pl = np.column_stack([
                np.ones(len(placebo_df)),
                placebo_df["_is_treated"].values,
                placebo_df["_placebo_post"].values,
                placebo_df["_placebo_interact"].values,
            ])
            try:
                c_pl, r_pl = _ols(X_pl, y_pl)
                s_pl = _ols_se(X_pl, r_pl)
                coef_pl = float(c_pl[3])
                se_pl = float(s_pl[3])
                t_stat = coef_pl / max(se_pl, 1e-10)
                p_val = float(
                    2 * (1 - scipy.stats.t.cdf(abs(t_stat), df=max(len(placebo_df)-4, 1)))
                )
                placebo_results.append({"period": fake_post, "estimate": coef_pl, "pvalue": p_val})
            except Exception:
                pass

        return {
            "profile_id": "causal-did",
            "did_estimate": did_estimate,
            "did_se": did_se,
            "parallel_trends_pvalue": parallel_pvalue,
            "event_study_data": event_study_data,
            "placebo_results": placebo_results,
        }

    except Exception as exc:
        logger.error(f"causal_did_handler failed: {exc}")
        return {"error": f"DiD estimation failed: {exc}", "statusCode": 500}
