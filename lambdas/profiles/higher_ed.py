"""
higher_ed.py — Higher-education-specific analysis profiles

equity_gap_handler      — Demographic disaggregation and gap indices (equity-gap)
dfwi_handler            — D/F/W/I rates by course/instructor/demographic (dfwi-analysis)
cohort_flow_handler     — Enrollment funnel stage flow (cohort-flow)
peer_benchmark_handler  — Z-scores and percentile ranking vs peer institutions (peer-benchmark)

Called by runner/handler.py via entrypoint strings in config/profiles/*.json.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# equity-gap
# ---------------------------------------------------------------------------

def equity_gap_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Disaggregate an outcome metric by demographic group(s) and compute gap indices.

    Gap index = (group mean) / (reference group mean). Values < 1 indicate
    underperformance relative to the reference group.

    Returns the original DataFrame with group summary columns appended,
    plus a compact diagnostics dict with the full gap table.
    """
    metric_col = parameters.get("metric_column", "")
    group_cols = parameters.get("group_columns") or []
    reference_group = parameters.get("reference_group")  # value in the first group col
    min_group_size = int(parameters.get("min_group_size", 10))

    if not metric_col or metric_col not in df.columns:
        raise ValueError(f"metric_column '{metric_col}' not found")
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if not group_cols:
        raise ValueError("group_columns must specify at least one column")
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"group_columns not found: {missing}")

    clean = df[[metric_col] + group_cols].dropna(subset=[metric_col])
    if len(clean) < min_group_size:
        raise ValueError(f"Too few rows ({len(clean)}) with non-null metric values")

    # Compute group statistics
    grouped = clean.groupby(group_cols)[metric_col].agg(["count", "mean", "std"]).reset_index()
    grouped.columns = list(group_cols) + ["n", "mean", "std"]
    grouped = grouped[grouped["n"] >= min_group_size].copy()

    if grouped.empty:
        raise ValueError(
            f"No groups have at least {min_group_size} members. "
            "Lower min_group_size or check your data."
        )

    # Determine reference group mean
    if reference_group is not None and len(group_cols) == 1:
        ref_rows = grouped[grouped[group_cols[0]].astype(str) == str(reference_group)]
        if ref_rows.empty:
            logger.warning(f"reference_group '{reference_group}' not found; using overall mean")
            ref_mean = float(clean[metric_col].mean())
        else:
            ref_mean = float(ref_rows["mean"].iloc[0])
    else:
        ref_mean = float(clean[metric_col].mean())

    grouped["gap_from_reference"] = grouped["mean"] - ref_mean
    grouped["equity_index"] = grouped["mean"] / ref_mean if ref_mean != 0 else np.nan
    grouped["reference_mean"] = ref_mean

    gap_records = grouped.to_dict(orient="records")

    # Annotate original DataFrame with group mean and equity index
    result = df.copy()
    merge_cols = group_cols + ["mean", "equity_index"]
    result = result.merge(
        grouped[merge_cols].rename(columns={"mean": "group_mean"}),
        on=group_cols,
        how="left",
    )

    diagnostics = {
        "metric_column": metric_col,
        "group_columns": group_cols,
        "reference_group": reference_group,
        "reference_mean": round(ref_mean, 4),
        "overall_mean": round(float(clean[metric_col].mean()), 4),
        "n_groups_analyzed": len(grouped),
        "n_groups_excluded_small": int(
            (clean.groupby(group_cols).size() < min_group_size).sum()
            if not clean.empty else 0
        ),
        "gap_table": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            for row in gap_records
        ],
    }

    logger.info(f"equity-gap complete: metric={metric_col}, groups={group_cols}, n_groups={len(grouped)}")
    return result, diagnostics


# ---------------------------------------------------------------------------
# dfwi-analysis
# ---------------------------------------------------------------------------

def dfwi_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Compute D/F/W/I (Drop/Fail/Withdraw/Incomplete) rates aggregated by
    specified grouping columns.

    Returns the original DataFrame with an is_dfwi boolean column appended,
    plus a summary table of DFWI rates per group in diagnostics.
    """
    grade_col = parameters.get("grade_column", "")
    dfwi_values = parameters.get("dfwi_values") or ["D", "F", "W", "I"]
    group_by = parameters.get("group_by") or []
    min_enrollment = int(parameters.get("min_enrollment", 5))

    if not grade_col or grade_col not in df.columns:
        raise ValueError(f"grade_column '{grade_col}' not found")
    if isinstance(group_by, str):
        group_by = [group_by]
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(f"group_by columns not found: {missing}")
    if isinstance(dfwi_values, str):
        dfwi_values = [dfwi_values]

    result = df.copy()
    result["is_dfwi"] = result[grade_col].astype(str).isin([str(v) for v in dfwi_values])

    overall_rate = float(result["is_dfwi"].mean())
    overall_n = len(result)

    if not group_by:
        summary = [{
            "group": "overall",
            "n": overall_n,
            "n_dfwi": int(result["is_dfwi"].sum()),
            "dfwi_rate": round(overall_rate, 4),
        }]
    else:
        agg = result.groupby(group_by).agg(
            n=("is_dfwi", "count"),
            n_dfwi=("is_dfwi", "sum"),
        ).reset_index()
        agg["dfwi_rate"] = (agg["n_dfwi"] / agg["n"]).round(4)
        agg = agg[agg["n"] >= min_enrollment]
        summary = agg.to_dict(orient="records")

    diagnostics = {
        "grade_column": grade_col,
        "dfwi_values": dfwi_values,
        "group_by": group_by,
        "overall_n": overall_n,
        "overall_dfwi_rate": round(overall_rate, 4),
        "n_groups": len(summary),
        "dfwi_summary": summary,
    }

    logger.info(
        f"dfwi-analysis complete: grade_col={grade_col}, "
        f"overall_rate={overall_rate:.2%}, groups={group_by}"
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# cohort-flow
# ---------------------------------------------------------------------------

def cohort_flow_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Track a cohort through an ordered sequence of stages and compute
    funnel flow counts and conversion rates at each transition.

    Returns the original DataFrame with a stage_reached column, plus a
    funnel summary table in diagnostics.
    """
    id_col = parameters.get("id_column", "")
    stage_col = parameters.get("stage_column", "")
    stage_order = parameters.get("stage_order") or []
    cohort_col = parameters.get("cohort_column")
    group_by = parameters.get("group_by")

    if not id_col or id_col not in df.columns:
        raise ValueError(f"id_column '{id_col}' not found")
    if not stage_col or stage_col not in df.columns:
        raise ValueError(f"stage_column '{stage_col}' not found")
    if not stage_order:
        raise ValueError("stage_order must list stages in sequence (e.g. ['applied','admitted','enrolled'])")

    # Determine the furthest stage each individual reached
    def _stage_rank(val: str) -> int:
        try:
            return stage_order.index(str(val))
        except ValueError:
            return -1

    result = df.copy()
    result["_stage_rank"] = result[stage_col].map(_stage_rank)

    # Per-individual: furthest stage reached
    id_cols = [id_col] + ([cohort_col] if cohort_col else [])
    max_stage = (
        result[result["_stage_rank"] >= 0]
        .groupby(id_cols)["_stage_rank"]
        .max()
        .reset_index()
        .rename(columns={"_stage_rank": "stage_reached_rank"})
    )
    max_stage["stage_reached"] = max_stage["stage_reached_rank"].apply(
        lambda r: stage_order[r] if 0 <= r < len(stage_order) else None
    )

    result = result.merge(max_stage[[id_col] + ["stage_reached"]], on=id_col, how="left")

    # Funnel summary
    _group_keys = ([group_by] if isinstance(group_by, str) else list(group_by)) if group_by else []
    funnel_rows = []
    for i, stage in enumerate(stage_order):
        reached = max_stage[max_stage["stage_reached_rank"] >= i]
        n = len(reached)
        if i == 0:
            conversion = 1.0
        else:
            prev = len(max_stage[max_stage["stage_reached_rank"] >= i - 1])
            conversion = n / prev if prev > 0 else 0.0
        funnel_rows.append({
            "stage": stage,
            "stage_order": i,
            "n_reached": n,
            "conversion_from_previous": round(conversion, 4),
            "pct_of_start": round(n / len(max_stage), 4) if max_stage.shape[0] > 0 else 0.0,
        })

    result.drop(columns=["_stage_rank"], inplace=True)

    diagnostics = {
        "id_column": id_col,
        "stage_column": stage_col,
        "stage_order": stage_order,
        "n_individuals": len(max_stage),
        "funnel": funnel_rows,
    }

    logger.info(
        f"cohort-flow complete: {len(max_stage)} individuals across {len(stage_order)} stages"
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# peer-benchmark
# ---------------------------------------------------------------------------

def peer_benchmark_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Compute z-scores and percentile ranks for specified metrics across a
    peer group. Identifies a focal institution by id and highlights its
    relative position.

    Returns the original DataFrame with {metric}_zscore and {metric}_percentile
    columns appended, plus a focal institution summary in diagnostics.
    """
    from scipy import stats as scipy_stats

    metric_cols = parameters.get("metric_columns") or []
    id_col = parameters.get("id_column", "")
    focal_id = parameters.get("focal_id")
    _label_col = parameters.get("label_column")

    if isinstance(metric_cols, str):
        metric_cols = [metric_cols]
    if not metric_cols:
        raise ValueError("metric_columns must specify at least one column")
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"metric_columns not found: {missing}")
    if id_col and id_col not in df.columns:
        raise ValueError(f"id_column '{id_col}' not found")

    result = df.copy()

    focal_summary = {}
    for col in metric_cols:
        values = result[col].dropna()
        if len(values) < 2:
            logger.warning(f"Not enough non-null values in '{col}' for benchmarking")
            result[f"{col}_zscore"] = np.nan
            result[f"{col}_percentile"] = np.nan
            continue

        z_scores = scipy_stats.zscore(result[col], nan_policy="omit")
        result[f"{col}_zscore"] = z_scores.round(4)
        result[f"{col}_percentile"] = result[col].rank(pct=True).round(4) * 100

        if focal_id is not None and id_col:
            focal_rows = result[result[id_col].astype(str) == str(focal_id)]
            if not focal_rows.empty:
                focal_summary[col] = {
                    "value": float(focal_rows[col].iloc[0]),
                    "zscore": float(focal_rows[f"{col}_zscore"].iloc[0]),
                    "percentile": float(focal_rows[f"{col}_percentile"].iloc[0]),
                }

    # Flag focal institution
    if focal_id is not None and id_col:
        result["is_focal"] = result[id_col].astype(str) == str(focal_id)

    n_peers = len(df)
    diagnostics = {
        "metric_columns": metric_cols,
        "id_column": id_col,
        "focal_id": str(focal_id) if focal_id is not None else None,
        "n_peers": n_peers,
        "focal_institution": focal_summary if focal_summary else None,
        "note": (
            f"Focal institution '{focal_id}' not found in dataset"
            if focal_id is not None and id_col and not focal_summary
            else None
        ),
    }

    logger.info(
        f"peer-benchmark complete: {n_peers} peers, {len(metric_cols)} metrics, focal={focal_id}"
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# intersectionality-equity
# ---------------------------------------------------------------------------

def intersectionality_equity_handler(df, params):
    """
    Cross-tabulate an outcome metric by multiple demographic dimensions and
    compute Disparate Impact ratios using the 80% rule.

    Cells with n < n_suppress are redacted to protect privacy.

    Returns a plain dict (not a DataFrame tuple) because the output is a
    variable-length cell table rather than a row-level annotation.
    """
    metric_col = params.get("metric_column", "")
    group_cols = params.get("group_columns") or []
    reference_group = params.get("reference_group") or []
    n_suppress = int(params.get("n_suppress", 10))

    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if len(group_cols) < 2:
        return {"error": "group_columns requires at least 2 columns"}
    if not metric_col or metric_col not in df.columns:
        return {"error": f"metric_column '{metric_col}' not found or not in dataset"}
    if not pd.api.types.is_numeric_dtype(df[metric_col]):
        return {"error": f"metric_column '{metric_col}' must be numeric"}

    clean = df[group_cols + [metric_col]].dropna(subset=[metric_col])

    # Overall mean used as fallback reference
    overall_mean = float(clean[metric_col].mean()) if len(clean) > 0 else 0.0

    # Compute per-cell stats
    grouped = (
        clean.groupby(group_cols)[metric_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    grouped.columns = list(group_cols) + ["n", "group_mean"]

    # Identify reference cell
    reference_mean = overall_mean
    if reference_group and len(reference_group) == len(group_cols):
        mask = pd.Series([True] * len(grouped), index=grouped.index)
        for col, val in zip(group_cols, reference_group):
            mask &= grouped[col].astype(str) == str(val)
        ref_rows = grouped[mask]
        if not ref_rows.empty:
            reference_mean = float(ref_rows["group_mean"].iloc[0])
        else:
            logger.warning(
                "reference_group values not found in data; falling back to overall mean"
            )

    rows = []
    suppressed_count = 0
    for _, row in grouped.iterrows():
        group_vals = [row[c] for c in group_cols]
        group_key = "|".join(str(v) for v in group_vals)
        n = int(row["n"])
        gm = float(row["group_mean"])

        if n < n_suppress:
            suppressed_count += 1
            rows.append({
                "group_key": group_key,
                "n": f"<{n_suppress}",
                "group_mean": None,
                "di_ratio": None,
                "adverse_impact_flag": None,
            })
        else:
            if reference_mean != 0:
                di_ratio = round(gm / reference_mean, 4)
                adverse_impact_flag = di_ratio < 0.80
            else:
                di_ratio = None
                adverse_impact_flag = None
            rows.append({
                "group_key": group_key,
                "n": n,
                "group_mean": round(gm, 4),
                "di_ratio": di_ratio,
                "adverse_impact_flag": adverse_impact_flag,
            })

    logger.info(
        f"intersectionality-equity complete: metric={metric_col}, "
        f"group_cols={group_cols}, cells={len(rows)}, suppressed={suppressed_count}"
    )
    return {
        "rows": rows,
        "suppressed_cells": suppressed_count,
        "reference_group_mean": round(reference_mean, 4),
        "profile_id": "intersectionality-equity",
    }


# ---------------------------------------------------------------------------
# assessment-irt
# ---------------------------------------------------------------------------

def assessment_irt_handler(df, params):
    """
    Fit a two-parameter logistic (2PL) IRT model to binary item-response data.

    Requires the optional 'girth' library (installed as a Lambda Layer).
    Returns item parameters, person ability estimates, and item information
    functions evaluated over a standard theta grid.
    """
    import math

    item_columns = params.get("item_columns") or []
    person_id_column = params.get("person_id_column", "")

    if isinstance(item_columns, str):
        item_columns = [item_columns]
    if len(item_columns) < 5:
        return {"error": "item_columns requires at least 5 columns"}
    if len(df) < 100:
        return {"error": f"assessment-irt requires at least 100 rows; got {len(df)}"}

    try:
        import girth  # noqa: PLC0415
    except ImportError:
        return {"error": "IRT library not available in this deployment", "requires_layer": "girth"}

    exp = math.exp

    # Build response matrix: persons × items, then transpose to items × persons
    data = df[item_columns].to_numpy(dtype=float)
    data_T = data.T

    # Fit 2PL model
    result = girth.twopl_mml(data_T)
    difficulties = result["Difficulty"]
    discriminations = result["Discrimination"]

    # EAP person ability estimates
    thetas, ses = girth.ability_eap(data_T, discriminations, difficulties)

    # Item information at a grid of theta values
    theta_grid = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    def _item_info(a, b, theta):
        p = 1.0 / (1.0 + exp(-a * (theta - b)))
        return a ** 2 * p * (1.0 - p)

    item_parameters = [
        {
            "item": col,
            "difficulty": float(b),
            "discrimination": float(a),
            "info_at_zero": float(_item_info(a, b, 0.0)),
        }
        for col, a, b in zip(item_columns, discriminations, difficulties)
    ]

    person_id_col = person_id_column if person_id_column and person_id_column in df.columns else None
    person_ids = df[person_id_col].tolist() if person_id_col else list(range(len(df)))
    person_abilities = [
        {"person_id": pid, "theta": float(t), "se": float(s)}
        for pid, t, s in zip(person_ids, thetas, ses)
    ]

    item_information = [
        {
            "item": col,
            "theta_range": theta_grid,
            "information": [float(_item_info(a, b, th)) for th in theta_grid],
        }
        for col, a, b in zip(item_columns, discriminations, difficulties)
    ]

    logger.info(
        f"assessment-irt complete: {len(item_columns)} items, {len(df)} persons"
    )
    return {
        "item_parameters": item_parameters,
        "person_abilities": person_abilities,
        "item_information": item_information,
        "profile_id": "assessment-irt",
    }


# ---------------------------------------------------------------------------
# financial-aid-effectiveness
# ---------------------------------------------------------------------------

def financial_aid_effectiveness_handler(
    df: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, dict]:
    """
    Model financial aid packaging effectiveness on persistence/graduation.

    Required columns (via parameters):
      student_id_col, aid_year_col, efc_col, total_grants_col,
      total_loans_col, persistence_col
    Optional: gpa_col, degree_col

    Returns DataFrame with aid_band and predicted_persistence_prob columns,
    plus diagnostics with cohort_table, regression coefficients, unmet_need_trend.
    """
    # Column mapping
    student_id = parameters.get("student_id_col", "student_id")
    aid_year = parameters.get("aid_year_col", "aid_year")
    efc = parameters.get("efc_col", "efc")
    grants = parameters.get("total_grants_col", "total_grants")
    loans = parameters.get("total_loans_col", "total_loans")
    persist = parameters.get("persistence_col", "persistence_flag")
    gpa_col = parameters.get("gpa_col")

    # Validate required columns
    required = [student_id, aid_year, efc, grants, loans, persist]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return df, {"error": f"Missing required columns: {missing}"}

    # Compute net price and aid band
    df = df.copy()
    df["net_price"] = df[efc] - df[grants]

    def _band(net):
        if net < 5000:
            return "<$5k"
        elif net < 10000:
            return "$5-10k"
        elif net < 15000:
            return "$10-15k"
        else:
            return ">$15k"

    df["aid_band"] = df["net_price"].apply(_band)

    # Cohort table by aid band
    cohort_rows = []
    for band, group in df.groupby("aid_band"):
        row = {
            "aid_band": band,
            "count": len(group),
            "persistence_rate": round(group[persist].mean(), 4),
            "mean_net_price": round(group["net_price"].mean(), 2),
        }
        if gpa_col and gpa_col in df.columns:
            row["mean_gpa"] = round(group[gpa_col].mean(), 4)
        cohort_rows.append(row)

    # Logistic regression: persistence ~ net_price + grants + loans
    try:
        y = df[persist].astype(float).values
        X_cols = ["net_price", grants, loans]
        X = df[X_cols].astype(float).values
        X = np.column_stack([np.ones(len(X)), X])  # add intercept

        # Iteratively reweighted least squares (simple logistic)
        beta = np.zeros(X.shape[1])
        for _ in range(25):
            p = 1.0 / (1.0 + np.exp(-X @ beta))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            W = np.diag(p * (1 - p))
            try:
                beta = beta + np.linalg.solve(X.T @ W @ X, X.T @ (y - p))
            except np.linalg.LinAlgError:
                break

        # Predicted probabilities
        df["predicted_persistence_prob"] = 1.0 / (1.0 + np.exp(-X @ beta))
        df["predicted_persistence_prob"] = df["predicted_persistence_prob"].round(4)

        coef_names = ["intercept", "net_price", grants, loans]
        regression_summary = [
            {"variable": name, "coefficient": round(float(b), 6)}
            for name, b in zip(coef_names, beta)
        ]
    except Exception as exc:
        logger.warning("Logistic regression failed: %s", exc)
        df["predicted_persistence_prob"] = None
        regression_summary = [{"error": str(exc)}]

    # Unmet-need trend by aid year
    df["unmet_need"] = df[efc] - df[grants] - df[loans]
    unmet_trend = []
    for year, group in df.groupby(aid_year):
        unmet_trend.append({
            "aid_year": str(year),
            "mean_unmet_need": round(float(group["unmet_need"].mean()), 2),
            "student_count": len(group),
        })
    unmet_trend.sort(key=lambda r: r["aid_year"])

    diagnostics = {
        "cohort_table": cohort_rows,
        "regression_summary": regression_summary,
        "unmet_need_trend": unmet_trend,
        "total_students": len(df),
        "persistence_rate_overall": round(float(df[persist].mean()), 4),
    }

    return df, diagnostics
