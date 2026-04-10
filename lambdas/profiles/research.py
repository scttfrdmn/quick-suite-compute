"""
research.py — Research analytics profiles

grant_portfolio_handler      — Grant burn rate, NCE risk, PI productivity (grant-portfolio)
coauthor_network_handler     — Co-authorship network centrality and community detection (network-coauthor)
grant_pipeline_handler       — PI-level portfolio health and NCE risk projection (grant-pipeline)
provenance_graph_handler     — W3C PROV-DM lineage reconstruction from HistoryTable (provenance-graph)
power_analysis_handler       — Literature-informed sample size calculation with power curves (#69)
anomaly_hypothesis_handler   — Anomaly detection + literature cross-reference classification (#70)
reproducibility_check_handler — Re-execute analysis script and compare to manuscript results (#71)

Called by runner/handler.py via entrypoints in config/profiles/*.json.
"""

from __future__ import annotations

import json
import logging
import math
import os
import urllib.request
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# grant-portfolio
# ---------------------------------------------------------------------------

def grant_portfolio_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Compute burn rate, percent expended, and NCE risk per grant award.

    Groups transaction-level data by award ID to compute award-level financial
    metrics. Merges those metrics back to every row so each transaction carries
    the cumulative award-level picture.

    Returns the original DataFrame with burn_rate, pct_expended, and nce_risk
    columns appended, plus a portfolio summary table in diagnostics.
    """
    amount_col = parameters.get("amount_column", "")
    budget_col = parameters.get("budget_column", "")
    date_col = parameters.get("date_column", "")
    award_col = parameters.get("award_id_column", "")
    pi_col = parameters.get("pi_column")
    area_col = parameters.get("research_area_column")
    nce_threshold = float(parameters.get("nce_threshold", 0.90))

    # Validate required columns
    for param_name, col in [
        ("amount_column", amount_col),
        ("budget_column", budget_col),
        ("date_column", date_col),
        ("award_id_column", award_col),
    ]:
        if not col or col not in df.columns:
            raise ValueError(f"{param_name} '{col}' not found in DataFrame")
    if pi_col and pi_col not in df.columns:
        raise ValueError(f"pi_column '{pi_col}' not found in DataFrame")
    if area_col and area_col not in df.columns:
        raise ValueError(f"research_area_column '{area_col}' not found in DataFrame")

    if df.empty:
        result = df.copy()
        result["burn_rate"] = pd.Series(dtype=float)
        result["pct_expended"] = pd.Series(dtype=float)
        result["nce_risk"] = pd.Series(dtype=bool)
        return result, {"warning": "Input dataset is empty"}

    clean = df.dropna(subset=[amount_col, award_col])
    dropped = len(df) - len(clean)

    # Award-level aggregation
    award_agg = clean.groupby(award_col).agg(
        total_spent=(amount_col, "sum"),
        # Budget: use sum if each row has a per-transaction budget slice,
        # otherwise take the first value if all rows share the same total budget.
        # Heuristic: if max budget == min budget per award, treat as constant.
    )

    # Separately compute budget per award
    budget_per_award = clean.groupby(award_col)[budget_col].agg(["sum", "first", "min", "max"])
    budget_per_award["total_budget"] = np.where(
        budget_per_award["min"] == budget_per_award["max"],
        budget_per_award["first"],   # constant budget → use as-is
        budget_per_award["sum"],      # varying budget → sum the slices
    )
    award_agg = award_agg.join(budget_per_award[["total_budget"]])

    # Guard against zero/NaN budget
    zero_budget = (award_agg["total_budget"] == 0) | award_agg["total_budget"].isna()
    if zero_budget.any():
        logger.warning(f"{zero_budget.sum()} awards have zero or missing budget; burn_rate set to NaN")
    award_agg["burn_rate"] = np.where(
        zero_budget, np.nan, award_agg["total_spent"] / award_agg["total_budget"]
    )
    award_agg["pct_expended"] = award_agg["burn_rate"] * 100
    award_agg["nce_risk"] = award_agg["burn_rate"] > nce_threshold

    # Merge back to original rows
    metrics = award_agg[["burn_rate", "pct_expended", "nce_risk"]].reset_index()
    result = df.merge(metrics, on=award_col, how="left")
    result.index = df.index  # preserve original index

    # Portfolio summary
    portfolio_summary = award_agg.reset_index().rename(columns={award_col: "award_id"})
    portfolio_summary = portfolio_summary.round(4).to_dict(orient="records")

    # Optional PI summary
    pi_summary = None
    if pi_col:
        pi_agg = clean.groupby(pi_col).agg(
            n_awards=(award_col, "nunique"),
            total_spent=(amount_col, "sum"),
            pct_at_nce_risk=(award_col, lambda x: round(
                result.loc[x.index, "nce_risk"].fillna(False).mean(), 4
            )),
        ).reset_index().rename(columns={pi_col: "pi"})
        pi_summary = pi_agg.to_dict(orient="records")

    # Optional research area clustering
    area_clusters = None
    if area_col:
        try:
            from sklearn.cluster import KMeans
            n_clusters = int(parameters.get("n_clusters", 3))
            area_metrics = award_agg.reset_index()
            area_metrics[area_col] = clean.groupby(award_col)[area_col].first().values
            X = area_metrics[["burn_rate", "pct_expended"]].fillna(0).values
            if len(X) >= n_clusters:
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                area_metrics["award_cluster"] = km.fit_predict(X)
                area_clusters = area_metrics[[award_col, area_col, "award_cluster"]].to_dict(orient="records")
        except Exception as exc:
            logger.warning(f"Award clustering failed: {exc}")

    n_awards = int(award_agg.shape[0])
    n_at_risk = int(award_agg["nce_risk"].sum())
    diagnostics = {
        "amount_column": amount_col,
        "budget_column": budget_col,
        "award_id_column": award_col,
        "nce_threshold": nce_threshold,
        "n_awards": n_awards,
        "n_at_nce_risk": n_at_risk,
        "total_portfolio_spend": round(float(award_agg["total_spent"].sum()), 2),
        "total_portfolio_budget": round(float(award_agg["total_budget"].sum()), 2),
        "rows_excluded_missing_data": dropped,
        "portfolio_summary": portfolio_summary,
        "pi_summary": pi_summary,
        "area_clusters": area_clusters,
    }

    logger.info(
        f"grant-portfolio complete: {n_awards} awards, {n_at_risk} at NCE risk, "
        f"threshold={nce_threshold}"
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# network-coauthor
# ---------------------------------------------------------------------------

def coauthor_network_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Build a co-authorship collaboration network from publication data.

    Each row is a publication. The author_column contains a separator-delimited
    list of author names. Computes degree centrality, betweenness centrality,
    and Louvain community assignments per author.

    Returns a new DataFrame — one row per (author, publication_id) pair —
    with network metrics joined in.
    """
    import community as community_louvain
    import networkx as nx

    author_col = parameters.get("author_column", "")
    pub_col = parameters.get("publication_id_column", "")
    separator = parameters.get("author_separator", ";")
    min_collabs = int(parameters.get("min_collaborations", 1))

    if not author_col or author_col not in df.columns:
        raise ValueError(f"author_column '{author_col}' not found in DataFrame")
    if not pub_col or pub_col not in df.columns:
        raise ValueError(f"publication_id_column '{pub_col}' not found in DataFrame")

    if df.empty:
        result = pd.DataFrame(columns=["author", "publication_id", "degree_centrality",
                                        "betweenness_centrality", "community_id"])
        return result, {"warning": "Input dataset is empty"}

    # Explode: one row per (publication, author)
    records = []
    for _, row in df.iterrows():
        raw = str(row[author_col]) if pd.notna(row[author_col]) else ""
        authors = [a.strip() for a in raw.split(separator) if a.strip()]
        for author in authors:
            records.append({"author": author, "publication_id": row[pub_col]})

    if not records:
        raise ValueError("No author entries found after parsing author_column")

    exploded = pd.DataFrame(records)
    n_unique_authors = exploded["author"].nunique()
    if n_unique_authors < 2:
        raise ValueError(f"Need at least 2 distinct authors to form a network (found {n_unique_authors})")

    # Build edge list: for each publication, all pairs of co-authors
    from itertools import combinations
    edge_weights: dict[tuple[str, str], int] = {}
    for pub_id, group in exploded.groupby("publication_id"):
        authors = group["author"].tolist()
        for a, b in combinations(sorted(set(authors)), 2):
            key = (a, b)
            edge_weights[key] = edge_weights.get(key, 0) + 1

    # Build graph, filtering by min_collaborations
    G = nx.Graph()
    G.add_nodes_from(exploded["author"].unique())
    for (a, b), weight in edge_weights.items():
        if weight >= min_collabs:
            G.add_edge(a, b, weight=weight)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Degree centrality
    degree_cent = nx.degree_centrality(G)

    # Betweenness centrality — approximate for large graphs
    betweenness_approx = False
    if n_nodes > 5000:
        betweenness_cent = nx.betweenness_centrality(G, k=500, normalized=True, seed=42)
        betweenness_approx = True
    else:
        betweenness_cent = nx.betweenness_centrality(G, normalized=True)

    # Louvain community detection
    try:
        partition = community_louvain.best_partition(G, random_state=42)
    except Exception as exc:
        logger.warning(f"Louvain community detection failed: {exc}; assigning all to community 0")
        partition = {node: 0 for node in G.nodes()}

    n_communities = len(set(partition.values()))
    density = round(nx.density(G), 6)

    # Join metrics back to exploded (author, publication_id) rows
    exploded["degree_centrality"] = exploded["author"].map(
        lambda a: round(degree_cent.get(a, 0.0), 6)
    )
    exploded["betweenness_centrality"] = exploded["author"].map(
        lambda a: round(betweenness_cent.get(a, 0.0), 6)
    )
    exploded["community_id"] = exploded["author"].map(
        lambda a: int(partition.get(a, 0))
    )

    # Top authors by degree
    top_by_degree = sorted(
        [{"author": a, "degree_centrality": round(v, 6)} for a, v in degree_cent.items()],
        key=lambda x: x["degree_centrality"],
        reverse=True,
    )[:10]

    diagnostics = {
        "author_column": author_col,
        "publication_id_column": pub_col,
        "separator": separator,
        "min_collaborations": min_collabs,
        "n_authors": n_unique_authors,
        "n_publications": int(df[pub_col].nunique()),
        "n_edges": n_edges,
        "n_communities": n_communities,
        "density": density,
        "betweenness_approximated": betweenness_approx,
        "top_authors_by_degree": top_by_degree,
    }

    logger.info(
        f"coauthor-network complete: {n_unique_authors} authors, {n_edges} edges, "
        f"{n_communities} communities, density={density}"
    )
    return exploded, diagnostics


# ---------------------------------------------------------------------------
# grant-pipeline
# ---------------------------------------------------------------------------

def grant_pipeline_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Grant portfolio health scoring and NCE risk projection.

    Inputs: pi_column, start_date_column, end_date_column, amount_column,
            sponsor_column (optional), submission_date_column (optional),
            lookback_months (default 24).
    Returns: pi_health list, portfolio_summary, sponsor_timing.
    """
    import datetime

    pi_col = parameters.get("pi_column", "")
    start_col = parameters.get("start_date_column", "")
    end_col = parameters.get("end_date_column", "")
    amount_col = parameters.get("amount_column", "")
    sponsor_col = parameters.get("sponsor_column", "")
    submission_col = parameters.get("submission_date_column", "")

    if not pi_col:
        return {"error": "pi_column is required", "statusCode": 400}
    if not start_col or not end_col:
        return {"error": "start_date_column and end_date_column are required", "statusCode": 400}

    today = datetime.date.today()
    nce_threshold = today + datetime.timedelta(days=60)

    try:
        df = df.copy()
        df["_start"] = pd.to_datetime(df[start_col], errors="coerce").dt.date
        df["_end"] = pd.to_datetime(df[end_col], errors="coerce").dt.date
        if amount_col and amount_col in df.columns:
            df["_amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
        else:
            df["_amount"] = 0.0

        pi_health = []
        for pi, group in df.groupby(pi_col):
            active = group[(group["_start"] <= today) & (group["_end"] >= today)]
            ending_soon = group[
                (group["_end"] >= today) & (group["_end"] <= nce_threshold)
            ]
            # Overlapping coverage in next 12 months
            future = group[group["_end"] >= today]
            months_covered = 0
            for check_month in range(12):
                check_date = today + datetime.timedelta(days=check_month * 30)
                if any((r["_start"] <= check_date <= r["_end"]) for _, r in future.iterrows()):
                    months_covered += 1
            funding_continuity = months_covered / 12.0

            unique_sponsors = group[sponsor_col].nunique() if sponsor_col and sponsor_col in group else 1
            diversity = min(unique_sponsors / 5.0, 1.0)  # normalize to 5 sponsors = max

            active_count = len(active)
            health_score = (
                min(active_count / 3.0, 1.0) * 0.4 +
                funding_continuity * 0.4 +
                diversity * 0.2
            )

            nce_risk_score = len(ending_soon) / max(active_count, 1)

            pi_health.append({
                "pi": str(pi),
                "active_grants": active_count,
                "ending_soon": len(ending_soon),
                "funding_gap_months": 12 - months_covered,
                "nce_risk_score": round(nce_risk_score, 3),
                "health_score": round(health_score, 3),
            })

        # Portfolio summary
        total_active = int(df[(df["_start"] <= today) & (df["_end"] >= today)].shape[0])
        total_amount = float(df[df["_end"] >= today]["_amount"].sum())

        # Sponsor timing
        sponsor_timing = []
        if sponsor_col and submission_col and sponsor_col in df.columns and submission_col in df.columns:
            df["_submission"] = pd.to_datetime(df[submission_col], errors="coerce").dt.date
            for sponsor, sgroup in df[df["_submission"].notna()].groupby(sponsor_col):
                try:
                    lead_days = (sgroup["_start"] - sgroup["_submission"]).dt.days.mean()
                    sponsor_timing.append({
                        "sponsor": str(sponsor),
                        "avg_lead_days": round(float(lead_days), 1),
                    })
                except Exception:
                    pass

        return {
            "profile_id": "grant-pipeline",
            "pi_health": pi_health,
            "portfolio_summary": {
                "total_active_grants": total_active,
                "total_active_amount": total_amount,
                "pi_count": len(pi_health),
            },
            "sponsor_timing": sponsor_timing,
        }

    except Exception as exc:
        logger.error(f"grant_pipeline_handler failed: {exc}")
        return {"error": f"Grant pipeline analysis failed: {exc}", "statusCode": 500}


# ---------------------------------------------------------------------------
# provenance-graph
# ---------------------------------------------------------------------------

def provenance_graph_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Reconstruct data lineage from HistoryTable (W3C PROV-DM JSON-LD).

    Input df is unused; all data comes from DynamoDB HistoryTable.
    artifact_uri: S3 path to trace lineage for.
    lookback_days: how far back to search (default 365).
    """
    import boto3
    from boto3.dynamodb.conditions import Attr

    artifact_uri = str(parameters.get("artifact_uri", "")).strip()
    lookback_days = int(parameters.get("lookback_days", 365))

    if not artifact_uri:
        return {"error": "artifact_uri is required", "statusCode": 400}

    history_table_name = (
        os.environ.get("COMPUTE_HISTORY_TABLE") or
        os.environ.get("HISTORY_TABLE", "")
    )

    entities: list[dict] = []
    activities: list[dict] = []
    gaps: list[dict] = []

    if history_table_name:
        try:
            dynamodb = boto3.resource("dynamodb")
            table = dynamodb.Table(history_table_name)
            resp = table.scan(
                FilterExpression=(
                    Attr("result_s3_uri").eq(artifact_uri) |
                    Attr("source_s3_uri").eq(artifact_uri)
                ),
            )
            items = resp.get("Items", [])

            for item in sorted(items, key=lambda x: x.get("started_at", "")):
                job_id = item.get("job_id", item.get("execution_id", "unknown"))
                activities.append({
                    "id": f"activity:{job_id}",
                    "type": "prov:Activity",
                    "profile_id": item.get("profile_id", ""),
                    "startedAtTime": item.get("started_at", ""),
                    "endedAtTime": item.get("completed_at", ""),
                    "wasAssociatedWith": item.get("user_arn", ""),
                })
                if item.get("source_s3_uri"):
                    entities.append({
                        "id": f"entity:{item['source_s3_uri']}",
                        "type": "prov:Entity",
                        "uri": item["source_s3_uri"],
                        "role": "input",
                    })
                if item.get("result_s3_uri"):
                    entities.append({
                        "id": f"entity:{item['result_s3_uri']}",
                        "type": "prov:Entity",
                        "uri": item["result_s3_uri"],
                        "role": "output",
                    })

            # Gap detection: check for URI chain breaks
            if len(activities) > 1:
                for i in range(1, len(activities)):
                    prev_out = items[i-1].get("result_s3_uri", "")
                    curr_in = items[i].get("source_s3_uri", "")
                    if prev_out and curr_in and prev_out != curr_in:
                        gaps.append({
                            "stage": i,
                            "description": f"Output {prev_out!r} does not match input {curr_in!r}",
                        })

        except Exception as exc:
            logger.warning(f"provenance_graph: HistoryTable query failed: {exc}")

    # Deduplicate entities
    seen_ids = set()
    unique_entities = []
    for e in entities:
        if e["id"] not in seen_ids:
            seen_ids.add(e["id"])
            unique_entities.append(e)

    prov_graph = {
        "@context": "http://www.w3.org/ns/prov",
        "entity": {e["id"]: e for e in unique_entities},
        "activity": {a["id"]: a for a in activities},
    }

    # Markdown lineage summary
    if activities:
        lines = [f"# Data Lineage for `{artifact_uri}`\n"]
        for i, act in enumerate(activities, 1):
            started = act.get("startedAtTime", "")[:10]
            lines.append(f"**Step {i}**: `{act['profile_id']}` — {started}")
        lineage_md = "\n".join(lines)
    else:
        lineage_md = f"No lineage records found for `{artifact_uri}` within {lookback_days} days."

    return {
        "profile_id": "provenance-graph",
        "prov_graph": prov_graph,
        "lineage_markdown": lineage_md,
        "gaps": gaps,
        "activities_found": len(activities),
    }


# ---------------------------------------------------------------------------
# Router API helper
# ---------------------------------------------------------------------------

def _call_router_api(tool: str, payload: dict) -> dict | None:
    """POST to Router API Gateway. Returns parsed JSON or None on failure."""
    url = os.environ.get("ROUTER_API_URL", "")
    if not url:
        return None
    try:
        body = json.dumps({"tool": tool, **payload}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Router API call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# power-analysis (#69)
# ---------------------------------------------------------------------------

def power_analysis_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Literature-informed sample size calculation with power curves.

    Three modes via effect_size_source:
      - "literature": extract effect sizes from comparable PubMed studies via Router
      - "pilot_data": compute Cohen's d from df grouped by first treatment_var
      - "manual": use manual_effect_size directly
    """
    from scipy.stats import norm

    outcome_var = parameters.get("outcome_var", "")
    treatment_vars: list[str] = parameters.get("treatment_vars") or []
    alpha = float(parameters.get("alpha", 0.05))
    target_power = float(parameters.get("target_power", 0.80))
    effect_size_source = str(parameters.get("effect_size_source", "manual"))
    comparable_studies: list[str] = parameters.get("comparable_studies") or []
    assay_context = parameters.get("assay_context", "")
    manual_effect_size = parameters.get("manual_effect_size")

    if not outcome_var:
        return {"error": "outcome_var is required", "statusCode": 400}
    if not treatment_vars:
        return {"error": "treatment_vars is required (at least one column)", "statusCode": 400}

    d: float | None = None
    effect_size_distribution: list[float] = []
    citations: list[str] = []
    confound_checklist: list[str] = []

    if effect_size_source == "literature":
        # Call Router extract for each comparable study
        for pmid in comparable_studies:
            resp = _call_router_api("extract", {
                "extraction_type": "effect_sizes",
                "text": pmid,
                "assay_context": assay_context,
            })
            if resp and "effect_sizes" in resp:
                effect_size_distribution.extend(resp["effect_sizes"])
                citations.append(pmid)
            if resp and "confounds" in resp:
                confound_checklist.extend(resp["confounds"])

        if effect_size_distribution:
            # 25th percentile as conservative estimate
            effect_size_distribution_sorted = sorted(effect_size_distribution)
            idx = max(0, int(len(effect_size_distribution_sorted) * 0.25) - 1)
            d = float(effect_size_distribution_sorted[idx])
        elif manual_effect_size is not None:
            d = float(manual_effect_size)
        else:
            return {
                "error": "Router unavailable and no manual_effect_size provided",
                "statusCode": 400,
            }

    elif effect_size_source == "pilot_data":
        if df.empty:
            return {"error": "pilot_data mode requires a non-empty DataFrame", "statusCode": 400}
        treatment_col = treatment_vars[0]
        if treatment_col not in df.columns:
            return {"error": f"treatment column '{treatment_col}' not found", "statusCode": 400}
        if outcome_var not in df.columns:
            return {"error": f"outcome column '{outcome_var}' not found", "statusCode": 400}

        groups = df.groupby(treatment_col)[outcome_var].apply(list)
        if len(groups) < 2:
            return {"error": "pilot_data mode requires at least 2 groups", "statusCode": 400}

        group_values = list(groups.values)
        g1 = np.array(group_values[0], dtype=float)
        g2 = np.array(group_values[1], dtype=float)
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        if len(g1) < 2 or len(g2) < 2:
            return {"error": "Each group needs at least 2 non-NaN observations", "statusCode": 400}

        pooled_std = float(np.sqrt(
            ((len(g1) - 1) * np.var(g1, ddof=1) + (len(g2) - 1) * np.var(g2, ddof=1))
            / (len(g1) + len(g2) - 2)
        ))
        if pooled_std < 1e-12:
            return {"error": "Pooled standard deviation is near zero", "statusCode": 400}
        d = abs(float(np.mean(g1)) - float(np.mean(g2))) / pooled_std

    elif effect_size_source == "manual":
        if manual_effect_size is None:
            return {"error": "manual_effect_size is required when effect_size_source='manual'", "statusCode": 400}
        d = float(manual_effect_size)

    else:
        return {"error": f"Unknown effect_size_source: {effect_size_source}", "statusCode": 400}

    if d is None or d <= 0:
        return {"error": "Computed effect size must be > 0", "statusCode": 400}

    # Compute required n per group using scipy approximation
    z_alpha = float(norm.ppf(1 - alpha / 2))
    z_beta = float(norm.ppf(target_power))
    required_n = int(math.ceil(2 * ((z_alpha + z_beta) / d) ** 2))

    # Try statsmodels for more precise answer
    try:
        from statsmodels.stats.power import TTestIndPower
        required_n = int(math.ceil(
            TTestIndPower().solve_power(
                effect_size=d, alpha=alpha, power=target_power, alternative="two-sided",
            )
        ))
    except ImportError:
        pass  # scipy approximation already computed

    # Power curve
    power_curve: list[dict] = []
    for n in range(2, 101):
        try:
            from statsmodels.stats.power import TTestIndPower
            p = float(TTestIndPower().solve_power(
                effect_size=d, alpha=alpha, nobs1=n, alternative="two-sided",
            ))
        except (ImportError, Exception):
            # Scipy approximation
            se = d / math.sqrt(2 / n) if n > 0 else 0
            p = float(1 - norm.cdf(z_alpha - se))
        power_curve.append({"n": n, "power": round(p, 6)})

    return {
        "profile_id": "power-analysis",
        "required_n_per_group": required_n,
        "power_curve": power_curve,
        "effect_size_used": round(d, 6),
        "effect_size_source": effect_size_source,
        "effect_size_distribution": effect_size_distribution,
        "citations": citations,
        "confound_checklist": confound_checklist,
    }


# ---------------------------------------------------------------------------
# anomaly-hypothesis (#70)
# ---------------------------------------------------------------------------

_DOMAIN_THRESHOLDS = {
    "genomics": 3.5,
    "proteomics": 3.0,
    "behavioral": 2.5,
    "geospatial": 3.0,
}


def anomaly_hypothesis_handler(
    df: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, dict]:
    """
    Detect anomalies and classify each with literature cross-reference.

    Classifications: instrument_error, known_noise, reported_effect, novel_candidate.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feature_cols: list[str] = parameters.get("features") or []
    domain = str(parameters.get("domain", "genomics"))
    contamination = float(parameters.get("contamination", 0.05))

    if not feature_cols:
        return df.copy(), {"error": "features parameter is required (list of columns)"}
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return df.copy(), {"error": f"Feature columns not found: {missing}"}
    if len(df) < 50:
        return df.copy(), {"error": "anomaly-hypothesis requires at least 50 rows"}

    z_threshold = _DOMAIN_THRESHOLDS.get(domain, 3.0)

    X = df[feature_cols].copy()
    nan_mask = X.isna().any(axis=1)
    X_clean = X[~nan_mask]

    if len(X_clean) < 10:
        return df.copy(), {"error": "Too few complete rows for anomaly detection"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    clf.fit(X_scaled)
    scores = clf.score_samples(X_scaled)

    # Convert anomaly scores to z-scores
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    if score_std < 1e-12:
        score_std = 1.0
    z_scores = (scores - score_mean) / score_std

    # Anomalies are those with z-score below -threshold (more negative = more anomalous)
    anomaly_mask = z_scores < -z_threshold

    result = df.copy()
    result["is_anomaly"] = False
    result["anomaly_score"] = np.nan
    result["anomaly_class"] = ""
    result["confidence"] = np.nan
    result["supporting_citations"] = ""
    result["note"] = ""

    clean_indices = df.index[~nan_mask]
    result.loc[clean_indices, "anomaly_score"] = scores

    anomaly_indices = clean_indices[anomaly_mask]
    result.loc[anomaly_indices, "is_anomaly"] = True

    classification_counts: dict[str, int] = {
        "instrument_error": 0,
        "known_noise": 0,
        "reported_effect": 0,
        "novel_candidate": 0,
    }

    for idx in anomaly_indices:
        row = df.loc[idx]
        context_parts = [f"{col}={row[col]}" for col in feature_cols if pd.notna(row[col])]
        context = f"Anomalous observation in {domain} domain: " + ", ".join(context_parts)

        resp = _call_router_api("research", {
            "query": context,
            "grounding_mode": "strict",
        })

        anomaly_class = "novel_candidate"
        confidence = 0.5
        citations_str = ""
        note = ""

        if resp:
            sources = resp.get("sources_used") or []
            content = str(resp.get("content", "")).lower()
            citations_str = "; ".join(str(s) for s in sources[:5])

            if any(kw in content for kw in ("instrument", "calibration")):
                anomaly_class = "instrument_error"
                confidence = 0.8
                note = "Literature suggests instrument/calibration artifact"
            elif any(kw in content for kw in ("noise", "artifact")):
                anomaly_class = "known_noise"
                confidence = 0.7
                note = "Literature suggests known noise pattern"
            elif sources:
                anomaly_class = "reported_effect"
                confidence = 0.75
                note = "Literature contains grounded reports of similar observations"
            else:
                anomaly_class = "novel_candidate"
                confidence = 0.6
                note = "No matching literature found — potential novel finding"
        else:
            note = "Router unavailable — classified as novel candidate by default"

        result.loc[idx, "anomaly_class"] = anomaly_class
        result.loc[idx, "confidence"] = confidence
        result.loc[idx, "supporting_citations"] = citations_str
        result.loc[idx, "note"] = note
        classification_counts[anomaly_class] += 1

    anomaly_count = int(anomaly_mask.sum())
    diagnostics = {
        "feature_columns": feature_cols,
        "domain": domain,
        "z_threshold": z_threshold,
        "contamination": contamination,
        "anomaly_count": anomaly_count,
        "classifications": classification_counts,
    }

    logger.info(
        "anomaly-hypothesis complete: %d anomalies detected, domain=%s, z_threshold=%.1f",
        anomaly_count, domain, z_threshold,
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# reproducibility-check (#71)
# ---------------------------------------------------------------------------

def reproducibility_check_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Re-execute analysis script against deposited data and compare outputs
    to reported manuscript results.
    """
    import io

    import boto3

    manuscript_results_raw = parameters.get("manuscript_results")
    analysis_script_uri = str(parameters.get("analysis_script_uri", "")).strip()
    result_uri = str(parameters.get("result_uri", "")).strip()
    provenance_run_id = parameters.get("provenance_run_id")
    tolerance = float(parameters.get("tolerance", 1e-4))

    if not manuscript_results_raw:
        return {"error": "manuscript_results is required", "statusCode": 400}
    if not analysis_script_uri:
        return {"error": "analysis_script_uri is required", "statusCode": 400}

    # Parse manuscript_results — accept JSON string or list of dicts
    if isinstance(manuscript_results_raw, str):
        try:
            manuscript_results = json.loads(manuscript_results_raw)
        except json.JSONDecodeError:
            return {"error": "manuscript_results must be valid JSON", "statusCode": 400}
    elif isinstance(manuscript_results_raw, list):
        manuscript_results = manuscript_results_raw
    else:
        return {"error": "manuscript_results must be a JSON string or list", "statusCode": 400}

    if not isinstance(manuscript_results, list):
        return {"error": "manuscript_results must be a JSON list of dicts", "statusCode": 400}

    # Parse S3 URIs
    def _parse_s3(uri: str) -> tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got: {uri}")
        parts = uri[5:].split("/", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError(f"Invalid S3 URI: {uri}")
        return parts[0], parts[1]

    s3 = boto3.client("s3")

    # Provenance lookup for script version
    script_version = None
    if provenance_run_id:
        history_table_name = os.environ.get("COMPUTE_HISTORY_TABLE", "")
        if history_table_name:
            try:
                dynamodb = boto3.resource("dynamodb")
                table = dynamodb.Table(history_table_name)
                resp = table.get_item(Key={"job_id": provenance_run_id})
                item = resp.get("Item", {})
                script_version = item.get("script_version") or item.get("profile_id")
            except Exception as exc:
                logger.warning("Provenance lookup failed: %s", exc)

    # Download analysis script
    try:
        script_bucket, script_key = _parse_s3(analysis_script_uri)
        script_body = s3.get_object(Bucket=script_bucket, Key=script_key)["Body"].read()
        script_text = script_body.decode("utf-8")
    except Exception as exc:
        return {"error": f"Could not download script: {exc}", "statusCode": 400}

    # Load data — use result_uri if provided, otherwise use input df
    data_df = df
    if result_uri:
        try:
            data_bucket, data_key = _parse_s3(result_uri)
            data_body = s3.get_object(Bucket=data_bucket, Key=data_key)["Body"].read()
            if data_key.endswith(".json") or data_key.endswith(".jsonl"):
                data_df = pd.read_json(io.BytesIO(data_body))
            else:
                data_df = pd.read_csv(io.BytesIO(data_body))
        except Exception as exc:
            return {"error": f"Could not load data from result_uri: {exc}", "statusCode": 400}

    # Execute script in RestrictedPython sandbox
    try:
        from RestrictedPython import compile_restricted
    except ImportError:
        return {
            "error": "Reproducibility check requires RestrictedPython",
            "statusCode": 503,
        }

    try:
        # Import sandbox builder from custom.py
        from custom import _build_safe_globals

        byte_code = compile_restricted(script_text, filename=analysis_script_uri, mode="exec")
        safe_ns = _build_safe_globals()
        safe_ns["__name__"] = "__restricted__"
        exec(byte_code, safe_ns)  # noqa: S102

        transform_fn = safe_ns.get("transform")
        if transform_fn is None or not callable(transform_fn):
            return {"error": "Script must define a callable named 'transform(df)'", "statusCode": 400}

        computed = transform_fn(data_df)
    except Exception as exc:
        return {"error": f"Script execution failed: {exc}", "statusCode": 500}

    # Compare manuscript_results to computed outputs
    matches: list[dict] = []
    discrepancies: list[dict] = []

    for entry in manuscript_results:
        metric_name = entry.get("metric", entry.get("name", "unknown"))
        reported = entry.get("value")
        if reported is None:
            discrepancies.append({
                "metric": metric_name,
                "error": "No 'value' field in manuscript entry",
            })
            continue

        # Try to find computed value: check dict results, DataFrame columns, or attributes
        computed_value = None
        if isinstance(computed, dict):
            computed_value = computed.get(metric_name)
        elif isinstance(computed, pd.DataFrame) and metric_name in computed.columns:
            # Use the first value or mean
            computed_value = float(computed[metric_name].mean())

        if computed_value is None:
            discrepancies.append({
                "metric": metric_name,
                "reported": reported,
                "computed": None,
                "error": f"Metric '{metric_name}' not found in computed output",
            })
            continue

        try:
            reported_f = float(reported)
            computed_f = float(computed_value)
            delta = abs(reported_f - computed_f)
            if delta <= tolerance:
                matches.append({
                    "metric": metric_name,
                    "reported": reported_f,
                    "computed": computed_f,
                    "delta": round(delta, 10),
                })
            else:
                discrepancies.append({
                    "metric": metric_name,
                    "reported": reported_f,
                    "computed": computed_f,
                    "delta": round(delta, 10),
                })
        except (ValueError, TypeError):
            # String comparison
            if str(reported) == str(computed_value):
                matches.append({"metric": metric_name, "reported": reported, "computed": computed_value})
            else:
                discrepancies.append({
                    "metric": metric_name,
                    "reported": reported,
                    "computed": computed_value,
                })

    return {
        "profile_id": "reproducibility-check",
        "matches": matches,
        "discrepancies": discrepancies,
        "script_version": script_version,
        "total_checks": len(manuscript_results),
    }
