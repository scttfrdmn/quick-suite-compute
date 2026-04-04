"""
research.py — Research analytics profiles

grant_portfolio_handler  — Grant burn rate, NCE risk, PI productivity (grant-portfolio)
coauthor_network_handler — Co-authorship network centrality and community detection (network-coauthor)

Called by runner/handler.py via entrypoints in config/profiles/*.json.
"""

from __future__ import annotations

import logging
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
    import networkx as nx
    import community as community_louvain

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
