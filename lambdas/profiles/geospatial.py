"""
geospatial.py — Geospatial analysis profiles

spatial_aggregate_handler  — Point-in-polygon rollup (spatial-aggregate)
isochrone_handler          — Travel-time catchment via haversine distance (isochrone)

Called by runner/handler.py via entrypoints in config/profiles/*.json.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: float, lon2: float) -> np.ndarray:
    """Vectorised haversine distance from each (lat1, lon1) to a single (lat2, lon2)."""
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# spatial-aggregate
# ---------------------------------------------------------------------------

def spatial_aggregate_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Assign each lat/lon point to a polygon from a GeoJSON boundary file,
    then aggregate specified metrics by polygon.

    Downloads the GeoJSON from S3 `boundary_uri`. Requires shapely.

    Returns the original DataFrame annotated with polygon_id, polygon_label,
    plus a polygon-level summary table in diagnostics.
    """
    import boto3
    from shapely.geometry import Point, shape

    lat_col = parameters.get("lat_column", "")
    lon_col = parameters.get("lon_column", "")
    boundary_uri = parameters.get("boundary_uri", "")
    metric_cols = parameters.get("metric_columns") or []
    agg_functions = parameters.get("agg_functions") or ["count", "mean"]
    label_property = parameters.get("label_column") or "name"

    if not lat_col or lat_col not in df.columns:
        raise ValueError(f"lat_column '{lat_col}' not found")
    if not lon_col or lon_col not in df.columns:
        raise ValueError(f"lon_column '{lon_col}' not found")
    if not boundary_uri:
        raise ValueError("boundary_uri is required (S3 path to GeoJSON file)")
    if isinstance(metric_cols, str):
        metric_cols = [metric_cols]
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"metric_columns not found: {missing}")

    # Validate agg functions
    valid_aggs = {"count", "mean", "sum", "median", "min", "max"}
    bad = [a for a in agg_functions if a not in valid_aggs]
    if bad:
        raise ValueError(f"agg_functions must be subset of {sorted(valid_aggs)}, got: {bad}")

    # Load GeoJSON from S3
    if not boundary_uri.startswith("s3://"):
        raise ValueError("boundary_uri must be an s3:// path")
    bucket, key = boundary_uri[len("s3://"):].split("/", 1)
    s3 = boto3.client("s3")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        geojson = json.loads(body)
    except Exception as exc:
        raise ValueError(f"Could not load GeoJSON from {boundary_uri}: {exc}") from exc

    features = geojson.get("features", [])
    if not features:
        raise ValueError(f"GeoJSON at {boundary_uri} has no features")

    polygons = []
    for i, feat in enumerate(features):
        props = feat.get("properties") or {}
        geom = feat.get("geometry")
        if geom is None:
            continue
        try:
            poly = shape(geom)
        except Exception:
            continue
        polygons.append({
            "polygon_id": i,
            "polygon_label": str(props.get(label_property, f"polygon_{i}")),
            "shape": poly,
        })

    if not polygons:
        raise ValueError("No valid polygon geometries found in GeoJSON")

    # Point-in-polygon assignment
    result = df.copy()
    result["polygon_id"] = -1
    result["polygon_label"] = None

    lats = pd.to_numeric(result[lat_col], errors="coerce")
    lons = pd.to_numeric(result[lon_col], errors="coerce")
    valid_mask = lats.notna() & lons.notna()

    n_assigned = 0
    for idx in result[valid_mask].index:
        pt = Point(float(lons.loc[idx]), float(lats.loc[idx]))  # shapely uses (lon, lat)
        for poly_info in polygons:
            if poly_info["shape"].contains(pt):
                result.loc[idx, "polygon_id"] = poly_info["polygon_id"]
                result.loc[idx, "polygon_label"] = poly_info["polygon_label"]
                n_assigned += 1
                break

    # Aggregate metrics by polygon
    agg_map: dict[str, str | list] = {}
    for col in metric_cols:
        if col in result.columns:
            col_aggs = [a for a in agg_functions if a != "count"]
            if col_aggs:
                agg_map[col] = col_aggs

    if agg_map and n_assigned > 0:
        assigned = result[result["polygon_id"] >= 0]
        agg_df = assigned.groupby(["polygon_id", "polygon_label"]).agg(
            n=("polygon_id", "count"),
            **{f"{col}_{agg}": (col, agg)
               for col, aggs in agg_map.items()
               for agg in aggs}
        ).reset_index()
        summary = agg_df.to_dict(orient="records")
    else:
        count_df = result[result["polygon_id"] >= 0].groupby(
            ["polygon_id", "polygon_label"]
        ).size().reset_index(name="n")
        summary = count_df.to_dict(orient="records")

    diagnostics = {
        "lat_column": lat_col,
        "lon_column": lon_col,
        "boundary_uri": boundary_uri,
        "n_polygons": len(polygons),
        "n_points_total": int(valid_mask.sum()),
        "n_points_assigned": n_assigned,
        "n_points_unassigned": int(valid_mask.sum()) - n_assigned,
        "polygon_summary": summary,
    }

    logger.info(
        f"spatial-aggregate complete: {n_assigned}/{int(valid_mask.sum())} points assigned "
        f"across {len(polygons)} polygons"
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# isochrone
# ---------------------------------------------------------------------------

def isochrone_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Assign each lat/lon record to the nearest origin point within a maximum
    distance threshold (straight-line haversine; no routing API required).

    Downloads a CSV of origin points from S3 `origins_uri`. The origins CSV
    must have columns: lat, lon, and a label column (default: 'label').

    Returns the original DataFrame with nearest_origin, distance_km, and
    within_catchment columns appended.
    """
    import boto3

    lat_col = parameters.get("lat_column", "")
    lon_col = parameters.get("lon_column", "")
    origins_uri = parameters.get("origins_uri", "")
    max_distance_km = float(parameters.get("max_distance_km", 80.0))
    label_col_name = parameters.get("label_column", "label")

    if not lat_col or lat_col not in df.columns:
        raise ValueError(f"lat_column '{lat_col}' not found")
    if not lon_col or lon_col not in df.columns:
        raise ValueError(f"lon_column '{lon_col}' not found")
    if not origins_uri:
        raise ValueError("origins_uri is required (S3 path to origins CSV)")

    # Load origins from S3
    if not origins_uri.startswith("s3://"):
        raise ValueError("origins_uri must be an s3:// path")
    bucket, key = origins_uri[len("s3://"):].split("/", 1)
    s3 = boto3.client("s3")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        origins = pd.read_csv(pd.io.common.BytesIO(body))
    except Exception as exc:
        raise ValueError(f"Could not load origins CSV from {origins_uri}: {exc}") from exc

    for required in ("lat", "lon"):
        if required not in origins.columns:
            raise ValueError(f"Origins CSV must have a '{required}' column")
    if label_col_name not in origins.columns:
        raise ValueError(f"Origins CSV must have a '{label_col_name}' column")

    lats = pd.to_numeric(df[lat_col], errors="coerce").values
    lons = pd.to_numeric(df[lon_col], errors="coerce").values
    valid_mask = ~(np.isnan(lats) | np.isnan(lons))

    nearest_origin = np.full(len(df), None, dtype=object)
    min_distance = np.full(len(df), np.inf)

    for _, origin in origins.iterrows():
        olat = float(origin["lat"])
        olon = float(origin["lon"])
        olabel = str(origin[label_col_name])

        dist = np.where(
            valid_mask,
            _haversine_km(lats, lons, olat, olon),
            np.inf,
        )
        closer = dist < min_distance
        min_distance = np.where(closer, dist, min_distance)
        nearest_origin = np.where(closer, olabel, nearest_origin)

    result = df.copy()
    result["nearest_origin"] = nearest_origin
    result["distance_km"] = np.where(np.isinf(min_distance), np.nan, min_distance.round(3))
    result["within_catchment"] = (min_distance <= max_distance_km) & valid_mask

    n_within = int(result["within_catchment"].sum())
    n_valid = int(valid_mask.sum())

    diagnostics = {
        "lat_column": lat_col,
        "lon_column": lon_col,
        "origins_uri": origins_uri,
        "n_origins": len(origins),
        "max_distance_km": max_distance_km,
        "n_records": len(df),
        "n_valid_coordinates": n_valid,
        "n_within_catchment": n_within,
        "pct_within_catchment": round(n_within / n_valid, 4) if n_valid else 0,
        "origins": origins[[label_col_name, "lat", "lon"]].to_dict(orient="records"),
    }

    logger.info(
        f"isochrone complete: {n_within}/{n_valid} within {max_distance_km}km "
        f"of {len(origins)} origins"
    )
    return result, diagnostics
