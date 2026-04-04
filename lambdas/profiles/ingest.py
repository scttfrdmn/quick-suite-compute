"""
ingest.py — Data ingest profiles for non-native formats

netcdf_handler      — NetCDF4 → tabular DataFrame (ingest-netcdf)
pdf_extract_handler — PDF text extraction, one row per page (ingest-pdf-extract)
geojson_handler     — GeoJSON FeatureCollection → tabular with WKT geometry (ingest-geojson)

All three handlers download their source file from S3 via the `source_uri` parameter
and ignore the standard input DataFrame (which will be empty for ingest jobs).

Called by runner/handler.py via entrypoints in config/profiles/*.json.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse 's3://bucket/key' into (bucket, key). Raises ValueError on bad format."""
    if not uri.startswith("s3://"):
        raise ValueError(f"source_uri must be an s3:// path (got '{uri}')")
    parts = uri[len("s3://"):].split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"source_uri must be s3://bucket/key (got '{uri}')")
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# ingest-netcdf
# ---------------------------------------------------------------------------

def netcdf_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Convert a NetCDF4 file stored in S3 to a tabular Parquet-friendly DataFrame.

    Downloads the .nc file to a temporary path (xarray requires a file path,
    not a file-like object), opens it with xarray, subsets variables if requested,
    and flattens all dimensions to rows via .to_dataframe().reset_index().

    The input DataFrame `df` is ignored — this handler operates entirely from
    the `source_uri` parameter.
    """
    import boto3
    import xarray as xr

    source_uri = parameters.get("source_uri", "")
    variables = parameters.get("variables")  # list or None
    if isinstance(variables, str):
        variables = [variables]

    bucket, key = _parse_s3_uri(source_uri)

    # Download to a named temp file (xarray needs a real path)
    s3 = boto3.client("s3")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:
        raise ValueError(f"Could not download NetCDF from {source_uri}: {exc}") from exc

    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    try:
        tmp.write(body)
        tmp.flush()
        tmp.close()

        try:
            ds = xr.open_dataset(tmp.name)
        except Exception as exc:
            raise ValueError(f"Could not parse NetCDF file {source_uri}: {exc}") from exc

        # Subset variables
        all_vars = list(ds.data_vars)
        if variables:
            missing = [v for v in variables if v not in all_vars]
            if missing:
                raise ValueError(f"Variables not found in dataset: {missing}. Available: {all_vars}")
            ds = ds[variables]
            vars_extracted = variables
        else:
            vars_extracted = all_vars

        dims = {k: int(v) for k, v in ds.dims.items()}
        global_attrs = dict(list(ds.attrs.items())[:10])  # cap at 10 attrs

        try:
            result_df = ds.to_dataframe().reset_index()
        except Exception as exc:
            ds.close()
            raise ValueError(f"Failed to flatten NetCDF dataset: {exc}") from exc

        ds.close()

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    n_before = len(result_df)
    result_df = result_df.dropna(how="all", subset=[c for c in result_df.columns
                                                     if c not in list(dims.keys())])
    n_after = len(result_df)
    result_df = result_df.reset_index(drop=True)

    diagnostics = {
        "source_uri": source_uri,
        "variables_extracted": vars_extracted,
        "dimensions": dims,
        "n_rows_before_dropna": n_before,
        "n_rows_after_dropna": n_after,
        "global_attrs": {str(k): str(v) for k, v in global_attrs.items()},
    }

    logger.info(
        f"ingest-netcdf complete: {n_after} rows extracted "
        f"({n_before - n_after} dropped), vars={vars_extracted}"
    )
    return result_df, diagnostics


# ---------------------------------------------------------------------------
# ingest-pdf-extract
# ---------------------------------------------------------------------------

def pdf_extract_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Extract text from a PDF file stored in S3, producing one row per page.

    Downloads the PDF bytes from S3 and parses with pypdf. Filters out pages
    shorter than `min_page_length` characters (near-blank pages, headers-only, etc.).

    The input DataFrame `df` is ignored — this handler operates entirely from
    the `source_uri` parameter.
    """
    import boto3
    import pypdf

    source_uri = parameters.get("source_uri", "")
    include_page_numbers = bool(parameters.get("include_page_numbers", True))
    min_page_length = int(parameters.get("min_page_length", 50))

    bucket, key = _parse_s3_uri(source_uri)

    s3 = boto3.client("s3")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:
        raise ValueError(f"Could not download PDF from {source_uri}: {exc}") from exc

    try:
        reader = pypdf.PdfReader(BytesIO(body))
    except Exception as exc:
        raise ValueError(f"Could not parse PDF {source_uri}: {exc}") from exc

    total_pages = len(reader.pages)
    pdf_meta = {}
    if reader.metadata:
        for k, v in reader.metadata.items():
            if isinstance(v, str):
                pdf_meta[str(k).lstrip("/")] = v

    rows = []
    n_filtered = 0
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        char_count = len(text)
        if char_count < min_page_length:
            n_filtered += 1
            continue
        row: dict[str, Any] = {"text": text, "char_count": char_count}
        if include_page_numbers:
            row["page_number"] = i + 1
        rows.append(row)

    if not rows:
        raise ValueError(
            f"No pages extracted from {source_uri} after filtering "
            f"(total={total_pages}, filtered={n_filtered}, min_length={min_page_length})"
        )

    result_df = pd.DataFrame(rows)
    if include_page_numbers:
        result_df = result_df[["page_number", "text", "char_count"]]

    avg_chars = round(float(result_df["char_count"].mean()), 1)
    diagnostics = {
        "source_uri": source_uri,
        "total_pages": total_pages,
        "pages_extracted": len(rows),
        "pages_filtered_short": n_filtered,
        "avg_chars_per_page": avg_chars,
        "pdf_metadata": pdf_meta,
    }

    logger.info(
        f"ingest-pdf-extract complete: {len(rows)}/{total_pages} pages extracted "
        f"(filtered={n_filtered})"
    )
    return result_df, diagnostics


# ---------------------------------------------------------------------------
# ingest-geojson
# ---------------------------------------------------------------------------

def geojson_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Convert a GeoJSON FeatureCollection stored in S3 to a tabular DataFrame.

    Each GeoJSON feature becomes one row. All feature properties become columns.
    The geometry is converted to a WKT string via shapely.

    The input DataFrame `df` is ignored — this handler operates entirely from
    the `source_uri` parameter.
    """
    import boto3
    from shapely.geometry import shape

    source_uri = parameters.get("source_uri", "")
    geometry_column = parameters.get("geometry_column", "geometry_wkt")
    include_bbox = bool(parameters.get("include_bbox", False))

    bucket, key = _parse_s3_uri(source_uri)

    s3 = boto3.client("s3")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        geojson = json.loads(body)
    except Exception as exc:
        raise ValueError(f"Could not load GeoJSON from {source_uri}: {exc}") from exc

    if geojson.get("type") != "FeatureCollection":
        raise ValueError(
            f"Expected a GeoJSON FeatureCollection, got type='{geojson.get('type')}'"
        )

    features = geojson.get("features") or []
    if not features:
        raise ValueError(f"GeoJSON FeatureCollection at {source_uri} has no features")

    # Determine CRS from top-level field if present
    crs = None
    if "crs" in geojson:
        crs = geojson["crs"].get("properties", {}).get("name") or str(geojson["crs"])

    rows = []
    geometry_types: set[str] = set()
    n_null_geometries = 0
    all_property_keys: list[str] = []

    for feat in features:
        props = dict(feat.get("properties") or {})
        geom_obj = feat.get("geometry")

        # Track all property keys (maintain insertion order)
        for k in props:
            if k not in all_property_keys:
                all_property_keys.append(k)

        # Convert geometry to WKT
        wkt = None
        if geom_obj is not None:
            geometry_types.add(geom_obj.get("type", "Unknown"))
            try:
                geom = shape(geom_obj)
                wkt = geom.wkt
                if include_bbox:
                    bounds = geom.bounds  # (minx, miny, maxx, maxy)
                    props["bbox_minx"] = bounds[0]
                    props["bbox_miny"] = bounds[1]
                    props["bbox_maxx"] = bounds[2]
                    props["bbox_maxy"] = bounds[3]
            except Exception as exc:
                logger.warning(f"shapely failed on geometry: {exc}")
                n_null_geometries += 1
        else:
            n_null_geometries += 1

        props[geometry_column] = wkt
        rows.append(props)

    result_df = pd.DataFrame(rows)
    # Reorder: property columns first, geometry last
    ordered_cols = [c for c in all_property_keys if c in result_df.columns]
    if include_bbox:
        ordered_cols += [c for c in ["bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"]
                         if c in result_df.columns]
    if geometry_column in result_df.columns:
        ordered_cols.append(geometry_column)
    result_df = result_df[ordered_cols]

    diagnostics = {
        "source_uri": source_uri,
        "n_features": len(features),
        "geometry_types": sorted(geometry_types),
        "property_columns": all_property_keys,
        "n_null_geometries": n_null_geometries,
        "crs": crs,
    }

    logger.info(
        f"ingest-geojson complete: {len(features)} features, "
        f"types={sorted(geometry_types)}, null_geoms={n_null_geometries}"
    )
    return result_df, diagnostics
