"""
geo_enrich.py — Geographic Enrichment (Census) profile

Appends US Census Bureau ACS 5-year demographic variables using lat/lon
or ZIP codes. Uses the Census Bureau public API (no key required for basic use).

Called by runner/handler.py via the entrypoint "geo_enrich.geo_enrich_handler".
"""

import json
import logging
import time
import urllib.request
from typing import Any
from urllib.parse import quote, urlencode

import pandas as pd

logger = logging.getLogger(__name__)

_CENSUS_API_BASE = "https://geocoding.geo.census.gov/geocoder"

# Default ACS 5-year variable mappings
_ACS_VARIABLE_MAP = {
    "median_income": "B19013_001E",
    "population": "B01003_001E",
    "pct_bachelor_plus": "B15003_022E",
    "median_age": "B01002_001E",
    "pct_poverty": "B17001_002E",
}


def geo_enrich_handler(
    df: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, dict]:
    """
    Append Census demographic variables to each row using lat/lon or ZIP.

    Returns (result_df, diagnostics).
    """
    lat_col = (parameters.get("lat_column") or "").strip() or None
    lon_col = (parameters.get("lon_column") or "").strip() or None
    zip_col = (parameters.get("zip_column") or "").strip() or None
    enrichment_source = parameters.get("enrichment_source", "acs5")
    variables = parameters.get("variables") or ["median_income", "population", "pct_bachelor_plus"]

    MAX_ROWS = 8_000
    if len(df) > MAX_ROWS:
        raise ValueError(
            f"Geographic enrichment is limited to {MAX_ROWS} rows due to Census API "
            f"rate limits (received {len(df)}). Filter your dataset to a sample first."
        )

    if df.empty:
        result = df.copy()
        for v in variables:
            result[v] = None
        result["census_tract"] = None
        return result, {"warning": "Input dataset is empty (extract step stub)"}

    if not (lat_col and lon_col) and not zip_col:
        raise ValueError(
            "Either lat_column + lon_column, or zip_column, is required"
        )

    if lat_col and lat_col not in df.columns:
        raise ValueError(f"Lat column '{lat_col}' not found in dataset")
    if lon_col and lon_col not in df.columns:
        raise ValueError(f"Lon column '{lon_col}' not found in dataset")
    if zip_col and zip_col not in df.columns:
        raise ValueError(f"ZIP column '{zip_col}' not found in dataset")

    result = df.copy()

    # Initialize output columns
    result["census_tract"] = None
    result["geo_state_fips"] = None
    result["geo_county_fips"] = None
    for v in variables:
        result[v] = None

    enriched_count = 0
    skipped_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        try:
            geo = _geocode_row(row, lat_col, lon_col, zip_col)
            if geo is None:
                skipped_count += 1
                continue

            result.at[idx, "census_tract"] = geo.get("tract")
            result.at[idx, "geo_state_fips"] = geo.get("state")
            result.at[idx, "geo_county_fips"] = geo.get("county")

            if enrichment_source == "acs5" and geo.get("state") and geo.get("county"):
                acs_data = _fetch_acs5(
                    geo["state"], geo["county"], geo.get("tract", "*"), variables
                )
                for v in variables:
                    result.at[idx, v] = acs_data.get(v)

            enriched_count += 1
            # Rate-limit to ~10 req/sec (Census API limit)
            time.sleep(0.1)

        except Exception as exc:
            logger.warning(f"Geo enrichment failed for row {idx}: {exc}")
            error_count += 1

    diagnostics = {
        "lat_column": lat_col,
        "lon_column": lon_col,
        "zip_column": zip_col,
        "enrichment_source": enrichment_source,
        "variables": variables,
        "rows_enriched": enriched_count,
        "rows_skipped": skipped_count,
        "rows_with_errors": error_count,
    }

    logger.info(
        f"Geo enrichment complete: enriched={enriched_count}, "
        f"skipped={skipped_count}, errors={error_count}"
    )
    return result, diagnostics


def _geocode_row(row, lat_col, lon_col, zip_col) -> dict | None:
    """Return {'state': '06', 'county': '037', 'tract': '123456'} or None."""
    if lat_col and lon_col:
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        if pd.isna(lat) or pd.isna(lon):
            return None
        return _geocode_latlon(float(lat), float(lon))
    elif zip_col:
        zip_code = str(row.get(zip_col, "")).strip().zfill(5)[:5]
        if not zip_code or zip_code == "00000":
            return None
        return _geocode_zip(zip_code)
    return None


def _geocode_latlon(lat: float, lon: float) -> dict | None:
    """Reverse-geocode lat/lon to Census geographies."""
    url = (
        f"{_CENSUS_API_BASE}/locations/coordinates"
        f"?x={lon}&y={lat}&benchmark=Public_AR_Current"
        f"&vintage=Current_Current&format=json&layers=Census+Tracts"
    )
    data = _get_json(url)
    try:
        geo = data["result"]["geographies"]["Census Tracts"][0]
        return {
            "state": geo["STATE"],
            "county": geo["COUNTY"],
            "tract": geo["TRACT"],
        }
    except (KeyError, IndexError, TypeError):
        return None


def _geocode_zip(zip_code: str) -> dict | None:
    """Convert ZIP to Census state/county using Tiger geocoder."""
    url = (
        f"{_CENSUS_API_BASE}/locations/address?"
        + urlencode({"address": zip_code, "benchmark": "Public_AR_Current",
                     "format": "json"})
    )
    data = _get_json(url)
    try:
        matches = data["result"]["addressMatches"]
        if not matches:
            return None
        coords = matches[0]["coordinates"]
        return _geocode_latlon(coords["y"], coords["x"])
    except (KeyError, IndexError, TypeError):
        return None


def _fetch_acs5(state: str, county: str, tract: str, variables: list[str]) -> dict:
    """Fetch ACS 5-year estimates for a tract."""
    census_vars = [_ACS_VARIABLE_MAP.get(v, v) for v in variables]
    safe_vars = quote(",".join(census_vars), safe=",")
    url = (
        f"https://api.census.gov/data/2022/acs/acs5"
        f"?get={safe_vars}"
        f"&for=tract:{quote(str(tract), safe='')}"
        f"&in=state:{quote(str(state), safe='')}+county:{quote(str(county), safe='')}"
    )
    data = _get_json(url)
    if not data or len(data) < 2:
        return {}
    headers = data[0]
    values = data[1]
    raw = dict(zip(headers, values))
    return {v: _safe_float(raw.get(_ACS_VARIABLE_MAP.get(v, v))) for v in variables}


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if f >= 0 else None
    except (TypeError, ValueError):
        return None


def _get_json(url: str) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": "QuickSuiteCompute/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())
