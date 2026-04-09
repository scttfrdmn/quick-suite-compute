"""
runner — Step Functions Lambda

Single dispatcher Lambda. Receives the profile_id and full execution context,
reads the input Parquet from S3, dispatches to the appropriate profile module,
writes the result Parquet back to S3, and returns result metadata.

Profile modules live in lambdas/profiles/ (bundled into the Lambda Layer
alongside heavy deps). Each module exports a function matching the profile's
entrypoint field (e.g. "clustering.kmeans_handler").

Input (from Step Functions — full context):
  {
    "execution_id": str,
    "profile": {"profile_id": str, "entrypoint": str, ...},
    "parameters": {...},
    "extract": {"input_s3_uri": str, ...},
    ...
  }

Output:
  {
    "execution_id": str,
    "result_s3_uri": str,
    "metadata_s3_uri": str,
    "row_count": int,
    "columns": list[str],
    "actual_cost_usd": float
  }
"""

import importlib
import json
import logging
import os
import time

import boto3
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key) from s3://bucket/key."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _read_input(input_s3_uri: str) -> pd.DataFrame:
    """Read Parquet, CSV, or JSON stub from S3 into a DataFrame."""
    bucket, key = _parse_s3_uri(input_s3_uri)

    # CSV path (from real extract step implementation)
    csv_key = key.replace(".parquet", ".csv")
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=csv_key)
        return pd.read_csv(resp["Body"])
    except s3_client.exceptions.NoSuchKey:
        pass

    # Check for stub JSON (from extract step stub implementation)
    json_key = key.replace(".parquet", ".json")
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=json_key)
        stub = json.loads(resp["Body"].read())
        logger.warning(json.dumps({"warning": "input_is_stub", "stub": stub}))
        # Return empty DataFrame for now; profiles handle gracefully
        return pd.DataFrame()
    except s3_client.exceptions.NoSuchKey:
        pass

    # Real Parquet path
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(resp["Body"])


def _write_result(df: pd.DataFrame, metadata: dict, execution_id: str) -> tuple[str, str]:
    """Write result Parquet and metadata JSON to S3. Returns (data_uri, meta_uri)."""
    compute_bucket = os.environ["COMPUTE_BUCKET"]
    data_key = f"results/{execution_id}/data.parquet"
    meta_key = f"results/{execution_id}/metadata.json"

    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    s3_client.put_object(Bucket=compute_bucket, Key=data_key, Body=buf.getvalue())
    s3_client.put_object(
        Bucket=compute_bucket,
        Key=meta_key,
        Body=json.dumps(metadata).encode(),
        ContentType="application/json",
    )

    return (
        f"s3://{compute_bucket}/{data_key}",
        f"s3://{compute_bucket}/{meta_key}",
    )


def _validate_columns(profile: dict, parameters: dict, available_columns: list) -> list:
    """Return list of missing column values, or [] if all referenced columns exist."""
    column_params = profile.get("input_requirements", {}).get("column_parameters", [])
    missing = []
    for param_name in column_params:
        val = parameters.get(param_name)
        if val is None:
            continue  # param not supplied — skip (handles optional / either-or params)
        cols = val if isinstance(val, list) else [val]
        for col in cols:
            if col and col not in available_columns:
                missing.append(col)
    return missing


def _dispatch(profile: dict, df: pd.DataFrame, parameters: dict) -> tuple[pd.DataFrame, dict]:
    """Dispatch to the profile module and return (result_df, diagnostics)."""
    entrypoint = profile["entrypoint"]  # e.g. "clustering.kmeans_handler"
    module_name, func_name = entrypoint.rsplit(".", 1)

    # Profile modules are on sys.path via Lambda Layer
    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name)
    return func(df, parameters)


def handler(event: dict, context) -> dict:
    logger.info(json.dumps({"event": {
        "execution_id": event.get("execution_id"),
        "profile_id": event.get("profile", {}).get("profile_id"),
    }}))

    execution_id = event["execution_id"]
    profile = event["profile"]
    parameters = event.get("parameters") or {}
    extract = event.get("extract") or {}
    input_s3_uri = extract.get("input_s3_uri", "")

    if not input_s3_uri:
        raise ValueError(
            "extract.input_s3_uri is empty — the extract step may have failed "
            "or returned an unsupported_source error"
        )

    # Validate that column parameters reference columns that actually exist
    available_columns = extract.get("columns") or []
    if available_columns:
        missing = _validate_columns(profile, parameters, available_columns)
        if missing:
            return {
                "status": "validation_error",
                "missing_columns": missing,
                "available_columns": available_columns,
                "execution_id": execution_id,
            }

    start_time = time.monotonic()

    # Read input data
    try:
        df = _read_input(input_s3_uri)
    except Exception as exc:
        logger.error(f"Failed to read input: {exc}")
        raise RuntimeError(f"Failed to read input dataset: {exc}") from exc

    # Dispatch to profile
    try:
        result_df, diagnostics = _dispatch(profile, df, parameters)
    except Exception as exc:
        logger.error(f"Profile execution failed: {exc}")
        raise RuntimeError(f"Analysis failed: {exc}") from exc

    # Coerce plain-dict result (handlers that return {key: val} rather than DataFrame)
    if isinstance(result_df, dict):
        diagnostics = result_df
        result_df = pd.DataFrame()

    elapsed = time.monotonic() - start_time

    # Write results
    row_count = diagnostics.get("row_count", len(result_df))
    metadata = {
        "execution_id": execution_id,
        "profile_id": profile["profile_id"],
        "elapsed_seconds": elapsed,
        "row_count": row_count,
        "columns": diagnostics.get("columns", list(result_df.columns)),
        "diagnostics": diagnostics,
    }

    data_uri, meta_uri = _write_result(result_df, metadata, execution_id)

    # Estimate cost: Lambda at $0.0000166667/GB-second, 3GB memory
    gb_seconds = (elapsed * 3.0)
    actual_cost_usd = gb_seconds * 0.0000166667

    logger.info(json.dumps({
        "execution_id": execution_id,
        "profile_id": profile["profile_id"],
        "row_count": row_count,
        "elapsed_seconds": elapsed,
        "actual_cost_usd": actual_cost_usd,
    }))

    return {
        "execution_id": execution_id,
        "result_s3_uri": data_uri,
        "metadata_s3_uri": meta_uri,
        "row_count": row_count,
        "columns": diagnostics.get("columns", list(result_df.columns)),
        "actual_cost_usd": actual_cost_usd,
        "duration_seconds": elapsed,
    }
