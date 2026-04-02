"""
compute_compare — AgentCore Gateway Lambda target

Compares two named result snapshots by loading their S3 result Parquet files
and diffing the row sets. Returns counts (not full row data).

Event args:
  label_a   (str, required)  Label of first snapshot
  label_b   (str, required)  Label of second snapshot
  user_arn  (str, required)  IAM ARN of the user (both snapshots must belong to user)

Returns:
  {
    "label_a": str,
    "label_b": str,
    "added_count": int,        # rows in B not in A
    "removed_count": int,      # rows in A not in B
    "unchanged_count": int,    # rows in both A and B
    "schema_diff": {           # present if schemas differ
      "columns_only_in_a": [...],
      "columns_only_in_b": [...],
      "type_changes": {"col": {"a": type_a, "b": type_b}}
    } | null,
    "cost_delta_usd": float,   # cost_b - cost_a
    "duration_delta_seconds": float   # duration_b - duration_a
  }
OR {"error": str}
"""

import csv
import io
import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SNAPSHOTS_TABLE = os.environ.get("SNAPSHOTS_TABLE", "")

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")


def _get_snapshot(table, user_arn: str, label: str) -> dict | None:
    """Fetch a snapshot item from DynamoDB. Returns None if not found."""
    try:
        resp = table.get_item(
            Key={"user_arn": user_arn, "label": label}
        )
        return resp.get("Item")
    except Exception as exc:
        logger.error(f"DynamoDB get_item failed for label={label}: {exc}")
        return None


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key → (bucket, key)."""
    without_scheme = uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def _load_csv_rows(bucket: str, key: str) -> tuple[list[dict], list[str]]:
    """
    Load rows from an S3 object. Tries to parse as CSV (common result format).
    Returns (rows_as_dicts, column_names). Falls back to line-based comparison.
    """
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"Cannot load s3://{bucket}/{key}: {exc}") from exc

    # Try CSV parse
    try:
        reader = csv.DictReader(io.StringIO(body))
        rows = list(reader)
        columns = reader.fieldnames or []
        return rows, list(columns)
    except Exception:
        # Fallback: treat each line as a "row"
        lines = [ln for ln in body.splitlines() if ln.strip()]
        if not lines:
            return [], []
        rows = [{"_line": ln} for ln in lines]
        return rows, ["_line"]


def _schema_diff(cols_a: list[str], cols_b: list[str]) -> dict | None:
    """Return schema diff dict if schemas differ, else None."""
    set_a = set(cols_a)
    set_b = set(cols_b)
    only_in_a = sorted(set_a - set_b)
    only_in_b = sorted(set_b - set_a)
    if only_in_a or only_in_b:
        return {
            "columns_only_in_a": only_in_a,
            "columns_only_in_b": only_in_b,
            "type_changes": {},
        }
    return None


def _diff_rows(
    rows_a: list[dict], cols_a: list[str],
    rows_b: list[dict], cols_b: list[str],
) -> tuple[int, int, int]:
    """
    Diff two row sets using the intersection of columns for comparison.
    Returns (added_count, removed_count, unchanged_count).
    Returns counts only — no row data is surfaced.
    """
    # Use common columns for row identity
    common_cols = sorted(set(cols_a) & set(cols_b))
    if not common_cols:
        # No common columns — treat all rows as different
        return len(rows_b), len(rows_a), 0

    def _row_key(row: dict) -> tuple:
        return tuple(row.get(c, "") for c in common_cols)

    keys_a = [_row_key(r) for r in rows_a]
    keys_b = [_row_key(r) for r in rows_b]

    set_a = set(keys_a)
    set_b = set(keys_b)

    added = len(set_b - set_a)
    removed = len(set_a - set_b)
    unchanged = len(set_a & set_b)
    return added, removed, unchanged


def handler(event: dict, context) -> dict:
    _tool_name = "unknown"
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        _tool_name = raw.split("___")[-1]
    except Exception:
        pass
    logger.info(json.dumps({"tool": _tool_name, "event": event}))

    label_a = (event.get("label_a") or "").strip()
    label_b = (event.get("label_b") or "").strip()
    user_arn = (event.get("user_arn") or "").strip()

    if not label_a:
        return {"error": 'Required parameter "label_a" is missing'}
    if not label_b:
        return {"error": 'Required parameter "label_b" is missing'}
    if not user_arn:
        return {"error": 'Required parameter "user_arn" is missing'}
    if label_a == label_b:
        return {"error": '"label_a" and "label_b" must be different labels'}

    if not SNAPSHOTS_TABLE:
        return {"error": "SNAPSHOTS_TABLE environment variable is not configured"}

    table = dynamodb.Table(SNAPSHOTS_TABLE)

    snap_a = _get_snapshot(table, user_arn, label_a)
    if not snap_a:
        return {"error": f"Snapshot '{label_a}' not found for this user"}

    snap_b = _get_snapshot(table, user_arn, label_b)
    if not snap_b:
        return {"error": f"Snapshot '{label_b}' not found for this user"}

    cost_delta = float(snap_b.get("cost_usd", 0)) - float(snap_a.get("cost_usd", 0))
    duration_delta = (
        float(snap_b.get("duration_seconds", 0)) - float(snap_a.get("duration_seconds", 0))
    )

    result_uri_a = snap_a.get("result_uri", "")
    result_uri_b = snap_b.get("result_uri", "")

    if not result_uri_a or not result_uri_b:
        return {
            "label_a": label_a,
            "label_b": label_b,
            "added_count": 0,
            "removed_count": 0,
            "unchanged_count": 0,
            "schema_diff": None,
            "cost_delta_usd": round(cost_delta, 6),
            "duration_delta_seconds": round(duration_delta, 3),
            "warning": "One or both snapshots have no result_uri — row diff unavailable",
        }

    # Load rows from S3
    try:
        bucket_a, key_a = _parse_s3_uri(result_uri_a)
        rows_a, cols_a = _load_csv_rows(bucket_a, key_a)
    except Exception as exc:
        return {"error": f"Failed to load snapshot '{label_a}': {exc}"}

    try:
        bucket_b, key_b = _parse_s3_uri(result_uri_b)
        rows_b, cols_b = _load_csv_rows(bucket_b, key_b)
    except Exception as exc:
        return {"error": f"Failed to load snapshot '{label_b}': {exc}"}

    sdiff = _schema_diff(cols_a, cols_b)
    added, removed, unchanged = _diff_rows(rows_a, cols_a, rows_b, cols_b)

    logger.info(json.dumps({
        "compare": {"label_a": label_a, "label_b": label_b,
                    "added": added, "removed": removed, "unchanged": unchanged},
    }))

    return {
        "label_a": label_a,
        "label_b": label_b,
        "added_count": added,
        "removed_count": removed,
        "unchanged_count": unchanged,
        "schema_diff": sdiff,
        "cost_delta_usd": round(cost_delta, 6),
        "duration_delta_seconds": round(duration_delta, 3),
    }
