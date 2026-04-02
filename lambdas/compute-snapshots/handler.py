"""
compute_snapshots — AgentCore Gateway Lambda target

Lists a user's named result snapshots, sorted by completed_at descending.
Only jobs submitted with a result_label are written to the snapshots table.

Event args:
  user_arn  (str, required)  IAM ARN of the user
  limit     (int, optional)  Max results to return (1–50, default 20)

Returns:
  {
    "user_arn": str,
    "snapshots": [
      {
        "label": str,
        "completed_at": str,
        "job_id": str,
        "profile_id": str,
        "result_uri": str,
        "row_count": int,
        "cost_usd": float,
        "duration_seconds": float
      },
      ...
    ],
    "count": int
  }
OR {"error": str}
"""

import json
import logging
import os

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SNAPSHOTS_TABLE = os.environ.get("SNAPSHOTS_TABLE", "")
MAX_ITEMS = 50

dynamodb = boto3.resource("dynamodb")


def handler(event: dict, context) -> dict:
    _tool_name = "unknown"
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        _tool_name = raw.split("___")[-1]
    except Exception:
        pass
    logger.info(json.dumps({"tool": _tool_name, "event": event}))

    user_arn = (event.get("user_arn") or "").strip()
    if not user_arn:
        return {"error": 'Required parameter "user_arn" is missing'}

    try:
        limit = min(int(event.get("limit", 20)), MAX_ITEMS)
        if limit < 1:
            limit = 1
    except (TypeError, ValueError):
        limit = 20

    if not SNAPSHOTS_TABLE:
        return {"error": "SNAPSHOTS_TABLE environment variable is not configured"}

    table = dynamodb.Table(SNAPSHOTS_TABLE)
    try:
        resp = table.query(
            KeyConditionExpression=Key("user_arn").eq(user_arn),
            ScanIndexForward=False,  # most recent first
            Limit=limit,
        )
    except Exception as exc:
        logger.error(f"DynamoDB query failed: {exc}")
        return {"error": f"Failed to retrieve snapshots: {exc}"}

    snapshots = []
    for item in resp.get("Items", []):
        snapshots.append({
            "label": item.get("label", ""),
            "completed_at": item.get("completed_at", ""),
            "job_id": item.get("job_id", ""),
            "profile_id": item.get("profile_id", ""),
            "result_uri": item.get("result_uri", ""),
            "row_count": int(item.get("row_count", 0)),
            "cost_usd": float(item.get("cost_usd", 0)),
            "duration_seconds": float(item.get("duration_seconds", 0)),
        })

    logger.info(json.dumps({"user_arn": user_arn, "snapshot_count": len(snapshots)}))
    return {
        "user_arn": user_arn,
        "snapshots": snapshots,
        "count": len(snapshots),
    }
