"""
compute_history: List job history for a user.

AgentCore Lambda target — invoked directly by the Gateway.
Returns the user's recent compute jobs from the history table,
most recent first.

Tool arguments:
  user_arn: str (required) — IAM ARN of the user
  limit:    int (optional, 1–20, default 10) — max results to return
"""

import json
import logging
import os

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

HISTORY_TABLE = os.environ["HISTORY_TABLE"]
MAX_ITEMS = 20

dynamodb = boto3.resource("dynamodb")


def handler(event: dict, context) -> dict:
    _tool_name = "unknown"
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        _tool_name = raw.split("___")[-1]
    except Exception:
        pass
    logger.info(json.dumps({"tool": _tool_name, "event": event}))

    user_arn = event.get("user_arn", "").strip()
    if not user_arn:
        return {"error": "user_arn is required"}

    try:
        limit = min(int(event.get("limit", 10)), MAX_ITEMS)
    except (TypeError, ValueError):
        limit = 10

    table = dynamodb.Table(HISTORY_TABLE)
    try:
        resp = table.query(
            KeyConditionExpression=Key("user_arn").eq(user_arn),
            ScanIndexForward=False,
            Limit=limit,
        )
    except Exception as exc:
        logger.error(f"DynamoDB query failed: {exc}")
        return {"error": f"Failed to retrieve job history: {exc}"}

    jobs = []
    for item in resp.get("Items", []):
        jobs.append({
            "execution_id": item.get("execution_id", ""),
            "profile_id": item.get("profile_id", ""),
            "started_at": item.get("started_at", ""),
            "cost_usd": float(item.get("cost_usd", 0)),
            "duration_seconds": float(item.get("duration_seconds", 0)),
            "status": item.get("status", ""),
        })

    return {
        "user_arn": user_arn,
        "jobs": jobs,
        "count": len(jobs),
    }
