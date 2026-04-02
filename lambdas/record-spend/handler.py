"""
record-spend — Step Functions Lambda

Updates the user's monthly spend in DynamoDB after a job completes.
Uses ADD for atomic increment in case of concurrent jobs.

Input (from Step Functions — full context):
  {
    "user_arn": str,
    "profile": {"cost_estimate": {"typical_cost_usd": float}},
    "compute": {"actual_cost_usd": float},   # from runner (optional)
    ...
  }

Output:
  {"recorded": true, "spend_usd": float, "month": str}
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from decimal import Decimal

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")


def handler(event: dict, context) -> dict:
    logger.info(json.dumps({"event": {
        "user_arn": event.get("user_arn"),
        "execution_id": event.get("execution_id"),
    }}))

    user_arn = event.get("user_arn", "unknown")
    table_name = os.environ["SPEND_TABLE"]
    month = datetime.now(timezone.utc).strftime("%Y-%m")

    # Use actual cost from runner if available, fall back to profile estimate
    compute_result = event.get("compute") or {}
    if "actual_cost_usd" in compute_result:
        cost = float(compute_result["actual_cost_usd"])
    else:
        profile = event.get("profile") or {}
        cost = float(profile.get("cost_estimate", {}).get("typical_cost_usd", 0))

    if cost < 0:
        logger.error(json.dumps({"negative_cost": cost, "execution_id": event.get("execution_id"), "clamped_to": 0}))
        cost = 0.0

    table = dynamodb.Table(table_name)
    try:
        table.update_item(
            Key={"user_arn": user_arn, "month": month},
            UpdateExpression="ADD spend_usd :cost SET last_updated = :ts",
            ExpressionAttributeValues={
                ":cost": Decimal(str(cost)),
                ":ts": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info(json.dumps({"recorded_spend": cost, "user_arn": user_arn, "month": month}))
    except Exception as exc:
        logger.error(f"Failed to record spend: {exc}")
        return {"recorded": False, "spend_usd": cost, "month": month,
                "error": str(exc)}

    profile_id = (event.get("profile") or {}).get("profile_id", "unknown")
    execution_id = event.get("execution_id", "")
    duration_seconds = float((event.get("compute") or {}).get("duration_seconds", 0))

    # Emit CloudWatch metrics for spend and duration per profile
    try:
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace="QuickSuiteCompute",
            MetricData=[
                {
                    "MetricName": "JobCost",
                    "Dimensions": [
                        {"Name": "ProfileId", "Value": profile_id},
                        {"Name": "UserArn", "Value": user_arn},
                    ],
                    "Value": cost,
                    "Unit": "None",
                },
                {
                    "MetricName": "JobDuration",
                    "Dimensions": [{"Name": "ProfileId", "Value": profile_id}],
                    "Value": duration_seconds,
                    "Unit": "Seconds",
                },
            ],
        )
    except Exception as exc:
        logger.warning(f"Failed to emit CW metrics: {exc}")

    # Write job history record
    history_table_name = os.environ.get("HISTORY_TABLE", "")
    if history_table_name:
        try:
            hist = dynamodb.Table(history_table_name)
            started_at = event.get("started_at") or datetime.now(timezone.utc).isoformat()
            hist.put_item(Item={
                "user_arn": user_arn,
                "started_at": started_at,
                "execution_id": execution_id,
                "profile_id": profile_id,
                "cost_usd": Decimal(str(cost)),
                "duration_seconds": Decimal(str(duration_seconds)),
                "status": "succeeded",
                "ttl": int(time.time()) + 90 * 86400,
            })
        except Exception as exc:
            logger.warning(f"Failed to write history: {exc}")

    return {"recorded": True, "spend_usd": cost, "month": month}
