"""
record-spend — Step Functions Lambda

Updates the user's monthly spend in DynamoDB after a job completes.
Uses ADD for atomic increment in case of concurrent jobs.

Input (from Step Functions — full context):
  {
    "user_arn": str,
    "profile": {"cost_estimate": {"typical_cost_usd": float}},
    "compute": {"actual_cost_usd": float},   # from runner (optional)
    "result_label": str,                     # optional; if set, writes snapshot
    "deliver": {"result_uri": str, "row_count": int},  # from deliver step
    ...
  }

Output:
  {"recorded": true, "spend_usd": float, "month": str}
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

_LABEL_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

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

    budget_limit = float(os.environ.get("MONTHLY_BUDGET_USD", "50"))
    remaining = Decimal(str(budget_limit)) - Decimal(str(cost))

    table = dynamodb.Table(table_name)
    try:
        table.update_item(
            Key={"user_arn": user_arn, "month": month},
            UpdateExpression="ADD spend_usd :cost SET last_updated = :ts",
            ConditionExpression="attribute_not_exists(spend_usd) OR spend_usd <= :remaining",
            ExpressionAttributeValues={
                ":cost": Decimal(str(cost)),
                ":ts": datetime.now(timezone.utc).isoformat(),
                ":remaining": remaining,
            },
        )
        logger.info(json.dumps({"recorded_spend": cost, "user_arn": user_arn, "month": month}))
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            logger.warning(json.dumps({
                "budget_exceeded_at_record": True,
                "user_arn": user_arn,
                "cost": cost,
                "budget_limit": budget_limit,
            }))
            # Fail open — budget was checked before job started; log but don't crash
        else:
            logger.error(f"Failed to record spend: {exc}")
            return {"recorded": False, "spend_usd": cost, "month": month, "error": str(exc)}
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
                    "MetricName": "JobCost",
                    "Dimensions": [{"Name": "UserArn", "Value": user_arn}],
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

    # Write named snapshot if result_label is provided (Issue 19)
    result_label = (event.get("result_label") or "").strip()
    if result_label and not _LABEL_RE.match(result_label):
        logger.warning(json.dumps({"invalid_result_label": True, "label": result_label[:128]}))
        result_label = ""  # skip snapshot write — don't fail the workflow
    snapshots_table_name = os.environ.get("SNAPSHOTS_TABLE", "")
    if result_label and snapshots_table_name:
        try:
            deliver_result = event.get("deliver") or {}
            result_uri = deliver_result.get("result_uri", "")
            row_count = int(deliver_result.get("row_count", 0))
            completed_at = datetime.now(timezone.utc).isoformat()
            snap = dynamodb.Table(snapshots_table_name)
            snap.put_item(Item={
                "user_arn": user_arn,
                "label": result_label,
                "completed_at": completed_at,
                "job_id": execution_id,
                "profile_id": profile_id,
                "result_uri": result_uri,
                "row_count": row_count,
                "cost_usd": Decimal(str(cost)),
                "duration_seconds": Decimal(str(duration_seconds)),
            })
            logger.info(json.dumps({
                "snapshot_written": True,
                "label": result_label,
                "user_arn": user_arn,
            }))
        except Exception as exc:
            logger.warning(f"Failed to write snapshot: {exc}")

    # Write to data source registry (v0.18.0 #58) — fail-open
    registry_table = os.environ.get("DATA_REGISTRY_TABLE", "")
    if registry_table:
        try:
            deliver_output = event.get("deliver", {})
            result_uri = deliver_output.get("result_uri") or deliver_output.get("result_s3_uri", "")
            now_str = datetime.now(timezone.utc).isoformat()
            dynamodb.Table(registry_table).put_item(Item={
                "source_id": f"compute-result-{event.get('execution_id', 'unknown')}",
                "source_type": "s3",
                "uri": result_uri,
                "name": f"{profile_id} results — {now_str}",
                "created_by": user_arn,
                "tags": [profile_id, "compute-result"],
                "data_classification": "internal",
                "registered_at": now_str,
            })
            logger.info(json.dumps({"registry_write": "success", "source_id": f"compute-result-{event.get('execution_id')}"}))
        except Exception as exc:
            logger.warning("Registry write-back failed (non-fatal): %s", exc)

    return {"recorded": True, "spend_usd": cost, "month": month}
