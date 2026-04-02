"""
check-budget — Step Functions Lambda

Reads the user's current-month spend from DynamoDB and checks it against
the monthly budget limit. Returns budget_ok=True/False so the state machine
can gate on it before provisioning any compute.

Input (from Step Functions):
  {
    "user_arn": str,
    "estimated_cost_usd": float   (from profile.cost_estimate.typical_cost_usd)
  }

Output:
  {
    "budget_ok": bool,
    "spend_usd": float,
    "budget_limit_usd": float,
    "user_arn": str
  }
"""

import json
import logging
import os
from datetime import datetime, timezone

import boto3

ALERT_THRESHOLD = 0.80

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")


def handler(event: dict, context) -> dict:
    logger.info(json.dumps({"event": event}))

    user_arn = event.get("user_arn", "")
    try:
        estimated_cost = float(event.get("estimated_cost_usd", 0) or 0)
    except (TypeError, ValueError):
        logger.warning(json.dumps({"invalid_estimated_cost": event.get("estimated_cost_usd")}))
        estimated_cost = 0.0
    budget_limit = float(os.environ.get("MONTHLY_BUDGET_USD", "50"))
    table_name = os.environ["SPEND_TABLE"]

    month = datetime.now(timezone.utc).strftime("%Y-%m")

    table = dynamodb.Table(table_name)
    try:
        resp = table.get_item(
            Key={"user_arn": user_arn, "month": month}
        )
        item = resp.get("Item") or {}
        spend_usd = float(item.get("spend_usd", 0))
    except Exception as exc:
        logger.error(f"DynamoDB get_item failed: {exc}")
        # Fail open — allow job to proceed, spend will be recorded on completion
        spend_usd = 0.0

    projected = spend_usd + estimated_cost
    budget_ok = projected <= budget_limit

    # Fire a one-time SNS alert when the user crosses 80% of their monthly budget
    threshold_alert_sent = False
    sns_topic_arn = os.environ.get("NOTIFICATION_TOPIC_ARN", "")
    if (
        sns_topic_arn
        and budget_limit > 0
        and projected / budget_limit >= ALERT_THRESHOLD
        and spend_usd / budget_limit < ALERT_THRESHOLD
    ):
        try:
            boto3.client("sns").publish(
                TopicArn=sns_topic_arn,
                Subject="Compute budget 80% threshold reached",
                Message=(
                    f"User {user_arn} has reached {ALERT_THRESHOLD:.0%} of their "
                    f"${budget_limit:.2f} monthly compute budget. "
                    f"Current projected spend: ${projected:.2f}."
                ),
            )
            threshold_alert_sent = True
        except Exception as exc:
            logger.warning(f"Failed to send budget threshold alert: {exc}")

    logger.info(json.dumps({
        "user_arn": user_arn,
        "month": month,
        "spend_usd": spend_usd,
        "estimated_cost": estimated_cost,
        "budget_limit": budget_limit,
        "budget_ok": budget_ok,
        "threshold_alert_sent": threshold_alert_sent,
    }))

    result = {
        "budget_ok": budget_ok,
        "spend_usd": spend_usd,
        "budget_limit_usd": budget_limit,
        "user_arn": user_arn,
    }
    if threshold_alert_sent:
        result["threshold_alert_sent"] = True
    return result
