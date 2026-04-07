"""
compute-schedule-trigger — Internal Lambda invoked by EventBridge Scheduler

Receives the trigger payload from EventBridge, reads the full schedule config
from DynamoDB, and starts a Step Functions execution.

Event (from EventBridge Scheduler):
  {
    "user_arn": str,
    "schedule_name": str
  }
"""

import json
import logging
import os
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SCHEDULES_TABLE = os.environ.get("SCHEDULES_TABLE", "")
STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN", "")

dynamodb = boto3.resource("dynamodb")
sfn = boto3.client("stepfunctions")


def handler(event: dict, context) -> dict:
    logger.info(json.dumps({"event": event}))

    user_arn = (event.get("user_arn") or "").strip()
    schedule_name = (event.get("schedule_name") or "").strip()

    if not user_arn or not schedule_name:
        logger.error("Missing user_arn or schedule_name in trigger event")
        return {"error": "user_arn and schedule_name are required"}

    if not SCHEDULES_TABLE:
        logger.error("SCHEDULES_TABLE not configured")
        return {"error": "SCHEDULES_TABLE not configured"}

    if not STATE_MACHINE_ARN:
        logger.error("STATE_MACHINE_ARN not configured")
        return {"error": "STATE_MACHINE_ARN not configured"}

    # Load schedule config from DynamoDB
    table = dynamodb.Table(SCHEDULES_TABLE)
    try:
        resp = table.get_item(Key={"user_arn": user_arn, "schedule_name": schedule_name})
    except Exception as exc:
        logger.error(f"DynamoDB get_item failed: {exc}")
        return {"error": f"Could not load schedule config: {exc}"}

    item = resp.get("Item")
    if not item:
        logger.error(f"Schedule not found: user_arn={user_arn}, schedule_name={schedule_name}")
        return {"error": f"Schedule '{schedule_name}' not found for user"}

    if not item.get("enabled", True):
        logger.info(f"Schedule '{schedule_name}' is disabled — skipping execution")
        return {"status": "skipped", "reason": "disabled"}

    profile_id = item.get("profile_id", "")
    parameters = item.get("parameters") or {}
    dataset_id = item.get("dataset_id")
    source_uri = item.get("source_uri")

    # Build execution input
    execution_input = {
        "user_arn": user_arn,
        "profile_id": profile_id,
        "parameters": parameters,
        "triggered_by": "schedule",
        "schedule_name": schedule_name,
    }
    if dataset_id:
        execution_input["dataset_id"] = dataset_id
    if source_uri:
        execution_input["source_uri"] = source_uri

    # Derive execution name (schedule-based jobs use a timestamp suffix)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_schedule = schedule_name.replace(" ", "-")[:30]
    execution_name = f"sched-{safe_schedule}-{ts}"

    try:
        start_resp = sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=execution_name,
            input=json.dumps(execution_input),
        )
    except Exception as exc:
        logger.error(f"SFN start_execution failed: {exc}")
        return {"error": f"Could not start execution: {exc}"}

    execution_arn = start_resp.get("executionArn", "")
    logger.info(json.dumps({
        "action": "triggered",
        "schedule_name": schedule_name,
        "user_arn": user_arn,
        "execution_arn": execution_arn,
    }))

    return {
        "status": "started",
        "execution_arn": execution_arn,
        "execution_name": execution_name,
    }
