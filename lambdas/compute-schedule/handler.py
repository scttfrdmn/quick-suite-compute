"""
compute_schedule — AgentCore Gateway Lambda target

Create, list, and delete scheduled compute jobs via EventBridge Scheduler.

Event args:
  action              (str, required)  "create" | "list" | "delete"
  schedule_name       (str)            for create/delete: unique name within user scope
  profile_id          (str)            for create: which profile to run
  parameters          (dict)           for create: profile parameters
  schedule_expression (str)            for create: "rate(1 day)" or "cron(0 9 ? * MON-FRI *)"
  dataset_id          (str)            for create: Quick Sight dataset to use as input
  source_uri          (str)            for create: S3 URI (alternative to dataset_id)
  timezone            (str)            for create: IANA timezone (default "UTC")
  enabled             (bool)           for create: activate immediately (default true)

Returns (create):
  {"schedule_name": str, "status": "created", "next_run": str|null}

Returns (list):
  {"schedules": [{schedule_name, profile_id, schedule_expression, enabled, created_at}]}

Returns (delete):
  {"schedule_name": str, "status": "deleted"}
"""

import json
import logging
import os
from datetime import datetime
from datetime import timezone as _tz

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SCHEDULES_TABLE = os.environ.get("SCHEDULES_TABLE", "")
SCHEDULER_GROUP_NAME = os.environ.get("SCHEDULER_GROUP_NAME", "")
SCHEDULER_ROLE_ARN = os.environ.get("SCHEDULER_ROLE_ARN", "")
TRIGGER_LAMBDA_ARN = os.environ.get("TRIGGER_LAMBDA_ARN", "")

dynamodb = boto3.resource("dynamodb")
scheduler_client = boto3.client("scheduler")


def _user_arn_from_context(context) -> str:
    try:
        return context.invoked_function_arn.replace(":function:" + context.function_name, "")
    except Exception:
        return "arn:aws:iam::unknown:user/unknown"


def _schedule_name_scoped(user_arn: str, schedule_name: str) -> str:
    """EventBridge schedule names must be globally unique within a group.
    Scope them per-user using a sanitized prefix."""
    user_suffix = user_arn.split("/")[-1].replace(":", "-")[:20]
    clean = schedule_name.replace(" ", "-")[:40]
    return f"{user_suffix}-{clean}"


def _create_schedule(event: dict, user_arn: str) -> dict:
    schedule_name = (event.get("schedule_name") or "").strip()
    profile_id = (event.get("profile_id") or "").strip()
    parameters = event.get("parameters") or {}
    schedule_expression = (event.get("schedule_expression") or "").strip()
    dataset_id = (event.get("dataset_id") or "").strip()
    source_uri = (event.get("source_uri") or "").strip()
    timezone = (event.get("timezone") or "UTC").strip()
    enabled = event.get("enabled", True)

    if not schedule_name:
        return {"error": "schedule_name is required"}
    if not profile_id:
        return {"error": "profile_id is required"}
    if not schedule_expression:
        return {"error": "schedule_expression is required"}
    if not dataset_id and not source_uri:
        return {"error": "Either dataset_id or source_uri is required"}

    if not SCHEDULES_TABLE or not SCHEDULER_GROUP_NAME or not SCHEDULER_ROLE_ARN or not TRIGGER_LAMBDA_ARN:
        return {"error": "Scheduler infrastructure not configured"}

    eb_name = _schedule_name_scoped(user_arn, schedule_name)
    created_at = datetime.now(_tz.utc).isoformat()

    item = {
        "user_arn": user_arn,
        "schedule_name": schedule_name,
        "eb_schedule_name": eb_name,
        "profile_id": profile_id,
        "parameters": parameters,
        "schedule_expression": schedule_expression,
        "timezone": timezone,
        "enabled": bool(enabled),
        "created_at": created_at,
    }
    if dataset_id:
        item["dataset_id"] = dataset_id
    if source_uri:
        item["source_uri"] = source_uri

    # Write to DynamoDB first
    table = dynamodb.Table(SCHEDULES_TABLE)
    try:
        table.put_item(Item=item)
    except Exception as exc:
        logger.error(f"DynamoDB put_item failed: {exc}")
        return {"error": f"Could not save schedule: {exc}"}

    # Create EventBridge Scheduler schedule
    trigger_payload = {
        "user_arn": user_arn,
        "schedule_name": schedule_name,
    }
    try:
        scheduler_client.create_schedule(
            Name=eb_name,
            GroupName=SCHEDULER_GROUP_NAME,
            ScheduleExpression=schedule_expression,
            ScheduleExpressionTimezone=timezone,
            FlexibleTimeWindow={"Mode": "OFF"},
            Target={
                "Arn": TRIGGER_LAMBDA_ARN,
                "RoleArn": SCHEDULER_ROLE_ARN,
                "Input": json.dumps(trigger_payload),
            },
            State="ENABLED" if enabled else "DISABLED",
        )
    except Exception as exc:
        logger.error(f"EventBridge create_schedule failed: {exc}")
        # Roll back DynamoDB write
        try:
            table.delete_item(Key={"user_arn": user_arn, "schedule_name": schedule_name})
        except Exception:
            pass
        return {"error": f"Could not create EventBridge schedule: {exc}"}

    logger.info(json.dumps({"action": "create", "schedule_name": schedule_name, "user_arn": user_arn}))
    return {"schedule_name": schedule_name, "status": "created"}


def _list_schedules(user_arn: str) -> dict:
    if not SCHEDULES_TABLE:
        return {"error": "Scheduler infrastructure not configured"}

    table = dynamodb.Table(SCHEDULES_TABLE)
    try:
        resp = table.query(
            KeyConditionExpression=Key("user_arn").eq(user_arn),
        )
    except Exception as exc:
        logger.error(f"DynamoDB query failed: {exc}")
        return {"error": f"Could not list schedules: {exc}"}

    schedules = [
        {
            "schedule_name": item["schedule_name"],
            "profile_id": item.get("profile_id", ""),
            "schedule_expression": item.get("schedule_expression", ""),
            "timezone": item.get("timezone", "UTC"),
            "enabled": item.get("enabled", True),
            "created_at": item.get("created_at", ""),
            "dataset_id": item.get("dataset_id"),
            "source_uri": item.get("source_uri"),
        }
        for item in resp.get("Items", [])
    ]
    return {"schedules": schedules, "count": len(schedules)}


def _delete_schedule(event: dict, user_arn: str) -> dict:
    schedule_name = (event.get("schedule_name") or "").strip()
    if not schedule_name:
        return {"error": "schedule_name is required"}

    if not SCHEDULES_TABLE or not SCHEDULER_GROUP_NAME:
        return {"error": "Scheduler infrastructure not configured"}

    # Look up the EventBridge schedule name
    table = dynamodb.Table(SCHEDULES_TABLE)
    try:
        resp = table.get_item(Key={"user_arn": user_arn, "schedule_name": schedule_name})
    except Exception as exc:
        return {"error": f"Could not look up schedule: {exc}"}

    item = resp.get("Item")
    if not item:
        return {"error": f"Schedule '{schedule_name}' not found"}

    eb_name = item.get("eb_schedule_name", _schedule_name_scoped(user_arn, schedule_name))

    # Delete from EventBridge
    try:
        scheduler_client.delete_schedule(Name=eb_name, GroupName=SCHEDULER_GROUP_NAME)
    except scheduler_client.exceptions.ResourceNotFoundException:
        pass  # Already gone — continue to clean up DynamoDB
    except Exception as exc:
        return {"error": f"Could not delete EventBridge schedule: {exc}"}

    # Delete from DynamoDB
    try:
        table.delete_item(Key={"user_arn": user_arn, "schedule_name": schedule_name})
    except Exception as exc:
        logger.error(f"DynamoDB delete_item failed: {exc}")

    logger.info(json.dumps({"action": "delete", "schedule_name": schedule_name, "user_arn": user_arn}))
    return {"schedule_name": schedule_name, "status": "deleted"}


def handler(event: dict, context) -> dict:
    _tool_name = "unknown"
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        _tool_name = raw.split("___")[-1]
    except Exception:
        pass
    logger.info(json.dumps({"tool": _tool_name, "event": {k: v for k, v in event.items() if k != "parameters"}}))

    action = (event.get("action") or "").strip().lower()
    if not action:
        return {"error": 'Required parameter "action" is missing (create | list | delete)'}

    # Derive user ARN from execution context
    try:
        invoked_arn = context.invoked_function_arn
        parts = invoked_arn.split(":")
        user_arn = ":".join(parts[:5]) + ":user/" + (event.get("user_arn_suffix") or "agent")
    except Exception:
        user_arn = event.get("user_arn") or "arn:aws:iam::unknown:user/agent"

    if action == "create":
        return _create_schedule(event, user_arn)
    elif action == "list":
        return _list_schedules(user_arn)
    elif action == "delete":
        return _delete_schedule(event, user_arn)
    else:
        return {"error": f"Unknown action '{action}'. Use: create | list | delete"}
