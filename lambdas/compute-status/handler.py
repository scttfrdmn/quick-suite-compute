"""
compute_status — AgentCore Gateway Lambda target

Polls the status of a compute job. Maps Step Functions execution state
to a user-friendly status and returns result dataset info when done.

Event args:
  job_id  (str, required)  Job ID returned by compute_run

Returns:
  {
    "status": "RUNNING" | "SUCCEEDED" | "FAILED" | "TIMED_OUT" | "ABORTED",
    "job_id": str,
    "execution_arn": str,
    "started_at": str | null,
    "stopped_at": str | null,
    "elapsed_seconds": float | null,
    "result_dataset_id": str | null,       # present when SUCCEEDED
    "result_dataset_name": str | null,     # present when SUCCEEDED
    "error": str | null,                   # present when FAILED/TIMED_OUT
    "message": str
  }
"""

import json
import logging
import os
from datetime import timezone

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sfn = boto3.client("stepfunctions")
dynamodb = boto3.resource("dynamodb")

HISTORY_TABLE = os.environ.get("HISTORY_TABLE", "")


def _find_execution_arn(job_id: str) -> str | None:
    """
    job_id is either a full execution ARN or the execution name
    (e.g. 'job-abc12345-20250101').
    """
    if job_id.startswith("arn:aws"):
        return job_id
    # Reconstruct ARN from execution name using state machine ARN in env
    state_machine_arn = os.environ.get("STATE_MACHINE_ARN", "")
    if not state_machine_arn:
        return None
    # Convert stateMachine ARN to execution ARN
    # arn:aws:states:region:account:stateMachine:name
    # → arn:aws:states:region:account:execution:name:job_id
    parts = state_machine_arn.split(":")
    parts[-2] = "execution"
    parts.append(job_id)
    return ":".join(parts)


def handler(event: dict, context) -> dict:
    _tool_name = "unknown"
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        _tool_name = raw.split("___")[-1]
    except Exception:
        pass
    logger.info(json.dumps({"tool": _tool_name, "event": event}))

    job_id = (event.get("job_id") or "").strip()
    if not job_id:
        return {"error": 'Required parameter "job_id" is missing'}

    execution_arn = _find_execution_arn(job_id)
    if not execution_arn:
        return {"error": "Cannot resolve execution ARN — STATE_MACHINE_ARN not configured"}

    try:
        resp = sfn.describe_execution(executionArn=execution_arn)
    except sfn.exceptions.ExecutionDoesNotExist:
        return {"error": f"Job '{job_id}' not found"}
    except Exception as exc:
        logger.error(f"describe_execution failed: {exc}")
        return {"error": f"Failed to check job status: {exc}"}

    sfn_status = resp.get("status", "UNKNOWN")
    started_at = resp.get("startDate")
    stopped_at = resp.get("stopDate")

    started_str = started_at.astimezone(timezone.utc).isoformat() if started_at else None
    stopped_str = stopped_at.astimezone(timezone.utc).isoformat() if stopped_at else None

    elapsed = None
    if started_at and stopped_at:
        elapsed = (stopped_at - started_at).total_seconds()

    result: dict = {
        "status": sfn_status,
        "job_id": job_id,
        "execution_arn": execution_arn,
        "started_at": started_str,
        "stopped_at": stopped_str,
        "elapsed_seconds": elapsed,
        "result_dataset_id": None,
        "result_dataset_name": None,
        "error": None,
    }

    if sfn_status == "SUCCEEDED":
        # Extract result dataset info from execution output
        try:
            output = json.loads(resp.get("output") or "{}")
            deliver = output.get("deliver", {})
            result["result_dataset_id"] = deliver.get("dataset_id")
            result["result_dataset_name"] = deliver.get("result_dataset_name")

            # Issue 21: surface chain step info if present
            chain_step = output.get("chain_step")
            if chain_step:
                result["step"] = chain_step

            # Issue 21: surface total cost across both steps
            chain_spend = output.get("chain_spend")
            if chain_spend:
                result["total_cost_usd"] = float(chain_spend.get("total_cost_usd", 0))

        except (json.JSONDecodeError, AttributeError):
            pass

        # Enrich with cost/duration from HistoryTable
        if HISTORY_TABLE:
            try:
                history_resp = dynamodb.Table(HISTORY_TABLE).query(
                    IndexName="by-execution-arn",
                    KeyConditionExpression=Key("execution_arn").eq(execution_arn),
                    Limit=1,
                )
                history_items = history_resp.get("Items", [])
                if history_items:
                    h = history_items[0]
                    result["actual_cost_usd"] = float(h.get("cost_usd", 0))
                    result["duration_seconds"] = float(h.get("duration_seconds", 0))
                    result["profile_id"] = h.get("profile_id")
            except Exception:
                pass  # non-fatal

        result["message"] = (
            f"Job completed successfully in {elapsed:.0f}s. "
            f"Results are available as a new Quick Sight dataset."
            if elapsed else "Job completed successfully."
        )

    elif sfn_status == "FAILED":
        try:
            output = json.loads(resp.get("output") or "{}")
            failure = output.get("failure", {})
            result["error"] = failure.get("error_message") or resp.get("cause", "Unknown error")
        except (json.JSONDecodeError, AttributeError):
            result["error"] = resp.get("cause", "Unknown error")
        result["message"] = f"Job failed: {result['error']}"

    elif sfn_status == "TIMED_OUT":
        result["error"] = "Job exceeded the maximum allowed duration"
        result["message"] = "Job timed out"

    elif sfn_status == "ABORTED":
        result["error"] = "Job was cancelled"
        result["message"] = "Job was aborted"

    else:  # RUNNING
        elapsed_running = None
        if started_at:
            from datetime import datetime
            elapsed_running = (
                datetime.now(timezone.utc) - started_at.astimezone(timezone.utc)
            ).total_seconds()

        # Issue 21: try to surface chain step from execution input
        try:
            inp = json.loads(resp.get("input") or "{}")
            if inp.get("chain_profile"):
                # chain job — determine which step based on SFN output if available
                partial_output = json.loads(resp.get("output") or "{}")
                chain_step = partial_output.get("chain_step", "profile_1")
                result["step"] = chain_step
        except (json.JSONDecodeError, AttributeError):
            pass

        # Issue #24: cumulative spend so far this month for RUNNING jobs.
        # Query HistoryTable for completed jobs for this user in the current month.
        if HISTORY_TABLE:
            try:
                from datetime import datetime as _dt
                from boto3.dynamodb.conditions import Key as _Key, Attr as _Attr
                inp_data = json.loads(resp.get("input") or "{}")
                running_user_arn = inp_data.get("user_arn", "")
                current_month_prefix = _dt.now(timezone.utc).strftime("%Y-%m")
                history_running_resp = dynamodb.Table(HISTORY_TABLE).query(
                    KeyConditionExpression=(
                        _Key("user_arn").eq(running_user_arn)
                        & _Key("started_at").begins_with(current_month_prefix)
                    ),
                    ProjectionExpression="cost_usd",
                )
                cost_so_far = sum(
                    float(h.get("cost_usd", 0))
                    for h in history_running_resp.get("Items", [])
                )
                result["cost_usd_so_far"] = cost_so_far
            except Exception:
                pass  # non-fatal

        result["message"] = (
            f"Job is running ({elapsed_running:.0f}s elapsed). "
            "Check again in 15–30 seconds."
            if elapsed_running else "Job is running. Check again in 15–30 seconds."
        )

    logger.info(json.dumps({"job_id": job_id, "status": sfn_status}))
    return result
