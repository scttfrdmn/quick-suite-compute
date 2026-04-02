"""
compute_cancel: Cancel a running Step Functions job.

AgentCore Lambda target — invoked directly by the Gateway.
Calls sfn.stop_execution() to halt a job in progress.

Tool arguments:
  job_id: str (required) — the job ID returned by compute_run
"""

import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]

sfn = boto3.client("stepfunctions")


def handler(event: dict, context) -> dict:
    _tool_name = "unknown"
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        _tool_name = raw.split("___")[-1]
    except Exception:
        pass
    logger.info(json.dumps({"tool": _tool_name, "event": event}))

    job_id = event.get("job_id", "").strip()
    if not job_id:
        return {"error": "job_id is required"}

    # Reconstruct the execution ARN from the state machine ARN and job ID.
    # State machine ARN:  arn:aws:states:REGION:ACCOUNT:stateMachine:NAME
    # Execution ARN:      arn:aws:states:REGION:ACCOUNT:execution:NAME:JOB_ID
    execution_arn = STATE_MACHINE_ARN.replace(":stateMachine:", ":execution:") + ":" + job_id

    try:
        sfn.stop_execution(
            executionArn=execution_arn,
            error="UserCancelled",
            cause="Cancelled by user via compute_cancel tool",
        )
        logger.info(json.dumps({"cancelled": job_id}))
        return {"status": "cancelled", "job_id": job_id}
    except sfn.exceptions.ExecutionDoesNotExist:
        return {"error": f"Job {job_id!r} not found"}
    except Exception as exc:
        error_name = type(exc).__name__
        if "AlreadyComplete" in error_name or "InvalidState" in error_name:
            return {"error": f"Job {job_id!r} has already completed and cannot be cancelled"}
        logger.error(f"stop_execution failed: {exc}")
        return {"error": f"Failed to cancel job: {exc}"}
