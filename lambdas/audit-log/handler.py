"""
audit-log — Step Functions Lambda

Writes an immutable audit record to S3 at the end of every execution path
(SUCCEEDED, FAILED, TIMED_OUT).  Audit objects contain URIs only — no PII,
no data content.

S3 key layout:
  s3://{compute-results-bucket}/audit/{year}/{month}/{job_id}.json

Input (full Step Functions execution context, enriched by previous steps):
  {
    "execution_id": str,
    "user_arn": str,
    "profile": {"profile_id": str, ...},
    "dataset_uri": str,
    "params": dict,
    "deliver": {"result_uri": str} | missing on failure,
    "compute": {"actual_cost_usd": float, "duration_seconds": float} | missing on failure,
    "status": "SUCCEEDED" | "FAILED" | "TIMED_OUT"   # injected by SFN pass state
  }

Output:
  {"audit_written": true, "audit_key": str}
"""

import json
import logging
import os
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")


def handler(event: dict, context) -> dict:
    logger.info(json.dumps({
        "event": {
            "execution_id": event.get("execution_id"),
            "user_arn": event.get("user_arn"),
            "status": event.get("status"),
        }
    }))

    bucket = os.environ["COMPUTE_BUCKET"]

    job_id = event.get("execution_id", "unknown")
    user_arn = event.get("user_arn", "unknown")
    profile = event.get("profile") or {}
    profile_id = profile.get("profile_id", "unknown")
    dataset_uri = event.get("dataset_uri", "")
    params = event.get("params") or event.get("parameters") or {}
    status = event.get("status", "UNKNOWN")
    timestamp = datetime.now(timezone.utc).isoformat()

    # Cost and duration only present on SUCCEEDED path
    compute = event.get("compute") or {}
    cost_usd = compute.get("actual_cost_usd")
    if cost_usd is None:
        cost_usd = (profile.get("cost_estimate") or {}).get("typical_cost_usd")

    duration_seconds = compute.get("duration_seconds")

    # result_uri only present on SUCCEEDED path
    deliver = event.get("deliver") or {}
    result_uri = deliver.get("result_uri") or deliver.get("result_dataset_name") or None

    audit = {
        "job_id": job_id,
        "profile_id": profile_id,
        "user_arn": user_arn,
        "dataset_uri": dataset_uri,
        "params": params,
        "result_uri": result_uri,
        "cost_usd": cost_usd,
        "duration_seconds": duration_seconds,
        "status": status,
        "timestamp": timestamp,
    }

    # S3 key: audit/{year}/{month}/{job_id}.json
    now = datetime.now(timezone.utc)
    audit_key = f"audit/{now.year}/{now.month:02d}/{job_id}.json"

    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=audit_key,
            Body=json.dumps(audit),
            ContentType="application/json",
        )
        logger.info(json.dumps({"audit_written": True, "audit_key": audit_key}))
    except Exception as exc:
        logger.error(json.dumps({"audit_write_failed": str(exc), "job_id": job_id}))
        raise

    return {"audit_written": True, "audit_key": audit_key}
