"""
compute_run — AgentCore Gateway Lambda target

Validates job parameters against the selected profile, checks the user's
remaining monthly compute budget, and starts a Step Functions execution.
Returns a job_id immediately.

Single-profile event args:
  profile_id   (str, required)  Profile to run
  dataset_id   (str, required)  Quick Sight dataset ID as input (or use source_uri)
  user_arn     (str, required)  Caller ARN for budget tracking
  parameters   (dict)           Profile-specific parameters
  dataset_name (str)            Optional result dataset name
  source_uri   (str)            Optional direct S3 URI (s3://bucket/key) or claws:// URI

Parallel-profile event args (supply profiles list instead of profile_id):
  profiles     (list[str])      List of profile_id values to run in parallel
  dataset_id   (str, required)  Quick Sight dataset ID as input (or use source_uri)
  user_arn     (str, required)  Caller ARN for budget tracking
  parameters   (dict)           Shared profile parameters

Returns (single):
  {"status": "started", "job_id": str, "execution_arn": str, ...}
Returns (parallel):
  {"status": "started", "count": int, "jobs": [{"job_id": str, "profile_id": str}, ...]}
OR {"error": str}
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone

import boto3

_ARN_RE = re.compile(
    r"^arn:aws:iam::\d{12}:(user|role|assumed-role)/[\w+=,.@/-]+$"
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sfn = boto3.client("stepfunctions")

_PROFILES: dict[str, dict] | None = None


def _load_profiles() -> dict[str, dict]:
    global _PROFILES
    if _PROFILES is None:
        raw = os.environ.get("PROFILES_CONFIG", "[]")
        _PROFILES = {p["profile_id"]: p for p in json.loads(raw)}
    return _PROFILES


def _validate_params(profile: dict, user_params: dict) -> list[str]:
    """Return list of validation error messages (empty = valid)."""
    errors = []
    schema = profile.get("parameters", {})
    for param_name, spec in schema.items():
        val = user_params.get(param_name)
        if val is None:
            val = spec.get("default")
        if val is None and spec.get("type") not in ("column", "column_list", "string_list"):
            errors.append(f"Parameter '{param_name}' is required")
            continue

        if val is None:
            continue

        ptype = spec.get("type")
        if ptype == "integer":
            if not isinstance(val, int):
                errors.append(f"Parameter '{param_name}' must be an integer")
                continue
            if "min" in spec and val < spec["min"]:
                errors.append(f"Parameter '{param_name}' must be >= {spec['min']}")
            if "max" in spec and val > spec["max"]:
                errors.append(f"Parameter '{param_name}' must be <= {spec['max']}")
        elif ptype == "float":
            if not isinstance(val, (int, float)):
                errors.append(f"Parameter '{param_name}' must be numeric")
                continue
            if "min" in spec and val < spec["min"]:
                errors.append(f"Parameter '{param_name}' must be >= {spec['min']}")
            if "max" in spec and val > spec["max"]:
                errors.append(f"Parameter '{param_name}' must be <= {spec['max']}")
        elif ptype == "enum":
            if val not in spec.get("values", []):
                errors.append(
                    f"Parameter '{param_name}' must be one of: {spec.get('values')}"
                )
        elif ptype == "column_list":
            if not isinstance(val, list):
                errors.append(f"Parameter '{param_name}' must be a list of column names")
            elif not all(isinstance(c, str) for c in val):
                errors.append(f"Parameter '{param_name}' must contain only string column names")
            elif len(val) < spec.get("min_columns", 0):
                errors.append(
                    f"Parameter '{param_name}' requires at least {spec['min_columns']} columns"
                )
        elif ptype == "string_list":
            if not isinstance(val, list):
                errors.append(f"Parameter '{param_name}' must be a list of strings")
            elif not all(isinstance(s, str) for s in val):
                errors.append(f"Parameter '{param_name}' must contain only strings")
            elif "max_items" in spec and len(val) > spec["max_items"]:
                errors.append(f"Parameter '{param_name}' exceeds max {spec['max_items']} items")
        elif ptype == "column":
            if not isinstance(val, str):
                errors.append(f"Parameter '{param_name}' must be a column name string")

    return errors


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
    if not _ARN_RE.match(user_arn):
        return {"error": "user_arn must be a valid IAM ARN "
                         "(arn:aws:iam::ACCOUNT:(user|role|assumed-role)/NAME)"}

    # Validate source_uri if provided
    source_uri = (event.get("source_uri") or "").strip()
    if source_uri and not (source_uri.startswith("s3://") or source_uri.startswith("claws://")):
        return {"error": "source_uri must start with 's3://' or 'claws://'"}

    dataset_id = (event.get("dataset_id") or "").strip()
    if not dataset_id and not source_uri:
        return {"error": 'Either "dataset_id" or "source_uri" is required'}

    # Check for parallel-profile invocation
    profiles_list = event.get("profiles")
    if profiles_list:
        return _run_parallel(event, user_arn, dataset_id, source_uri)

    # Single-profile path
    profile_id = (event.get("profile_id") or "").strip()
    if not profile_id:
        return {"error": 'Required parameter "profile_id" is missing'}

    return _run_single(event, profile_id, user_arn, dataset_id, source_uri)


def _run_single(event: dict, profile_id: str, user_arn: str,
                dataset_id: str, source_uri: str) -> dict:
    profiles = _load_profiles()
    profile = profiles.get(profile_id)
    if not profile:
        available = sorted(profiles.keys())
        return {
            "error": f"Profile '{profile_id}' not found",
            "available_profiles": available,
        }

    # Check EMR requirement
    enable_emr = os.environ.get("ENABLE_EMR", "false").lower() == "true"
    if profile.get("backend") == "emr_serverless" and not enable_emr:
        return {
            "status": "requires_emr",
            "profile_id": profile_id,
            "message": (
                f"Profile '{profile['display_name']}' requires EMR Serverless, "
                "which is not enabled in this deployment. "
                "Redeploy with --context enable_emr=true to enable."
            ),
        }

    # Fast budget pre-check (avoids queuing jobs that will immediately fail in SFN)
    # The Step Functions CheckBudget step remains the authoritative gate.
    try:
        dynamodb = boto3.resource("dynamodb")
        spend_table = dynamodb.Table(os.environ["SPEND_TABLE"])
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        item = spend_table.get_item(Key={"user_arn": user_arn, "month": month}).get("Item") or {}
        current_spend = float(item.get("spend_usd", 0))
        budget_limit = float(os.environ.get("MONTHLY_BUDGET_USD", "50"))
        estimated_cost = float(profile.get("cost_estimate", {}).get("typical_cost_usd", 0))
        if current_spend + estimated_cost > budget_limit:
            return {
                "status": "budget_exceeded",
                "spend_usd": current_spend,
                "budget_limit_usd": budget_limit,
                "message": (
                    f"Monthly compute budget of ${budget_limit:.0f} reached "
                    f"(${current_spend:.2f} spent this month)."
                ),
            }
    except Exception as exc:
        # Fail open — Step Functions CheckBudget is the authoritative gate
        logger.warning(json.dumps({"budget_precheck_error": str(exc)}))

    # Validate parameters
    user_params = event.get("parameters") or {}
    errors = _validate_params(profile, user_params)
    if errors:
        return {
            "error": "Parameter validation failed",
            "validation_errors": errors,
        }

    job_id, execution_arn, err = _start_execution(
        profile, user_arn, dataset_id, source_uri,
        event.get("dataset_name"), event.get("parameters") or {},
    )
    if err:
        return {"error": err}

    logger.info(json.dumps({
        "started": True,
        "job_id": job_id,
        "execution_arn": execution_arn,
        "profile_id": profile_id,
    }))

    return {
        "status": "started",
        "job_id": job_id,
        "execution_arn": execution_arn,
        "profile_id": profile_id,
        "profile_display_name": profile["display_name"],
        "estimated_cost_usd": profile.get("cost_estimate", {}).get("typical_cost_usd", 0),
        "message": (
            f"Started {profile['display_name']} job. "
            f"Call compute_status with job_id='{job_id}' to check progress."
        ),
    }


def _run_parallel(event: dict, user_arn: str, dataset_id: str, source_uri: str) -> dict:
    """Start one Step Functions execution per profile and return all job IDs."""
    profiles_list = event.get("profiles", [])
    if not isinstance(profiles_list, list) or not profiles_list:
        return {"error": '"profiles" must be a non-empty list of profile_id strings'}
    if len(profiles_list) > 10:
        return {"error": '"profiles" list must not exceed 10 items'}

    profiles = _load_profiles()
    unknown = [p for p in profiles_list if p not in profiles]
    if unknown:
        return {
            "error": f"Unknown profile(s): {unknown}",
            "available_profiles": sorted(profiles.keys()),
        }

    enable_emr = os.environ.get("ENABLE_EMR", "false").lower() == "true"
    jobs = []
    errors = []
    for pid in profiles_list:
        profile = profiles[pid]
        if profile.get("backend") == "emr_serverless" and not enable_emr:
            errors.append(f"Profile '{pid}' requires EMR Serverless (not enabled)")
            continue
        job_id, execution_arn, err = _start_execution(
            profile, user_arn, dataset_id, source_uri,
            event.get("dataset_name"), event.get("parameters") or {},
        )
        if err:
            errors.append(f"Profile '{pid}': {err}")
        else:
            jobs.append({"job_id": job_id, "execution_arn": execution_arn, "profile_id": pid})

    result = {
        "status": "started" if jobs else "failed",
        "count": len(jobs),
        "jobs": jobs,
    }
    if errors:
        result["errors"] = errors
    return result


def _start_execution(profile: dict, user_arn: str, dataset_id: str,
                     source_uri: str, dataset_name, user_params: dict) -> tuple:
    """Start a single Step Functions execution. Returns (job_id, execution_arn, error_str)."""
    execution_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Merge parameter defaults
    merged_params = {}
    for param_name, spec in profile.get("parameters", {}).items():
        merged_params[param_name] = user_params.get(param_name, spec.get("default"))

    name = dataset_name or f"{profile['display_name']} — {timestamp}"
    if len(name) > 128:
        name = name[:128]

    execution_input = {
        "execution_id": execution_id,
        "profile": profile,
        "dataset_id": dataset_id,
        "source_uri": source_uri,
        "dataset_name": name,
        "user_arn": user_arn,
        "parameters": merged_params,
        "enable_emr": os.environ.get("ENABLE_EMR", "false").lower() == "true",
        "started_at": timestamp,
    }

    state_machine_arn = os.environ["STATE_MACHINE_ARN"]
    execution_name = f"job-{execution_id[:8]}-{timestamp[:10].replace('-', '')}"

    try:
        response = sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=execution_name,
            input=json.dumps(execution_input),
        )
        return execution_name, response["executionArn"], None
    except Exception as exc:
        logger.error(f"Failed to start execution for {profile['profile_id']}: {exc}")
        return None, None, f"Failed to start compute job: {exc}"
