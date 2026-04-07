"""
compute_run — AgentCore Gateway Lambda target

Validates job parameters against the selected profile, checks the user's
remaining monthly compute budget, and starts a Step Functions execution.
Returns a job_id immediately.

Single-profile event args:
  profile_id      (str, required)  Profile to run
  dataset_id      (str, required)  Quick Sight dataset ID as input (or use source_uri)
  user_arn        (str, required)  Caller ARN for budget tracking
  parameters      (dict)           Profile-specific parameters
  dataset_name    (str)            Optional result dataset name
  source_uri      (str)            Optional direct S3 URI (s3://bucket/key) or claws:// URI
  result_label    (str)            Optional label for named snapshot (Issue 19)
  chain_profile_id (str)           Optional second profile to run after first completes (Issue 21)

Parallel-profile event args (supply profiles list instead of profile_id):
  profiles     (list[str])      List of profile_id values to run in parallel
  dataset_id   (str, required)  Quick Sight dataset ID as input (or use source_uri)
  user_arn     (str, required)  Caller ARN for budget tracking
  parameters   (dict)           Shared profile parameters

Returns (single):
  {
    "status": "started",
    "job_id": str,
    "execution_arn": str,
    "estimated_cost_usd": float,        # Issue 22: pre-submission estimate
    "estimated_duration_seconds": float, # Issue 22: pre-submission estimate
    ...
  }
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

_S3_BYTES_PER_ROW_ESTIMATE = 200  # rough estimate for cost scaling

_ARN_RE = re.compile(
    r"^arn:aws:iam::\d{12}:(user|role|assumed-role)/[\w+=,.@/-]+$"
)

_LABEL_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

MAX_STRING_PARAM_LENGTH = 256

# Comma-separated bucket names; when non-empty, source_uri bucket must be in the list.
ALLOWED_BUCKETS: set[str] = set(
    b.strip() for b in os.environ.get("COMPUTE_ALLOWED_BUCKETS", "").split(",") if b.strip()
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sfn = boto3.client("stepfunctions")

_PROFILES: dict[str, dict] | None = None


def _check_concurrent_limit(user_arn: str, state_machine_arn: str, limit: int) -> bool:
    """Return True if user is at or above concurrent job limit."""
    try:
        paginator = sfn.get_paginator("list_executions")
        running = 0
        for page in paginator.paginate(
            stateMachineArn=state_machine_arn,
            statusFilter="RUNNING",
            PaginationConfig={"MaxItems": 200},
        ):
            for ex in page.get("executions", []):
                try:
                    detail = sfn.describe_execution(executionArn=ex["executionArn"])
                    inp = json.loads(detail.get("input", "{}"))
                    if inp.get("user_arn") == user_arn:
                        running += 1
                        if running >= limit:
                            return True
                except Exception:
                    pass
        return False
    except Exception as exc:
        logger.warning(json.dumps({"concurrent_check_error": str(exc)}))
        return False  # Fail open


def _load_profiles() -> dict[str, dict]:
    global _PROFILES
    if _PROFILES is None:
        s3_uri = os.environ.get("PROFILES_S3_URI")
        if s3_uri:
            import boto3
            s3 = boto3.client("s3")
            bucket, key = s3_uri[len("s3://"):].split("/", 1)
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            _PROFILES = {p["profile_id"]: p for p in json.loads(body)}
        else:
            raw = os.environ.get("PROFILES_CONFIG", "[]")
            _PROFILES = {p["profile_id"]: p for p in json.loads(raw)}
    return _PROFILES


def _estimate_cost_from_dataset(profile: dict, source_uri: str) -> tuple[float, float]:
    """
    Return (estimated_cost_usd, estimated_duration_seconds) for Issue 22.
    Uses profile cost_estimate as base. If source_uri is an S3 path, calls
    head_object to get actual dataset size and scales the estimate proportionally
    relative to a 10 MB typical baseline. Fails open — returns profile estimate on error.
    """
    base_cost = float(profile.get("cost_estimate", {}).get("typical_cost_usd", 0))
    base_duration = float(profile.get("cost_estimate", {}).get("typical_duration_seconds", 30))

    if not source_uri or not source_uri.startswith("s3://"):
        return base_cost, base_duration

    try:
        without_scheme = source_uri[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        if not bucket or not key:
            return base_cost, base_duration
        if ALLOWED_BUCKETS and bucket not in ALLOWED_BUCKETS:
            return base_cost, base_duration
        s3_client = boto3.client("s3")
        head = s3_client.head_object(Bucket=bucket, Key=key)
        size_bytes = head.get("ContentLength", 0)
        if size_bytes <= 0:
            return base_cost, base_duration
        # Scale relative to 10 MB typical baseline (clamped to 0.1x–10x)
        baseline_bytes = 10 * 1024 * 1024
        scale = max(0.1, min(10.0, size_bytes / baseline_bytes))
        return round(base_cost * scale, 6), round(base_duration * scale, 1)
    except Exception as exc:
        logger.warning(json.dumps({"cost_estimate_s3_error": str(exc)}))
        return base_cost, base_duration


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
            elif len(val) < spec.get("min_columns", 0):
                errors.append(
                    f"Parameter '{param_name}' requires at least {spec['min_columns']} columns"
                )
            else:
                for i, c in enumerate(val):
                    if not isinstance(c, str) or len(c) > MAX_STRING_PARAM_LENGTH:
                        errors.append(
                            f"Parameter '{param_name}[{i}]' must be a string "
                            f"≤ {MAX_STRING_PARAM_LENGTH} characters"
                        )
        elif ptype == "string_list":
            if not isinstance(val, list):
                errors.append(f"Parameter '{param_name}' must be a list of strings")
            elif "max_items" in spec and len(val) > spec["max_items"]:
                errors.append(f"Parameter '{param_name}' exceeds max {spec['max_items']} items")
            else:
                for i, s in enumerate(val):
                    if not isinstance(s, str) or len(s) > MAX_STRING_PARAM_LENGTH:
                        errors.append(
                            f"Parameter '{param_name}[{i}]' must be a string "
                            f"≤ {MAX_STRING_PARAM_LENGTH} characters"
                        )
        elif ptype == "column":
            if not isinstance(val, str):
                errors.append(f"Parameter '{param_name}' must be a column name string")
            elif len(val) > MAX_STRING_PARAM_LENGTH:
                errors.append(
                    f"Parameter '{param_name}' exceeds max length {MAX_STRING_PARAM_LENGTH}"
                )

    return errors


def handler(event: dict, context) -> dict:
    _tool_name = "unknown"
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        _tool_name = raw.split("___")[-1]
    except Exception:
        pass
    logger.info(json.dumps({"tool": _tool_name, "event": event}))

    # Check for parallel-profile invocation or single profile_id
    profiles_list = event.get("profiles")
    profile_id = (event.get("profile_id") or "").strip()
    if not profiles_list and not profile_id:
        return {"error": 'Required parameter "profile_id" is missing'}

    # Validate source_uri if provided
    source_uri = (event.get("source_uri") or "").strip()
    if source_uri:
        if not (source_uri.startswith("s3://") or source_uri.startswith("claws://")):
            return {"error": "source_uri must start with 's3://' or 'claws://'"}
        if source_uri.startswith("s3://") and ALLOWED_BUCKETS:
            bucket = source_uri[5:].split("/")[0]
            if bucket not in ALLOWED_BUCKETS:
                return {"error": "source_uri bucket is not in the allowed list"}

    dataset_id = (event.get("dataset_id") or "").strip()
    if not dataset_id and not source_uri:
        return {"error": 'Either "dataset_id" or "source_uri" is required'}

    user_arn = (event.get("user_arn") or "").strip()
    if not user_arn:
        return {"error": 'Required parameter "user_arn" is missing'}
    if not _ARN_RE.match(user_arn):
        return {"error": "user_arn must be a valid IAM ARN "
                         "(arn:aws:iam::ACCOUNT:(user|role|assumed-role)/NAME)"}

    if profiles_list:
        return _run_parallel(event, user_arn, dataset_id, source_uri)

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

    # Issue #23: Cross-stack router spend pre-check against qs-router-spend table.
    # Reads the router's spend ledger for the request's department (current month).
    # Blocked only when department spend + estimated cost exceeds the budget cap.
    # Fails open on any AWS error — SFN CheckBudget remains the authoritative gate.
    department = (event.get("department") or "default").strip() or "default"
    router_spend_table_name = os.environ.get("ROUTER_SPEND_TABLE", "").strip()
    if router_spend_table_name:
        try:
            from boto3.dynamodb.conditions import Attr
            dynamodb_router = boto3.resource("dynamodb")
            router_table = dynamodb_router.Table(router_spend_table_name)
            month = datetime.now(timezone.utc).strftime("%Y-%m")
            # Scan for records matching this department and current month (YYYY-MM prefix on date)
            scan_resp = router_table.scan(
                FilterExpression=(
                    Attr("department").eq(department) & Attr("date").begins_with(month)
                ),
                ProjectionExpression="cost_usd",
            )
            router_dept_spend = sum(
                float(item.get("cost_usd", 0)) for item in scan_resp.get("Items", [])
            )
            budget_limit = float(os.environ.get("MONTHLY_BUDGET_USD", "50"))
            estimated_cost = float(profile.get("cost_estimate", {}).get("typical_cost_usd", 0))
            if router_dept_spend + estimated_cost > budget_limit:
                logger.info(json.dumps({
                    "router_budget_blocked": True,
                    "department": department,
                    "router_dept_spend": router_dept_spend,
                    "estimated_cost": estimated_cost,
                    "cap_usd": budget_limit,
                }))
                return {
                    "status": "budget_exceeded",
                    "department": department,
                    "cap_usd": budget_limit,
                    "spent_usd": router_dept_spend,
                }
        except Exception as exc:
            # Fail open — cross-stack spend table is advisory
            logger.warning(json.dumps({"router_spend_check_error": str(exc)}))

    # Check concurrent job limit
    max_concurrent = int(os.environ.get("MAX_CONCURRENT_JOBS_PER_USER", "2"))
    sfn_arn = os.environ.get("STATE_MACHINE_ARN", "")
    if sfn_arn and _check_concurrent_limit(user_arn, sfn_arn, max_concurrent):
        return {
            "status": "concurrent_limit_exceeded",
            "message": f"You already have {max_concurrent} job(s) running.",
            "limit": max_concurrent,
        }

    # Validate parameters
    user_params = event.get("parameters") or {}
    errors = _validate_params(profile, user_params)
    if errors:
        return {
            "error": "Parameter validation failed",
            "validation_errors": errors,
        }

    # Issue 21: validate chain_profile_id if provided
    chain_profile_id = (event.get("chain_profile_id") or "").strip()
    chain_profile = None
    if chain_profile_id:
        profiles = _load_profiles()
        chain_profile = profiles.get(chain_profile_id)
        if not chain_profile:
            return {
                "error": f"chain_profile_id '{chain_profile_id}' not found",
                "available_profiles": sorted(profiles.keys()),
            }
        chain_enable_emr = os.environ.get("ENABLE_EMR", "false").lower() == "true"
        if chain_profile.get("backend") == "emr_serverless" and not chain_enable_emr:
            return {
                "status": "requires_emr",
                "profile_id": chain_profile_id,
                "message": (
                    f"chain_profile_id '{chain_profile['display_name']}' requires EMR Serverless, "
                    "which is not enabled in this deployment."
                ),
            }

    # Issue 22: compute pre-submission cost estimate using profile + dataset size
    result_label = (event.get("result_label") or "").strip()
    if result_label and not _LABEL_RE.match(result_label):
        return {"error": "result_label must be 1–64 alphanumeric, hyphen, or underscore characters"}
    est_cost, est_duration = _estimate_cost_from_dataset(profile, source_uri)
    if chain_profile:
        chain_cost = float(chain_profile.get("cost_estimate", {}).get("typical_cost_usd", 0))
        chain_dur = float(chain_profile.get("cost_estimate", {}).get("typical_duration_seconds", 30))
        est_cost = round(est_cost + chain_cost, 6)
        est_duration = round(est_duration + chain_dur, 1)

    job_id, execution_arn, err = _start_execution(
        profile, user_arn, dataset_id, source_uri,
        event.get("dataset_name"), event.get("parameters") or {},
        result_label=result_label,
        chain_profile=chain_profile,
    )
    if err:
        return {"error": err}

    logger.info(json.dumps({
        "started": True,
        "job_id": job_id,
        "execution_arn": execution_arn,
        "profile_id": profile_id,
        "result_label": result_label or None,
        "chain_profile_id": chain_profile_id or None,
    }))

    resp = {
        "status": "started",
        "job_id": job_id,
        "execution_arn": execution_arn,
        "profile_id": profile_id,
        "profile_display_name": profile["display_name"],
        "estimated_cost_usd": est_cost,
        "estimated_duration_seconds": est_duration,
        "message": (
            f"Started {profile['display_name']} job. "
            f"Call compute_status with job_id='{job_id}' to check progress."
        ),
    }
    if result_label:
        resp["result_label"] = result_label
    if chain_profile_id:
        resp["chain_profile_id"] = chain_profile_id
    return resp


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
                     source_uri: str, dataset_name, user_params: dict,
                     result_label: str = "", chain_profile: dict | None = None) -> tuple:
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

    # Issue 19: pass result_label so record-spend can write to snapshots table
    if result_label:
        execution_input["result_label"] = result_label

    # Issue 21: pass chain_profile so SFN workflow can run second profile
    if chain_profile:
        execution_input["chain_profile"] = chain_profile

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
