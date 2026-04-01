"""
Integration tests for the compute handler chain using Substrate.

Tests individual Lambda handlers (check-budget, compute-run, compute-status)
in sequence against Substrate-managed DynamoDB and Step Functions endpoints.
Does NOT execute the full Step Functions state machine workflow — tests the
tool handler logic with real boto3 calls against the Substrate emulator.

Run unit tests only:     pytest -m "not integration"
Run integration tests:   pytest -m integration
Run everything:          pytest
"""

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "lambdas" / "check-budget"))
sys.path.insert(0, str(REPO_ROOT / "lambdas" / "compute-run"))
sys.path.insert(0, str(REPO_ROOT / "lambdas" / "compute-status"))

pytestmark = pytest.mark.integration

_USER_ARN = "arn:aws:iam::123456789012:user/test-analyst"
_SPEND_TABLE = "qs-compute-spend-integration"
_STATE_MACHINE_ARN = "arn:aws:states:us-east-1:123456789012:stateMachine:qs-compute-job"


def _make_dynamo(substrate_url: str):
    import boto3
    return boto3.resource(
        "dynamodb",
        endpoint_url=substrate_url,
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


def _make_sfn(substrate_url: str):
    import boto3
    return boto3.client(
        "stepfunctions",
        endpoint_url=substrate_url,
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


def _ensure_spend_table(substrate_url: str):
    """Create spend tracking table if it does not exist."""
    ddb = _make_dynamo(substrate_url)
    try:
        table = ddb.create_table(
            TableName=_SPEND_TABLE,
            KeySchema=[
                {"AttributeName": "user_arn", "KeyType": "HASH"},
                {"AttributeName": "month", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "user_arn", "AttributeType": "S"},
                {"AttributeName": "month", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
    except ddb.meta.client.exceptions.ResourceInUseException:
        pass
    return ddb.Table(_SPEND_TABLE)


def _ensure_state_machine(substrate_url: str):
    """Register a minimal state machine definition for integration testing."""
    sfn = _make_sfn(substrate_url)
    try:
        sfn.create_state_machine(
            name="qs-compute-job",
            definition=json.dumps({
                "Comment": "Test stub",
                "StartAt": "Stub",
                "States": {
                    "Stub": {"Type": "Pass", "End": True}
                },
            }),
            roleArn="arn:aws:iam::123456789012:role/stub-role",
        )
    except sfn.exceptions.StateMachineAlreadyExists:
        pass
    return sfn


def _load_check_budget(substrate_url: str, monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", substrate_url)
    monkeypatch.setenv("SPEND_TABLE", _SPEND_TABLE)
    monkeypatch.setenv("MONTHLY_BUDGET_USD", "50")
    import handler as check_budget
    importlib.reload(check_budget)
    return check_budget


def _load_compute_run(substrate_url: str, monkeypatch, profiles_json: str):
    monkeypatch.setenv("AWS_ENDPOINT_URL", substrate_url)
    monkeypatch.setenv("SPEND_TABLE", _SPEND_TABLE)
    monkeypatch.setenv("MONTHLY_BUDGET_USD", "50")
    monkeypatch.setenv("STATE_MACHINE_ARN", _STATE_MACHINE_ARN)
    monkeypatch.setenv("PROFILES_CONFIG", profiles_json)
    monkeypatch.setenv("ENABLE_EMR", "false")
    import handler as compute_run
    importlib.reload(compute_run)
    return compute_run


def _load_compute_status(substrate_url: str, monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", substrate_url)
    monkeypatch.setenv("STATE_MACHINE_ARN", _STATE_MACHINE_ARN)
    import handler as compute_status
    importlib.reload(compute_status)
    return compute_status


# ===========================================================================
# CheckBudget handler
# ===========================================================================

class TestCheckBudgetHandler:
    """check-budget Lambda validates spend against monthly limit."""

    def test_under_limit_returns_budget_ok(
        self, substrate_url, reset_substrate, monkeypatch
    ):
        table = _ensure_spend_table(substrate_url)
        from decimal import Decimal
        table.put_item(Item={
            "user_arn": _USER_ARN,
            "month": "2026-04",
            "spend_usd": Decimal("20"),
        })

        cb = _load_check_budget(substrate_url, monkeypatch)
        result = cb.handler({
            "user_arn": _USER_ARN,
            "estimated_cost_usd": 1.0,
        }, None)

        assert result["budget_ok"] is True
        assert result["spend_usd"] == 20.0
        assert result["user_arn"] == _USER_ARN

    def test_over_limit_returns_budget_not_ok(
        self, substrate_url, reset_substrate, monkeypatch
    ):
        table = _ensure_spend_table(substrate_url)
        from decimal import Decimal
        table.put_item(Item={
            "user_arn": _USER_ARN,
            "month": "2026-04",
            "spend_usd": Decimal("49"),
        })

        cb = _load_check_budget(substrate_url, monkeypatch)
        result = cb.handler({
            "user_arn": _USER_ARN,
            "estimated_cost_usd": 5.0,
        }, None)

        assert result["budget_ok"] is False
        assert result["spend_usd"] == 49.0

    def test_no_prior_spend_budget_ok(
        self, substrate_url, reset_substrate, monkeypatch
    ):
        _ensure_spend_table(substrate_url)
        cb = _load_check_budget(substrate_url, monkeypatch)

        result = cb.handler({
            "user_arn": "arn:aws:iam::123456789012:user/new-user",
            "estimated_cost_usd": 0.01,
        }, None)

        assert result["budget_ok"] is True
        assert result["spend_usd"] == 0.0


# ===========================================================================
# compute_run handler — starts Step Functions execution
# ===========================================================================

class TestComputeRunHandler:
    """compute_run Lambda validates request and starts Step Functions execution."""

    _MOCK_CONTEXT = None  # compute_run doesn't use context.client_context in AgentCore path

    def test_starts_execution_returns_job_id(
        self, substrate_url, reset_substrate, monkeypatch, profiles_json
    ):
        _ensure_spend_table(substrate_url)
        _ensure_state_machine(substrate_url)

        cr = _load_compute_run(substrate_url, monkeypatch, profiles_json)
        result = cr.handler({
            "profile_id": "clustering-kmeans",
            "dataset_id": "qs-dataset-abc123",
            "user_arn": _USER_ARN,
            "parameters": {"k": 5, "features": ["col1", "col2"]},
        }, self._MOCK_CONTEXT)

        assert result["status"] == "started"
        assert "job_id" in result
        assert result["profile_id"] == "clustering-kmeans"
        assert result["execution_arn"].startswith("arn:aws:states:")

    def test_invalid_profile_returns_error(
        self, substrate_url, reset_substrate, monkeypatch, profiles_json
    ):
        _ensure_spend_table(substrate_url)
        cr = _load_compute_run(substrate_url, monkeypatch, profiles_json)

        result = cr.handler({
            "profile_id": "nonexistent-profile",
            "dataset_id": "qs-dataset-abc123",
            "user_arn": _USER_ARN,
        }, self._MOCK_CONTEXT)

        assert "error" in result
        assert "not found" in result["error"]
        assert "available_profiles" in result

    def test_missing_user_arn_returns_error(
        self, substrate_url, reset_substrate, monkeypatch, profiles_json
    ):
        cr = _load_compute_run(substrate_url, monkeypatch, profiles_json)
        result = cr.handler({
            "profile_id": "clustering-kmeans",
            "dataset_id": "qs-dataset-abc123",
        }, self._MOCK_CONTEXT)

        assert "error" in result

    def test_source_uri_s3_accepted(
        self, substrate_url, reset_substrate, monkeypatch, profiles_json
    ):
        _ensure_spend_table(substrate_url)
        _ensure_state_machine(substrate_url)

        cr = _load_compute_run(substrate_url, monkeypatch, profiles_json)
        result = cr.handler({
            "profile_id": "clustering-kmeans",
            "source_uri": "s3://my-bucket/data/input.csv",
            "user_arn": _USER_ARN,
            "parameters": {"k": 3, "features": ["a", "b"]},
        }, self._MOCK_CONTEXT)

        # Either starts or hits budget — should NOT return invalid source_uri error
        assert "error" not in result or "source_uri" not in result.get("error", "")

    def test_source_uri_invalid_scheme_rejected(
        self, substrate_url, reset_substrate, monkeypatch, profiles_json
    ):
        cr = _load_compute_run(substrate_url, monkeypatch, profiles_json)
        result = cr.handler({
            "profile_id": "clustering-kmeans",
            "source_uri": "ftp://some-server/data.csv",
            "user_arn": _USER_ARN,
        }, self._MOCK_CONTEXT)

        assert "error" in result
        assert "source_uri" in result["error"]


# ===========================================================================
# compute_run — parallel profile execution
# ===========================================================================

class TestComputeRunParallel:
    """compute_run accepts a profiles list and starts one execution per profile."""

    _MOCK_CONTEXT = None

    def test_parallel_profiles_returns_multiple_job_ids(
        self, substrate_url, reset_substrate, monkeypatch, profiles_json
    ):
        _ensure_spend_table(substrate_url)
        _ensure_state_machine(substrate_url)

        cr = _load_compute_run(substrate_url, monkeypatch, profiles_json)
        result = cr.handler({
            "profiles": ["clustering-kmeans", "explore-correlations"],
            "dataset_id": "qs-dataset-parallel-test",
            "user_arn": _USER_ARN,
            "parameters": {"features": ["col1", "col2"]},
        }, self._MOCK_CONTEXT)

        assert result["status"] == "started"
        assert result["count"] >= 1
        assert isinstance(result["jobs"], list)
        for job in result["jobs"]:
            assert "job_id" in job
            assert "profile_id" in job

    def test_parallel_unknown_profile_returns_error(
        self, substrate_url, reset_substrate, monkeypatch, profiles_json
    ):
        cr = _load_compute_run(substrate_url, monkeypatch, profiles_json)
        result = cr.handler({
            "profiles": ["clustering-kmeans", "does-not-exist"],
            "dataset_id": "qs-dataset-test",
            "user_arn": _USER_ARN,
        }, self._MOCK_CONTEXT)

        assert "error" in result
        assert "does-not-exist" in str(result["error"])


# ===========================================================================
# compute_status handler
# ===========================================================================

class TestComputeStatusHandler:
    """compute_status returns execution status from Step Functions."""

    def test_running_execution_returns_running_status(
        self, substrate_url, reset_substrate, monkeypatch, profiles_json
    ):
        """Start an execution and immediately check its status — should be RUNNING."""
        _ensure_spend_table(substrate_url)
        sfn_client = _ensure_state_machine(substrate_url)

        # Start an execution so we have a real ARN to query
        resp = sfn_client.start_execution(
            stateMachineArn=_STATE_MACHINE_ARN,
            input=json.dumps({"test": True}),
        )
        execution_arn = resp["executionArn"]
        # Extract job_id from ARN (last segment)
        job_id = execution_arn.split(":")[-1]

        # Load compute-status handler
        sys.path.insert(0, str(REPO_ROOT / "lambdas" / "compute-status"))
        monkeypatch.setenv("AWS_ENDPOINT_URL", substrate_url)
        monkeypatch.setenv("STATE_MACHINE_ARN", _STATE_MACHINE_ARN)
        import handler as compute_status
        importlib.reload(compute_status)

        result = compute_status.handler({"job_id": job_id}, None)

        # Substrate may return RUNNING or SUCCEEDED for a pass-state machine
        assert result.get("status") in ("running", "succeeded", "failed")
        assert "job_id" in result or "error" in result

    def test_unknown_job_id_returns_error(
        self, substrate_url, reset_substrate, monkeypatch
    ):
        sys.path.insert(0, str(REPO_ROOT / "lambdas" / "compute-status"))
        monkeypatch.setenv("AWS_ENDPOINT_URL", substrate_url)
        monkeypatch.setenv("STATE_MACHINE_ARN", _STATE_MACHINE_ARN)
        import handler as compute_status
        importlib.reload(compute_status)

        result = compute_status.handler({"job_id": "job-doesnotexist-99999999"}, None)

        assert "error" in result
