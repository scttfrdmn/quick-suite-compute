"""
Security hardening tests — quick-suite-compute v0.14.0

Covers:
  #82  compute-cancel ownership check
  #79  record-spend conditional write budget backstop
  #75  compute-run source_uri bucket allowlist (SSRF)
  #76  result_label validation (compute-run + record-spend)
  #80  _validate_params max-length for string/column types
  #81  CDK stack: DynamoDB PITR + deletion_protection on all tables
  #84  CDK stack: log retention on all Lambda functions
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

REPO_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(REPO_ROOT))

from stacks.compute_stack import ComputeStack  # noqa: E402

# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------

def _load(lambda_dir: str, alias: str):
    path = REPO_ROOT / "lambdas" / lambda_dir / "handler.py"
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_cancel = _load("compute-cancel", "_sec_cancel_handler")
_record = _load("record-spend", "_sec_record_handler")
_run = _load("compute-run", "_sec_run_handler")


# ---------------------------------------------------------------------------
# CDK stack fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stack_template():
    app = cdk.App()
    stack = ComputeStack(app, "TestComputeStack")
    return Template.from_stack(stack)


# ---------------------------------------------------------------------------
# #82 — TestCancelOwnership
# ---------------------------------------------------------------------------

_USER_ARN = "arn:aws:iam::123456789012:user/alice"
_OTHER_ARN = "arn:aws:iam::123456789012:user/bob"
_JOB_ID = "job-deadbeef-20260401"

ExecutionDoesNotExist = type("ExecutionDoesNotExist", (Exception,), {})


def _mock_sfn_for_cancel(owner_arn=_USER_ARN, stop_side_effect=None, describe_side_effect=None):
    mock = MagicMock()
    if describe_side_effect:
        mock.describe_execution.side_effect = describe_side_effect
    else:
        mock.describe_execution.return_value = {"input": json.dumps({"user_arn": owner_arn})}
    if stop_side_effect:
        mock.stop_execution.side_effect = stop_side_effect
    else:
        mock.stop_execution.return_value = {}
    mock.exceptions.ExecutionDoesNotExist = ExecutionDoesNotExist
    return mock


class TestCancelOwnership:

    def test_owner_can_cancel(self):
        with patch.object(_cancel, "sfn", _mock_sfn_for_cancel()):
            result = _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        assert result == {"status": "cancelled", "job_id": _JOB_ID}

    def test_non_owner_gets_not_found(self):
        with patch.object(_cancel, "sfn", _mock_sfn_for_cancel(owner_arn=_OTHER_ARN)):
            result = _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        assert "error" in result
        assert "not found" in result["error"].lower()
        # stop_execution must NOT have been called
    def test_non_owner_stop_not_called(self):
        mock_sfn = _mock_sfn_for_cancel(owner_arn=_OTHER_ARN)
        with patch.object(_cancel, "sfn", mock_sfn):
            _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        mock_sfn.stop_execution.assert_not_called()

    def test_missing_user_arn_returns_error(self):
        with patch.object(_cancel, "sfn", _mock_sfn_for_cancel()):
            result = _cancel.handler({"job_id": _JOB_ID}, None)
        assert "error" in result
        assert "user_arn" in result["error"]

    def test_execution_not_found_at_describe(self):
        mock_sfn = _mock_sfn_for_cancel(describe_side_effect=ExecutionDoesNotExist())
        with patch.object(_cancel, "sfn", mock_sfn):
            result = _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_no_info_leak_on_ownership_mismatch(self):
        with patch.object(_cancel, "sfn", _mock_sfn_for_cancel(owner_arn=_OTHER_ARN)):
            result = _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        # Same message as "not found" — no "unauthorized" or "forbidden"
        assert "not found" in result["error"].lower()
        assert "unauthorized" not in result["error"].lower()
        assert "forbidden" not in result["error"].lower()


# ---------------------------------------------------------------------------
# #79 — TestBudgetConditionalWrite
# ---------------------------------------------------------------------------

from botocore.exceptions import ClientError  # noqa: E402


def _make_conditional_check_failed():
    error_response = {"Error": {"Code": "ConditionalCheckFailedException", "Message": "cond"}}
    return ClientError(error_response, "UpdateItem")


class TestBudgetConditionalWrite:

    def _make_event(self):
        return {
            "user_arn": _USER_ARN,
            "execution_id": "exec-001",
            "profile": {"profile_id": "clustering-kmeans", "cost_estimate": {"typical_cost_usd": 0.01}},
        }

    def test_conditional_check_failed_logs_warning_not_raises(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.update_item.side_effect = _make_conditional_check_failed()
        env = {"SPEND_TABLE": "test-spend"}
        with patch.object(_record, "dynamodb", mock_ddb), \
             patch.dict(os.environ, env):
            result = _record.handler(self._make_event(), None)
        # Should not surface as an error to the caller
        assert result.get("recorded") is True or "error" not in result

    def test_successful_update_proceeds_normally(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.update_item.return_value = {}
        env = {"SPEND_TABLE": "test-spend"}
        with patch.object(_record, "dynamodb", mock_ddb), \
             patch.dict(os.environ, env):
            result = _record.handler(self._make_event(), None)
        assert result.get("recorded") is True

    def test_condition_expression_present_in_update_call(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.update_item.return_value = {}
        env = {"SPEND_TABLE": "test-spend"}
        with patch.object(_record, "dynamodb", mock_ddb), \
             patch.dict(os.environ, env):
            _record.handler(self._make_event(), None)
        call_kwargs = mock_ddb.Table.return_value.update_item.call_args[1]
        assert "ConditionExpression" in call_kwargs

    def test_remaining_attribute_in_expression_values(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.update_item.return_value = {}
        env = {"SPEND_TABLE": "test-spend", "MONTHLY_BUDGET_USD": "50"}
        with patch.object(_record, "dynamodb", mock_ddb), \
             patch.dict(os.environ, env):
            _record.handler(self._make_event(), None)
        call_kwargs = mock_ddb.Table.return_value.update_item.call_args[1]
        assert ":remaining" in call_kwargs["ExpressionAttributeValues"]

    def test_non_conditional_client_error_returns_error(self):
        mock_ddb = MagicMock()
        err = ClientError({"Error": {"Code": "ProvisionedThroughputExceededException", "Message": "x"}}, "UpdateItem")
        mock_ddb.Table.return_value.update_item.side_effect = err
        env = {"SPEND_TABLE": "test-spend"}
        with patch.object(_record, "dynamodb", mock_ddb), \
             patch.dict(os.environ, env):
            result = _record.handler(self._make_event(), None)
        assert result.get("recorded") is False


# ---------------------------------------------------------------------------
# #75 — TestSourceUriAllowlist
# ---------------------------------------------------------------------------

class TestSourceUriAllowlist:

    def _run(self, source_uri: str, allowed_buckets: str = ""):
        env = {
            "COMPUTE_ALLOWED_BUCKETS": allowed_buckets,
            "SPEND_TABLE": "t",
            "STATE_MACHINE_ARN": "arn:aws:states:us-east-1:123456789012:stateMachine:qs-compute-job",
        }
        event = {
            "profile_id": "clustering-kmeans",
            "source_uri": source_uri,
            "user_arn": _USER_ARN,
        }
        with patch.dict(os.environ, env):
            # Patch ALLOWED_BUCKETS to reflect env (module-level constant)
            buckets = set(b.strip() for b in allowed_buckets.split(",") if b.strip())
            with patch.object(_run, "ALLOWED_BUCKETS", buckets):
                return _run.handler(event, None)

    def test_disallowed_bucket_rejected(self):
        result = self._run("s3://secret-internal-bucket/data.parquet", "approved-bucket")
        assert "error" in result
        assert "allowed" in result["error"].lower()

    def test_allowed_bucket_passes_validation(self):
        # Should not return an allowlist error (may fail later for other reasons)
        result = self._run("s3://approved-bucket/data.parquet", "approved-bucket")
        assert "bucket" not in result.get("error", "").lower() or "allowed" not in result.get("error", "").lower()

    def test_empty_allowlist_permits_any_bucket(self):
        # No allowlist set — backward compatible; any s3:// passes
        result = self._run("s3://any-bucket/data.parquet", "")
        assert "allowed" not in result.get("error", "")

    def test_claws_uri_always_permitted(self):
        result = self._run("claws://roda-noaa-ghcn", "approved-bucket")
        # claws:// URIs bypass the bucket allowlist check
        assert "allowed" not in result.get("error", "")

    def test_invalid_scheme_still_rejected(self):
        result = self._run("http://evil.com/data.parquet", "approved-bucket")
        assert "error" in result
        assert "s3://" in result["error"] or "claws://" in result["error"]


# ---------------------------------------------------------------------------
# #76 — TestResultLabelValidation
# ---------------------------------------------------------------------------

class TestResultLabelValidation:

    def _run_event(self, result_label: str):
        env = {
            "COMPUTE_ALLOWED_BUCKETS": "",
            "SPEND_TABLE": "t",
            "STATE_MACHINE_ARN": "arn:aws:states:us-east-1:123456789012:stateMachine:qs-compute-job",
        }
        event = {
            "profile_id": "clustering-kmeans",
            "dataset_id": "ds-001",
            "user_arn": _USER_ARN,
            "result_label": result_label,
        }
        with patch.dict(os.environ, env), patch.object(_run, "ALLOWED_BUCKETS", set()):
            return _run.handler(event, None)

    def test_label_too_long_returns_error(self):
        result = self._run_event("a" * 65)
        assert "error" in result
        assert "result_label" in result["error"]

    def test_label_with_slashes_returns_error(self):
        result = self._run_event("folder/name")
        assert "error" in result
        assert "result_label" in result["error"]

    def test_label_with_spaces_returns_error(self):
        result = self._run_event("my label")
        assert "error" in result
        assert "result_label" in result["error"]

    def test_valid_label_passes(self):
        env = {
            "COMPUTE_ALLOWED_BUCKETS": "",
            "SPEND_TABLE": "t",
            "STATE_MACHINE_ARN": "arn:aws:states:us-east-1:123456789012:stateMachine:qs-compute-job",
        }
        event = {
            "profile_id": "clustering-kmeans",
            "dataset_id": "ds-001",
            "user_arn": _USER_ARN,
            "result_label": "my-label_2026",
        }
        with patch.dict(os.environ, env), patch.object(_run, "ALLOWED_BUCKETS", set()):
            result = _run.handler(event, None)
        # Label validation passed (other errors may still occur)
        assert "result_label" not in result.get("error", "")

    def test_record_spend_skips_snapshot_on_invalid_label(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.update_item.return_value = {}
        env = {"SPEND_TABLE": "test-spend", "SNAPSHOTS_TABLE": "test-snaps"}
        event = {
            "user_arn": _USER_ARN,
            "execution_id": "exec-001",
            "profile": {"profile_id": "clustering-kmeans", "cost_estimate": {"typical_cost_usd": 0.01}},
            "result_label": "bad label/with/slashes",
        }
        with patch.object(_record, "dynamodb", mock_ddb), patch.dict(os.environ, env):
            result = _record.handler(event, None)
        # Snapshot table should NOT have been written
        table_calls = [str(c) for c in mock_ddb.Table.call_args_list]
        snap_table_written = any("test-snaps" in c for c in table_calls)
        # If the table was accessed, put_item should not have been called
        if snap_table_written:
            mock_ddb.Table.return_value.put_item.assert_not_called()
        assert result.get("recorded") is True  # overall success unaffected


# ---------------------------------------------------------------------------
# #80 — TestParamLengthValidation
# ---------------------------------------------------------------------------

_KMEANS_PROFILE = {
    "profile_id": "clustering-kmeans",
    "display_name": "K-Means",
    "parameters": {
        "k": {"type": "integer", "default": 5, "min": 2, "max": 20},
        "features": {"type": "column_list", "min_columns": 2},
        "target": {"type": "column"},
    },
}


class TestParamLengthValidation:

    def _validate(self, params: dict) -> list[str]:
        return _run._validate_params(_KMEANS_PROFILE, params)

    def test_column_too_long_rejected(self):
        errors = self._validate({
            "k": 3,
            "features": ["col1", "col2"],
            "target": "x" * 257,
        })
        assert any("target" in e for e in errors)

    def test_column_at_max_length_accepted(self):
        errors = self._validate({
            "k": 3,
            "features": ["col1", "col2"],
            "target": "x" * 256,
        })
        assert not any("target" in e for e in errors)

    def test_column_list_item_too_long_rejected(self):
        errors = self._validate({
            "k": 3,
            "features": ["col1", "x" * 257],
        })
        assert any("features" in e for e in errors)

    def test_column_list_item_at_max_length_accepted(self):
        errors = self._validate({
            "k": 3,
            "features": ["col1", "x" * 256],
        })
        assert not any("features" in e for e in errors)

    def test_valid_params_no_errors(self):
        errors = self._validate({
            "k": 5,
            "features": ["col_a", "col_b", "col_c"],
            "target": "outcome",
        })
        assert errors == []


# ---------------------------------------------------------------------------
# #81 — TestDynamoDbProtection
# ---------------------------------------------------------------------------

class TestDynamoDbProtection:

    def test_spend_table_has_pitr(self, stack_template):
        stack_template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": Match.string_like_regexp(".*spend.*"),
                "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True},
            },
        )

    def test_spend_table_has_deletion_protection(self, stack_template):
        stack_template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": Match.string_like_regexp(".*spend.*"),
                "DeletionProtectionEnabled": True,
            },
        )

    def test_snapshots_table_has_pitr(self, stack_template):
        stack_template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": Match.string_like_regexp(".*snapshots.*"),
                "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True},
            },
        )

    def test_snapshots_table_has_deletion_protection(self, stack_template):
        stack_template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": Match.string_like_regexp(".*snapshots.*"),
                "DeletionProtectionEnabled": True,
            },
        )

    def test_history_table_has_pitr(self, stack_template):
        stack_template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": Match.string_like_regexp(".*history.*"),
                "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True},
            },
        )

    def test_history_table_has_deletion_protection(self, stack_template):
        stack_template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": Match.string_like_regexp(".*history.*"),
                "DeletionProtectionEnabled": True,
            },
        )

    def test_schedules_table_has_pitr(self, stack_template):
        stack_template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": Match.string_like_regexp(".*schedules.*"),
                "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True},
            },
        )

    def test_schedules_table_has_deletion_protection(self, stack_template):
        stack_template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": Match.string_like_regexp(".*schedules.*"),
                "DeletionProtectionEnabled": True,
            },
        )

    def test_no_table_has_destroy_removal_policy(self, stack_template):
        tables = stack_template.find_resources("AWS::DynamoDB::Table")
        for logical_id, table in tables.items():
            policy = table.get("DeletionPolicy", "Delete")
            assert policy != "Delete", (
                f"Table {logical_id} has DeletionPolicy=Delete (RemovalPolicy.DESTROY)"
            )


# ---------------------------------------------------------------------------
# #84 — TestLogRetention
# ---------------------------------------------------------------------------

class TestLogRetention:

    def test_all_lambda_functions_have_log_retention(self, stack_template):
        """Every app Lambda function should have a corresponding log retention custom resource."""
        # CDK log_retention creates a Custom::LogRetention resource per function.
        # Count these resources and verify at least as many as app Lambdas.
        log_retentions = stack_template.find_resources("Custom::LogRetention")
        # There should be at least 16 retention configs (one per Lambda function)
        assert len(log_retentions) >= 16, (
            f"Expected ≥16 log retention configs, found {len(log_retentions)}"
        )

    def test_retention_period_is_90_days(self, stack_template):
        """Verify at least one log retention resource specifies 90-day retention."""
        stack_template.has_resource_properties(
            "Custom::LogRetention",
            {"RetentionInDays": 90},
        )
