"""
Tests for compute-run and compute-status Lambda handlers.

Unit tests mock Step Functions. Integration tests use Substrate.
"""

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).parent.parent


def _load_handler(lambda_dir: str, module_alias: str):
    path = REPO_ROOT / "lambdas" / lambda_dir / "handler.py"
    spec = importlib.util.spec_from_file_location(module_alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = mod
    spec.loader.exec_module(mod)
    return mod


_run = _load_handler("compute-run", "_compute_run_handler")
_status = _load_handler("compute-status", "_compute_status_handler")
_budget = _load_handler("check-budget", "_check_budget_handler")
_record = _load_handler("record-spend", "_record_spend_handler")


# ===========================================================================
# compute-run
# ===========================================================================

class TestComputeRun:
    def test_missing_profile_id_returns_error(self):
        result = _run.handler({}, None)
        assert "error" in result
        assert "profile_id" in result["error"]

    def test_missing_dataset_id_returns_error(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _run._PROFILES = None
            result = _run.handler({"profile_id": "clustering-kmeans"}, None)
        assert "error" in result
        assert "dataset_id" in result["error"]

    def test_missing_user_arn_returns_error(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _run._PROFILES = None
            result = _run.handler(
                {"profile_id": "clustering-kmeans", "dataset_id": "ds-123"}, None
            )
        assert "error" in result
        assert "user_arn" in result["error"]

    def test_unknown_profile_returns_error_with_available(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _run._PROFILES = None
            result = _run.handler(
                {"profile_id": "does-not-exist", "dataset_id": "ds", "user_arn": "arn:aws:iam::123456789012:user/u"},
                None,
            )
        assert "error" in result
        assert "available_profiles" in result

    def test_emr_profile_without_emr_returns_requires_emr(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json, "ENABLE_EMR": "false"}):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "transform-spark",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                },
                None,
            )
        assert result["status"] == "requires_emr"

    def test_valid_job_starts_execution(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123456789012:execution:qs-compute-job:job-abc"
        }
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/analyst",
                    "parameters": {"k": 5, "features": ["gpa", "credits"]},
                },
                None,
            )
        assert result["status"] == "started"
        assert "job_id" in result
        assert "execution_arn" in result
        assert result["profile_id"] == "clustering-kmeans"
        mock_sfn.start_execution.assert_called_once()

    def test_sfn_failure_returns_error(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.side_effect = Exception("SFN throttled")
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                },
                None,
            )
        assert "error" in result

    def test_k_param_out_of_range_returns_error(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 100},
                },
                None,
            )
        assert "error" in result
        assert "validation_errors" in result

    def test_custom_dataset_name_in_execution_input(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {"executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"}
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "dataset_name": "My Clusters",
                },
                None,
            )
        call_kwargs = mock_sfn.start_execution.call_args[1]
        execution_input = json.loads(call_kwargs["input"])
        assert execution_input["dataset_name"] == "My Clusters"

    def test_defaults_merged_into_execution_params(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {"executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"}
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    # k not specified — should use default=5
                },
                None,
            )
        call_kwargs = mock_sfn.start_execution.call_args[1]
        execution_input = json.loads(call_kwargs["input"])
        assert execution_input["parameters"].get("k") == 5


# ===========================================================================
# compute-status
# ===========================================================================

class TestComputeStatus:
    def test_missing_job_id_returns_error(self):
        result = _status.handler({}, None)
        assert "error" in result
        assert "job_id" in result["error"]

    def test_nonexistent_job_returns_error(self):
        mock_sfn = MagicMock()
        mock_sfn.exceptions.ExecutionDoesNotExist = type("ExecutionDoesNotExist", (Exception,), {})
        mock_sfn.describe_execution.side_effect = mock_sfn.exceptions.ExecutionDoesNotExist()
        with patch.object(_status, "sfn", mock_sfn):
            result = _status.handler({"job_id": "job-missing-123"}, None)
        assert "error" in result

    def test_running_job_returns_running_status(self):
        from datetime import datetime, timezone
        mock_sfn = MagicMock()
        mock_sfn.exceptions.ExecutionDoesNotExist = type("ExecutionDoesNotExist", (Exception,), {})
        mock_sfn.describe_execution.return_value = {
            "status": "RUNNING",
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:job-abc",
            "startDate": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            "stopDate": None,
        }
        with patch.object(_status, "sfn", mock_sfn):
            result = _status.handler(
                {"job_id": "arn:aws:states:us-east-1:123:execution:sm:job-abc"}, None
            )
        assert result["status"] == "RUNNING"
        assert "message" in result

    def test_succeeded_job_returns_result_dataset(self):
        from datetime import datetime, timezone
        mock_sfn = MagicMock()
        mock_sfn.exceptions.ExecutionDoesNotExist = type("ExecutionDoesNotExist", (Exception,), {})
        mock_sfn.describe_execution.return_value = {
            "status": "SUCCEEDED",
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:job-abc",
            "startDate": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            "stopDate": datetime(2024, 1, 1, 10, 0, 30, tzinfo=timezone.utc),
            "output": json.dumps({
                "deliver": {
                    "dataset_id": "qs-compute-abc123-dataset",
                    "result_dataset_name": "K-Means Clustering — 2024-01-01",
                }
            }),
        }
        with patch.object(_status, "sfn", mock_sfn):
            result = _status.handler(
                {"job_id": "arn:aws:states:us-east-1:123:execution:sm:job-abc"}, None
            )
        assert result["status"] == "SUCCEEDED"
        assert result["result_dataset_id"] == "qs-compute-abc123-dataset"
        assert result["elapsed_seconds"] == 30.0

    def test_failed_job_returns_error_message(self):
        from datetime import datetime, timezone
        mock_sfn = MagicMock()
        mock_sfn.exceptions.ExecutionDoesNotExist = type("ExecutionDoesNotExist", (Exception,), {})
        mock_sfn.describe_execution.return_value = {
            "status": "FAILED",
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:job-abc",
            "startDate": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            "stopDate": datetime(2024, 1, 1, 10, 0, 10, tzinfo=timezone.utc),
            "output": json.dumps({"failure": {"error_message": "Too few rows for k=5"}}),
        }
        with patch.object(_status, "sfn", mock_sfn):
            result = _status.handler(
                {"job_id": "arn:aws:states:us-east-1:123:execution:sm:job-abc"}, None
            )
        assert result["status"] == "FAILED"
        assert result["error"] is not None

    def test_full_execution_arn_used_directly(self):
        from datetime import datetime, timezone
        mock_sfn = MagicMock()
        mock_sfn.exceptions.ExecutionDoesNotExist = type("ExecutionDoesNotExist", (Exception,), {})
        mock_sfn.describe_execution.return_value = {
            "status": "RUNNING",
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:job-xyz",
            "startDate": datetime.now(timezone.utc),
            "stopDate": None,
        }
        arn = "arn:aws:states:us-east-1:123:execution:sm:job-xyz"
        with patch.object(_status, "sfn", mock_sfn):
            _status.handler({"job_id": arn}, None)
        call_kwargs = mock_sfn.describe_execution.call_args[1]
        assert call_kwargs["executionArn"] == arn


# ===========================================================================
# check-budget
# ===========================================================================

class TestCheckBudget:
    def test_under_budget_returns_ok(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.get_item.return_value = {
            "Item": {"user_arn": "arn:iam::123:user/u", "month": "2024-01", "spend_usd": 10.0}
        }
        with patch.object(_budget, "dynamodb", mock_ddb):
            result = _budget.handler(
                {"user_arn": "arn:iam::123:user/u", "estimated_cost_usd": 0.01}, None
            )
        assert result["budget_ok"] is True
        assert result["spend_usd"] == 10.0

    def test_over_budget_returns_not_ok(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.get_item.return_value = {
            "Item": {"spend_usd": 49.99}
        }
        with patch.object(_budget, "dynamodb", mock_ddb):
            result = _budget.handler(
                {"user_arn": "arn:iam::123:user/u", "estimated_cost_usd": 1.0}, None
            )
        assert result["budget_ok"] is False

    def test_no_prior_spend_treats_as_zero(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.get_item.return_value = {"Item": None}
        with patch.object(_budget, "dynamodb", mock_ddb):
            result = _budget.handler(
                {"user_arn": "arn:iam::123:user/u", "estimated_cost_usd": 0.01}, None
            )
        assert result["budget_ok"] is True
        assert result["spend_usd"] == 0.0

    def test_dynamodb_error_fails_open(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.get_item.side_effect = Exception("DDB unavailable")
        with patch.object(_budget, "dynamodb", mock_ddb):
            result = _budget.handler(
                {"user_arn": "arn:iam::123:user/u", "estimated_cost_usd": 0.01}, None
            )
        assert result["budget_ok"] is True  # fail open


# ===========================================================================
# record-spend
# ===========================================================================

class TestRecordSpend:
    def test_records_profile_estimate_cost(self):
        mock_ddb = MagicMock()
        with patch.object(_record, "dynamodb", mock_ddb):
            result = _record.handler(
                {
                    "user_arn": "arn:iam::123:user/u",
                    "profile": {"cost_estimate": {"typical_cost_usd": 0.01}},
                    "execution_id": "exec-123",
                },
                None,
            )
        assert result["recorded"] is True
        assert result["spend_usd"] == 0.01
        mock_ddb.Table.return_value.update_item.assert_called_once()

    def test_uses_actual_cost_if_available(self):
        mock_ddb = MagicMock()
        with patch.object(_record, "dynamodb", mock_ddb):
            result = _record.handler(
                {
                    "user_arn": "arn:iam::123:user/u",
                    "profile": {"cost_estimate": {"typical_cost_usd": 0.01}},
                    "compute": {"actual_cost_usd": 0.05},
                    "execution_id": "exec-123",
                },
                None,
            )
        assert result["spend_usd"] == 0.05

    def test_dynamodb_error_does_not_raise(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.update_item.side_effect = Exception("DDB error")
        with patch.object(_record, "dynamodb", mock_ddb):
            result = _record.handler(
                {
                    "user_arn": "arn:iam::123:user/u",
                    "profile": {"cost_estimate": {"typical_cost_usd": 0.01}},
                    "execution_id": "exec-123",
                },
                None,
            )
        assert result["recorded"] is False  # non-fatal — error is surfaced in return value
