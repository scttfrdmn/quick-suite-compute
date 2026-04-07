"""
Unit tests for compute-history and compute-cancel handlers.
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).parent.parent


def _load_handler(lambda_dir: str, alias: str):
    path = REPO_ROOT / "lambdas" / lambda_dir / "handler.py"
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_history = _load_handler("compute-history", "_history_handler")
_cancel = _load_handler("compute-cancel", "_cancel_handler")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_ITEMS = [
    {
        "user_arn": "arn:aws:iam::123456789012:user/alice",
        "started_at": "2026-04-01T10:00:00",
        "execution_id": "exec-001",
        "profile_id": "clustering-kmeans",
        "cost_usd": Decimal("0.01"),
        "duration_seconds": Decimal("25"),
        "status": "succeeded",
    },
    {
        "user_arn": "arn:aws:iam::123456789012:user/alice",
        "started_at": "2026-03-31T09:00:00",
        "execution_id": "exec-002",
        "profile_id": "regression-glm",
        "cost_usd": Decimal("0.02"),
        "duration_seconds": Decimal("45"),
        "status": "succeeded",
    },
]


def _mock_ddb(items=None):
    ddb = MagicMock()
    items = items if items is not None else []
    ddb.Table.return_value.query.return_value = {
        "Items": items,
        "Count": len(items),
    }
    return ddb


# ---------------------------------------------------------------------------
# TestComputeHistory
# ---------------------------------------------------------------------------

class TestComputeHistory:

    def test_happy_path_returns_jobs(self):
        with patch.object(_history, "dynamodb", _mock_ddb(SAMPLE_ITEMS)):
            result = _history.handler(
                {"user_arn": "arn:aws:iam::123456789012:user/alice"}, None
            )
        assert result["count"] == 2
        assert result["jobs"][0]["execution_id"] == "exec-001"
        assert result["jobs"][0]["profile_id"] == "clustering-kmeans"
        assert result["jobs"][0]["status"] == "succeeded"
        assert isinstance(result["jobs"][0]["cost_usd"], float)
        assert isinstance(result["jobs"][0]["duration_seconds"], float)

    def test_missing_user_arn_returns_error(self):
        with patch.object(_history, "dynamodb", _mock_ddb()):
            result = _history.handler({}, None)
        assert "error" in result

    def test_whitespace_user_arn_returns_error(self):
        with patch.object(_history, "dynamodb", _mock_ddb()):
            result = _history.handler({"user_arn": "   "}, None)
        assert "error" in result

    def test_limit_clamped_to_max_items(self):
        mock_ddb = _mock_ddb(SAMPLE_ITEMS)
        with patch.object(_history, "dynamodb", mock_ddb):
            _history.handler(
                {"user_arn": "arn:aws:iam::123456789012:user/alice", "limit": 25}, None
            )
        _, kwargs = mock_ddb.Table.return_value.query.call_args
        assert kwargs.get("Limit", None) == 20 or mock_ddb.Table.return_value.query.call_args[1].get("Limit") == 20

    def test_limit_defaults_to_10_when_absent(self):
        mock_ddb = _mock_ddb(SAMPLE_ITEMS)
        with patch.object(_history, "dynamodb", mock_ddb):
            _history.handler(
                {"user_arn": "arn:aws:iam::123456789012:user/alice"}, None
            )
        call_kwargs = mock_ddb.Table.return_value.query.call_args[1]
        assert call_kwargs.get("Limit") == 10

    def test_limit_string_not_a_number_defaults_10(self):
        mock_ddb = _mock_ddb(SAMPLE_ITEMS)
        with patch.object(_history, "dynamodb", mock_ddb):
            result = _history.handler(
                {"user_arn": "arn:aws:iam::123456789012:user/alice", "limit": "bad"}, None
            )
        call_kwargs = mock_ddb.Table.return_value.query.call_args[1]
        assert call_kwargs.get("Limit") == 10
        assert "error" not in result

    def test_empty_results_returns_empty_list(self):
        with patch.object(_history, "dynamodb", _mock_ddb([])):
            result = _history.handler(
                {"user_arn": "arn:aws:iam::123456789012:user/alice"}, None
            )
        assert result["jobs"] == []
        assert result["count"] == 0

    def test_dynamodb_error_returns_error_dict(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.query.side_effect = Exception("throttle")
        with patch.object(_history, "dynamodb", mock_ddb):
            result = _history.handler(
                {"user_arn": "arn:aws:iam::123456789012:user/alice"}, None
            )
        assert "error" in result

    def test_no_next_cursor_when_no_last_evaluated_key(self):
        with patch.object(_history, "dynamodb", _mock_ddb(SAMPLE_ITEMS)):
            result = _history.handler(
                {"user_arn": "arn:aws:iam::123456789012:user/alice"}, None
            )
        assert "next_cursor" not in result

    def test_next_cursor_returned_when_last_evaluated_key_present(self):
        import base64
        import json
        last_key = {"user_arn": {"S": "arn:aws:iam::123456789012:user/alice"}, "started_at": {"S": "2026-01-01"}}
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.query.return_value = {
            "Items": SAMPLE_ITEMS,
            "Count": len(SAMPLE_ITEMS),
            "LastEvaluatedKey": last_key,
        }
        with patch.object(_history, "dynamodb", mock_ddb):
            result = _history.handler(
                {"user_arn": "arn:aws:iam::123456789012:user/alice"}, None
            )
        assert "next_cursor" in result
        decoded = json.loads(base64.b64decode(result["next_cursor"]).decode())
        assert decoded == last_key

    def test_valid_cursor_passed_as_exclusive_start_key(self):
        import base64
        import json
        start_key = {"user_arn": {"S": "arn:aws:iam::123456789012:user/alice"}, "started_at": {"S": "2026-01-01"}}
        cursor = base64.b64encode(json.dumps(start_key).encode()).decode()
        mock_ddb = _mock_ddb(SAMPLE_ITEMS)
        with patch.object(_history, "dynamodb", mock_ddb):
            _history.handler(
                {"user_arn": "arn:aws:iam::123456789012:user/alice", "cursor": cursor}, None
            )
        call_kwargs = mock_ddb.Table.return_value.query.call_args[1]
        assert call_kwargs.get("ExclusiveStartKey") == start_key

    def test_malformed_cursor_silently_ignored(self):
        mock_ddb = _mock_ddb(SAMPLE_ITEMS)
        with patch.object(_history, "dynamodb", mock_ddb):
            result = _history.handler(
                {"user_arn": "arn:aws:iam::123456789012:user/alice", "cursor": "not-valid-base64!!!"}, None
            )
        call_kwargs = mock_ddb.Table.return_value.query.call_args[1]
        assert "ExclusiveStartKey" not in call_kwargs
        assert "error" not in result


# ---------------------------------------------------------------------------
# TestComputeCancel
# ---------------------------------------------------------------------------

# Custom exception classes to simulate sfn exception types.
ExecutionDoesNotExist = type("ExecutionDoesNotExist", (Exception,), {})
ExecutionAlreadyComplete = type("AlreadyCompleteException", (Exception,), {})
ExecutionInvalidState = type("InvalidStateForOperation", (Exception,), {})

_USER_ARN = "arn:aws:iam::123456789012:user/alice"
_SM_ARN = "arn:aws:states:us-east-1:123456789012:stateMachine:qs-compute-job"
_JOB_ID = "job-deadbeef-20260401"


def _mock_sfn(stop_side_effect=None, describe_side_effect=None, owner_arn=None):
    mock = MagicMock()
    if stop_side_effect is not None:
        mock.stop_execution.side_effect = stop_side_effect
    else:
        mock.stop_execution.return_value = {}
    if describe_side_effect is not None:
        mock.describe_execution.side_effect = describe_side_effect
    else:
        import json as _json
        mock.describe_execution.return_value = {
            "input": _json.dumps({"user_arn": owner_arn or _USER_ARN})
        }
    mock.exceptions.ExecutionDoesNotExist = ExecutionDoesNotExist
    return mock


class TestComputeCancel:

    def test_happy_path_cancels_execution(self):
        with patch.object(_cancel, "sfn", _mock_sfn()):
            result = _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        assert result == {"status": "cancelled", "job_id": _JOB_ID}

    def test_correct_execution_arn_constructed(self):
        mock_sfn = _mock_sfn()
        with patch.object(_cancel, "sfn", mock_sfn):
            _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        call_kwargs = mock_sfn.stop_execution.call_args[1]
        arn = call_kwargs["executionArn"]
        assert ":execution:" in arn
        assert ":stateMachine:" not in arn
        assert arn.endswith(f":{_JOB_ID}")

    def test_missing_job_id_returns_error(self):
        with patch.object(_cancel, "sfn", _mock_sfn()):
            result = _cancel.handler({"user_arn": _USER_ARN}, None)
        assert "error" in result

    def test_whitespace_job_id_returns_error(self):
        with patch.object(_cancel, "sfn", _mock_sfn()):
            result = _cancel.handler({"job_id": "  ", "user_arn": _USER_ARN}, None)
        assert "error" in result

    def test_missing_user_arn_returns_error(self):
        with patch.object(_cancel, "sfn", _mock_sfn()):
            result = _cancel.handler({"job_id": _JOB_ID}, None)
        assert "error" in result

    def test_execution_does_not_exist_at_describe(self):
        mock_sfn = _mock_sfn(describe_side_effect=ExecutionDoesNotExist())
        with patch.object(_cancel, "sfn", mock_sfn):
            result = _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_execution_does_not_exist_at_stop(self):
        mock_sfn = _mock_sfn(stop_side_effect=ExecutionDoesNotExist())
        with patch.object(_cancel, "sfn", mock_sfn):
            result = _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_execution_already_complete(self):
        mock_sfn = _mock_sfn(stop_side_effect=ExecutionAlreadyComplete())
        with patch.object(_cancel, "sfn", mock_sfn):
            result = _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        assert "error" in result
        assert "already completed" in result["error"].lower() or "cannot be cancelled" in result["error"].lower()

    def test_generic_exception_returns_error_dict(self):
        mock_sfn = _mock_sfn(stop_side_effect=Exception("network error"))
        with patch.object(_cancel, "sfn", mock_sfn):
            result = _cancel.handler({"job_id": _JOB_ID, "user_arn": _USER_ARN}, None)
        assert "error" in result
