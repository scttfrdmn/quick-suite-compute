"""
Tests for lambdas/audit-log/handler.py (Issue #28).

Covers:
  - SUCCEEDED audit written with correct fields
  - FAILED audit written (result_uri and cost_usd may be None)
  - TIMED_OUT audit written
  - Audit object contains no PII (only URIs)
  - Audit key follows the correct S3 path layout
  - S3 write failure is re-raised (so SFN marks the step as failed)
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _load_handler(lambda_dir: str, alias: str):
    path = REPO_ROOT / "lambdas" / lambda_dir / "handler.py"
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_audit = _load_handler("audit-log", "_audit_log_handler")


def _make_succeeded_event():
    return {
        "execution_id": "exec-abc-123",
        "user_arn": "arn:aws:iam::123456789012:user/analyst",
        "profile": {
            "profile_id": "clustering-kmeans",
            "display_name": "K-Means Clustering",
            "cost_estimate": {"typical_cost_usd": 0.01},
        },
        "dataset_uri": "s3://my-bucket/inputs/data.parquet",
        "params": {"k": 5, "features": ["gpa", "credits"]},
        "deliver": {
            "result_uri": "s3://qs-compute-bucket/results/exec-abc-123/",
            "result_dataset_name": "clustering-result-xyz",
        },
        "compute": {
            "actual_cost_usd": 0.0082,
            "duration_seconds": 18.4,
        },
        "status": "SUCCEEDED",
    }


def _make_failed_event():
    return {
        "execution_id": "exec-fail-456",
        "user_arn": "arn:aws:iam::123456789012:user/analyst",
        "profile": {
            "profile_id": "regression-glm",
            "display_name": "GLM Regression",
            "cost_estimate": {"typical_cost_usd": 0.01},
        },
        "dataset_uri": "s3://my-bucket/inputs/data.parquet",
        "params": {"target": "gpa", "features": ["credits", "year"]},
        # deliver and compute are absent on failure paths
        "status": "FAILED",
    }


class TestAuditLogSucceeded:
    def test_writes_audit_to_s3(self):
        mock_s3 = MagicMock()
        with patch.object(_audit, "s3_client", mock_s3):
            result = _audit.handler(_make_succeeded_event(), None)

        assert result["audit_written"] is True
        mock_s3.put_object.assert_called_once()

    def test_audit_key_prefix(self):
        mock_s3 = MagicMock()
        with patch.object(_audit, "s3_client", mock_s3):
            result = _audit.handler(_make_succeeded_event(), None)

        assert result["audit_key"].startswith("audit/")
        assert result["audit_key"].endswith("exec-abc-123.json")

    def test_audit_object_contains_required_fields(self):
        captured = {}
        mock_s3 = MagicMock()

        def _capture(**kwargs):
            captured.update(kwargs)
            return {}

        mock_s3.put_object.side_effect = _capture
        with patch.object(_audit, "s3_client", mock_s3):
            _audit.handler(_make_succeeded_event(), None)

        body = json.loads(captured["Body"])
        for field in ("job_id", "profile_id", "user_arn", "dataset_uri",
                      "params", "result_uri", "cost_usd", "duration_seconds",
                      "status", "timestamp"):
            assert field in body, f"Missing field: {field}"

    def test_audit_object_correct_values(self):
        captured = {}
        mock_s3 = MagicMock()

        def _capture(**kwargs):
            captured.update(kwargs)
            return {}

        mock_s3.put_object.side_effect = _capture
        with patch.object(_audit, "s3_client", mock_s3):
            _audit.handler(_make_succeeded_event(), None)

        body = json.loads(captured["Body"])
        assert body["job_id"] == "exec-abc-123"
        assert body["profile_id"] == "clustering-kmeans"
        assert body["status"] == "SUCCEEDED"
        assert body["cost_usd"] == pytest.approx(0.0082, rel=1e-3)
        assert body["duration_seconds"] == pytest.approx(18.4, rel=1e-3)
        assert body["result_uri"] is not None

    def test_audit_bucket_is_compute_bucket(self):
        mock_s3 = MagicMock()
        with patch.dict(os.environ, {"COMPUTE_BUCKET": "qs-compute-test-bucket"}), \
             patch.object(_audit, "s3_client", mock_s3):
            _audit.handler(_make_succeeded_event(), None)

        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "qs-compute-test-bucket"

    def test_no_pii_in_audit_params_are_uris_or_primitives(self):
        """Params field contains user-supplied analysis parameters (scalars/lists),
        not raw data content.  dataset_uri and result_uri are S3 URIs, not row data."""
        captured = {}
        mock_s3 = MagicMock()

        def _capture(**kwargs):
            captured.update(kwargs)
            return {}

        mock_s3.put_object.side_effect = _capture
        with patch.object(_audit, "s3_client", mock_s3):
            _audit.handler(_make_succeeded_event(), None)

        body = json.loads(captured["Body"])
        # dataset_uri and result_uri must be string URIs, not dicts with row data
        assert isinstance(body["dataset_uri"], str)
        assert body["result_uri"] is None or isinstance(body["result_uri"], str)
        # params must not contain actual data rows
        assert isinstance(body["params"], dict)


class TestAuditLogFailed:
    def test_writes_audit_to_s3_on_failure(self):
        mock_s3 = MagicMock()
        with patch.object(_audit, "s3_client", mock_s3):
            result = _audit.handler(_make_failed_event(), None)

        assert result["audit_written"] is True
        mock_s3.put_object.assert_called_once()

    def test_result_uri_is_none_on_failure(self):
        captured = {}
        mock_s3 = MagicMock()

        def _capture(**kwargs):
            captured.update(kwargs)
            return {}

        mock_s3.put_object.side_effect = _capture
        with patch.object(_audit, "s3_client", mock_s3):
            _audit.handler(_make_failed_event(), None)

        body = json.loads(captured["Body"])
        assert body["result_uri"] is None

    def test_status_is_failed(self):
        captured = {}
        mock_s3 = MagicMock()

        def _capture(**kwargs):
            captured.update(kwargs)
            return {}

        mock_s3.put_object.side_effect = _capture
        with patch.object(_audit, "s3_client", mock_s3):
            _audit.handler(_make_failed_event(), None)

        body = json.loads(captured["Body"])
        assert body["status"] == "FAILED"

    def test_cost_usd_falls_back_to_profile_estimate_on_failure(self):
        """When compute block is missing (failure path), cost falls back to profile estimate."""
        captured = {}
        mock_s3 = MagicMock()

        def _capture(**kwargs):
            captured.update(kwargs)
            return {}

        mock_s3.put_object.side_effect = _capture
        with patch.object(_audit, "s3_client", mock_s3):
            _audit.handler(_make_failed_event(), None)

        body = json.loads(captured["Body"])
        # Falls back to profile.cost_estimate.typical_cost_usd = 0.01
        assert body["cost_usd"] == pytest.approx(0.01, rel=1e-3)


class TestAuditLogTimedOut:
    def test_timed_out_status_written(self):
        captured = {}
        mock_s3 = MagicMock()

        def _capture(**kwargs):
            captured.update(kwargs)
            return {}

        mock_s3.put_object.side_effect = _capture
        event = _make_failed_event()
        event["status"] = "TIMED_OUT"
        event["execution_id"] = "exec-timeout-789"
        with patch.object(_audit, "s3_client", mock_s3):
            result = _audit.handler(event, None)

        body = json.loads(captured["Body"])
        assert body["status"] == "TIMED_OUT"
        assert result["audit_key"].endswith("exec-timeout-789.json")


class TestAuditLogS3Failure:
    def test_s3_error_is_reraised(self):
        """If S3 write fails the step should raise so SFN marks it as failed."""
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = Exception("S3 write error")
        with patch.object(_audit, "s3_client", mock_s3):
            with pytest.raises(Exception, match="S3 write error"):
                _audit.handler(_make_succeeded_event(), None)
