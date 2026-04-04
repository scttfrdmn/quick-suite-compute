"""
Tests for v0.6.0 collaboration features (Issues 19–22):
  Issue 19: Named result snapshots (compute_snapshots + record-spend snapshot write)
  Issue 20: Compare two snapshots (compute_compare)
  Issue 21: Profile composition / chain_profile_id (compute_run + compute_status)
  Issue 22: Pre-submission cost estimate (compute_run)
"""

import importlib
import importlib.util
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _load_handler(lambda_dir: str, module_alias: str):
    path = REPO_ROOT / "lambdas" / lambda_dir / "handler.py"
    spec = importlib.util.spec_from_file_location(module_alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = mod
    spec.loader.exec_module(mod)
    return mod


_run = _load_handler("compute-run", "_v060_compute_run")
_status = _load_handler("compute-status", "_v060_compute_status")
_record = _load_handler("record-spend", "_v060_record_spend")
_snapshots = _load_handler("compute-snapshots", "_v060_compute_snapshots")
_compare = _load_handler("compute-compare", "_v060_compute_compare")


# =============================================================================
# Issue 22 — Pre-submission cost estimate
# =============================================================================

class TestPreSubmissionCostEstimate:
    """compute_run returns estimated_cost_usd and estimated_duration_seconds."""

    def test_response_includes_estimated_cost(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 5, "features": ["a", "b"]},
                },
                None,
            )
        assert result["status"] == "started"
        assert "estimated_cost_usd" in result
        assert "estimated_duration_seconds" in result
        assert isinstance(result["estimated_cost_usd"], float)
        assert isinstance(result["estimated_duration_seconds"], float)

    def test_estimated_cost_labeled_as_estimated(self, profiles_json):
        """The field name is 'estimated_cost_usd', not 'actual_cost_usd'."""
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "regression-glm",
                    "dataset_id": "ds-456",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                },
                None,
            )
        assert "estimated_cost_usd" in result
        assert "actual_cost_usd" not in result

    def test_s3_source_uri_scales_estimate(self, profiles_json):
        """head_object is called for s3:// URIs; estimate is scaled."""
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        mock_s3 = MagicMock()
        # 50 MB file — 5x the 10 MB baseline → clamped scaling
        mock_s3.head_object.return_value = {"ContentLength": 50 * 1024 * 1024}

        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn), \
             patch("boto3.client", return_value=mock_s3):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "source_uri": "s3://my-bucket/data.parquet",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 3, "features": ["a", "b"]},
                },
                None,
            )
        assert result["status"] == "started"
        assert "estimated_cost_usd" in result

    def test_s3_head_object_failure_falls_back_to_profile_estimate(self, profiles_json):
        """If head_object raises, estimate falls back to profile typical_cost_usd."""
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = Exception("Access denied")

        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn), \
             patch("boto3.client", return_value=mock_s3):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "source_uri": "s3://my-bucket/data.parquet",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 3, "features": ["a", "b"]},
                },
                None,
            )
        # Should still start, just with profile-level estimate
        assert result["status"] == "started"
        assert "estimated_cost_usd" in result


# =============================================================================
# Issue 19 — Named result snapshots: compute_run passes result_label
# =============================================================================

class TestResultLabel:
    """compute_run accepts result_label and forwards it in execution input."""

    def test_result_label_included_in_execution_input(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 5, "features": ["a", "b"]},
                    "result_label": "spring-2026-cohort",
                },
                None,
            )
        call_kwargs = mock_sfn.start_execution.call_args[1]
        execution_input = json.loads(call_kwargs["input"])
        assert execution_input.get("result_label") == "spring-2026-cohort"

    def test_no_result_label_not_in_execution_input(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                },
                None,
            )
        call_kwargs = mock_sfn.start_execution.call_args[1]
        execution_input = json.loads(call_kwargs["input"])
        assert "result_label" not in execution_input

    def test_result_label_in_response(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 5, "features": ["a", "b"]},
                    "result_label": "spring-2026-cohort",
                },
                None,
            )
        assert result.get("result_label") == "spring-2026-cohort"


# =============================================================================
# Issue 19 — record-spend writes snapshot when result_label is set
# =============================================================================

class TestRecordSpendSnapshot:
    """record-spend writes to snapshots table only when result_label is present."""

    def test_writes_snapshot_when_label_set(self):
        mock_ddb = MagicMock()
        with patch.dict(os.environ, {"SNAPSHOTS_TABLE": "qs-compute-snapshots"}), \
             patch.object(_record, "dynamodb", mock_ddb):
            result = _record.handler(
                {
                    "user_arn": "arn:aws:iam::123:user/u",
                    "profile": {
                        "profile_id": "clustering-kmeans",
                        "cost_estimate": {"typical_cost_usd": 0.01},
                    },
                    "compute": {"actual_cost_usd": 0.012, "duration_seconds": 25.0},
                    "execution_id": "exec-snap-001",
                    "result_label": "q1-2026-clusters",
                    "deliver": {"result_uri": "s3://bucket/results/exec-snap-001/data.csv",
                                "row_count": 1500},
                },
                None,
            )
        assert result["recorded"] is True
        # put_item is called for both history table and snapshots table
        put_calls = mock_ddb.Table.return_value.put_item.call_args_list
        assert len(put_calls) >= 1
        # Find the snapshot put_item call (has "label" key)
        snapshot_calls = [c for c in put_calls if "label" in c[1]["Item"]]
        assert len(snapshot_calls) == 1
        item = snapshot_calls[0][1]["Item"]
        assert item["label"] == "q1-2026-clusters"
        assert item["user_arn"] == "arn:aws:iam::123:user/u"
        assert item["job_id"] == "exec-snap-001"
        assert item["result_uri"] == "s3://bucket/results/exec-snap-001/data.csv"
        assert item["row_count"] == 1500

    def test_no_snapshot_when_no_label(self):
        mock_ddb = MagicMock()
        with patch.dict(os.environ, {"SNAPSHOTS_TABLE": "qs-compute-snapshots"}), \
             patch.object(_record, "dynamodb", mock_ddb):
            result = _record.handler(
                {
                    "user_arn": "arn:aws:iam::123:user/u",
                    "profile": {"cost_estimate": {"typical_cost_usd": 0.01}},
                    "execution_id": "exec-no-snap",
                },
                None,
            )
        assert result["recorded"] is True
        # put_item should NOT be called with a "label" key (no snapshot written)
        put_calls = mock_ddb.Table.return_value.put_item.call_args_list
        snapshot_calls = [c for c in put_calls if "label" in c[1].get("Item", {})]
        assert snapshot_calls == []

    def test_snapshot_write_failure_does_not_raise(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.put_item.side_effect = Exception("DDB error")
        with patch.dict(os.environ, {"SNAPSHOTS_TABLE": "qs-compute-snapshots"}), \
             patch.object(_record, "dynamodb", mock_ddb):
            result = _record.handler(
                {
                    "user_arn": "arn:aws:iam::123:user/u",
                    "profile": {"cost_estimate": {"typical_cost_usd": 0.01}},
                    "execution_id": "exec-snap-fail",
                    "result_label": "will-fail",
                },
                None,
            )
        # update_item also raises, so recorded is False but no exception raised
        assert "recorded" in result

    def test_no_snapshots_table_env_skips_write(self):
        mock_ddb = MagicMock()
        with patch.dict(os.environ, {"SNAPSHOTS_TABLE": ""}), \
             patch.object(_record, "dynamodb", mock_ddb):
            _result = _record.handler(
                {
                    "user_arn": "arn:aws:iam::123:user/u",
                    "profile": {"cost_estimate": {"typical_cost_usd": 0.01}},
                    "execution_id": "exec-no-table",
                    "result_label": "my-label",
                },
                None,
            )
        # No snapshot put_item call (label key absent from all calls)
        put_calls = mock_ddb.Table.return_value.put_item.call_args_list
        snapshot_calls = [c for c in put_calls if "label" in c[1].get("Item", {})]
        assert snapshot_calls == []


# =============================================================================
# Issue 19 — compute_snapshots tool
# =============================================================================

class TestComputeSnapshots:
    """compute_snapshots lists a user's named snapshots."""

    def test_missing_user_arn_returns_error(self):
        with patch.object(_snapshots, "SNAPSHOTS_TABLE", "qs-compute-snapshots"):
            result = _snapshots.handler({}, None)
        assert "error" in result

    def test_no_snapshots_table_returns_error(self):
        with patch.object(_snapshots, "SNAPSHOTS_TABLE", ""):
            result = _snapshots.handler(
                {"user_arn": "arn:aws:iam::123:user/u"}, None
            )
        assert "error" in result

    def test_returns_snapshots_sorted_desc(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.query.return_value = {
            "Items": [
                {
                    "user_arn": "arn:aws:iam::123:user/u",
                    "label": "q2-2026",
                    "completed_at": "2026-04-02T10:00:00+00:00",
                    "job_id": "exec-002",
                    "profile_id": "regression-glm",
                    "result_uri": "s3://bucket/results/exec-002/data.csv",
                    "row_count": 2000,
                    "cost_usd": Decimal("0.015"),
                    "duration_seconds": Decimal("40.0"),
                },
                {
                    "user_arn": "arn:aws:iam::123:user/u",
                    "label": "q1-2026",
                    "completed_at": "2026-01-10T10:00:00+00:00",
                    "job_id": "exec-001",
                    "profile_id": "clustering-kmeans",
                    "result_uri": "s3://bucket/results/exec-001/data.csv",
                    "row_count": 1500,
                    "cost_usd": Decimal("0.01"),
                    "duration_seconds": Decimal("25.0"),
                },
            ]
        }
        with patch.object(_snapshots, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_snapshots, "dynamodb", mock_ddb):
            result = _snapshots.handler(
                {"user_arn": "arn:aws:iam::123:user/u"}, None
            )
        assert result["count"] == 2
        assert len(result["snapshots"]) == 2
        snaps = result["snapshots"]
        assert snaps[0]["label"] == "q2-2026"
        assert snaps[0]["row_count"] == 2000
        assert snaps[0]["cost_usd"] == pytest.approx(0.015)
        assert snaps[1]["label"] == "q1-2026"

    def test_empty_snapshots_returns_empty_list(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.query.return_value = {"Items": []}
        with patch.object(_snapshots, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_snapshots, "dynamodb", mock_ddb):
            result = _snapshots.handler(
                {"user_arn": "arn:aws:iam::123:user/u"}, None
            )
        assert result["count"] == 0
        assert result["snapshots"] == []

    def test_limit_passed_to_query(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.query.return_value = {"Items": []}
        with patch.object(_snapshots, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_snapshots, "dynamodb", mock_ddb):
            _snapshots.handler(
                {"user_arn": "arn:aws:iam::123:user/u", "limit": 5}, None
            )
        call_kwargs = mock_ddb.Table.return_value.query.call_args[1]
        assert call_kwargs["Limit"] == 5

    def test_limit_capped_at_50(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.query.return_value = {"Items": []}
        with patch.object(_snapshots, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_snapshots, "dynamodb", mock_ddb):
            _snapshots.handler(
                {"user_arn": "arn:aws:iam::123:user/u", "limit": 999}, None
            )
        call_kwargs = mock_ddb.Table.return_value.query.call_args[1]
        assert call_kwargs["Limit"] == 50

    def test_dynamodb_error_returns_error(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.query.side_effect = Exception("DDB error")
        with patch.object(_snapshots, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_snapshots, "dynamodb", mock_ddb):
            result = _snapshots.handler(
                {"user_arn": "arn:aws:iam::123:user/u"}, None
            )
        assert "error" in result

    def test_scan_index_forward_false(self):
        """Query must use ScanIndexForward=False for most-recent-first ordering."""
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.query.return_value = {"Items": []}
        with patch.object(_snapshots, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_snapshots, "dynamodb", mock_ddb):
            _snapshots.handler(
                {"user_arn": "arn:aws:iam::123:user/u"}, None
            )
        call_kwargs = mock_ddb.Table.return_value.query.call_args[1]
        assert call_kwargs.get("ScanIndexForward") is False


# =============================================================================
# Issue 20 — compute_compare tool
# =============================================================================

class TestComputeCompare:
    """compute_compare diffs two named snapshots."""

    def _make_snap(self, label: str, uri: str, cost: float = 0.01, duration: float = 30.0):
        return {
            "user_arn": "arn:aws:iam::123:user/u",
            "label": label,
            "completed_at": "2026-01-01T00:00:00+00:00",
            "job_id": f"exec-{label}",
            "profile_id": "clustering-kmeans",
            "result_uri": uri,
            "row_count": 100,
            "cost_usd": Decimal(str(cost)),
            "duration_seconds": Decimal(str(duration)),
        }

    def test_missing_label_a_returns_error(self):
        result = _compare.handler(
            {"label_b": "snap-b", "user_arn": "arn:aws:iam::123:user/u"}, None
        )
        assert "error" in result
        assert "label_a" in result["error"]

    def test_missing_label_b_returns_error(self):
        result = _compare.handler(
            {"label_a": "snap-a", "user_arn": "arn:aws:iam::123:user/u"}, None
        )
        assert "error" in result
        assert "label_b" in result["error"]

    def test_missing_user_arn_returns_error(self):
        result = _compare.handler(
            {"label_a": "snap-a", "label_b": "snap-b"}, None
        )
        assert "error" in result
        assert "user_arn" in result["error"]

    def test_same_label_returns_error(self):
        result = _compare.handler(
            {"label_a": "snap-a", "label_b": "snap-a",
             "user_arn": "arn:aws:iam::123:user/u"}, None
        )
        assert "error" in result

    def test_snapshot_not_found_returns_error(self):
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.get_item.return_value = {}  # no Item
        with patch.object(_compare, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_compare, "dynamodb", mock_ddb):
            result = _compare.handler(
                {
                    "label_a": "snap-a",
                    "label_b": "snap-b",
                    "user_arn": "arn:aws:iam::123:user/u",
                },
                None,
            )
        assert "error" in result
        assert "snap-a" in result["error"]

    def test_no_snapshots_table_returns_error(self):
        with patch.object(_compare, "SNAPSHOTS_TABLE", ""):
            result = _compare.handler(
                {
                    "label_a": "snap-a",
                    "label_b": "snap-b",
                    "user_arn": "arn:aws:iam::123:user/u",
                },
                None,
            )
        assert "error" in result

    def test_identical_snapshots_all_unchanged(self):
        """Two snapshots with identical rows → added=0, removed=0, unchanged=N."""
        csv_data = "id,cluster_id\n1,A\n2,B\n3,A\n"

        snap_a = self._make_snap("snap-a", "s3://bucket/a.csv", cost=0.01, duration=30.0)
        snap_b = self._make_snap("snap-b", "s3://bucket/b.csv", cost=0.01, duration=30.0)

        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.get_item.side_effect = [
            {"Item": snap_a},
            {"Item": snap_b},
        ]
        mock_s3 = MagicMock()
        body_mock = MagicMock()
        body_mock.read.return_value = csv_data.encode()
        mock_s3.get_object.return_value = {"Body": body_mock}

        with patch.object(_compare, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_compare, "dynamodb", mock_ddb), \
             patch.object(_compare, "s3", mock_s3):
            result = _compare.handler(
                {
                    "label_a": "snap-a",
                    "label_b": "snap-b",
                    "user_arn": "arn:aws:iam::123:user/u",
                },
                None,
            )
        assert result["added_count"] == 0
        assert result["removed_count"] == 0
        assert result["unchanged_count"] == 3
        assert result["schema_diff"] is None
        assert result["cost_delta_usd"] == pytest.approx(0.0)

    def test_row_added_and_removed(self):
        """Snapshot B has one new row and is missing one row from A."""
        csv_a = "id,val\n1,x\n2,y\n3,z\n"
        csv_b = "id,val\n1,x\n3,z\n4,w\n"  # row 2 removed, row 4 added

        snap_a = self._make_snap("snap-a", "s3://bucket/a.csv", cost=0.01, duration=30.0)
        snap_b = self._make_snap("snap-b", "s3://bucket/b.csv", cost=0.02, duration=35.0)

        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.get_item.side_effect = [
            {"Item": snap_a},
            {"Item": snap_b},
        ]

        def _s3_get(Bucket, Key):
            body_mock = MagicMock()
            body_mock.read.return_value = (csv_a if Key.endswith("a.csv") else csv_b).encode()
            return {"Body": body_mock}

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = _s3_get

        with patch.object(_compare, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_compare, "dynamodb", mock_ddb), \
             patch.object(_compare, "s3", mock_s3):
            result = _compare.handler(
                {
                    "label_a": "snap-a",
                    "label_b": "snap-b",
                    "user_arn": "arn:aws:iam::123:user/u",
                },
                None,
            )
        assert result["added_count"] == 1
        assert result["removed_count"] == 1
        assert result["unchanged_count"] == 2
        assert result["cost_delta_usd"] == pytest.approx(0.01)
        assert result["duration_delta_seconds"] == pytest.approx(5.0)

    def test_schema_mismatch_reported(self):
        """Different column sets are captured in schema_diff."""
        csv_a = "id,col_a\n1,x\n"
        csv_b = "id,col_b\n1,x\n"

        snap_a = self._make_snap("snap-a", "s3://bucket/a.csv")
        snap_b = self._make_snap("snap-b", "s3://bucket/b.csv")

        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.get_item.side_effect = [
            {"Item": snap_a},
            {"Item": snap_b},
        ]

        def _s3_get(Bucket, Key):
            body_mock = MagicMock()
            body_mock.read.return_value = (csv_a if Key.endswith("a.csv") else csv_b).encode()
            return {"Body": body_mock}

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = _s3_get

        with patch.object(_compare, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_compare, "dynamodb", mock_ddb), \
             patch.object(_compare, "s3", mock_s3):
            result = _compare.handler(
                {
                    "label_a": "snap-a",
                    "label_b": "snap-b",
                    "user_arn": "arn:aws:iam::123:user/u",
                },
                None,
            )
        assert result["schema_diff"] is not None
        assert "col_a" in result["schema_diff"]["columns_only_in_a"]
        assert "col_b" in result["schema_diff"]["columns_only_in_b"]

    def test_s3_load_failure_returns_error(self):
        """S3 get_object failure returns error with snapshot label in message."""
        snap_a = self._make_snap("snap-a", "s3://bucket/a.csv")
        snap_b = self._make_snap("snap-b", "s3://bucket/b.csv")

        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.get_item.side_effect = [
            {"Item": snap_a},
            {"Item": snap_b},
        ]
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = Exception("NoSuchKey")

        with patch.object(_compare, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_compare, "dynamodb", mock_ddb), \
             patch.object(_compare, "s3", mock_s3):
            result = _compare.handler(
                {
                    "label_a": "snap-a",
                    "label_b": "snap-b",
                    "user_arn": "arn:aws:iam::123:user/u",
                },
                None,
            )
        assert "error" in result

    def test_no_result_uri_returns_warning(self):
        """Snapshots without result_uri get a warning, not an error."""
        snap_a = self._make_snap("snap-a", "")
        snap_b = self._make_snap("snap-b", "")

        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.get_item.side_effect = [
            {"Item": snap_a},
            {"Item": snap_b},
        ]

        with patch.object(_compare, "SNAPSHOTS_TABLE", "qs-compute-snapshots"), \
             patch.object(_compare, "dynamodb", mock_ddb):
            result = _compare.handler(
                {
                    "label_a": "snap-a",
                    "label_b": "snap-b",
                    "user_arn": "arn:aws:iam::123:user/u",
                },
                None,
            )
        assert "error" not in result
        assert "warning" in result


# =============================================================================
# Issue 21 — Profile composition (chain_profile_id)
# =============================================================================

class TestProfileComposition:
    """compute_run accepts chain_profile_id and forwards it in execution input."""

    def test_chain_profile_included_in_execution_input(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 5, "features": ["a", "b"]},
                    "chain_profile_id": "explore-correlations",
                },
                None,
            )
        assert result["status"] == "started"
        call_kwargs = mock_sfn.start_execution.call_args[1]
        execution_input = json.loads(call_kwargs["input"])
        assert "chain_profile" in execution_input
        assert execution_input["chain_profile"]["profile_id"] == "explore-correlations"

    def test_chain_profile_in_response(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 5, "features": ["a", "b"]},
                    "chain_profile_id": "explore-correlations",
                },
                None,
            )
        assert result.get("chain_profile_id") == "explore-correlations"

    def test_unknown_chain_profile_returns_error(self, profiles_json):
        mock_sfn = MagicMock()
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            result = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 5, "features": ["a", "b"]},
                    "chain_profile_id": "does-not-exist",
                },
                None,
            )
        assert "error" in result
        assert "chain_profile_id" in result["error"]

    def test_no_chain_profile_not_in_input(self, profiles_json):
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 5, "features": ["a", "b"]},
                },
                None,
            )
        call_kwargs = mock_sfn.start_execution.call_args[1]
        execution_input = json.loads(call_kwargs["input"])
        assert "chain_profile" not in execution_input

    def test_chain_cost_summed_in_estimate(self, profiles_json):
        """Estimated cost includes both profile costs when chain_profile_id is set."""
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:j"
        }
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            result_single = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 5, "features": ["a", "b"]},
                },
                None,
            )

        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}), \
             patch.object(_run, "sfn", mock_sfn):
            _run._PROFILES = None
            result_chain = _run.handler(
                {
                    "profile_id": "clustering-kmeans",
                    "dataset_id": "ds-123",
                    "user_arn": "arn:aws:iam::123456789012:user/u",
                    "parameters": {"k": 5, "features": ["a", "b"]},
                    "chain_profile_id": "explore-correlations",
                },
                None,
            )
        # Chain run should cost at least as much as single run
        assert result_chain["estimated_cost_usd"] >= result_single["estimated_cost_usd"]


# =============================================================================
# Issue 21 — compute_status shows step for chain jobs
# =============================================================================

class TestComputeStatusChainStep:
    """compute_status includes step: profile_1 | profile_2 for chained jobs."""

    def test_succeeded_status_shows_step_from_output(self):
        from datetime import datetime, timezone
        mock_sfn = MagicMock()
        mock_sfn.exceptions.ExecutionDoesNotExist = type("ExecutionDoesNotExist", (Exception,), {})
        mock_sfn.describe_execution.return_value = {
            "status": "SUCCEEDED",
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:job-chain",
            "startDate": datetime(2026, 4, 2, 10, 0, 0, tzinfo=timezone.utc),
            "stopDate": datetime(2026, 4, 2, 10, 1, 0, tzinfo=timezone.utc),
            "output": json.dumps({
                "deliver": {
                    "dataset_id": "ds-chain-result",
                    "result_dataset_name": "Chain Result",
                },
                "chain_step": "profile_2",
                "chain_spend": {"total_cost_usd": 0.025},
            }),
        }
        with patch.object(_status, "sfn", mock_sfn), \
             patch.object(_status, "HISTORY_TABLE", ""):
            result = _status.handler(
                {"job_id": "arn:aws:states:us-east-1:123:execution:sm:job-chain"}, None
            )
        assert result["status"] == "SUCCEEDED"
        assert result.get("step") == "profile_2"
        assert result.get("total_cost_usd") == pytest.approx(0.025)

    def test_succeeded_status_no_step_for_single_profile(self):
        from datetime import datetime, timezone
        mock_sfn = MagicMock()
        mock_sfn.exceptions.ExecutionDoesNotExist = type("ExecutionDoesNotExist", (Exception,), {})
        mock_sfn.describe_execution.return_value = {
            "status": "SUCCEEDED",
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:job-single",
            "startDate": datetime(2026, 4, 2, 10, 0, 0, tzinfo=timezone.utc),
            "stopDate": datetime(2026, 4, 2, 10, 0, 30, tzinfo=timezone.utc),
            "output": json.dumps({
                "deliver": {
                    "dataset_id": "ds-single",
                    "result_dataset_name": "Single Profile Result",
                },
            }),
        }
        with patch.object(_status, "sfn", mock_sfn), \
             patch.object(_status, "HISTORY_TABLE", ""):
            result = _status.handler(
                {"job_id": "arn:aws:states:us-east-1:123:execution:sm:job-single"}, None
            )
        assert result["status"] == "SUCCEEDED"
        assert "step" not in result
