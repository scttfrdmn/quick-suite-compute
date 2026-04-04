"""
E2E tests for compute_snapshots and compute_compare.

Snapshots are only written on job SUCCEEDED with a result_label set.
These tests verify the API shape and error handling without requiring
a completed job.
"""

import pytest

from tests.e2e.conftest import _E2E_USER_ARN, invoke

pytestmark = pytest.mark.e2e


class TestComputeSnapshotsE2E:
    def test_snapshots_returns_list(self, lam, tool_arns):
        """compute_snapshots returns a list (possibly empty) for the test user."""
        result = invoke(lam, tool_arns["compute_snapshots"], {"user_arn": _E2E_USER_ARN})
        assert "snapshots" in result or "error" not in result, \
            f"Unexpected snapshots response: {result}"
        snapshots = result.get("snapshots", result if isinstance(result, list) else [])
        assert isinstance(snapshots, list), \
            f"Expected list of snapshots: {result}"

    def test_snapshots_missing_user_arn_returns_error(self, lam, tool_arns):
        """compute_snapshots without user_arn returns an error."""
        result = invoke(lam, tool_arns["compute_snapshots"], {})
        assert "error" in result, f"Expected error for missing user_arn: {result}"


class TestComputeCompareE2E:
    def test_compare_missing_labels_returns_error(self, lam, tool_arns):
        """compute_compare without label_a or label_b returns an error."""
        result = invoke(lam, tool_arns["compute_compare"], {"user_arn": _E2E_USER_ARN})
        assert "error" in result, f"Expected error for missing labels: {result}"

    def test_compare_nonexistent_labels_returns_error(self, lam, tool_arns):
        """compute_compare with labels that don't exist returns an error."""
        result = invoke(lam, tool_arns["compute_compare"], {
            "user_arn": _E2E_USER_ARN,
            "label_a": "nonexistent-snapshot-aaa",
            "label_b": "nonexistent-snapshot-bbb",
        })
        assert "error" in result, f"Expected error for nonexistent labels: {result}"
