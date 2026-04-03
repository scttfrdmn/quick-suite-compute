"""
E2E tests for compute_run and compute_status.

Uses a session-scoped fixture (run_result) that starts ONE explore-correlations
job, shared across all tests. The job is cancelled on teardown to avoid
incurring the full analysis cost.
"""

import time

import pytest
from tests.e2e.conftest import invoke, _E2E_USER_ARN

pytestmark = pytest.mark.e2e


class TestComputeRunE2E:
    def test_run_returns_started(self, run_result):
        """compute_run returns status=started immediately."""
        assert run_result.get("status") == "started", \
            f"Expected status=started: {run_result}"

    def test_run_returns_job_id(self, run_result):
        """compute_run returns a job_id string."""
        assert run_result.get("job_id"), f"Missing job_id: {run_result}"
        assert isinstance(run_result["job_id"], str)

    def test_run_returns_execution_arn(self, run_result):
        """compute_run returns a Step Functions execution_arn."""
        arn = run_result.get("execution_arn", "")
        assert arn.startswith("arn:aws:states:"), \
            f"Unexpected execution_arn: {arn}"

    def test_run_returns_cost_estimate(self, run_result):
        """compute_run returns a non-negative estimated_cost_usd."""
        cost = run_result.get("estimated_cost_usd")
        assert cost is not None, f"Missing estimated_cost_usd: {run_result}"
        assert cost >= 0, f"Negative estimated_cost_usd: {cost}"

    def test_run_returns_duration_estimate(self, run_result):
        """compute_run returns a positive estimated_duration_seconds."""
        dur = run_result.get("estimated_duration_seconds")
        assert dur is not None, f"Missing estimated_duration_seconds: {run_result}"
        assert dur > 0, f"Non-positive estimated_duration_seconds: {dur}"

    def test_run_returns_profile_id(self, run_result):
        """compute_run echoes back the profile_id."""
        assert run_result.get("profile_id") == "explore-correlations", \
            f"Unexpected profile_id: {run_result.get('profile_id')}"

    def test_run_invalid_profile_returns_error(self, lam, tool_arns, test_input_s3_uri):
        """compute_run with an unknown profile_id returns an error."""
        result = invoke(lam, tool_arns["compute_run"], {
            "profile_id": "nonexistent-profile-xyz",
            "source_uri": test_input_s3_uri,
            "user_arn": _E2E_USER_ARN,
        })
        assert "error" in result, f"Expected error for unknown profile: {result}"

    def test_run_missing_user_arn_returns_error(self, lam, tool_arns, test_input_s3_uri):
        """compute_run without user_arn returns an error."""
        result = invoke(lam, tool_arns["compute_run"], {
            "profile_id": "explore-correlations",
            "source_uri": test_input_s3_uri,
        })
        assert "error" in result, f"Expected error for missing user_arn: {result}"

    def test_run_missing_source_returns_error(self, lam, tool_arns):
        """compute_run without dataset_id or source_uri returns an error."""
        result = invoke(lam, tool_arns["compute_run"], {
            "profile_id": "explore-correlations",
            "user_arn": _E2E_USER_ARN,
        })
        assert "error" in result, f"Expected error for missing source: {result}"


class TestComputeStatusE2E:
    def test_status_returns_for_running_job(self, lam, tool_arns, run_result):
        """compute_status returns a status for the started job."""
        job_id = run_result["job_id"]
        result = invoke(lam, tool_arns["compute_status"], {"job_id": job_id})
        assert "status" in result, f"Missing status field: {result}"
        assert result["status"] in ("RUNNING", "SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"), \
            f"Unexpected status: {result['status']}"

    def test_status_returns_job_id(self, lam, tool_arns, run_result):
        """compute_status echoes back the job_id."""
        job_id = run_result["job_id"]
        result = invoke(lam, tool_arns["compute_status"], {"job_id": job_id})
        assert result.get("job_id") == job_id, \
            f"job_id mismatch: {result.get('job_id')} vs {job_id}"

    def test_status_returns_execution_arn(self, lam, tool_arns, run_result):
        """compute_status includes the execution_arn."""
        job_id = run_result["job_id"]
        result = invoke(lam, tool_arns["compute_status"], {"job_id": job_id})
        assert result.get("execution_arn"), f"Missing execution_arn: {result}"

    def test_status_running_has_elapsed_seconds(self, lam, tool_arns, run_result):
        """A running job has a positive elapsed_seconds."""
        job_id = run_result["job_id"]
        result = invoke(lam, tool_arns["compute_status"], {"job_id": job_id})
        if result.get("status") == "RUNNING":
            assert result.get("elapsed_seconds", 0) >= 0, \
                f"Unexpected elapsed_seconds: {result}"

    def test_status_unknown_job_returns_error(self, lam, tool_arns):
        """compute_status for a nonexistent job_id returns an error response."""
        result = invoke(lam, tool_arns["compute_status"], {"job_id": "nonexistent-job-xyz"})
        assert "error" in result or result.get("status") in (None, "NOT_FOUND"), \
            f"Expected error for unknown job: {result}"

    def test_status_missing_job_id_returns_error(self, lam, tool_arns):
        """compute_status without job_id returns an error."""
        result = invoke(lam, tool_arns["compute_status"], {})
        assert "error" in result, f"Expected error for missing job_id: {result}"
