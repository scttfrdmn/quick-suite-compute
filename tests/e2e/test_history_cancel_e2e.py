"""
E2E tests for compute_history and compute_cancel.

compute_history is read-only.
compute_cancel is tested by cancelling the shared job from run_result.
"""

import pytest
from tests.e2e.conftest import invoke, _E2E_USER_ARN

pytestmark = pytest.mark.e2e


class TestComputeHistoryE2E:
    def test_history_returns_list(self, lam, tool_arns):
        """compute_history returns a list of jobs."""
        result = invoke(lam, tool_arns["compute_history"], {"user_arn": _E2E_USER_ARN})
        assert "jobs" in result or "history" in result or isinstance(result, list), \
            f"Unexpected history response shape: {result}"

    def test_history_missing_user_arn_returns_error(self, lam, tool_arns):
        """compute_history without user_arn returns an error."""
        result = invoke(lam, tool_arns["compute_history"], {})
        assert "error" in result, f"Expected error for missing user_arn: {result}"

    def test_history_response_has_jobs_field(self, lam, tool_arns, run_result):
        """compute_history returns a well-formed response (jobs written by record-spend
        on completion; cancelled jobs may not appear)."""
        result = invoke(lam, tool_arns["compute_history"], {"user_arn": _E2E_USER_ARN})
        # Accept list or {"jobs": [...]} response shapes
        jobs = result if isinstance(result, list) else result.get("jobs", result.get("history", None))
        assert jobs is not None, f"Unexpected history response shape: {result}"
        assert isinstance(jobs, list), f"Expected list, got: {type(jobs)}"


class TestComputeCancelE2E:
    def test_cancel_running_job(self, lam, tool_arns, run_result, sfn_client):
        """compute_cancel stops the running job."""
        job_id = run_result["job_id"]
        exec_arn = run_result.get("execution_arn", "")

        # Check current status first — skip if already terminal
        status_result = invoke(lam, tool_arns["compute_status"], {"job_id": job_id})
        current_status = status_result.get("status", "")
        if current_status in ("SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"):
            pytest.skip(f"Job already in terminal state {current_status} — cancel not testable")

        result = invoke(lam, tool_arns["compute_cancel"], {
            "job_id": job_id,
            "user_arn": _E2E_USER_ARN,
        })
        # Accept success or already-terminal responses
        assert "error" not in result or "already" in str(result.get("error", "")).lower(), \
            f"Unexpected cancel error: {result}"

    def test_cancel_unknown_job_returns_error(self, lam, tool_arns):
        """compute_cancel for a nonexistent job returns an error."""
        result = invoke(lam, tool_arns["compute_cancel"], {
            "job_id": "nonexistent-job-xyz",
            "user_arn": _E2E_USER_ARN,
        })
        assert "error" in result, f"Expected error for unknown job: {result}"

    def test_cancel_missing_job_id_returns_error(self, lam, tool_arns):
        """compute_cancel without job_id returns an error."""
        result = invoke(lam, tool_arns["compute_cancel"], {"user_arn": _E2E_USER_ARN})
        assert "error" in result, f"Expected error for missing job_id: {result}"
