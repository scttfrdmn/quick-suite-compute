"""
E2E tests for the compute_profiles Lambda.

Read-only — lists profiles from the deployed catalog.
No compute jobs are started.
"""

import pytest
from tests.e2e.conftest import invoke

pytestmark = pytest.mark.e2e

_EXPECTED_PROFILES = {
    "clustering-kmeans",
    "regression-glm",
    "forecast-prophet",
    "retention-cohort",
    "text-topics",
    "anomaly-isolation-forest",
    "transform-spark",
    "explore-correlations",
    "geo-enrich",
    "survival-kaplan-meier",
}


class TestProfilesE2E:
    def test_profiles_returns_list(self, lam, tool_arns):
        """compute_profiles returns a list of profiles."""
        result = invoke(lam, tool_arns["compute_profiles"], {})
        assert "profiles" in result, f"Missing 'profiles' key: {result}"
        assert isinstance(result["profiles"], list)
        assert len(result["profiles"]) > 0

    def test_profiles_count_matches(self, lam, tool_arns):
        """count field matches the length of the profiles list."""
        result = invoke(lam, tool_arns["compute_profiles"], {})
        assert result.get("count") == len(result["profiles"]), \
            f"count mismatch: {result.get('count')} vs {len(result['profiles'])}"

    def test_all_expected_profiles_present(self, lam, tool_arns):
        """All 10 standard profiles are present."""
        result = invoke(lam, tool_arns["compute_profiles"], {})
        ids = {p["profile_id"] for p in result["profiles"]}
        missing = _EXPECTED_PROFILES - ids
        assert not missing, f"Missing profiles: {missing}"

    def test_each_profile_has_required_fields(self, lam, tool_arns):
        """Every profile has the required structural fields."""
        result = invoke(lam, tool_arns["compute_profiles"], {})
        required = {"profile_id", "display_name", "description", "category",
                    "backend", "parameters", "cost_estimate"}
        for p in result["profiles"]:
            missing = required - set(p.keys())
            assert not missing, f"Profile {p.get('profile_id')} missing fields: {missing}"

    def test_cost_estimate_has_duration_and_cost(self, lam, tool_arns):
        """Every profile's cost_estimate includes duration and USD cost."""
        result = invoke(lam, tool_arns["compute_profiles"], {})
        for p in result["profiles"]:
            est = p.get("cost_estimate", {})
            assert "typical_duration_seconds" in est, \
                f"Profile {p['profile_id']} missing typical_duration_seconds"
            assert "typical_cost_usd" in est, \
                f"Profile {p['profile_id']} missing typical_cost_usd"

    def test_category_filter(self, lam, tool_arns):
        """Filtering by category returns only profiles in that category."""
        result = invoke(lam, tool_arns["compute_profiles"], {"category": "segmentation"})
        assert result.get("applied_category") == "segmentation"
        for p in result["profiles"]:
            assert p["category"] == "segmentation", \
                f"Profile {p['profile_id']} has category {p['category']}, expected segmentation"

    def test_tag_filter(self, lam, tool_arns):
        """Filtering by tag returns only profiles with that tag."""
        result = invoke(lam, tool_arns["compute_profiles"], {"tags": ["clustering"]})
        assert result.get("applied_tags") == ["clustering"]
        for p in result["profiles"]:
            assert "clustering" in p.get("tags", []), \
                f"Profile {p['profile_id']} doesn't have tag 'clustering': {p.get('tags')}"

    def test_unknown_category_returns_empty(self, lam, tool_arns):
        """Filtering by a nonexistent category returns empty profiles list."""
        result = invoke(lam, tool_arns["compute_profiles"], {"category": "nonexistent-xyz"})
        assert result.get("profiles") == [] or result.get("count") == 0, \
            f"Expected empty result for unknown category: {result}"

    def test_emr_profile_has_emr_backend(self, lam, tool_arns):
        """transform-spark uses emr_serverless backend."""
        result = invoke(lam, tool_arns["compute_profiles"], {})
        spark = next((p for p in result["profiles"] if p["profile_id"] == "transform-spark"), None)
        assert spark is not None
        assert spark["backend"] == "emr_serverless", \
            f"Expected emr_serverless backend, got: {spark['backend']}"
