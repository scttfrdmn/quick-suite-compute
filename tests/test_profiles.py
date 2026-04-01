"""
Tests for profile JSON config files and the compute-profiles Lambda handler.
"""

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent

EXPECTED_PROFILE_IDS = {
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


def _load_handler(module_alias="_compute_profiles_handler"):
    path = REPO_ROOT / "lambdas" / "compute-profiles" / "handler.py"
    spec = importlib.util.spec_from_file_location(module_alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = mod
    spec.loader.exec_module(mod)
    return mod


_profiles_handler = _load_handler()


# ===========================================================================
# Profile JSON file validation
# ===========================================================================

class TestProfileJsonSchema:
    @pytest.fixture(params=list(EXPECTED_PROFILE_IDS))
    def profile_path(self, request):
        p = REPO_ROOT / "config" / "profiles" / f"{request.param}.json"
        assert p.exists(), f"Profile file missing: {p}"
        return p

    def test_all_required_fields_present(self, profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        required = [
            "profile_id", "display_name", "description", "category",
            "backend", "entrypoint", "parameters", "input_requirements",
            "output_schema", "cost_estimate", "tags",
        ]
        for field in required:
            assert field in profile, f"{profile_path.name}: missing field '{field}'"

    def test_cost_estimate_has_required_keys(self, profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        cost = profile["cost_estimate"]
        assert "typical_duration_seconds" in cost
        assert "typical_cost_usd" in cost

    def test_backend_is_valid(self, profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        assert profile["backend"] in ("lambda", "emr_serverless")

    def test_tags_is_non_empty_list(self, profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        assert isinstance(profile["tags"], list)
        assert len(profile["tags"]) > 0

    def test_entrypoint_format(self, profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        if profile["backend"] == "lambda":
            assert "." in profile["entrypoint"], (
                f"Lambda entrypoint should be 'module.function': {profile['entrypoint']}"
            )

    def test_all_10_profiles_exist(self):
        profiles_dir = REPO_ROOT / "config" / "profiles"
        found = {p.stem for p in profiles_dir.glob("*.json")}
        assert EXPECTED_PROFILE_IDS == found


# ===========================================================================
# compute-profiles Lambda handler
# ===========================================================================

class TestComputeProfilesHandler:
    def test_no_filter_returns_all_profiles(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _profiles_handler._PROFILES = None  # reset cache
            result = _profiles_handler.handler({}, None)
        assert result["count"] == 10
        assert len(result["profiles"]) == 10

    def test_applied_category_and_tags_in_response(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _profiles_handler._PROFILES = None
            result = _profiles_handler.handler({}, None)
        assert "applied_category" in result
        assert "applied_tags" in result

    def test_category_filter(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _profiles_handler._PROFILES = None
            result = _profiles_handler.handler({"category": "segmentation"}, None)
        assert result["count"] == 1
        assert result["profiles"][0]["profile_id"] == "clustering-kmeans"

    def test_category_filter_case_insensitive(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _profiles_handler._PROFILES = None
            result = _profiles_handler.handler({"category": "FORECASTING"}, None)
        assert result["count"] == 1

    def test_tag_filter(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _profiles_handler._PROFILES = None
            result = _profiles_handler.handler({"tags": ["enrollment"]}, None)
        # Multiple profiles are tagged with "enrollment"
        assert result["count"] >= 2

    def test_tag_filter_any_match(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _profiles_handler._PROFILES = None
            result = _profiles_handler.handler({"tags": ["clustering", "survival"]}, None)
        profile_ids = {p["profile_id"] for p in result["profiles"]}
        assert "clustering-kmeans" in profile_ids
        assert "survival-kaplan-meier" in profile_ids

    def test_entrypoint_not_in_projected_output(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _profiles_handler._PROFILES = None
            result = _profiles_handler.handler({}, None)
        for p in result["profiles"]:
            assert "entrypoint" not in p, "entrypoint should not be exposed to agent"

    def test_empty_profiles_config(self):
        with patch.dict(os.environ, {"PROFILES_CONFIG": "[]"}):
            _profiles_handler._PROFILES = None
            result = _profiles_handler.handler({}, None)
        assert result["count"] == 0
        assert result["profiles"] == []

    def test_nonexistent_category_returns_empty(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _profiles_handler._PROFILES = None
            result = _profiles_handler.handler({"category": "nonexistent"}, None)
        assert result["count"] == 0

    def test_all_profiles_have_required_output_keys(self, profiles_json):
        with patch.dict(os.environ, {"PROFILES_CONFIG": profiles_json}):
            _profiles_handler._PROFILES = None
            result = _profiles_handler.handler({}, None)
        required_keys = {
            "profile_id", "display_name", "description", "category",
            "backend", "parameters", "cost_estimate", "tags",
        }
        for p in result["profiles"]:
            assert required_keys.issubset(p.keys()), (
                f"Profile {p.get('profile_id')} missing required output keys"
            )
