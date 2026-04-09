"""
Tests for v0.17.0 science research profiles:
  - power-analysis (power_analysis_handler)
  - anomaly-hypothesis (anomaly_hypothesis_handler)
  - reproducibility-check (reproducibility_check_handler)
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent

# Ensure profile modules are importable
sys.path.insert(0, str(REPO_ROOT / "lambdas" / "profiles"))

import research  # noqa: E402


# ===========================================================================
# TestPowerAnalysis
# ===========================================================================

class TestPowerAnalysis:
    def test_manual_effect_size(self):
        """Manual effect size produces a reasonable required_n_per_group."""
        df = pd.DataFrame({"outcome": [1, 2, 3], "treatment": [0, 1, 0]})
        result = research.power_analysis_handler(df, {
            "outcome_var": "outcome",
            "treatment_vars": ["treatment"],
            "manual_effect_size": 0.5,
            "effect_size_source": "manual",
        })
        assert "error" not in result
        assert result["required_n_per_group"] > 10
        assert result["required_n_per_group"] < 200
        assert result["effect_size_used"] == 0.5
        assert result["effect_size_source"] == "manual"

    @patch.object(research, "_call_router_api")
    def test_literature_mode_mocked(self, mock_router):
        """Literature mode extracts effect sizes from Router and uses 25th percentile."""
        mock_router.return_value = {"effect_sizes": [0.3, 0.5, 0.7]}
        df = pd.DataFrame({"outcome": [1], "treatment": [0]})
        result = research.power_analysis_handler(df, {
            "outcome_var": "outcome",
            "treatment_vars": ["treatment"],
            "effect_size_source": "literature",
            "comparable_studies": ["PMID123", "PMID456"],
        })
        assert "error" not in result
        # 25th percentile of [0.3, 0.5, 0.7, 0.3, 0.5, 0.7] — first call returns 3 values, second too
        # The sorted combined list is [0.3, 0.3, 0.5, 0.5, 0.7, 0.7]
        # idx = max(0, int(6 * 0.25) - 1) = max(0, 0) = 0 → value 0.3
        assert result["effect_size_used"] == 0.3
        assert result["required_n_per_group"] > 0
        assert len(result["citations"]) == 2

    @patch.object(research, "_call_router_api")
    def test_router_unavailable_fallback(self, mock_router):
        """When Router returns None and no manual_effect_size, returns error."""
        mock_router.return_value = None
        df = pd.DataFrame({"outcome": [1], "treatment": [0]})
        result = research.power_analysis_handler(df, {
            "outcome_var": "outcome",
            "treatment_vars": ["treatment"],
            "effect_size_source": "literature",
            "comparable_studies": ["PMID123"],
        })
        assert "error" in result

    def test_pilot_data_mode(self):
        """Pilot data mode computes Cohen's d from two groups."""
        np.random.seed(42)
        g1 = np.random.normal(10, 2, 30)
        g2 = np.random.normal(12, 2, 30)
        df = pd.DataFrame({
            "outcome": np.concatenate([g1, g2]),
            "treatment": [0] * 30 + [1] * 30,
        })
        result = research.power_analysis_handler(df, {
            "outcome_var": "outcome",
            "treatment_vars": ["treatment"],
            "effect_size_source": "pilot_data",
        })
        assert "error" not in result
        assert result["effect_size_used"] > 0
        assert result["required_n_per_group"] > 0
        assert result["effect_size_source"] == "pilot_data"

    def test_invalid_inputs(self):
        """Missing required params produces an error dict."""
        df = pd.DataFrame({"x": [1]})
        # Missing outcome_var
        result = research.power_analysis_handler(df, {
            "treatment_vars": ["x"],
            "manual_effect_size": 0.5,
        })
        assert "error" in result

        # Missing treatment_vars
        result = research.power_analysis_handler(df, {
            "outcome_var": "x",
            "manual_effect_size": 0.5,
        })
        assert "error" in result

    def test_power_curve_shape(self):
        """Power curve has entries and is monotonically non-decreasing."""
        df = pd.DataFrame({"outcome": [1], "treatment": [0]})
        result = research.power_analysis_handler(df, {
            "outcome_var": "outcome",
            "treatment_vars": ["treatment"],
            "manual_effect_size": 0.5,
            "effect_size_source": "manual",
        })
        assert "error" not in result
        curve = result["power_curve"]
        assert len(curve) > 0
        powers = [p["power"] for p in curve]
        # Monotonically non-decreasing (allowing small float imprecision)
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1] - 1e-6


# ===========================================================================
# TestAnomalyHypothesis
# ===========================================================================

class TestAnomalyHypothesis:
    def _make_df(self, n=100, seed=42):
        np.random.seed(seed)
        df = pd.DataFrame({
            "feature_a": np.random.normal(0, 1, n),
            "feature_b": np.random.normal(0, 1, n),
        })
        # Inject a few outliers
        df.loc[0, "feature_a"] = 10.0
        df.loc[1, "feature_b"] = -10.0
        return df

    @patch.object(research, "_call_router_api")
    def test_happy_path_mocked_router(self, mock_router):
        """Mocked Router returns grounded response, anomaly_class populated."""
        mock_router.return_value = {
            "content": "This appears to be an instrument calibration drift artifact.",
            "sources_used": ["doi:10.1234/test"],
        }
        df = self._make_df()
        result_df, diag = research.anomaly_hypothesis_handler(df, {
            "features": ["feature_a", "feature_b"],
            "domain": "genomics",
        })
        assert "anomaly_count" in diag
        assert diag["anomaly_count"] >= 0
        assert "classifications" in diag
        # Check that anomaly columns exist
        assert "anomaly_class" in result_df.columns
        assert "confidence" in result_df.columns

    @patch.object(research, "_call_router_api")
    def test_router_unavailable_all_novel(self, mock_router):
        """When Router returns None, all anomalies classified as novel_candidate."""
        mock_router.return_value = None
        df = self._make_df()
        result_df, diag = research.anomaly_hypothesis_handler(df, {
            "features": ["feature_a", "feature_b"],
            "domain": "genomics",
        })
        anomalies = result_df[result_df["is_anomaly"] == True]  # noqa: E712
        if len(anomalies) > 0:
            assert all(anomalies["anomaly_class"] == "novel_candidate")

    def test_domain_z_thresholds(self):
        """Verify domain-specific z-thresholds are configured correctly."""
        assert research._DOMAIN_THRESHOLDS["genomics"] == 3.5
        assert research._DOMAIN_THRESHOLDS["behavioral"] == 2.5
        assert research._DOMAIN_THRESHOLDS["proteomics"] == 3.0
        assert research._DOMAIN_THRESHOLDS["geospatial"] == 3.0

    @patch.object(research, "_call_router_api")
    def test_empty_features(self, mock_router):
        """Empty features list produces error in diagnostics."""
        df = self._make_df()
        _, diag = research.anomaly_hypothesis_handler(df, {
            "features": [],
            "domain": "genomics",
        })
        assert "error" in diag

    @patch.object(research, "_call_router_api")
    def test_too_few_rows(self, mock_router):
        """Fewer than 50 rows produces error."""
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        _, diag = research.anomaly_hypothesis_handler(df, {
            "features": ["a", "b"],
            "domain": "genomics",
        })
        assert "error" in diag

    @patch.object(research, "_call_router_api")
    def test_classification_summary_in_diagnostics(self, mock_router):
        """Diagnostics dict has classification counts."""
        mock_router.return_value = None
        df = self._make_df()
        _, diag = research.anomaly_hypothesis_handler(df, {
            "features": ["feature_a", "feature_b"],
            "domain": "genomics",
        })
        assert "classifications" in diag
        assert isinstance(diag["classifications"], dict)
        for key in ("instrument_error", "known_noise", "reported_effect", "novel_candidate"):
            assert key in diag["classifications"]


# ===========================================================================
# TestReproducibilityCheck
# ===========================================================================

class TestReproducibilityCheck:
    def _mock_s3_script(self, mock_client, script_text: bytes):
        """Configure mock_client (boto3.client) to return script_text from S3."""
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=script_text))
        }
        return mock_s3

    @patch("boto3.resource")
    @patch("boto3.client")
    def test_matching_results(self, mock_client, mock_resource):
        """Script produces same values as manuscript — all matches."""
        script_text = b"def transform(df):\n    return {'mean_x': float(df['x'].mean())}\n"
        self._mock_s3_script(mock_client, script_text)

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = research.reproducibility_check_handler(df, {
            "manuscript_results": json.dumps([{"metric": "mean_x", "value": 2.0}]),
            "analysis_script_uri": "s3://bucket/script.py",
        })
        assert "error" not in result
        assert len(result["matches"]) == 1
        assert len(result["discrepancies"]) == 0

    @patch("boto3.resource")
    @patch("boto3.client")
    def test_discrepancy_detected(self, mock_client, mock_resource):
        """Script produces different value — discrepancy with delta."""
        script_text = b"def transform(df):\n    return {'mean_x': 999.0}\n"
        self._mock_s3_script(mock_client, script_text)

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = research.reproducibility_check_handler(df, {
            "manuscript_results": json.dumps([{"metric": "mean_x", "value": 2.0}]),
            "analysis_script_uri": "s3://bucket/script.py",
        })
        assert "error" not in result
        assert len(result["discrepancies"]) == 1
        assert result["discrepancies"][0]["delta"] > 0

    @patch("boto3.resource")
    @patch("boto3.client")
    def test_provenance_lookup(self, mock_client, mock_resource):
        """Provenance run_id triggers HistoryTable query."""
        script_text = b"def transform(df):\n    return {'val': 1.0}\n"
        self._mock_s3_script(mock_client, script_text)

        mock_table = MagicMock()
        mock_table.get_item.return_value = {
            "Item": {"job_id": "run-123", "script_version": "v2.1"}
        }
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value = mock_table
        mock_resource.return_value = mock_ddb

        os.environ["COMPUTE_HISTORY_TABLE"] = "test-history"
        try:
            df = pd.DataFrame({"x": [1]})
            result = research.reproducibility_check_handler(df, {
                "manuscript_results": json.dumps([{"metric": "val", "value": 1.0}]),
                "analysis_script_uri": "s3://bucket/script.py",
                "provenance_run_id": "run-123",
            })
            assert result.get("script_version") == "v2.1"
        finally:
            os.environ.pop("COMPUTE_HISTORY_TABLE", None)

    @patch("boto3.resource")
    @patch("boto3.client")
    def test_script_execution_error(self, mock_client, mock_resource):
        """Bad script produces error in output."""
        script_text = b"def transform(df):\n    raise ValueError('boom')\n"
        self._mock_s3_script(mock_client, script_text)

        df = pd.DataFrame({"x": [1]})
        result = research.reproducibility_check_handler(df, {
            "manuscript_results": json.dumps([{"metric": "val", "value": 1.0}]),
            "analysis_script_uri": "s3://bucket/script.py",
        })
        assert "error" in result

    def test_missing_manuscript_results(self):
        """No manuscript_results produces error."""
        df = pd.DataFrame({"x": [1]})
        result = research.reproducibility_check_handler(df, {
            "analysis_script_uri": "s3://bucket/script.py",
        })
        assert "error" in result

    @patch("boto3.resource")
    @patch("boto3.client")
    def test_tolerance_edge(self, mock_client, mock_resource):
        """Value within tolerance matches; outside does not."""
        script_text = b"def transform(df):\n    return {'val': 1.00005}\n"
        self._mock_s3_script(mock_client, script_text)

        df = pd.DataFrame({"x": [1]})
        # tolerance=0.001 → within tolerance
        result = research.reproducibility_check_handler(df, {
            "manuscript_results": json.dumps([{"metric": "val", "value": 1.0}]),
            "analysis_script_uri": "s3://bucket/script.py",
            "tolerance": 0.001,
        })
        assert "error" not in result
        assert len(result["matches"]) == 1

        # tolerance=0.00001 → outside tolerance
        result2 = research.reproducibility_check_handler(df, {
            "manuscript_results": json.dumps([{"metric": "val", "value": 1.0}]),
            "analysis_script_uri": "s3://bucket/script.py",
            "tolerance": 0.00001,
        })
        assert "error" not in result2
        assert len(result2["discrepancies"]) == 1
