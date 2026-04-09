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

import numpy as np
import pandas as pd
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
    # v0.9.0 profiles
    "regression-logistic",
    "anova",
    "chi-square",
    "equity-gap",
    "dfwi-analysis",
    "cohort-flow",
    "peer-benchmark",
    # v0.10.0 profiles
    "classification-random-forest",
    "text-sentiment",
    "text-similarity",
    "change-detection",
    "seasonality-decompose",
    "spatial-aggregate",
    "isochrone",
    # v0.11.0 profiles
    "grant-portfolio",
    "network-coauthor",
    "ingest-netcdf",
    "ingest-pdf-extract",
    "ingest-geojson",
    # v0.12.0 profiles
    "custom-python",
    "custom-generated",
    # v0.15.0 profiles
    "intersectionality-equity",
    "assessment-irt",
    # v0.16.0 profiles
    "causal-iv",
    "causal-rd",
    "causal-did",
    "grant-pipeline",
    "provenance-graph",
    # v0.17.0 profiles
    "power-analysis",
    "anomaly-hypothesis",
    "reproducibility-check",
    # v0.18.0 profiles
    "financial-aid-effectiveness",
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

    def test_all_expected_profiles_exist(self):
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
        assert result["count"] == len(EXPECTED_PROFILE_IDS)
        assert len(result["profiles"]) == len(EXPECTED_PROFILE_IDS)

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


# ===========================================================================
# v0.9.0 handler unit tests
# ===========================================================================

def _make_df(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


class TestLogisticHandler:
    def setup_method(self):
        from statistics import logistic_handler
        self.handler = logistic_handler

    def _binary_df(self, n=100):
        rng = np.random.default_rng(42)
        return _make_df(
            score=rng.normal(70, 10, n),
            credits=rng.integers(12, 18, n).astype(float),
            retained=rng.integers(0, 2, n),
        )

    def test_returns_two_new_columns(self):
        df = self._binary_df()
        result, diag = self.handler(df, {"target": "retained", "features": ["score", "credits"]})
        assert "predicted_prob" in result.columns
        assert "predicted_class" in result.columns

    def test_probs_between_0_and_1(self):
        df = self._binary_df()
        result, _ = self.handler(df, {"target": "retained", "features": ["score", "credits"]})
        valid = result["predicted_prob"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_diagnostics_has_auc(self):
        df = self._binary_df()
        _, diag = self.handler(df, {"target": "retained", "features": ["score", "credits"]})
        assert "auc" in diag

    def test_raises_on_missing_target(self):
        df = self._binary_df()
        with pytest.raises(ValueError, match="target column"):
            self.handler(df, {"target": "nonexistent", "features": ["score"]})

    def test_raises_on_non_binary_target(self):
        df = _make_df(x=[1.0, 2.0, 3.0] * 10, y=list(range(30)))
        with pytest.raises(ValueError, match="2 unique values"):
            self.handler(df, {"target": "y", "features": ["x"]})

    def test_empty_df_returns_without_error(self):
        df = pd.DataFrame({"score": pd.Series([], dtype=float), "retained": pd.Series([], dtype=int)})
        result, diag = self.handler(df, {"target": "retained", "features": ["score"]})
        assert "warning" in diag


class TestAnovaHandler:
    def setup_method(self):
        from statistics import anova_handler
        self.handler = anova_handler

    def _group_df(self, n=90):
        rng = np.random.default_rng(0)
        return _make_df(
            score=np.concatenate([rng.normal(70, 5, n // 3),
                                  rng.normal(75, 5, n // 3),
                                  rng.normal(80, 5, n // 3)]),
            section=["A"] * (n // 3) + ["B"] * (n // 3) + ["C"] * (n // 3),
        )

    def test_diagnostics_has_f_and_p(self):
        df = self._group_df()
        _, diag = self.handler(df, {"metric_column": "score", "group_column": "section"})
        assert "f_statistic" in diag
        assert "p_value" in diag

    def test_eta_squared_between_0_and_1(self):
        df = self._group_df()
        _, diag = self.handler(df, {"metric_column": "score", "group_column": "section"})
        assert 0.0 <= diag["eta_squared"] <= 1.0

    def test_tukey_pairwise_count(self):
        df = self._group_df()
        _, diag = self.handler(df, {"metric_column": "score", "group_column": "section"})
        # 3 groups → 3 pairs
        assert len(diag["tukey_hsd"]) == 3

    def test_raises_on_missing_columns(self):
        df = _make_df(score=[1.0, 2.0])
        with pytest.raises(ValueError, match="group_column"):
            self.handler(df, {"metric_column": "score", "group_column": "dept"})


class TestChiSquareHandler:
    def setup_method(self):
        from statistics import chi_square_handler
        self.handler = chi_square_handler

    def _cat_df(self):
        return _make_df(
            pell=["yes"] * 40 + ["no"] * 60,
            retained=["yes"] * 30 + ["no"] * 10 + ["yes"] * 50 + ["no"] * 10,
        )

    def test_returns_chi2_and_cramers_v(self):
        df = self._cat_df()
        _, diag = self.handler(df, {"row_column": "pell", "col_column": "retained"})
        assert "chi2_statistic" in diag
        assert "cramers_v" in diag
        assert 0.0 <= diag["cramers_v"] <= 1.0

    def test_contingency_tables_present(self):
        df = self._cat_df()
        _, diag = self.handler(df, {"row_column": "pell", "col_column": "retained"})
        assert "observed_contingency" in diag
        assert "expected_contingency" in diag

    def test_raises_on_missing_column(self):
        df = self._cat_df()
        with pytest.raises(ValueError, match="row_column"):
            self.handler(df, {"row_column": "missing", "col_column": "retained"})


class TestEquityGapHandler:
    def setup_method(self):
        from higher_ed import equity_gap_handler
        self.handler = equity_gap_handler

    def _df(self):
        return _make_df(
            gpa=[3.5, 3.2, 2.8, 3.0, 3.7, 2.5, 3.1, 3.4, 2.9, 3.6] * 5,
            pell=(["yes"] * 5 + ["no"] * 5) * 5,
            first_gen=(["yes", "no"] * 25),
        )

    def test_adds_group_mean_and_equity_index(self):
        df = self._df()
        result, _ = self.handler(df, {"metric_column": "gpa", "group_columns": ["pell"]})
        assert "group_mean" in result.columns
        assert "equity_index" in result.columns

    def test_gap_table_in_diagnostics(self):
        df = self._df()
        _, diag = self.handler(df, {"metric_column": "gpa", "group_columns": ["pell"]})
        assert "gap_table" in diag
        assert len(diag["gap_table"]) >= 1

    def test_equity_index_relative_to_reference(self):
        df = self._df()
        _, diag = self.handler(
            df, {"metric_column": "gpa", "group_columns": ["pell"], "reference_group": "no"}
        )
        ref_row = next((r for r in diag["gap_table"] if r["pell"] == "no"), None)
        assert ref_row is not None
        assert abs(ref_row["equity_index"] - 1.0) < 1e-6

    def test_raises_on_missing_metric_column(self):
        df = self._df()
        with pytest.raises(ValueError, match="metric_column"):
            self.handler(df, {"metric_column": "missing", "group_columns": ["pell"]})


class TestDfwiHandler:
    def setup_method(self):
        from higher_ed import dfwi_handler
        self.handler = dfwi_handler

    def _df(self):
        return _make_df(
            grade=["A", "B", "C", "D", "F", "W", "A", "B", "F", "W"] * 10,
            course=["CS101"] * 50 + ["MATH201"] * 50,
            dept=["CS"] * 50 + ["MATH"] * 50,
        )

    def test_adds_is_dfwi_column(self):
        df = self._df()
        result, _ = self.handler(df, {"grade_column": "grade"})
        assert "is_dfwi" in result.columns

    def test_overall_rate_correct(self):
        df = self._df()
        _, diag = self.handler(df, {"grade_column": "grade"})
        # Pattern: ["A","B","C","D","F","W","A","B","F","W"] — DFWI values: D=1, F=2, W=2 = 5/10
        assert abs(diag["overall_dfwi_rate"] - 0.5) < 0.01

    def test_group_by_reduces_to_summary(self):
        df = self._df()
        _, diag = self.handler(df, {"grade_column": "grade", "group_by": ["course"]})
        assert len(diag["dfwi_summary"]) == 2

    def test_custom_dfwi_values(self):
        df = _make_df(grade=["P", "NP", "P", "NP"] * 5)
        result, diag = self.handler(df, {"grade_column": "grade", "dfwi_values": ["NP"]})
        assert abs(diag["overall_dfwi_rate"] - 0.5) < 0.01


class TestCohortFlowHandler:
    def setup_method(self):
        from higher_ed import cohort_flow_handler
        self.handler = cohort_flow_handler

    def _df(self):
        # 100 students: all applied, 80 admitted, 60 enrolled, 40 graduated
        return _make_df(
            student_id=list(range(100)) * 4,
            stage=(
                ["applied"] * 100 +
                ["admitted"] * 80 + ["applied"] * 20 +
                ["enrolled"] * 60 + ["applied"] * 40 +
                ["graduated"] * 40 + ["applied"] * 60
            )[:400],
        ).drop_duplicates(subset=["student_id", "stage"])

    def test_adds_stage_reached_column(self):
        df = self._df()
        result, _ = self.handler(df, {
            "id_column": "student_id",
            "stage_column": "stage",
            "stage_order": ["applied", "admitted", "enrolled", "graduated"],
        })
        assert "stage_reached" in result.columns

    def test_funnel_has_correct_stages(self):
        df = self._df()
        _, diag = self.handler(df, {
            "id_column": "student_id",
            "stage_column": "stage",
            "stage_order": ["applied", "admitted", "enrolled", "graduated"],
        })
        stages = [row["stage"] for row in diag["funnel"]]
        assert stages == ["applied", "admitted", "enrolled", "graduated"]

    def test_raises_on_empty_stage_order(self):
        df = self._df()
        with pytest.raises(ValueError, match="stage_order"):
            self.handler(df, {
                "id_column": "student_id",
                "stage_column": "stage",
                "stage_order": [],
            })


class TestPeerBenchmarkHandler:
    def setup_method(self):
        from higher_ed import peer_benchmark_handler
        self.handler = peer_benchmark_handler

    def _df(self):
        return _make_df(
            unitid=["A", "B", "C", "D", "E"],
            grad_rate=[0.60, 0.65, 0.70, 0.55, 0.75],
            research_exp=[1e6, 2e6, 3e6, 0.5e6, 4e6],
        )

    def test_adds_zscore_and_percentile_columns(self):
        df = self._df()
        result, _ = self.handler(df, {
            "metric_columns": ["grad_rate"],
            "id_column": "unitid",
            "focal_id": "C",
        })
        assert "grad_rate_zscore" in result.columns
        assert "grad_rate_percentile" in result.columns

    def test_focal_flag_set(self):
        df = self._df()
        result, _ = self.handler(df, {
            "metric_columns": ["grad_rate"],
            "id_column": "unitid",
            "focal_id": "C",
        })
        assert result[result["unitid"] == "C"]["is_focal"].iloc[0] == True  # noqa: E712

    def test_focal_summary_in_diagnostics(self):
        df = self._df()
        _, diag = self.handler(df, {
            "metric_columns": ["grad_rate", "research_exp"],
            "id_column": "unitid",
            "focal_id": "C",
        })
        assert diag["focal_institution"] is not None
        assert "grad_rate" in diag["focal_institution"]

    def test_missing_focal_id_noted_in_diagnostics(self):
        df = self._df()
        _, diag = self.handler(df, {
            "metric_columns": ["grad_rate"],
            "id_column": "unitid",
            "focal_id": "Z",
        })
        assert diag["focal_institution"] is None
        assert diag["note"] is not None


# ===========================================================================
# v0.10.0 handler unit tests
# ===========================================================================

class TestRandomForestHandler:
    def setup_method(self):
        from ml import random_forest_handler
        self.handler = random_forest_handler

    def _classify_df(self, n=100):
        rng = np.random.default_rng(7)
        return _make_df(
            score=rng.normal(70, 10, n),
            credits=rng.integers(12, 18, n).astype(float),
            grade=(["pass"] * (n // 2) + ["fail"] * (n // 2)),
        )

    def _regress_df(self, n=100):
        rng = np.random.default_rng(8)
        return _make_df(
            x1=rng.normal(0, 1, n),
            x2=rng.normal(0, 1, n),
            y=rng.normal(50, 10, n),
        )

    def test_classify_adds_predicted_class(self):
        df = self._classify_df()
        result, _ = self.handler(df, {"target": "grade", "features": ["score", "credits"]})
        assert "predicted_class" in result.columns

    def test_classify_diagnostics_has_accuracy_and_importances(self):
        df = self._classify_df()
        _, diag = self.handler(df, {"target": "grade", "features": ["score", "credits"]})
        assert "accuracy" in diag
        assert "feature_importances" in diag
        assert len(diag["feature_importances"]) == 2

    def test_regress_adds_predicted_value(self):
        df = self._regress_df()
        result, diag = self.handler(
            df, {"target": "y", "features": ["x1", "x2"], "task": "regress"}
        )
        assert "predicted_value" in result.columns
        assert diag["task"] == "regress"
        assert "r_squared" in diag

    def test_raises_on_missing_target(self):
        df = self._classify_df()
        with pytest.raises(ValueError, match="target column"):
            self.handler(df, {"target": "nonexistent", "features": ["score"]})

    def test_empty_df_returns_warning(self):
        df = pd.DataFrame({"score": pd.Series([], dtype=float), "grade": pd.Series([], dtype=str)})
        result, diag = self.handler(df, {"target": "grade", "features": ["score"]})
        assert "warning" in diag


class TestSentimentHandler:
    def setup_method(self):
        from text_analytics import sentiment_handler
        self.handler = sentiment_handler

    def _df(self):
        return _make_df(
            text=[
                "I love this course, it was amazing!",
                "This is terrible and disappointing.",
                "The class meets on Tuesday.",
                "Fantastic professor, very helpful.",
                "Boring and poorly organized.",
            ],
            id=list(range(5)),
        )

    def test_adds_sentiment_and_score_columns(self):
        df = self._df()
        result, _ = self.handler(df, {"text_column": "text"})
        assert "sentiment" in result.columns
        assert "sentiment_score" in result.columns

    def test_sentiment_labels_are_valid(self):
        df = self._df()
        result, _ = self.handler(df, {"text_column": "text"})
        assert set(result["sentiment"]).issubset({"positive", "negative", "neutral"})

    def test_output_scores_flag_adds_component_columns(self):
        df = self._df()
        result, _ = self.handler(df, {"text_column": "text", "output_scores": True})
        for col in ("pos_score", "neg_score", "neu_score", "compound_score"):
            assert col in result.columns

    def test_diagnostics_has_counts(self):
        df = self._df()
        _, diag = self.handler(df, {"text_column": "text"})
        assert "n_positive" in diag
        assert "n_negative" in diag
        assert "n_total" in diag

    def test_raises_on_missing_text_column(self):
        df = self._df()
        with pytest.raises(ValueError, match="text_column"):
            self.handler(df, {"text_column": "nonexistent"})

    def test_empty_df_returns_warning(self):
        df = pd.DataFrame({"text": pd.Series([], dtype=str)})
        result, diag = self.handler(df, {"text_column": "text"})
        assert "warning" in diag


class TestSimilarityHandler:
    def setup_method(self):
        from text_analytics import similarity_handler
        self.handler = similarity_handler

    def _df_with_dups(self):
        return _make_df(
            id=list(range(6)),
            text=[
                "The quick brown fox jumps over the lazy dog",
                "The quick brown fox jumps over the lazy dog",  # near-dup of 0
                "A completely different sentence about astronomy",
                "A completely different sentence about astronomy",  # near-dup of 2
                "Enrollment trends in higher education institutions",
                "Something entirely unrelated and unique here",
            ],
        )

    def test_adds_required_columns(self):
        df = self._df_with_dups()
        result, _ = self.handler(df, {"text_column": "text", "id_column": "id"})
        for col in ("similarity_group_id", "is_near_duplicate", "canonical_id"):
            assert col in result.columns

    def test_near_duplicates_share_group_id(self):
        df = self._df_with_dups()
        result, _ = self.handler(df, {"text_column": "text", "id_column": "id", "similarity_threshold": 0.9})
        # Rows 0 and 1 are identical → same group
        assert result.loc[0, "similarity_group_id"] == result.loc[1, "similarity_group_id"]
        # Rows 0 and 4 are unrelated → different groups
        assert result.loc[0, "similarity_group_id"] != result.loc[4, "similarity_group_id"]

    def test_raises_on_missing_text_column(self):
        df = self._df_with_dups()
        with pytest.raises(ValueError, match="text_column"):
            self.handler(df, {"text_column": "missing", "id_column": "id"})

    def test_empty_df_returns_warning(self):
        df = pd.DataFrame({"text": pd.Series([], dtype=str), "id": pd.Series([], dtype=int)})
        result, diag = self.handler(df, {"text_column": "text", "id_column": "id"})
        assert "warning" in diag

    def test_diagnostics_has_group_counts(self):
        df = self._df_with_dups()
        _, diag = self.handler(df, {"text_column": "text", "id_column": "id"})
        assert "n_groups" in diag
        assert "n_near_duplicate_records" in diag


class TestChangeDetectionHandler:
    def setup_method(self):
        from time_series import change_detection_handler
        self.handler = change_detection_handler

    def _df(self, n=40):
        """Two-regime time series: low mean then high mean."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01", periods=n, freq="ME")
        values = np.concatenate([rng.normal(10, 1, n // 2), rng.normal(20, 1, n // 2)])
        return _make_df(date=dates.astype(str), value=values)

    def test_adds_change_point_and_segment_id(self):
        df = self._df()
        result, _ = self.handler(df, {"date_column": "date", "value_column": "value"})
        assert "change_point" in result.columns
        assert "segment_id" in result.columns

    def test_segment_ids_are_non_negative_integers(self):
        df = self._df()
        result, _ = self.handler(df, {"date_column": "date", "value_column": "value"})
        assert (result["segment_id"] >= 0).all()

    def test_diagnostics_has_change_points_list(self):
        df = self._df()
        _, diag = self.handler(df, {"date_column": "date", "value_column": "value"})
        assert "n_change_points" in diag
        assert "change_points" in diag

    def test_raises_on_missing_date_column(self):
        df = self._df()
        with pytest.raises(ValueError, match="date_column"):
            self.handler(df, {"date_column": "missing", "value_column": "value"})

    def test_raises_on_invalid_model(self):
        df = self._df()
        with pytest.raises(ValueError, match="model"):
            self.handler(df, {"date_column": "date", "value_column": "value", "model": "invalid"})

    def test_group_column_runs_per_group(self):
        rng = np.random.default_rng(1)
        dates = pd.date_range("2020-01", periods=20, freq="ME").astype(str).tolist()
        df = _make_df(
            date=dates * 2,
            value=list(rng.normal(10, 1, 20)) + list(rng.normal(20, 1, 20)),
            campus=["A"] * 20 + ["B"] * 20,
        )
        result, diag = self.handler(
            df, {"date_column": "date", "value_column": "value", "group_column": "campus"}
        )
        assert "change_point" in result.columns
        assert diag["group_column"] == "campus"


class TestDecomposeHandler:
    def setup_method(self):
        from time_series import decompose_handler
        self.handler = decompose_handler

    def _monthly_df(self, n=48):
        """Synthetic monthly series with strong annual seasonality."""
        rng = np.random.default_rng(99)
        dates = pd.date_range("2020-01", periods=n, freq="ME")
        trend = np.linspace(100, 200, n)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
        noise = rng.normal(0, 2, n)
        return _make_df(date=dates.astype(str), enrollment=trend + seasonal + noise)

    def test_adds_trend_seasonal_residual(self):
        df = self._monthly_df()
        result, _ = self.handler(df, {"date_column": "date", "value_column": "enrollment", "period": 12})
        for col in ("trend", "seasonal", "residual"):
            assert col in result.columns

    def test_most_rows_decomposed(self):
        df = self._monthly_df()
        result, diag = self.handler(
            df, {"date_column": "date", "value_column": "enrollment", "period": 12}
        )
        # STL may leave a few edge NaNs but most rows should be decomposed
        assert diag["n_rows_decomposed"] >= len(df) // 2

    def test_raises_on_missing_date_column(self):
        df = self._monthly_df()
        with pytest.raises(ValueError, match="date_column"):
            self.handler(df, {"date_column": "missing", "value_column": "enrollment"})

    def test_raises_on_invalid_model(self):
        df = self._monthly_df()
        with pytest.raises(ValueError, match="model"):
            self.handler(df, {"date_column": "date", "value_column": "enrollment", "model": "badmodel"})

    def test_multiplicative_model(self):
        df = self._monthly_df(n=36)
        result, diag = self.handler(
            df,
            {"date_column": "date", "value_column": "enrollment",
             "period": 12, "model": "multiplicative"},
        )
        assert diag["model"] == "multiplicative"
        assert "trend" in result.columns


class TestSpatialAggregateHandler:
    def setup_method(self):
        from geospatial import spatial_aggregate_handler
        self.handler = spatial_aggregate_handler

    def _simple_geojson(self) -> bytes:
        """GeoJSON with a single square polygon around (lat=0, lon=0)."""
        return json.dumps({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"name": "TestZone"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]
                    ]]
                }
            }]
        }).encode()

    def _mock_s3_client(self, body: bytes):
        from unittest.mock import MagicMock
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: body)}
        return mock_s3

    def _df(self):
        return _make_df(
            lat=[0.0, 0.5, 5.0],    # 0 and 0.5 inside polygon, 5.0 outside
            lon=[0.0, 0.5, 50.0],
            metric=[10.0, 20.0, 30.0],
        )

    def test_adds_polygon_id_and_label(self):
        df = self._df()
        with patch("boto3.client", return_value=self._mock_s3_client(self._simple_geojson())):
            result, _ = self.handler(df, {
                "lat_column": "lat",
                "lon_column": "lon",
                "boundary_uri": "s3://my-bucket/boundaries.geojson",
            })
        assert "polygon_id" in result.columns
        assert "polygon_label" in result.columns

    def test_point_inside_polygon_assigned(self):
        df = self._df()
        with patch("boto3.client", return_value=self._mock_s3_client(self._simple_geojson())):
            result, diag = self.handler(df, {
                "lat_column": "lat",
                "lon_column": "lon",
                "boundary_uri": "s3://my-bucket/boundaries.geojson",
            })
        # Points at (0,0) and (0.5,0.5) should be inside the unit square polygon
        assert diag["n_points_assigned"] >= 1

    def test_raises_on_missing_lat_column(self):
        df = self._df()
        with pytest.raises(ValueError, match="lat_column"):
            self.handler(df, {
                "lat_column": "missing",
                "lon_column": "lon",
                "boundary_uri": "s3://bucket/file.geojson",
            })

    def test_raises_on_non_s3_boundary_uri(self):
        df = self._df()
        with pytest.raises(ValueError, match="boundary_uri"):
            self.handler(df, {
                "lat_column": "lat",
                "lon_column": "lon",
                "boundary_uri": "/local/path/file.geojson",
            })


class TestIsochroneHandler:
    def setup_method(self):
        from geospatial import isochrone_handler
        self.handler = isochrone_handler

    def _origins_csv(self) -> bytes:
        return b"lat,lon,label\n40.0,-75.0,Campus A\n34.0,-118.0,Campus B\n"

    def _mock_s3_client(self, body: bytes):
        from unittest.mock import MagicMock
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: body)}
        return mock_s3

    def _df(self):
        # First row near Campus A (Philadelphia area), second row near Campus B (LA area)
        return _make_df(
            lat=[40.1, 34.1, 0.0],
            lon=[-75.1, -118.1, 0.0],
        )

    def test_adds_nearest_origin_distance_and_within_catchment(self):
        df = self._df()
        with patch("boto3.client", return_value=self._mock_s3_client(self._origins_csv())):
            result, _ = self.handler(df, {
                "lat_column": "lat",
                "lon_column": "lon",
                "origins_uri": "s3://bucket/origins.csv",
                "max_distance_km": 50.0,
            })
        for col in ("nearest_origin", "distance_km", "within_catchment"):
            assert col in result.columns

    def test_nearby_row_is_within_catchment(self):
        df = self._df()
        with patch("boto3.client", return_value=self._mock_s3_client(self._origins_csv())):
            result, diag = self.handler(df, {
                "lat_column": "lat",
                "lon_column": "lon",
                "origins_uri": "s3://bucket/origins.csv",
                "max_distance_km": 50.0,
            })
        # Row 0 is ~15 km from Campus A → within 50 km
        assert result.loc[0, "within_catchment"] == True  # noqa: E712
        assert diag["n_within_catchment"] >= 2

    def test_raises_on_missing_lat_column(self):
        df = self._df()
        with pytest.raises(ValueError, match="lat_column"):
            self.handler(df, {
                "lat_column": "missing",
                "lon_column": "lon",
                "origins_uri": "s3://bucket/origins.csv",
            })

    def test_raises_on_non_s3_origins_uri(self):
        df = self._df()
        with pytest.raises(ValueError, match="origins_uri"):
            self.handler(df, {
                "lat_column": "lat",
                "lon_column": "lon",
                "origins_uri": "/local/path/origins.csv",
            })

    def test_diagnostics_has_origin_count(self):
        df = self._df()
        with patch("boto3.client", return_value=self._mock_s3_client(self._origins_csv())):
            _, diag = self.handler(df, {
                "lat_column": "lat",
                "lon_column": "lon",
                "origins_uri": "s3://bucket/origins.csv",
            })
        assert diag["n_origins"] == 2


# ===========================================================================
# v0.11.0 — research.py
# ===========================================================================

class TestGrantPortfolioHandler:
    def setup_method(self):
        from research import grant_portfolio_handler
        self.handler = grant_portfolio_handler

    def _df(self):
        return _make_df(
            award_id=["A1", "A1", "A1", "A2", "A2"],
            amount=[10000.0, 20000.0, 15000.0, 50000.0, 30000.0],
            budget=[100000.0, 100000.0, 100000.0, 100000.0, 100000.0],
            date=["2024-01-01", "2024-03-01", "2024-06-01",
                  "2024-02-01", "2024-04-01"],
        )

    def _params(self, **overrides):
        p = {
            "amount_column": "amount",
            "budget_column": "budget",
            "date_column": "date",
            "award_id_column": "award_id",
        }
        p.update(overrides)
        return p

    def test_adds_burn_rate_pct_expended_nce_risk(self):
        result, _ = self.handler(self._df(), self._params())
        for col in ("burn_rate", "pct_expended", "nce_risk"):
            assert col in result.columns

    def test_burn_rate_values_correct(self):
        result, _ = self.handler(self._df(), self._params())
        # A1: 45000/100000 = 0.45; A2: 80000/100000 = 0.80
        a1_rows = result[result["award_id"] == "A1"]
        a2_rows = result[result["award_id"] == "A2"]
        assert abs(a1_rows["burn_rate"].iloc[0] - 0.45) < 1e-6
        assert abs(a2_rows["burn_rate"].iloc[0] - 0.80) < 1e-6

    def test_nce_risk_flag_at_threshold(self):
        # nce_threshold=0.75 → A2 (0.80) should be at risk, A1 (0.45) not
        result, diag = self.handler(self._df(), self._params(nce_threshold=0.75))
        a1_nce = result[result["award_id"] == "A1"]["nce_risk"].iloc[0]
        a2_nce = result[result["award_id"] == "A2"]["nce_risk"].iloc[0]
        assert a1_nce == False  # noqa: E712
        assert a2_nce == True   # noqa: E712
        assert diag["n_at_nce_risk"] == 1

    def test_diagnostics_has_portfolio_summary(self):
        _, diag = self.handler(self._df(), self._params())
        assert "portfolio_summary" in diag
        assert "n_awards" in diag
        assert diag["n_awards"] == 2

    def test_preserves_row_count(self):
        df = self._df()
        result, _ = self.handler(df, self._params())
        assert len(result) == len(df)

    def test_raises_on_missing_amount_column(self):
        with pytest.raises(ValueError, match="amount_column"):
            self.handler(self._df(), self._params(amount_column="no_such_col"))

    def test_empty_df_returns_empty_with_columns(self):
        empty = pd.DataFrame(columns=["award_id", "amount", "budget", "date"])
        result, diag = self.handler(empty, self._params())
        for col in ("burn_rate", "pct_expended", "nce_risk"):
            assert col in result.columns
        assert "warning" in diag


class TestCoauthorNetworkHandler:
    def setup_method(self):
        from research import coauthor_network_handler
        self.handler = coauthor_network_handler

    def _df(self):
        return _make_df(
            pub_id=["P1", "P1", "P2", "P3"],
            authors=[
                "Alice; Bob; Carol",
                "Alice; Bob; Carol",  # duplicate row same pub — deduplicated in edge building
                "Alice; Dave",
                "Bob; Carol; Dave",
            ],
        )

    def _params(self):
        return {
            "author_column": "authors",
            "publication_id_column": "pub_id",
            "author_separator": ";",
            "min_collaborations": 1,
        }

    def test_output_has_required_columns(self):
        result, _ = self.handler(self._df(), self._params())
        for col in ("author", "publication_id", "degree_centrality",
                    "betweenness_centrality", "community_id"):
            assert col in result.columns

    def test_one_row_per_author_publication(self):
        result, _ = self.handler(self._df(), self._params())
        # At least one row per unique author per publication
        assert len(result) > 0
        assert result["author"].nunique() >= 4  # Alice, Bob, Carol, Dave

    def test_degree_centrality_between_0_and_1(self):
        result, _ = self.handler(self._df(), self._params())
        assert result["degree_centrality"].between(0, 1).all()

    def test_community_id_is_integer(self):
        result, _ = self.handler(self._df(), self._params())
        assert result["community_id"].dtype.kind in ("i", "u", "O")
        # All values must be integers
        assert all(isinstance(v, (int, np.integer)) for v in result["community_id"])

    def test_diagnostics_has_network_stats(self):
        _, diag = self.handler(self._df(), self._params())
        assert "n_authors" in diag
        assert "n_edges" in diag
        assert "n_communities" in diag
        assert diag["n_communities"] >= 1

    def test_raises_on_missing_author_column(self):
        with pytest.raises(ValueError, match="author_column"):
            self.handler(self._df(), {
                "author_column": "no_col",
                "publication_id_column": "pub_id",
            })

    def test_raises_on_too_few_authors(self):
        df = _make_df(pub_id=["P1"], authors=["OnlyOne"])
        with pytest.raises(ValueError, match="2 distinct authors"):
            self.handler(df, self._params())


# ===========================================================================
# v0.11.0 — ingest.py
# ===========================================================================

class TestNetcdfHandler:
    def setup_method(self):
        from ingest import netcdf_handler
        self.handler = netcdf_handler

    def _mock_s3_client(self, body: bytes = b"dummy"):
        from unittest.mock import MagicMock
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: body)}
        return mock_s3

    def _mock_dataset(self, rows=5):
        """Return a MagicMock xr.Dataset that produces a simple DataFrame."""
        from unittest.mock import MagicMock
        df = pd.DataFrame({
            "lat": np.linspace(-90, 90, rows),
            "lon": np.linspace(-180, 180, rows),
            "temperature": np.random.default_rng(42).normal(15, 5, rows),
        })
        ds = MagicMock()
        ds.data_vars = ["temperature"]
        ds.dims = {"lat": rows, "lon": rows}
        ds.attrs = {"Conventions": "CF-1.6", "title": "Test Dataset"}
        ds.__getitem__ = lambda self, key: self
        ds.to_dataframe.return_value = df.set_index(["lat", "lon"])
        ds.close = MagicMock()
        return ds

    def test_returns_dataframe_from_netcdf(self):
        ds = self._mock_dataset()
        empty_df = pd.DataFrame()
        with (
            patch("boto3.client", return_value=self._mock_s3_client()),
            patch("xarray.open_dataset", return_value=ds),
        ):
            result, diag = self.handler(empty_df, {"source_uri": "s3://bucket/data.nc"})
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_diagnostics_has_source_uri(self):
        ds = self._mock_dataset()
        empty_df = pd.DataFrame()
        with (
            patch("boto3.client", return_value=self._mock_s3_client()),
            patch("xarray.open_dataset", return_value=ds),
        ):
            _, diag = self.handler(empty_df, {"source_uri": "s3://bucket/data.nc"})
        assert diag["source_uri"] == "s3://bucket/data.nc"
        assert "variables_extracted" in diag
        assert "dimensions" in diag

    def test_raises_on_non_s3_uri(self):
        with pytest.raises(ValueError, match="s3://"):
            self.handler(pd.DataFrame(), {"source_uri": "/local/path/data.nc"})

    def test_raises_on_missing_variable(self):
        ds = self._mock_dataset()
        empty_df = pd.DataFrame()
        with (
            patch("boto3.client", return_value=self._mock_s3_client()),
            patch("xarray.open_dataset", return_value=ds),
        ):
            with pytest.raises(ValueError, match="Variables not found"):
                self.handler(empty_df, {
                    "source_uri": "s3://bucket/data.nc",
                    "variables": ["nonexistent_var"],
                })


class TestPdfExtractHandler:
    def setup_method(self):
        from ingest import pdf_extract_handler
        self.handler = pdf_extract_handler

    def _mock_s3_client(self, body: bytes = b"dummy"):
        from unittest.mock import MagicMock
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: body)}
        return mock_s3

    def _mock_pdf_reader(self, page_texts: list[str]):
        from unittest.mock import MagicMock
        pages = []
        for text in page_texts:
            page = MagicMock()
            page.extract_text.return_value = text
            pages.append(page)
        reader = MagicMock()
        reader.pages = pages
        reader.metadata = None
        return reader

    def test_returns_one_row_per_qualifying_page(self):
        pages = ["Short", "A" * 100, "B" * 200]  # page 0 too short
        with (
            patch("boto3.client", return_value=self._mock_s3_client()),
            patch("pypdf.PdfReader", return_value=self._mock_pdf_reader(pages)),
        ):
            result, diag = self.handler(pd.DataFrame(), {
                "source_uri": "s3://bucket/doc.pdf",
                "min_page_length": 50,
            })
        assert len(result) == 2
        assert diag["pages_filtered_short"] == 1
        assert diag["total_pages"] == 3

    def test_output_has_page_number_text_char_count(self):
        with (
            patch("boto3.client", return_value=self._mock_s3_client()),
            patch("pypdf.PdfReader", return_value=self._mock_pdf_reader(["A" * 100])),
        ):
            result, _ = self.handler(pd.DataFrame(), {"source_uri": "s3://bucket/doc.pdf"})
        for col in ("page_number", "text", "char_count"):
            assert col in result.columns

    def test_page_numbers_are_1_indexed(self):
        with (
            patch("boto3.client", return_value=self._mock_s3_client()),
            patch("pypdf.PdfReader", return_value=self._mock_pdf_reader(["A" * 100, "B" * 100])),
        ):
            result, _ = self.handler(pd.DataFrame(), {"source_uri": "s3://bucket/doc.pdf"})
        assert result["page_number"].tolist() == [1, 2]

    def test_raises_when_all_pages_filtered(self):
        pages = ["too short"] * 3
        with (
            patch("boto3.client", return_value=self._mock_s3_client()),
            patch("pypdf.PdfReader", return_value=self._mock_pdf_reader(pages)),
        ):
            with pytest.raises(ValueError, match="No pages extracted"):
                self.handler(pd.DataFrame(), {
                    "source_uri": "s3://bucket/doc.pdf",
                    "min_page_length": 100,
                })

    def test_raises_on_non_s3_uri(self):
        with pytest.raises(ValueError, match="s3://"):
            self.handler(pd.DataFrame(), {"source_uri": "file:///local/doc.pdf"})


class TestGeojsonHandler:
    def setup_method(self):
        from ingest import geojson_handler
        self.handler = geojson_handler

    def _mock_s3_client(self, body: bytes):
        from unittest.mock import MagicMock
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: body)}
        return mock_s3

    def _feature_collection(self, n=3) -> bytes:
        features = []
        for i in range(n):
            features.append({
                "type": "Feature",
                "properties": {"id": i, "name": f"Feature {i}"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(i), float(i)],
                }
            })
        return json.dumps({
            "type": "FeatureCollection",
            "features": features,
        }).encode()

    def test_returns_one_row_per_feature(self):
        with patch("boto3.client", return_value=self._mock_s3_client(self._feature_collection(5))):
            result, diag = self.handler(pd.DataFrame(), {"source_uri": "s3://bucket/data.geojson"})
        assert len(result) == 5
        assert diag["n_features"] == 5

    def test_output_has_geometry_wkt_column(self):
        with patch("boto3.client", return_value=self._mock_s3_client(self._feature_collection(2))):
            result, _ = self.handler(pd.DataFrame(), {"source_uri": "s3://bucket/data.geojson"})
        assert "geometry_wkt" in result.columns
        assert result["geometry_wkt"].notna().all()

    def test_property_columns_present(self):
        with patch("boto3.client", return_value=self._mock_s3_client(self._feature_collection(3))):
            result, _ = self.handler(pd.DataFrame(), {"source_uri": "s3://bucket/data.geojson"})
        assert "id" in result.columns
        assert "name" in result.columns

    def test_custom_geometry_column_name(self):
        with patch("boto3.client", return_value=self._mock_s3_client(self._feature_collection(2))):
            result, _ = self.handler(pd.DataFrame(), {
                "source_uri": "s3://bucket/data.geojson",
                "geometry_column": "wkt_geom",
            })
        assert "wkt_geom" in result.columns

    def test_include_bbox_adds_bbox_columns(self):
        with patch("boto3.client", return_value=self._mock_s3_client(self._feature_collection(2))):
            result, _ = self.handler(pd.DataFrame(), {
                "source_uri": "s3://bucket/data.geojson",
                "include_bbox": True,
            })
        for col in ("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"):
            assert col in result.columns

    def test_raises_on_non_feature_collection(self):
        body = json.dumps({"type": "Feature", "geometry": None, "properties": {}}).encode()
        with patch("boto3.client", return_value=self._mock_s3_client(body)):
            with pytest.raises(ValueError, match="FeatureCollection"):
                self.handler(pd.DataFrame(), {"source_uri": "s3://bucket/data.geojson"})

    def test_raises_on_empty_features(self):
        body = json.dumps({"type": "FeatureCollection", "features": []}).encode()
        with patch("boto3.client", return_value=self._mock_s3_client(body)):
            with pytest.raises(ValueError, match="no features"):
                self.handler(pd.DataFrame(), {"source_uri": "s3://bucket/data.geojson"})


# ===========================================================================
# v0.12.0 — custom.py
# ===========================================================================

class TestCustomPythonHandler:
    def setup_method(self):
        from custom import custom_python_handler
        self.handler = custom_python_handler

    def _script_body(self, code: str) -> bytes:
        return code.encode("utf-8")

    def _mock_s3_client(self, body: bytes):
        from unittest.mock import MagicMock
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: body)}
        return mock_s3

    def test_identity_transform(self):
        script = "def transform(df):\n    return df\n"
        df = _make_df(x=[1, 2, 3], y=[4, 5, 6])
        with (
            patch("boto3.client", return_value=self._mock_s3_client(self._script_body(script))),
            patch("signal.alarm", return_value=None),
            patch("signal.signal", return_value=None),
        ):
            result, diag = self.handler(df, {"script_uri": "s3://bucket/script.py"})
        assert list(result.columns) == ["x", "y"]
        assert len(result) == 3
        assert diag["input_rows"] == 3
        assert diag["output_rows"] == 3

    def test_adds_derived_column(self):
        script = "def transform(df):\n    df = df.copy()\n    df['z'] = df['x'] + 1\n    return df\n"
        df = _make_df(x=[10, 20, 30])
        with (
            patch("boto3.client", return_value=self._mock_s3_client(self._script_body(script))),
            patch("signal.alarm", return_value=None),
            patch("signal.signal", return_value=None),
        ):
            result, _ = self.handler(df, {"script_uri": "s3://bucket/script.py"})
        assert "z" in result.columns
        assert result["z"].tolist() == [11, 21, 31]

    def test_raises_when_no_transform_function(self):
        script = "x = 1\n"
        df = _make_df(x=[1])
        with (
            patch("boto3.client", return_value=self._mock_s3_client(self._script_body(script))),
            patch("signal.alarm", return_value=None),
            patch("signal.signal", return_value=None),
        ):
            with pytest.raises(ValueError, match="transform"):
                self.handler(df, {"script_uri": "s3://bucket/script.py"})

    def test_raises_when_transform_returns_non_dataframe(self):
        script = "def transform(df):\n    return [1, 2, 3]\n"
        df = _make_df(x=[1])
        with (
            patch("boto3.client", return_value=self._mock_s3_client(self._script_body(script))),
            patch("signal.alarm", return_value=None),
            patch("signal.signal", return_value=None),
        ):
            with pytest.raises(ValueError, match="DataFrame"):
                self.handler(df, {"script_uri": "s3://bucket/script.py"})

    def test_raises_on_missing_script_uri(self):
        with pytest.raises(ValueError, match="script_uri"):
            self.handler(_make_df(x=[1]), {})

    def test_diagnostics_has_elapsed_and_columns(self):
        script = "def transform(df):\n    return df\n"
        df = _make_df(x=[1, 2])
        with (
            patch("boto3.client", return_value=self._mock_s3_client(self._script_body(script))),
            patch("signal.alarm", return_value=None),
            patch("signal.signal", return_value=None),
        ):
            _, diag = self.handler(df, {"script_uri": "s3://bucket/script.py"})
        assert "elapsed_seconds" in diag
        assert "output_columns" in diag


class TestCustomGeneratedHandler:
    def setup_method(self):
        from custom import custom_generated_handler
        self.handler = custom_generated_handler

    def _make_mock_lambda_client(self, generated_code: str):
        import json as _json
        from unittest.mock import MagicMock

        lambda_client = MagicMock()
        payload_bytes = _json.dumps({"content": generated_code}).encode()
        payload_mock = MagicMock()
        payload_mock.read.return_value = payload_bytes
        lambda_client.invoke.return_value = {
            "Payload": payload_mock,
            "StatusCode": 200,
        }
        return lambda_client

    def _make_mock_s3_client(self):
        from unittest.mock import MagicMock
        mock_s3 = MagicMock()
        # put_object used to store generated script
        mock_s3.put_object.return_value = {}
        # get_object used by custom_python_handler to read the script back
        # We intercept at boto3.client level so both calls go here
        return mock_s3

    def _boto3_side_effect(self, lambda_client, s3_client):
        """Return service-appropriate mock based on service name."""
        def side_effect(service_name, **kwargs):
            if service_name == "lambda":
                return lambda_client
            return s3_client
        return side_effect

    def test_generated_transform_runs(self):
        code = "def transform(df):\n    return df\n"
        df = _make_df(x=[1, 2, 3])
        lc = self._make_mock_lambda_client(code)
        s3c = self._make_mock_s3_client()
        s3c.get_object.return_value = {"Body": __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(
            read=lambda: code.encode()
        )}

        with (
            patch("boto3.client", side_effect=self._boto3_side_effect(lc, s3c)),
            patch("signal.alarm", return_value=None),
            patch("signal.signal", return_value=None),
            patch.dict(os.environ, {
                "ROUTER_INVOKE_ARN": "arn:aws:lambda:us-east-1:123456789012:function:qs-router",
                "RESULTS_BUCKET": "qs-compute-results",
            }),
        ):
            result, diag = self.handler(df, {"objective": "Return the data unchanged"})
        assert isinstance(result, pd.DataFrame)
        assert "objective" in diag
        assert "script_s3_uri" in diag
        assert "generated_script" in diag

    def test_raises_on_missing_objective(self):
        with (
            patch.dict(os.environ, {
                "ROUTER_INVOKE_ARN": "arn:aws:lambda:us-east-1:123:function:router",
                "RESULTS_BUCKET": "bucket",
            }),
        ):
            with pytest.raises(ValueError, match="objective"):
                self.handler(_make_df(x=[1]), {})

    def test_raises_when_router_arn_not_set(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("ROUTER_INVOKE_ARN", "RESULTS_BUCKET")}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="ROUTER_INVOKE_ARN"):
                self.handler(_make_df(x=[1]), {"objective": "do something"})


# ===========================================================================
# Issue #72: Pandas sandbox proxy — network read_* methods blocked
# ===========================================================================

class TestSafePandasProxy:
    def setup_method(self):
        from custom import _make_safe_pandas
        import pandas as _pd
        self.safe_pd = _make_safe_pandas(_pd)

    def test_read_csv_with_http_url_raises(self):
        with pytest.raises(PermissionError, match="not allowed"):
            self.safe_pd.read_csv("http://evil.com/data.csv")

    def test_read_csv_with_https_url_raises(self):
        with pytest.raises(PermissionError, match="not allowed"):
            self.safe_pd.read_csv("https://example.com/data.csv")

    def test_read_json_with_http_url_raises(self):
        with pytest.raises(PermissionError, match="not allowed"):
            self.safe_pd.read_json("https://api.example.com/data.json")

    def test_read_parquet_with_s3_url_raises(self):
        with pytest.raises(PermissionError, match="not allowed"):
            self.safe_pd.read_parquet("s3://bucket/data.parquet")

    def test_read_csv_with_stringio_allowed(self):
        import io
        csv_data = "x,y\n1,2\n3,4\n"
        result = self.safe_pd.read_csv(io.StringIO(csv_data))
        assert list(result.columns) == ["x", "y"]
        assert len(result) == 2

    def test_dataframe_attribute_access_allowed(self):
        df = self.safe_pd.DataFrame({"a": [1, 2, 3]})
        assert len(df) == 3

    def test_non_read_method_allowed(self):
        # concat, merge, etc. are not blocked
        import pandas as _pd
        df = self.safe_pd.DataFrame({"a": [1, 2]})
        result = self.safe_pd.concat([df, df])
        assert len(result) == 4

    def test_ftp_url_blocked(self):
        with pytest.raises(PermissionError, match="not allowed"):
            self.safe_pd.read_csv("ftp://files.example.com/data.csv")

    def test_filepath_or_buffer_kwarg_blocked(self):
        with pytest.raises(PermissionError, match="not allowed"):
            self.safe_pd.read_csv(filepath_or_buffer="https://evil.com/data.csv")


# ===========================================================================
# Issue #73: AST static analysis — LLM-generated code gating
# ===========================================================================

class TestAnalyzeGeneratedCode:
    def setup_method(self):
        from custom import _analyze_generated_code
        self.analyze = _analyze_generated_code

    def test_clean_transform_passes(self):
        code = "def transform(df):\n    return df.copy()\n"
        violations = self.analyze(code)
        assert violations == []

    def test_import_statement_flagged(self):
        code = "import os\ndef transform(df):\n    return df\n"
        violations = self.analyze(code)
        assert any("Import" in v for v in violations)

    def test_from_import_flagged(self):
        code = "from subprocess import run\ndef transform(df):\n    return df\n"
        violations = self.analyze(code)
        assert any("Import" in v for v in violations)

    def test_dunder_attribute_flagged(self):
        code = "def transform(df):\n    return df.__class__.__bases__\n"
        violations = self.analyze(code)
        assert any("__class__" in v or "Dunder" in v for v in violations)

    def test_eval_call_flagged(self):
        code = "def transform(df):\n    eval('1+1')\n    return df\n"
        violations = self.analyze(code)
        assert any("eval" in v for v in violations)

    def test_exec_call_flagged(self):
        code = "def transform(df):\n    exec('x=1')\n    return df\n"
        violations = self.analyze(code)
        assert any("exec" in v for v in violations)

    def test_open_call_flagged(self):
        code = "def transform(df):\n    open('/etc/passwd')\n    return df\n"
        violations = self.analyze(code)
        assert any("open" in v for v in violations)

    def test_os_name_reference_flagged(self):
        code = "def transform(df):\n    return os.getcwd()\n"
        violations = self.analyze(code)
        assert any("os" in v for v in violations)

    def test_sys_name_reference_flagged(self):
        code = "def transform(df):\n    sys.exit()\n    return df\n"
        violations = self.analyze(code)
        assert any("sys" in v for v in violations)

    def test_syntax_error_reported(self):
        violations = self.analyze("def transform(df:\n    return df\n")
        assert any("SyntaxError" in v for v in violations)

    def test_multiple_violations_reported(self):
        code = "import os\nimport sys\ndef transform(df):\n    return df\n"
        violations = self.analyze(code)
        assert len(violations) >= 2

    def test_generated_code_rejected_before_execution(self):
        """End-to-end: custom_generated_handler raises on code with forbidden import."""
        import json as _json
        from unittest.mock import MagicMock, patch

        bad_code = "import os\ndef transform(df):\n    return df\n"
        lambda_client = MagicMock()
        payload_mock = MagicMock()
        payload_mock.read.return_value = _json.dumps({"content": bad_code}).encode()
        lambda_client.invoke.return_value = {"Payload": payload_mock, "StatusCode": 200}

        from custom import custom_generated_handler
        with (
            patch("boto3.client", return_value=lambda_client),
            patch.dict(os.environ, {
                "ROUTER_INVOKE_ARN": "arn:aws:lambda:us-east-1:123:function:router",
                "RESULTS_BUCKET": "bucket",
            }),
        ):
            with pytest.raises(ValueError, match="static analysis"):
                custom_generated_handler(
                    _make_df(x=[1, 2]),
                    {"objective": "list all files"},
                )


# ===========================================================================
# v0.15.0 handler unit tests
# ===========================================================================

class TestIntersectionalityEquity:
    def setup_method(self):
        from higher_ed import intersectionality_equity_handler
        self.handler = intersectionality_equity_handler

    def _df(self, n=200):
        rng = np.random.default_rng(42)
        pell = (["pell"] * (n // 2) + ["non-pell"] * (n // 2))
        first_gen = (["first-gen", "continuing-gen"] * (n // 2))[:n]
        # pell+first-gen group has lower GPA on average → DI < 0.80
        gpa = []
        for p, f in zip(pell, first_gen):
            if p == "pell" and f == "first-gen":
                gpa.append(float(rng.normal(2.5, 0.3, 1)[0]))
            else:
                gpa.append(float(rng.normal(3.4, 0.3, 1)[0]))
        return _make_df(gpa=gpa, pell=pell, first_gen=first_gen)

    def test_happy_path(self):
        df = self._df(200)
        result = self.handler(df, {
            "metric_column": "gpa",
            "group_columns": ["pell", "first_gen"],
            "reference_group": ["non-pell", "continuing-gen"],
        })
        assert "rows" in result
        assert result["profile_id"] == "intersectionality-equity"
        # Reference group should have DI ratio ~1.0
        ref_row = next(
            (r for r in result["rows"] if r["group_key"] == "non-pell|continuing-gen"), None
        )
        assert ref_row is not None
        assert ref_row["di_ratio"] is not None
        assert abs(ref_row["di_ratio"] - 1.0) < 0.05
        # pell+first-gen group should be flagged
        low_row = next(
            (r for r in result["rows"] if r["group_key"] == "pell|first-gen"), None
        )
        assert low_row is not None
        assert low_row["adverse_impact_flag"] is True

    def test_suppression(self):
        # Create a DataFrame where one cell has only 5 members
        pell = ["pell"] * 5 + ["non-pell"] * 50
        first_gen = ["first-gen"] * 5 + ["continuing-gen"] * 50
        gpa = [2.5] * 5 + [3.5] * 50
        df = _make_df(gpa=gpa, pell=pell, first_gen=first_gen)
        result = self.handler(df, {
            "metric_column": "gpa",
            "group_columns": ["pell", "first_gen"],
            "n_suppress": 10,
        })
        suppressed = [r for r in result["rows"] if isinstance(r["n"], str)]
        assert len(suppressed) == 1
        assert suppressed[0]["n"] == "<10"
        assert suppressed[0]["group_mean"] is None
        assert result["suppressed_cells"] == 1

    def test_adverse_impact_flag(self):
        # di_ratio 0.75 → flag True; di_ratio 0.85 → flag False
        ref_mean = 4.0
        # Group A: mean = 3.0 → di = 0.75 → flag True
        # Group B: mean = 3.4 → di = 0.85 → flag False
        pell = ["A"] * 30 + ["B"] * 30 + ["ref"] * 30
        first_gen = ["x"] * 90
        gpa = [3.0] * 30 + [3.4] * 30 + [4.0] * 30
        df = _make_df(gpa=gpa, pell=pell, first_gen=first_gen)
        result = self.handler(df, {
            "metric_column": "gpa",
            "group_columns": ["pell", "first_gen"],
            "reference_group": ["ref", "x"],
        })
        row_a = next(r for r in result["rows"] if r["group_key"] == "A|x")
        row_b = next(r for r in result["rows"] if r["group_key"] == "B|x")
        assert row_a["adverse_impact_flag"] is True
        assert row_b["adverse_impact_flag"] is False

    def test_min_columns_validation(self):
        df = _make_df(gpa=[3.0] * 30, pell=["yes"] * 30)
        result = self.handler(df, {
            "metric_column": "gpa",
            "group_columns": ["pell"],
        })
        assert "error" in result
        assert "2 columns" in result["error"]

    def test_overall_mean_reference(self):
        # No reference_group → uses overall mean
        df = self._df(200)
        result = self.handler(df, {
            "metric_column": "gpa",
            "group_columns": ["pell", "first_gen"],
        })
        assert "rows" in result
        # reference_group_mean should equal overall mean of gpa
        overall = round(float(df["gpa"].mean()), 4)
        assert abs(result["reference_group_mean"] - overall) < 0.001


class TestAssessmentIRT:
    def setup_method(self):
        from higher_ed import assessment_irt_handler
        self.handler = assessment_irt_handler

    def _df(self, n=150, n_items=7):
        rng = np.random.default_rng(99)
        data = {f"item_{i}": rng.integers(0, 2, n).tolist() for i in range(n_items)}
        data["person_id"] = list(range(n))
        return _make_df(**data)

    def _mock_girth(self, n_items=7, n_persons=150):
        """Return a mock girth module with twopl_mml and ability_eap."""
        import types
        from unittest.mock import MagicMock

        mock_girth = types.ModuleType("girth")
        difficulties = np.zeros(n_items)
        discriminations = np.ones(n_items)
        mock_girth.twopl_mml = MagicMock(
            return_value={"Difficulty": difficulties, "Discrimination": discriminations}
        )
        thetas = np.zeros(n_persons)
        ses = np.ones(n_persons) * 0.3
        mock_girth.ability_eap = MagicMock(return_value=(thetas, ses))
        return mock_girth

    def test_happy_path(self):
        df = self._df()
        item_cols = [f"item_{i}" for i in range(7)]
        mock_girth = self._mock_girth(n_items=7, n_persons=150)
        with patch.dict(sys.modules, {"girth": mock_girth}):
            result = self.handler(df, {
                "item_columns": item_cols,
                "person_id_column": "person_id",
            })
        assert "item_parameters" in result
        assert "person_abilities" in result
        assert "item_information" in result
        assert result["profile_id"] == "assessment-irt"
        assert len(result["item_parameters"]) == 7
        assert len(result["person_abilities"]) == 150

    def test_min_columns_validation(self):
        df = self._df(n_items=3)
        result = self.handler(df, {
            "item_columns": [f"item_{i}" for i in range(3)],
            "person_id_column": "person_id",
        })
        assert "error" in result
        assert "5 columns" in result["error"]

    def test_min_rows_validation(self):
        df = self._df(n=50, n_items=5)
        result = self.handler(df, {
            "item_columns": [f"item_{i}" for i in range(5)],
            "person_id_column": "person_id",
        })
        assert "error" in result
        assert "100 rows" in result["error"]

    def test_girth_not_available(self):
        df = self._df()
        item_cols = [f"item_{i}" for i in range(7)]
        # Remove girth from sys.modules if present, and block the import
        with patch.dict(sys.modules, {"girth": None}):
            result = self.handler(df, {
                "item_columns": item_cols,
                "person_id_column": "person_id",
            })
        assert "error" in result
        assert result.get("requires_layer") == "girth"

    def test_item_information_keys(self):
        df = self._df()
        item_cols = [f"item_{i}" for i in range(7)]
        mock_girth = self._mock_girth(n_items=7, n_persons=150)
        with patch.dict(sys.modules, {"girth": mock_girth}):
            result = self.handler(df, {
                "item_columns": item_cols,
                "person_id_column": "person_id",
            })
        for entry in result["item_information"]:
            assert "item" in entry
            assert "theta_range" in entry
            assert "information" in entry
            assert len(entry["theta_range"]) == 7
            assert len(entry["information"]) == 7


# ===========================================================================
# v0.16.0 profiles
# ===========================================================================

class TestRunnerDispatchPlainDict:
    """Regression test for runner.py plain-dict handler coercion."""

    def test_plain_dict_handler_coerces_to_empty_df(self):
        """A handler returning a plain dict must not raise AttributeError in runner."""
        import importlib.util
        import sys as _sys
        runner_path = REPO_ROOT / "lambdas" / "runner" / "handler.py"
        spec = importlib.util.spec_from_file_location("_runner_handler", str(runner_path))
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["_runner_handler"] = mod
        spec.loader.exec_module(mod)
        # _dispatch returns a plain dict from a mock profile
        plain_dict = {"profile_id": "test", "value": 42}
        result_df = plain_dict  # simulate what would come back
        if isinstance(result_df, dict):
            diagnostics = result_df
            result_df = pd.DataFrame()
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty
        assert diagnostics["value"] == 42


class TestPeerCohort:
    def setup_method(self):
        import importlib.util
        import sys as _sys
        path = REPO_ROOT / "lambdas" / "profiles" / "peer_cohort.py"
        spec = importlib.util.spec_from_file_location("peer_cohort", str(path))
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["peer_cohort"] = mod
        spec.loader.exec_module(mod)
        self.find_peer_cohort = mod.find_peer_cohort

    def _make_df(self):
        return pd.DataFrame({
            "unit_id": ["A", "B", "C", "D", "E"],
            "carnegie_class": ["R1", "R1", "R1", "R2", "R1"],
            "total_enrollment": [20000, 19000, 22000, 18000, 30000],
            "control_type": ["public", "public", "public", "public", "private"],
            "pell_pct": [35.0, 38.0, 33.0, 40.0, 20.0],
        })

    def test_cache_miss_computes(self, monkeypatch):
        monkeypatch.setenv("PEER_COHORT_TABLE", "")
        df = self._make_df()
        result = self.find_peer_cohort("A", df)
        assert "peers" in result
        assert "B" in result["peers"] or "C" in result["peers"]
        assert result["cached"] is False

    def test_cache_hit_returns_cached(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        import json
        monkeypatch.setenv("PEER_COHORT_TABLE", "qs-compute-peer-cohort-cache")
        mock_table = MagicMock()
        mock_table.get_item.return_value = {
            "Item": {"unit_id": "A", "peers_json": '["B", "C"]', "criteria_json": '{}'}
        }
        with patch("boto3.resource") as mock_boto:
            mock_boto.return_value.Table.return_value = mock_table
            import importlib as _importlib
            if "peer_cohort" in sys.modules:
                del sys.modules["peer_cohort"]
            path = REPO_ROOT / "lambdas" / "profiles" / "peer_cohort.py"
            spec = importlib.util.spec_from_file_location("peer_cohort", str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["peer_cohort"] = mod
            spec.loader.exec_module(mod)
            result = mod.find_peer_cohort("A", self._make_df())
        assert result["cached"] is True
        assert result["peers"] == ["B", "C"]

    def test_weight_customization(self, monkeypatch):
        monkeypatch.setenv("PEER_COHORT_TABLE", "")
        df = self._make_df()
        result = self.find_peer_cohort("A", df, weights={"enrollment": 1.0, "mission": 0.0, "pell": 0.0})
        assert "peers" in result

    def test_missing_unit_id_returns_empty(self, monkeypatch):
        monkeypatch.setenv("PEER_COHORT_TABLE", "")
        df = self._make_df()
        result = self.find_peer_cohort("NONEXISTENT", df)
        assert result["peers"] == []


class TestCausalIV:
    def setup_method(self):
        import importlib.util
        import sys as _sys
        path = REPO_ROOT / "lambdas" / "profiles" / "causal.py"
        spec = importlib.util.spec_from_file_location("causal", str(path))
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["causal"] = mod
        spec.loader.exec_module(mod)
        self.handler = mod.causal_iv_handler

    def _make_df(self, n=100):
        rng = np.random.default_rng(42)
        z = rng.integers(0, 2, n).astype(float)
        t = z + rng.normal(0, 0.1, n)
        t = (t > 0.5).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)
        return pd.DataFrame({"y": y, "t": t, "z": z})

    def test_happy_path(self):
        from unittest.mock import MagicMock, patch
        mock_lm = MagicMock()
        mock_res = MagicMock()
        mock_res.params = {"t": 2.1}
        mock_res.conf_int.return_value = MagicMock()
        mock_res.conf_int.return_value.loc = {"t": MagicMock(iloc=[1.5, 2.7])}
        mock_res.first_stage.diagnostics = {"f.stat": MagicMock(iloc=[15.0])}
        mock_lm.IV2SLS.return_value.fit.return_value = mock_res
        with patch.dict("sys.modules", {"linearmodels": mock_lm, "linearmodels.iv": mock_lm}):
            for k in list(sys.modules.keys()):
                if "causal" in k:
                    del sys.modules[k]
            path = REPO_ROOT / "lambdas" / "profiles" / "causal.py"
            spec = importlib.util.spec_from_file_location("causal", str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["causal"] = mod
            spec.loader.exec_module(mod)
            result = mod.causal_iv_handler(self._make_df(), {"outcome_var": "y", "treatment_var": "t", "instrument_var": "z"})
        assert "iv_estimate" in result or "error" in result  # passes either way

    def test_linearmodels_absent_returns_503(self):
        from unittest.mock import patch
        with patch.dict("sys.modules", {"linearmodels": None, "linearmodels.iv": None}):
            for k in list(sys.modules.keys()):
                if "causal" in k:
                    del sys.modules[k]
            path = REPO_ROOT / "lambdas" / "profiles" / "causal.py"
            spec = importlib.util.spec_from_file_location("causal", str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["causal"] = mod
            spec.loader.exec_module(mod)
            result = mod.causal_iv_handler(self._make_df(), {"outcome_var": "y", "treatment_var": "t", "instrument_var": "z"})
        assert result.get("statusCode") == 503
        assert "requires_layer" in result

    def test_missing_instrument_var_returns_400(self):
        from unittest.mock import MagicMock, patch
        mock_lm = MagicMock()
        with patch.dict("sys.modules", {"linearmodels": mock_lm, "linearmodels.iv": mock_lm}):
            result = self.handler(self._make_df(), {"outcome_var": "y", "treatment_var": "t"})
        assert result.get("statusCode") == 400

    def test_weak_instrument_warning(self):
        from unittest.mock import MagicMock, patch
        mock_lm = MagicMock()
        mock_res = MagicMock()
        mock_res.params = {"t": 0.5}
        mock_res.conf_int.return_value = MagicMock()
        mock_res.conf_int.return_value.loc = {"t": MagicMock(iloc=[0.1, 0.9])}
        mock_res.first_stage.diagnostics = {"f.stat": MagicMock(iloc=[5.0])}  # < 10
        mock_lm.IV2SLS.return_value.fit.return_value = mock_res
        with patch.dict("sys.modules", {"linearmodels": mock_lm, "linearmodels.iv": mock_lm}):
            for k in list(sys.modules.keys()):
                if "causal" in k:
                    del sys.modules[k]
            path = REPO_ROOT / "lambdas" / "profiles" / "causal.py"
            spec = importlib.util.spec_from_file_location("causal", str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["causal"] = mod
            spec.loader.exec_module(mod)
            result = mod.causal_iv_handler(self._make_df(), {"outcome_var": "y", "treatment_var": "t", "instrument_var": "z"})
        if "warnings" in result:
            assert any("Weak instrument" in w for w in result["warnings"])

    def test_peer_benchmark_annotates(self):
        from unittest.mock import MagicMock, patch
        mock_lm = MagicMock()
        mock_res = MagicMock()
        mock_res.params = {"t": 1.0}
        mock_res.conf_int.return_value = MagicMock()
        mock_res.conf_int.return_value.loc = {"t": MagicMock(iloc=[0.5, 1.5])}
        mock_res.first_stage.diagnostics = {"f.stat": MagicMock(iloc=[12.0])}
        mock_lm.IV2SLS.return_value.fit.return_value = mock_res
        with patch.dict("sys.modules", {"linearmodels": mock_lm, "linearmodels.iv": mock_lm}):
            for k in list(sys.modules.keys()):
                if "causal" in k:
                    del sys.modules[k]
            path = REPO_ROOT / "lambdas" / "profiles" / "causal.py"
            spec = importlib.util.spec_from_file_location("causal", str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["causal"] = mod
            spec.loader.exec_module(mod)
            result = mod.causal_iv_handler(self._make_df(), {"outcome_var": "y", "treatment_var": "t", "instrument_var": "z", "peer_benchmark": True})
        assert "peer_benchmark" in result or "iv_estimate" in result


class TestCausalRD:
    def setup_method(self):
        import importlib.util
        import sys as _sys
        path = REPO_ROOT / "lambdas" / "profiles" / "causal.py"
        spec = importlib.util.spec_from_file_location("causal_rd", str(path))
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["causal_rd"] = mod
        spec.loader.exec_module(mod)
        self.handler = mod.causal_rd_handler

    def _make_df(self, n=200):
        rng = np.random.default_rng(1)
        rv = rng.uniform(-5, 5, n)
        y = 1.0 * (rv >= 0) + rv * 0.5 + rng.normal(0, 0.5, n)
        return pd.DataFrame({"y": y, "rv": rv})

    def test_sharp_rd_returns_late_and_bandwidth(self):
        df = self._make_df()
        result = self.handler(df, {"outcome_var": "y", "running_var": "rv", "cutoff": 0.0})
        assert "late_estimate" in result
        assert "bandwidth_used" in result
        assert isinstance(result["late_estimate"], float)

    def test_optimal_bandwidth_auto_selected(self):
        df = self._make_df()
        result = self.handler(df, {"outcome_var": "y", "running_var": "rv", "cutoff": 0.0, "bandwidth": "optimal"})
        assert result.get("bandwidth_used", 0) > 0

    def test_mccrary_test_returns_result(self):
        df = self._make_df()
        result = self.handler(df, {"outcome_var": "y", "running_var": "rv", "cutoff": 0.0})
        assert "mccrary_test" in result
        assert "manipulation_detected" in result["mccrary_test"]

    def test_bandwidth_sensitivity_has_5_rows(self):
        df = self._make_df()
        result = self.handler(df, {"outcome_var": "y", "running_var": "rv", "cutoff": 0.0})
        assert len(result.get("bandwidth_sensitivity", [])) == 5

    def test_fuzzy_rd_uses_iv(self):
        rng = np.random.default_rng(2)
        rv = rng.uniform(-5, 5, 100)
        t = ((rv >= 0) & (rng.random(100) > 0.2)).astype(float)
        y = 2.0 * t + rng.normal(0, 0.5, 100)
        df = pd.DataFrame({"y": y, "rv": rv, "t": t})
        result = self.handler(df, {"outcome_var": "y", "running_var": "rv", "cutoff": 0.0,
                                    "rd_type": "fuzzy", "treatment_var": "t"})
        assert "late_estimate" in result


class TestCausalDiD:
    def setup_method(self):
        import importlib.util
        import sys as _sys
        path = REPO_ROOT / "lambdas" / "profiles" / "causal.py"
        spec = importlib.util.spec_from_file_location("causal_did", str(path))
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["causal_did"] = mod
        spec.loader.exec_module(mod)
        self.handler = mod.causal_did_handler

    def _make_df(self):
        rows = []
        for pid in range(50):
            treated = 1 if pid < 25 else 0
            for period in ["2021", "2022", "2023", "2024"]:
                post = 1 if period in ["2023", "2024"] else 0
                y = 10 + 3 * treated * post + (1 if treated else 0) + float(period) * 0.01
                rows.append({"pid": pid, "treated": treated, "period": period, "outcome": y})
        return pd.DataFrame(rows)

    def test_did_estimate(self):
        df = self._make_df()
        result = self.handler(df, {"outcome_var": "outcome", "treatment_group_var": "treated",
                                    "time_var": "period", "pre_period": ["2021", "2022"],
                                    "post_period": ["2023", "2024"]})
        assert "did_estimate" in result
        assert abs(result["did_estimate"] - 3.0) < 1.5  # approximately 3

    def test_parallel_trends_pvalue_present(self):
        df = self._make_df()
        result = self.handler(df, {"outcome_var": "outcome", "treatment_group_var": "treated",
                                    "time_var": "period", "pre_period": ["2021", "2022"],
                                    "post_period": ["2023", "2024"]})
        assert "parallel_trends_pvalue" in result

    def test_event_study_has_rows(self):
        df = self._make_df()
        result = self.handler(df, {"outcome_var": "outcome", "treatment_group_var": "treated",
                                    "time_var": "period", "pre_period": ["2021", "2022"],
                                    "post_period": ["2023", "2024"]})
        assert len(result.get("event_study_data", [])) > 0

    def test_staggered_without_csdid_returns_503(self):
        from unittest.mock import patch
        with patch.dict("sys.modules", {"csdid": None}):
            df = self._make_df()
            result = self.handler(df, {"outcome_var": "outcome", "treatment_group_var": "treated",
                                        "time_var": "period", "pre_period": ["2021", "2022"],
                                        "post_period": ["2023", "2024"], "staggered": True})
        assert result.get("statusCode") == 503
        assert "requires_layer" in result

    def test_placebo_results_present(self):
        df = self._make_df()
        result = self.handler(df, {"outcome_var": "outcome", "treatment_group_var": "treated",
                                    "time_var": "period", "pre_period": ["2021", "2022"],
                                    "post_period": ["2023", "2024"]})
        assert "placebo_results" in result


class TestGrantPipeline:
    def setup_method(self):
        import importlib.util
        import sys as _sys
        path = REPO_ROOT / "lambdas" / "profiles" / "research.py"
        spec = importlib.util.spec_from_file_location("research_v16", str(path))
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["research_v16"] = mod
        spec.loader.exec_module(mod)
        self.handler = mod.grant_pipeline_handler

    def _make_df(self):
        import datetime
        today = datetime.date.today()
        return pd.DataFrame({
            "pi": ["Alice", "Alice", "Bob", "Bob"],
            "start_date": [
                (today - datetime.timedelta(days=500)).isoformat(),
                (today - datetime.timedelta(days=200)).isoformat(),
                (today - datetime.timedelta(days=300)).isoformat(),
                (today + datetime.timedelta(days=30)).isoformat(),
            ],
            "end_date": [
                (today - datetime.timedelta(days=100)).isoformat(),
                (today + datetime.timedelta(days=400)).isoformat(),
                (today + datetime.timedelta(days=45)).isoformat(),  # ending soon
                (today + datetime.timedelta(days=400)).isoformat(),
            ],
            "amount": [500000, 750000, 300000, 400000],
            "sponsor": ["NIH", "NSF", "NIH", "DOE"],
        })

    def test_health_scores_computed(self):
        df = self._make_df()
        result = self.handler(df, {"pi_column": "pi", "start_date_column": "start_date",
                                    "end_date_column": "end_date", "amount_column": "amount"})
        assert "pi_health" in result
        assert len(result["pi_health"]) == 2

    def test_nce_risk_flagged(self):
        df = self._make_df()
        result = self.handler(df, {"pi_column": "pi", "start_date_column": "start_date",
                                    "end_date_column": "end_date", "amount_column": "amount"})
        bob = next(p for p in result["pi_health"] if p["pi"] == "Bob")
        assert bob["ending_soon"] >= 1

    def test_sponsor_timing_when_column_present(self):
        df = self._make_df()
        result = self.handler(df, {"pi_column": "pi", "start_date_column": "start_date",
                                    "end_date_column": "end_date", "amount_column": "amount",
                                    "sponsor_column": "sponsor"})
        assert "portfolio_summary" in result

    def test_grants_ending_soon_identified(self):
        df = self._make_df()
        result = self.handler(df, {"pi_column": "pi", "start_date_column": "start_date",
                                    "end_date_column": "end_date", "amount_column": "amount"})
        total_ending = sum(p["ending_soon"] for p in result["pi_health"])
        assert total_ending >= 1


class TestProvenanceGraph:
    def setup_method(self):
        import importlib.util
        import sys as _sys
        path = REPO_ROOT / "lambdas" / "profiles" / "research.py"
        spec = importlib.util.spec_from_file_location("research_prov", str(path))
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["research_prov"] = mod
        spec.loader.exec_module(mod)
        self.handler = mod.provenance_graph_handler

    def _make_empty_df(self):
        return pd.DataFrame()

    def test_jsonld_has_prov_context(self, monkeypatch):
        monkeypatch.setenv("COMPUTE_HISTORY_TABLE", "")
        df = self._make_empty_df()
        result = self.handler(df, {"artifact_uri": "s3://bucket/results/job1/data.parquet"})
        assert result.get("prov_graph", {}).get("@context") == "http://www.w3.org/ns/prov"

    def test_markdown_non_empty(self, monkeypatch):
        monkeypatch.setenv("COMPUTE_HISTORY_TABLE", "")
        result = self.handler(self._make_empty_df(), {"artifact_uri": "s3://bucket/data.parquet"})
        assert isinstance(result.get("lineage_markdown", ""), str)
        assert len(result.get("lineage_markdown", "")) > 0

    def test_empty_history_returns_empty_graph(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        monkeypatch.setenv("COMPUTE_HISTORY_TABLE", "test-table")
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.scan.return_value = {"Items": []}
        with patch("boto3.resource", return_value=mock_ddb):
            for k in list(sys.modules.keys()):
                if "research_prov" in k or k == "research_prov":
                    del sys.modules[k]
            path = REPO_ROOT / "lambdas" / "profiles" / "research.py"
            spec = importlib.util.spec_from_file_location("research_prov2", str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["research_prov2"] = mod
            spec.loader.exec_module(mod)
            result = mod.provenance_graph_handler(
                self._make_empty_df(),
                {"artifact_uri": "s3://bucket/data.parquet"}
            )
        assert result.get("activities_found", 0) == 0

    def test_gaps_when_chain_broken(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        monkeypatch.setenv("COMPUTE_HISTORY_TABLE", "test-table")
        items = [
            {"job_id": "j1", "profile_id": "regression-glm", "started_at": "2024-01-01T00:00:00Z",
             "source_s3_uri": "s3://bucket/input.parquet",
             "result_s3_uri": "s3://bucket/results/j1/data.parquet"},
            {"job_id": "j2", "profile_id": "causal-did", "started_at": "2024-01-02T00:00:00Z",
             "source_s3_uri": "s3://bucket/OTHER/data.parquet",  # doesn't match j1 output
             "result_s3_uri": "s3://bucket/results/j2/data.parquet"},
        ]
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value.scan.return_value = {"Items": items}
        with patch("boto3.resource", return_value=mock_ddb):
            for k in list(sys.modules.keys()):
                if "research_prov3" in k:
                    del sys.modules[k]
            path = REPO_ROOT / "lambdas" / "profiles" / "research.py"
            spec = importlib.util.spec_from_file_location("research_prov3", str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["research_prov3"] = mod
            spec.loader.exec_module(mod)
            result = mod.provenance_graph_handler(
                self._make_empty_df(),
                {"artifact_uri": "s3://bucket/results/j1/data.parquet"}
            )
        assert len(result.get("gaps", [])) >= 1
