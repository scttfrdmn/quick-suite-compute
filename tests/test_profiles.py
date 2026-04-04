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
