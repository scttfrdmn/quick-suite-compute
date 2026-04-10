"""
Tests for v0.18.0 features:
  #58: Results write-back to source registry
  #59: CSV/Excel export from deliver Lambda
  #60: Financial aid effectiveness profile
"""

import importlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent


def _load_handler(lambda_dir: str, module_alias: str):
    path = REPO_ROOT / "lambdas" / lambda_dir / "handler.py"
    spec = importlib.util.spec_from_file_location(module_alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = mod
    spec.loader.exec_module(mod)
    return mod


_record = _load_handler("record-spend", "_v18_record_spend")
_deliver = _load_handler("deliver", "_v18_deliver")
_status = _load_handler("compute-status", "_v18_compute_status")

# Profile module is on sys.path via conftest
from higher_ed import financial_aid_effectiveness_handler  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_aid_df(n=200):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(n)],
        "aid_year": rng.choice([2022, 2023, 2024], n),
        "efc": rng.integers(0, 30000, n).astype(float),
        "total_grants": rng.integers(0, 20000, n).astype(float),
        "total_loans": rng.integers(0, 10000, n).astype(float),
        "persistence_flag": rng.choice([0, 1], n, p=[0.3, 0.7]),
        "term_gpa": rng.uniform(1.5, 4.0, n).round(2),
    })


def _base_record_event():
    return {
        "user_arn": "arn:aws:iam::123456789012:user/testuser",
        "execution_id": "job-abc12345",
        "profile": {
            "profile_id": "clustering-kmeans",
            "cost_estimate": {"typical_cost_usd": 0.01},
        },
        "compute": {"actual_cost_usd": 0.008, "duration_seconds": 12},
        "deliver": {
            "result_uri": "s3://bucket/results/job-abc12345/data.parquet",
            "result_s3_uri": "s3://bucket/results/job-abc12345/data.parquet",
            "row_count": 100,
        },
    }


# ===========================================================================
# #58 — Registry write-back
# ===========================================================================

class TestRegistryWriteBack:
    def test_happy_path_writes_entry(self):
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        mock_table.update_item.return_value = {}
        mock_table.put_item.return_value = {}

        event = _base_record_event()
        with patch.dict(os.environ, {
            "DATA_REGISTRY_TABLE": "qs-data-source-registry",
            "SNAPSHOTS_TABLE": "",
        }), patch.object(_record, "dynamodb", mock_dynamodb), \
             patch.object(_record, "boto3", MagicMock()):
            _record.handler(event, None)

        # Find the put_item call to the registry table
        registry_calls = [
            call for call in mock_table.put_item.call_args_list
            if "source_id" in (call.kwargs.get("Item", {}) or call[1].get("Item", {}))
            and str(call).count("compute-result") > 0
        ]
        assert len(registry_calls) >= 1
        item = registry_calls[0].kwargs.get("Item") or registry_calls[0][1]["Item"]
        assert item["source_id"].startswith("compute-result-")
        assert item["source_type"] == "s3"

    def test_fail_open_on_dynamodb_error(self):
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        mock_table.update_item.return_value = {}

        # Make put_item raise for registry write only on specific calls
        call_count = {"n": 0}

        def _side_effect(**kwargs):
            call_count["n"] += 1
            item = kwargs.get("Item", {})
            if "source_id" in item and "compute-result" in str(item.get("source_id", "")):
                raise Exception("DynamoDB error")
            return {}

        mock_table.put_item.side_effect = _side_effect

        event = _base_record_event()
        with patch.dict(os.environ, {
            "DATA_REGISTRY_TABLE": "qs-data-source-registry",
            "SNAPSHOTS_TABLE": "",
        }), patch.object(_record, "dynamodb", mock_dynamodb), \
             patch.object(_record, "boto3", MagicMock()):
            result = _record.handler(event, None)

        # Handler should still succeed
        assert result["recorded"] is True

    def test_skips_when_no_registry_table(self):
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        mock_table.update_item.return_value = {}

        event = _base_record_event()
        with patch.dict(os.environ, {
            "DATA_REGISTRY_TABLE": "",
            "SNAPSHOTS_TABLE": "",
        }), patch.object(_record, "dynamodb", mock_dynamodb), \
             patch.object(_record, "boto3", MagicMock()):
            result = _record.handler(event, None)

        assert result["recorded"] is True
        # No registry write should have happened — only spend + history calls
        registry_calls = [
            call for call in mock_table.put_item.call_args_list
            if "compute-result" in str(call)
        ]
        assert len(registry_calls) == 0

    def test_correct_source_id_format(self):
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        mock_table.update_item.return_value = {}
        mock_table.put_item.return_value = {}

        event = _base_record_event()
        event["execution_id"] = "job-xyz99999"

        with patch.dict(os.environ, {
            "DATA_REGISTRY_TABLE": "qs-data-source-registry",
            "SNAPSHOTS_TABLE": "",
        }), patch.object(_record, "dynamodb", mock_dynamodb), \
             patch.object(_record, "boto3", MagicMock()):
            _record.handler(event, None)

        registry_calls = [
            call for call in mock_table.put_item.call_args_list
            if "compute-result" in str(call)
        ]
        assert len(registry_calls) >= 1
        item = registry_calls[0].kwargs.get("Item") or registry_calls[0][1]["Item"]
        assert item["source_id"] == "compute-result-job-xyz99999"


# ===========================================================================
# #59 — CSV/Excel export
# ===========================================================================

class TestCSVExcelExport:
    def _make_deliver_event(self):
        return {
            "execution_id": "job-export-001",
            "dataset_name": "Test Export",
            "user_arn": "arn:aws:iam::123456789012:user/u",
            "profile": {"display_name": "Test", "profile_id": "test"},
            "compute": {
                "result_s3_uri": "s3://test-bucket/results/job-export-001/data.parquet",
            },
        }

    def _mock_s3_with_parquet(self):
        """Create a mock S3 client that returns Parquet bytes for get_object."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        parquet_bytes = buf.getvalue()

        mock_s3 = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = parquet_bytes
        mock_s3.get_object.return_value = {"Body": mock_body}
        mock_s3.head_object.return_value = {}
        mock_s3.put_object.return_value = {}
        mock_s3.generate_presigned_url.return_value = "https://presigned.example.com/file"
        return mock_s3

    def test_csv_generated(self):
        mock_s3 = self._mock_s3_with_parquet()
        mock_qs = MagicMock()
        mock_qs.create_data_source.return_value = {
            "Arn": "arn:aws:quicksight:us-east-1:123:datasource/src",
            "CreationStatus": "CREATION_SUCCESSFUL",
        }
        mock_qs.describe_data_source.return_value = {
            "DataSource": {"Status": "CREATION_SUCCESSFUL"}
        }
        mock_qs.create_data_set.return_value = {"CreationStatus": "CREATED"}

        with patch.object(_deliver, "s3", mock_s3), \
             patch.object(_deliver, "quicksight", mock_qs):
            _deliver.handler(self._make_deliver_event(), None)

        csv_puts = [
            c for c in mock_s3.put_object.call_args_list
            if c.kwargs.get("ContentType") == "text/csv"
        ]
        assert len(csv_puts) == 1

    def test_xlsx_generated(self):
        mock_s3 = self._mock_s3_with_parquet()
        mock_qs = MagicMock()
        mock_qs.create_data_source.return_value = {
            "Arn": "arn:aws:quicksight:us-east-1:123:datasource/src",
            "CreationStatus": "CREATION_SUCCESSFUL",
        }
        mock_qs.describe_data_source.return_value = {
            "DataSource": {"Status": "CREATION_SUCCESSFUL"}
        }
        mock_qs.create_data_set.return_value = {"CreationStatus": "CREATED"}

        with patch.object(_deliver, "s3", mock_s3), \
             patch.object(_deliver, "quicksight", mock_qs):
            _deliver.handler(self._make_deliver_event(), None)

        xlsx_puts = [
            c for c in mock_s3.put_object.call_args_list
            if "spreadsheetml" in (c.kwargs.get("ContentType") or "")
        ]
        assert len(xlsx_puts) == 1

    def test_presigned_urls_in_deliver_output(self):
        mock_s3 = self._mock_s3_with_parquet()
        mock_qs = MagicMock()
        mock_qs.create_data_source.return_value = {
            "Arn": "arn:aws:quicksight:us-east-1:123:datasource/src",
            "CreationStatus": "CREATION_SUCCESSFUL",
        }
        mock_qs.describe_data_source.return_value = {
            "DataSource": {"Status": "CREATION_SUCCESSFUL"}
        }
        mock_qs.create_data_set.return_value = {"CreationStatus": "CREATED"}

        with patch.object(_deliver, "s3", mock_s3), \
             patch.object(_deliver, "quicksight", mock_qs):
            result = _deliver.handler(self._make_deliver_event(), None)

        assert "export_urls" in result
        assert "parquet" in result["export_urls"]
        assert "csv" in result["export_urls"]
        assert "xlsx" in result["export_urls"]

    def test_export_urls_in_status_response(self):
        mock_sfn = MagicMock()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        mock_sfn.describe_execution.return_value = {
            "status": "SUCCEEDED",
            "startDate": now,
            "stopDate": now,
            "output": json.dumps({
                "deliver": {
                    "dataset_id": "ds-123",
                    "result_dataset_name": "Test",
                    "export_urls": {
                        "parquet": "https://example.com/parquet",
                        "csv": "https://example.com/csv",
                        "xlsx": "https://example.com/xlsx",
                    },
                },
                "compute": {},
            }),
        }

        mock_ctx = MagicMock()
        mock_ctx.client_context.custom = {
            "bedrockAgentCoreToolName": "target___compute_status"
        }

        with patch.object(_status, "sfn", mock_sfn):
            result = _status.handler({"job_id": "arn:aws:states:us-east-1:123:execution:sm:j"}, mock_ctx)

        assert result["status"] == "SUCCEEDED"
        assert "export_urls" in result
        assert result["export_urls"]["csv"] == "https://example.com/csv"

    def test_export_failure_non_blocking(self):
        """If pandas/export fails, deliver still returns successfully."""
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {}
        mock_s3.put_object.return_value = {}
        # get_object will raise, simulating export failure
        mock_s3.get_object.side_effect = Exception("S3 read failed")

        mock_qs = MagicMock()
        mock_qs.create_data_source.return_value = {
            "Arn": "arn:aws:quicksight:us-east-1:123:datasource/src",
            "CreationStatus": "CREATION_SUCCESSFUL",
        }
        mock_qs.describe_data_source.return_value = {
            "DataSource": {"Status": "CREATION_SUCCESSFUL"}
        }
        mock_qs.create_data_set.return_value = {"CreationStatus": "CREATED"}

        with patch.object(_deliver, "s3", mock_s3), \
             patch.object(_deliver, "quicksight", mock_qs):
            result = _deliver.handler({
                "execution_id": "job-fail-001",
                "user_arn": "arn:aws:iam::123:user/u",
                "compute": {"result_s3_uri": "s3://b/results/j/data.parquet"},
            }, None)

        assert result["status"] == "delivered"
        # export_urls should not be present since export failed
        assert "export_urls" not in result


# ===========================================================================
# #60 — Financial Aid Effectiveness Profile
# ===========================================================================

class TestFinancialAidEffectiveness:
    def test_aid_bands_computed(self):
        df = _make_aid_df()
        result_df, diag = financial_aid_effectiveness_handler(df, {
            "student_id_col": "student_id",
            "aid_year_col": "aid_year",
            "efc_col": "efc",
            "total_grants_col": "total_grants",
            "total_loans_col": "total_loans",
            "persistence_col": "persistence_flag",
        })
        assert "aid_band" in result_df.columns
        valid_bands = {"<$5k", "$5-10k", "$10-15k", ">$15k"}
        actual_bands = set(result_df["aid_band"].unique())
        assert actual_bands.issubset(valid_bands)

    def test_persistence_rate_per_band(self):
        df = _make_aid_df()
        _, diag = financial_aid_effectiveness_handler(df, {
            "student_id_col": "student_id",
            "aid_year_col": "aid_year",
            "efc_col": "efc",
            "total_grants_col": "total_grants",
            "total_loans_col": "total_loans",
            "persistence_col": "persistence_flag",
        })
        assert "cohort_table" in diag
        assert len(diag["cohort_table"]) > 0
        for row in diag["cohort_table"]:
            assert "persistence_rate" in row
            assert 0 <= row["persistence_rate"] <= 1

    def test_logistic_regression_runs(self):
        df = _make_aid_df()
        result_df, diag = financial_aid_effectiveness_handler(df, {
            "student_id_col": "student_id",
            "aid_year_col": "aid_year",
            "efc_col": "efc",
            "total_grants_col": "total_grants",
            "total_loans_col": "total_loans",
            "persistence_col": "persistence_flag",
        })
        assert "predicted_persistence_prob" in result_df.columns
        probs = result_df["predicted_persistence_prob"].dropna()
        assert len(probs) > 0
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_unmet_need_trend_by_year(self):
        df = _make_aid_df()
        _, diag = financial_aid_effectiveness_handler(df, {
            "student_id_col": "student_id",
            "aid_year_col": "aid_year",
            "efc_col": "efc",
            "total_grants_col": "total_grants",
            "total_loans_col": "total_loans",
            "persistence_col": "persistence_flag",
        })
        assert "unmet_need_trend" in diag
        years = [r["aid_year"] for r in diag["unmet_need_trend"]]
        assert len(years) > 0
        # Should be sorted
        assert years == sorted(years)

    def test_predicted_prob_column_added(self):
        df = _make_aid_df(50)
        result_df, _ = financial_aid_effectiveness_handler(df, {
            "student_id_col": "student_id",
            "aid_year_col": "aid_year",
            "efc_col": "efc",
            "total_grants_col": "total_grants",
            "total_loans_col": "total_loans",
            "persistence_col": "persistence_flag",
        })
        assert "predicted_persistence_prob" in result_df.columns
        assert "net_price" in result_df.columns
        assert "unmet_need" in result_df.columns

    def test_missing_required_column_error(self):
        df = pd.DataFrame({
            "student_id": ["S001"],
            "aid_year": [2023],
            "efc": [10000.0],
            # missing total_grants, total_loans, persistence_flag
        })
        result_df, diag = financial_aid_effectiveness_handler(df, {
            "student_id_col": "student_id",
            "aid_year_col": "aid_year",
            "efc_col": "efc",
            "total_grants_col": "total_grants",
            "total_loans_col": "total_loans",
            "persistence_col": "persistence_flag",
        })
        assert "error" in diag
        assert "Missing required columns" in diag["error"]
