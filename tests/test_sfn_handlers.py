"""
Unit tests for Step Functions handler chain:
  extract, runner, deliver, handle-failure
"""

import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent


def _load_handler(lambda_dir: str, alias: str):
    path = REPO_ROOT / "lambdas" / lambda_dir / "handler.py"
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_extract = _load_handler("extract", "_extract_sfn")
_runner = _load_handler("runner", "_runner_sfn")
_deliver = _load_handler("deliver", "_deliver_sfn")
_failure = _load_handler("handle-failure", "_failure_sfn")


# ---------------------------------------------------------------------------
# Extract helpers
# ---------------------------------------------------------------------------

def _qs_describe_dataset(
    data_source_arn="arn:aws:quicksight:us-east-1:123456789012:datasource/ds1",
):
    return {
        "DataSet": {
            "PhysicalTableMap": {
                "table1": {
                    "S3Source": {
                        "DataSourceArn": data_source_arn,
                        "InputColumns": [
                            {"Name": "id", "Type": "STRING"},
                            {"Name": "value", "Type": "STRING"},
                        ],
                    }
                }
            }
        }
    }


def _qs_describe_datasource(bucket="manifest-bucket", key="manifests/test.json"):
    return {
        "DataSource": {
            "DataSourceParameters": {
                "S3Parameters": {
                    "ManifestFileLocation": {"Bucket": bucket, "Key": key}
                }
            }
        }
    }


def _s3_manifest_response(uris=None, fmt="CSV"):
    manifest = {
        "fileLocations": [{"URIs": uris or ["s3://data-bucket/data.csv"]}],
        "globalUploadSettings": {"format": fmt},
    }
    return {"Body": io.BytesIO(json.dumps(manifest).encode())}


def _s3_csv_response(header="id,value", rows=None):
    rows = rows or ["1,10", "2,20", "3,30"]
    data = header + "\n" + "\n".join(rows) + "\n"
    return {"Body": io.BytesIO(data.encode())}


def _make_extract_mocks(uris=None, csv_rows=None):
    mock_qs = MagicMock()
    mock_qs.describe_data_set.return_value = _qs_describe_dataset()
    mock_qs.describe_data_source.return_value = _qs_describe_datasource()
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = [
        _s3_manifest_response(uris),
        _s3_csv_response(rows=csv_rows),
    ]
    mock_s3.put_object.return_value = {}
    return mock_qs, mock_s3


# ---------------------------------------------------------------------------
# TestExtract
# ---------------------------------------------------------------------------

class TestExtract:
    def _event(self, **kwargs):
        return {"execution_id": "exec-abc123", "dataset_id": "ds-test", **kwargs}

    def test_happy_path_quicksight_dataset(self):
        mock_qs, mock_s3 = _make_extract_mocks()
        with patch.object(_extract, "qs", mock_qs), patch.object(_extract, "s3", mock_s3):
            result = _extract.handler(self._event(), None)
        assert result["row_count"] == 3
        assert "input_s3_uri" in result
        assert result["input_s3_uri"].startswith("s3://")
        assert result["columns"] == ["id", "value"]
        mock_s3.put_object.assert_called_once()

    def test_direct_s3_uri_bypasses_quicksight(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = _s3_csv_response()
        mock_s3.put_object.return_value = {}
        mock_qs = MagicMock()
        with patch.object(_extract, "s3", mock_s3), patch.object(_extract, "qs", mock_qs):
            result = _extract.handler(self._event(source_uri="s3://bucket/data.csv"), None)
        assert "input_s3_uri" in result
        mock_qs.describe_data_set.assert_not_called()

    def test_claws_uri_without_resolver_arn_returns_error(self):
        with patch.object(_extract, "qs", MagicMock()), patch.object(_extract, "s3", MagicMock()), \
             patch.object(_extract, "CLAWS_RESOLVER_ARN", ""):
            result = _extract.handler(self._event(source_uri="claws://dataset/x"), None)
        assert result["status"] == "error"
        assert "CLAWS_RESOLVER_ARN" in result["error"]
        assert result["execution_id"] == "exec-abc123"

    def test_qs_describe_dataset_error_raises(self):
        mock_qs = MagicMock()
        mock_qs.describe_data_set.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}},
            "DescribeDataSet",
        )
        with patch.object(_extract, "qs", mock_qs), patch.object(_extract, "s3", MagicMock()):
            with pytest.raises(RuntimeError, match="DescribeDataSet"):
                _extract.handler(self._event(), None)

    def test_no_s3source_in_physical_table_map_raises(self):
        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = {
            "DataSet": {
                "PhysicalTableMap": {"table1": {"RelationalTable": {"Catalog": "cat"}}}
            }
        }
        with patch.object(_extract, "qs", mock_qs), patch.object(_extract, "s3", MagicMock()):
            with pytest.raises(ValueError, match="S3 source"):
                _extract.handler(self._event(), None)

    def test_empty_manifest_file_locations_raises(self):
        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = _qs_describe_dataset()
        mock_qs.describe_data_source.return_value = _qs_describe_datasource()
        mock_s3 = MagicMock()
        empty_manifest = {"fileLocations": [{"URIs": []}]}
        mock_s3.get_object.return_value = {
            "Body": io.BytesIO(json.dumps(empty_manifest).encode())
        }
        with patch.object(_extract, "qs", mock_qs), patch.object(_extract, "s3", mock_s3):
            with pytest.raises(ValueError, match="no file URIs"):
                _extract.handler(self._event(), None)

    def test_single_file_download_failure_skipped(self):
        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = _qs_describe_dataset()
        mock_qs.describe_data_source.return_value = _qs_describe_datasource()
        mock_s3 = MagicMock()
        manifest = {
            "fileLocations": [{"URIs": ["s3://bucket/fail.csv", "s3://bucket/good.csv"]}],
            "globalUploadSettings": {"format": "CSV"},
        }
        error = botocore.exceptions.ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "nope"}}, "GetObject"
        )
        mock_s3.get_object.side_effect = [
            {"Body": io.BytesIO(json.dumps(manifest).encode())},
            error,
            _s3_csv_response(rows=["1,10", "2,20"]),
        ]
        mock_s3.put_object.return_value = {}
        with patch.object(_extract, "qs", mock_qs), patch.object(_extract, "s3", mock_s3):
            result = _extract.handler(self._event(), None)
        assert result["row_count"] == 2

    def test_schema_mismatch_second_file_skipped(self):
        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = _qs_describe_dataset()
        mock_qs.describe_data_source.return_value = _qs_describe_datasource()
        mock_s3 = MagicMock()
        manifest = {
            "fileLocations": [{"URIs": ["s3://bucket/a.csv", "s3://bucket/b.csv"]}],
            "globalUploadSettings": {"format": "CSV"},
        }
        mock_s3.get_object.side_effect = [
            {"Body": io.BytesIO(json.dumps(manifest).encode())},
            {"Body": io.BytesIO(b"id,value\n1,10\n")},
            {"Body": io.BytesIO(b"different_col,other\n2,20\n")},  # schema mismatch
        ]
        mock_s3.put_object.return_value = {}
        with patch.object(_extract, "qs", mock_qs), patch.object(_extract, "s3", mock_s3):
            result = _extract.handler(self._event(), None)
        assert result["row_count"] == 1  # only rows from first file

    def test_unsupported_format_returns_unsupported_source(self):
        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = _qs_describe_dataset()
        mock_qs.describe_data_source.return_value = _qs_describe_datasource()
        mock_s3 = MagicMock()
        manifest = {
            "fileLocations": [{"URIs": ["s3://bucket/data.xlsx"]}],
            "globalUploadSettings": {"format": "EXCEL"},
        }
        mock_s3.get_object.side_effect = [
            {"Body": io.BytesIO(json.dumps(manifest).encode())},
            {"Body": io.BytesIO(b"binary data")},
        ]
        mock_s3.put_object.return_value = {}
        with patch.object(_extract, "qs", mock_qs), patch.object(_extract, "s3", mock_s3):
            result = _extract.handler(self._event(), None)
        assert result["status"] == "unsupported_source"


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _runner_event(
    profile_id="clustering-kmeans",
    entrypoint="clustering.kmeans_handler",
    input_s3_uri="s3://qs-compute-test-bucket/inputs/exec-test/data.parquet",
):
    return {
        "execution_id": "exec-test-runner",
        "profile": {"profile_id": profile_id, "entrypoint": entrypoint},
        "parameters": {"k": 3},
        "extract": {"input_s3_uri": input_s3_uri},
    }


def _make_runner_s3(csv_data=b"id,value\n1,10\n2,20\n"):
    mock = MagicMock()
    NoSuchKey = type("NoSuchKey", (Exception,), {})
    mock.exceptions.NoSuchKey = NoSuchKey
    mock.get_object.return_value = {"Body": io.BytesIO(csv_data)}
    mock.put_object.return_value = {}
    return mock


def _make_profile_module(df_result=None):
    df = df_result if df_result is not None else pd.DataFrame(
        {"id": [1, 2], "cluster_id": [0, 1]}
    )
    mock_mod = MagicMock()
    mock_mod.kmeans_handler.return_value = (df, {"inertia": 10.5})
    return mock_mod


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------

class TestRunner:
    def test_happy_path_returns_result(self):
        mock_s3 = _make_runner_s3()
        mock_mod = _make_profile_module()
        with patch.object(_runner, "s3_client", mock_s3), \
             patch.object(_runner, "importlib") as mock_importlib:
            mock_importlib.import_module.return_value = mock_mod
            result = _runner.handler(_runner_event(), None)
        assert "result_s3_uri" in result
        assert result["row_count"] == 2
        assert result["actual_cost_usd"] >= 0
        assert "duration_seconds" in result

    def test_cost_formula(self):
        mock_s3 = _make_runner_s3()
        mock_mod = _make_profile_module()
        with patch.object(_runner, "s3_client", mock_s3), \
             patch.object(_runner, "importlib") as mock_importlib, \
             patch.object(_runner, "time") as mock_time:
            mock_importlib.import_module.return_value = mock_mod
            mock_time.monotonic.side_effect = [1000.0, 1005.0]  # 5-second run
            result = _runner.handler(_runner_event(), None)
        expected = 5.0 * 3.0 * 0.0000166667
        assert abs(result["actual_cost_usd"] - expected) < 1e-8

    def test_input_read_fails_raises_runtime_error(self):
        mock_s3 = _make_runner_s3()
        NoSuchKey = mock_s3.exceptions.NoSuchKey
        mock_s3.get_object.side_effect = NoSuchKey("not found")
        with patch.object(_runner, "s3_client", mock_s3), \
             patch.object(_runner, "importlib"):
            with pytest.raises(RuntimeError, match="Failed to read input dataset"):
                _runner.handler(_runner_event(), None)

    def test_profile_module_not_importable_raises(self):
        mock_s3 = _make_runner_s3()
        with patch.object(_runner, "s3_client", mock_s3), \
             patch.object(_runner, "importlib") as mock_importlib:
            mock_importlib.import_module.side_effect = ModuleNotFoundError("clustering")
            with pytest.raises(RuntimeError, match="Analysis failed"):
                _runner.handler(_runner_event(), None)

    def test_profile_function_raises_runtime_error(self):
        mock_s3 = _make_runner_s3()
        mock_mod = MagicMock()
        mock_mod.kmeans_handler.side_effect = ValueError("k must be <= n_samples")
        with patch.object(_runner, "s3_client", mock_s3), \
             patch.object(_runner, "importlib") as mock_importlib:
            mock_importlib.import_module.return_value = mock_mod
            with pytest.raises(RuntimeError, match="Analysis failed"):
                _runner.handler(_runner_event(), None)

    def test_empty_dataframe_result_row_count_zero(self):
        mock_s3 = _make_runner_s3()
        mock_mod = _make_profile_module(df_result=pd.DataFrame())
        with patch.object(_runner, "s3_client", mock_s3), \
             patch.object(_runner, "importlib") as mock_importlib:
            mock_importlib.import_module.return_value = mock_mod
            result = _runner.handler(_runner_event(), None)
        assert result["row_count"] == 0

    def test_missing_input_s3_uri_raises_value_error(self):
        event = _runner_event()
        event["extract"] = {}
        with patch.object(_runner, "s3_client", _make_runner_s3()), \
             patch.object(_runner, "importlib"):
            with pytest.raises(ValueError, match="input_s3_uri"):
                _runner.handler(event, None)

    def test_validation_error_when_feature_column_missing(self):
        """Runner returns validation_error before S3 access when a feature column is absent."""
        event = {
            "execution_id": "exec-col-validate",
            "profile": {
                "profile_id": "clustering-kmeans",
                "entrypoint": "clustering.kmeans_handler",
                "input_requirements": {"column_parameters": ["features"]},
            },
            "parameters": {"features": ["age", "gpa"], "k": 3},
            "extract": {
                "input_s3_uri": "s3://bucket/data.parquet",
                "columns": ["id", "age"],  # "gpa" is missing
            },
        }
        mock_s3 = _make_runner_s3()
        result = _runner.handler(event, None)
        assert result["status"] == "validation_error"
        assert "gpa" in result["missing_columns"]
        assert result["available_columns"] == ["id", "age"]
        mock_s3.get_object.assert_not_called()  # no S3 access before validation

    def test_validation_passes_when_all_columns_present(self):
        """Runner proceeds normally when all referenced columns exist."""
        event = {
            "execution_id": "exec-col-ok",
            "profile": {
                "profile_id": "clustering-kmeans",
                "entrypoint": "clustering.kmeans_handler",
                "input_requirements": {"column_parameters": ["features"]},
            },
            "parameters": {"features": ["age", "gpa"], "k": 3},
            "extract": {
                "input_s3_uri": "s3://bucket/data.parquet",
                "columns": ["id", "age", "gpa"],  # all present
            },
        }
        mock_s3 = _make_runner_s3(csv_data=b"id,age,gpa\n1,20,3.5\n2,21,3.8\n")
        mock_mod = _make_profile_module()
        write_result_return = ("s3://bucket/results/exec-col-ok/data.parquet", "s3://bucket/results/exec-col-ok/metadata.json")
        with patch.object(_runner, "s3_client", mock_s3), \
             patch.object(_runner, "importlib") as mock_importlib, \
             patch.object(_runner, "_write_result", return_value=write_result_return):
            mock_importlib.import_module.return_value = mock_mod
            result = _runner.handler(event, None)
        assert "result_s3_uri" in result
        assert result.get("status") != "validation_error"

    def test_validation_skipped_when_columns_not_in_extract(self):
        """Runner skips validation when extract step didn't provide column list."""
        event = _runner_event()
        # _runner_event() has no "columns" in extract — validation should be skipped
        mock_s3 = _make_runner_s3()
        mock_mod = _make_profile_module()
        write_result_return = ("s3://bucket/results/exec-test-runner/data.parquet", "s3://bucket/results/exec-test-runner/metadata.json")
        with patch.object(_runner, "s3_client", mock_s3), \
             patch.object(_runner, "importlib") as mock_importlib, \
             patch.object(_runner, "_write_result", return_value=write_result_return):
            mock_importlib.import_module.return_value = mock_mod
            result = _runner.handler(event, None)
        assert "result_s3_uri" in result  # proceeded normally


# ---------------------------------------------------------------------------
# Deliver helpers
# ---------------------------------------------------------------------------

def _deliver_event(execution_id="abc12345xyz99999"):
    return {
        "execution_id": execution_id,
        "user_arn": "arn:aws:iam::123456789012:user/testuser",
        "profile": {"display_name": "K-Means Clustering"},
        "compute": {
            "result_s3_uri": (
                f"s3://qs-compute-test-bucket/results/{execution_id[:8]}/data.parquet"
            )
        },
    }


def _mock_qs_deliver(creation_status="CREATION_SUCCESSFUL"):
    qs = MagicMock()
    qs.create_data_source.return_value = {
        "Arn": "arn:aws:quicksight:us-east-1:123456789012:datasource/test",
        "CreationStatus": "CREATION_IN_PROGRESS",
        "Status": 202,
    }
    qs.describe_data_source.return_value = {"DataSource": {"Status": creation_status}}
    qs.create_data_set.return_value = {
        "DataSetId": "ds1",
        "CreationStatus": "CREATION_SUCCESSFUL",
    }
    return qs


def _mock_s3_deliver():
    s3 = MagicMock()
    s3.head_object.return_value = {"ContentLength": 1024}
    s3.put_object.return_value = {}
    return s3


# ---------------------------------------------------------------------------
# TestDeliver
# ---------------------------------------------------------------------------

class TestDeliver:
    def test_happy_path_delivered(self):
        event = _deliver_event("abc12345xyz99999")
        with patch.object(_deliver, "quicksight", _mock_qs_deliver()), \
             patch.object(_deliver, "s3", _mock_s3_deliver()), \
             patch("time.sleep"):
            result = _deliver.handler(event, None)
        assert result["status"] == "delivered"
        assert result["dataset_id"] == "qs-compute-abc12345-dataset"

    def test_dataset_id_uses_first_8_chars(self):
        event = _deliver_event("deadbeef1234567890")
        with patch.object(_deliver, "quicksight", _mock_qs_deliver()), \
             patch.object(_deliver, "s3", _mock_s3_deliver()), \
             patch("time.sleep"):
            result = _deliver.handler(event, None)
        assert "deadbeef" in result["dataset_id"]

    def test_create_datasource_fails_returns_manifest_ready(self):
        mock_qs = MagicMock()
        mock_qs.create_data_source.side_effect = Exception("QuickSight error")
        with patch.object(_deliver, "quicksight", mock_qs), \
             patch.object(_deliver, "s3", _mock_s3_deliver()), \
             patch("time.sleep"):
            result = _deliver.handler(_deliver_event(), None)
        assert result["status"] == "manifest_ready"
        assert "manifest_uri" in result

    def test_data_source_creation_timeout_returns_manifest_ready(self):
        mock_qs = _mock_qs_deliver(creation_status="CREATION_IN_PROGRESS")
        with patch.object(_deliver, "quicksight", mock_qs), \
             patch.object(_deliver, "s3", _mock_s3_deliver()), \
             patch("time.sleep"):
            result = _deliver.handler(_deliver_event(), None)
        assert result["status"] == "manifest_ready"

    def test_data_source_creation_failed_returns_manifest_ready(self):
        mock_qs = _mock_qs_deliver(creation_status="CREATION_FAILED")
        with patch.object(_deliver, "quicksight", mock_qs), \
             patch.object(_deliver, "s3", _mock_s3_deliver()), \
             patch("time.sleep"):
            result = _deliver.handler(_deliver_event(), None)
        assert result["status"] == "manifest_ready"

    def test_create_dataset_fails_returns_manifest_ready(self):
        mock_qs = _mock_qs_deliver()
        mock_qs.create_data_set.side_effect = Exception("dataset creation failed")
        with patch.object(_deliver, "quicksight", mock_qs), \
             patch.object(_deliver, "s3", _mock_s3_deliver()), \
             patch("time.sleep"):
            result = _deliver.handler(_deliver_event(), None)
        assert result["status"] == "manifest_ready"

    def test_missing_result_s3_uri_raises(self):
        event = _deliver_event()
        event["compute"] = {}
        with patch.object(_deliver, "quicksight", _mock_qs_deliver()), \
             patch.object(_deliver, "s3", _mock_s3_deliver()):
            with pytest.raises(RuntimeError, match="result_s3_uri"):
                _deliver.handler(event, None)

    def test_s3_head_object_not_found_raises(self):
        mock_s3 = _mock_s3_deliver()
        mock_s3.head_object.side_effect = Exception("404 Not Found")
        with patch.object(_deliver, "quicksight", _mock_qs_deliver()), \
             patch.object(_deliver, "s3", mock_s3):
            with pytest.raises(RuntimeError, match="not found"):
                _deliver.handler(_deliver_event(), None)

    def test_no_user_arn_falls_back_to_root_principal(self):
        event = _deliver_event()
        del event["user_arn"]
        mock_qs = _mock_qs_deliver()
        with patch.object(_deliver, "quicksight", mock_qs), \
             patch.object(_deliver, "s3", _mock_s3_deliver()), \
             patch("time.sleep"):
            result = _deliver.handler(event, None)
        assert result["status"] in ("delivered", "manifest_ready")
        call_kwargs = mock_qs.create_data_source.call_args[1]
        principal = call_kwargs["Permissions"][0]["Principal"]
        assert "arn:aws:iam" in principal


# ---------------------------------------------------------------------------
# TestHandleFailure
# ---------------------------------------------------------------------------

class TestHandleFailure:
    def test_json_cause_extracts_error_message(self):
        cause = json.dumps(
            {"errorMessage": "NaN in column 'value'", "errorType": "ValueError"}
        )
        result = _failure.handler(
            {
                "execution_id": "exec-1",
                "error_info": {"Error": "States.TaskFailed", "Cause": cause},
            },
            None,
        )
        assert result["error_message"] == "NaN in column 'value'"
        assert result["error_code"] == "States.TaskFailed"

    def test_plain_text_cause_used_directly(self):
        cause = "Task timed out after 900 seconds"
        result = _failure.handler(
            {
                "execution_id": "exec-2",
                "error_info": {"Error": "States.Timeout", "Cause": cause},
            },
            None,
        )
        assert result["error_message"] == cause

    def test_malformed_json_cause_falls_back_to_raw(self):
        cause = '{"broken": '
        result = _failure.handler(
            {
                "execution_id": "exec-3",
                "error_info": {"Error": "RuntimeError", "Cause": cause},
            },
            None,
        )
        assert result["error_message"] == cause

    def test_missing_error_field_defaults_to_unknown_error(self):
        result = _failure.handler(
            {"execution_id": "exec-4", "error_info": {"Cause": "something broke"}},
            None,
        )
        assert result["error_code"] == "UnknownError"

    def test_empty_cause_uses_fallback_message(self):
        result = _failure.handler(
            {
                "execution_id": "exec-5",
                "error_info": {"Error": "BudgetExceeded", "Cause": ""},
            },
            None,
        )
        assert "BudgetExceeded" in result["error_message"]

    def test_all_fields_absent_no_raise(self):
        result = _failure.handler({}, None)
        assert "error_code" in result
        assert "execution_id" in result
        assert result["execution_id"] == "unknown"
        assert result["error_code"] == "UnknownError"


# ===========================================================================
# TestExtractClawsUri (CP-16)
# ===========================================================================

BASE_EVENT = {"execution_id": "exec-claws-test", "dataset_id": ""}
RESOLVER_ARN = "arn:aws:lambda:us-east-1:123456789012:function:claws-resolver"


class TestExtractClawsUri:
    def _make_resolver_response(self, dataset_id=None, error=None):
        payload = {"dataset_id": dataset_id} if dataset_id else {"error": error or "not found"}
        return {"Payload": io.BytesIO(json.dumps(payload).encode())}

    def test_happy_path_resolves_and_extracts(self):
        mock_lambda = MagicMock()
        mock_lambda.invoke.return_value = self._make_resolver_response(dataset_id="qs-ds-abc123")
        mock_qs, mock_s3 = _make_extract_mocks()
        with patch.object(_extract, "lambda_client", mock_lambda), \
             patch.object(_extract, "qs", mock_qs), \
             patch.object(_extract, "s3", mock_s3), \
             patch.object(_extract, "CLAWS_RESOLVER_ARN", RESOLVER_ARN):
            result = _extract.handler({**BASE_EVENT, "source_uri": "claws://roda-noaa-ghcn"}, None)
        assert "error" not in result
        assert "input_s3_uri" in result
        mock_lambda.invoke.assert_called_once_with(
            FunctionName=RESOLVER_ARN,
            InvocationType="RequestResponse",
            Payload=json.dumps({"source_id": "roda-noaa-ghcn"}).encode(),
        )

    def test_resolver_returns_error(self):
        mock_lambda = MagicMock()
        mock_lambda.invoke.return_value = self._make_resolver_response(error="source not found")
        with patch.object(_extract, "lambda_client", mock_lambda), \
             patch.object(_extract, "CLAWS_RESOLVER_ARN", RESOLVER_ARN):
            result = _extract.handler({**BASE_EVENT, "source_uri": "claws://roda-missing"}, None)
        assert result["status"] == "error"
        assert "source not found" in result["error"]

    def test_resolver_arn_not_configured(self):
        with patch.object(_extract, "CLAWS_RESOLVER_ARN", ""):
            result = _extract.handler({**BASE_EVENT, "source_uri": "claws://roda-noaa-ghcn"}, None)
        assert result["status"] == "error"
        assert "CLAWS_RESOLVER_ARN" in result["error"]

    def test_resolver_invocation_exception(self):
        mock_lambda = MagicMock()
        mock_lambda.invoke.side_effect = Exception("network timeout")
        with patch.object(_extract, "lambda_client", mock_lambda), \
             patch.object(_extract, "CLAWS_RESOLVER_ARN", RESOLVER_ARN):
            result = _extract.handler({**BASE_EVENT, "source_uri": "claws://roda-noaa-ghcn"}, None)
        assert result["status"] == "error"
        assert "network timeout" in result["error"]

    def test_missing_source_id_returns_error(self):
        with patch.object(_extract, "CLAWS_RESOLVER_ARN", RESOLVER_ARN):
            result = _extract.handler({**BASE_EVENT, "source_uri": "claws://"}, None)
        assert result["status"] == "error"
        assert "source_id" in result["error"]
