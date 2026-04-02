"""
extract — Step Functions Lambda

Fetches data from a Quick Sight dataset and writes it to S3 as Parquet
for the runner Lambda to consume.

Reads the Quick Sight dataset's physical source (S3Source → manifest),
downloads the files listed in the manifest, and writes a combined Parquet
to:
  s3://{COMPUTE_BUCKET}/inputs/{execution_id}/data.parquet

If the dataset source is not S3-backed (e.g., SPICE without accessible
manifest), returns an unsupported_source error so the failure is surfaced
clearly rather than producing silent empty results.

Input (Step Functions task input):
  {
    "execution_id": str,
    "profile": {...},
    "dataset_id": str,
    "parameters": {...},
    ...
  }

Output (merged into state via result_path):
  {
    "execution_id": str,
    "input_s3_uri": str,
    "row_count": int,
    "columns": list[str]
  }
"""

import csv
import io
import json
import logging
import os
import urllib.parse
import urllib.request

import boto3
import botocore

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
qs = boto3.client("quicksight")
lambda_client = boto3.client("lambda")

CLAWS_RESOLVER_ARN = os.environ.get("CLAWS_RESOLVER_ARN", "")


def handler(event: dict, context) -> dict:
    execution_id = event["execution_id"]
    dataset_id = event.get("dataset_id", "")
    source_uri = event.get("source_uri", "")
    compute_bucket = os.environ["COMPUTE_BUCKET"]
    account_id = os.environ["QUICKSIGHT_ACCOUNT_ID"]

    logger.info(json.dumps({
        "step": "extract",
        "execution_id": execution_id,
        "dataset_id": dataset_id,
        "source_uri": source_uri,
    }))

    # ------------------------------------------------------------------
    # clAWS URI — resolve via claws-resolver Lambda, then extract via QS
    # ------------------------------------------------------------------
    if source_uri.startswith("claws://"):
        return _extract_from_claws_uri(event, execution_id, compute_bucket, account_id)

    # ------------------------------------------------------------------
    # Direct S3 URI — bypass Quick Sight dataset lookup
    # ------------------------------------------------------------------
    if source_uri.startswith("s3://"):
        return _extract_from_s3_uri(source_uri, execution_id, compute_bucket)

    # ------------------------------------------------------------------
    # 1. Describe the Quick Sight dataset to find the S3 source
    # ------------------------------------------------------------------
    try:
        resp = qs.describe_data_set(
            AwsAccountId=account_id,
            DataSetId=dataset_id,
        )
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        raise RuntimeError(f"quicksight:DescribeDataSet failed: {error_code}: {e}") from e

    physical_table_map = resp["DataSet"].get("PhysicalTableMap", {})

    # ------------------------------------------------------------------
    # 2. Find S3Source entries in PhysicalTableMap
    # ------------------------------------------------------------------
    s3_sources = []
    for _table_id, table_def in physical_table_map.items():
        if "S3Source" in table_def:
            s3_src = table_def["S3Source"]
            _manifest_loc = s3_src.get("InputColumns", None)
            upload_settings = s3_src.get("UploadSettings", {})
            manifest_file_location = s3_src.get("DataSourceArn")  # indirect via data source
            s3_sources.append({
                "data_source_arn": s3_src.get("DataSourceArn", ""),
                "upload_settings": upload_settings,
                "input_columns": s3_src.get("InputColumns", []),
            })

    if not s3_sources:
        raise ValueError(
            f"Unsupported source: Dataset {dataset_id} does not have an S3 source in PhysicalTableMap. "
            "Only S3-backed Quick Sight datasets are supported. "
            "SPICE-only datasets or RDS/Redshift sources are not currently supported."
        )

    # ------------------------------------------------------------------
    # 3. Resolve the manifest via the DataSource ARN
    #    DataSource ARN → describe_data_source → S3Parameters → ManifestFileLocation
    # ------------------------------------------------------------------
    data_source_arn = s3_sources[0]["data_source_arn"]
    # ARN format: arn:aws:quicksight:{region}:{account}:datasource/{id}
    data_source_id = data_source_arn.split("/")[-1]

    try:
        ds_resp = qs.describe_data_source(
            AwsAccountId=account_id,
            DataSourceId=data_source_id,
        )
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        raise RuntimeError(f"quicksight:DescribeDataSource failed: {error_code}: {e}") from e

    data_source_parameters = ds_resp["DataSource"].get("DataSourceParameters", {})
    s3_params = data_source_parameters.get("S3Parameters", {})
    manifest_file_location = s3_params.get("ManifestFileLocation", {})
    manifest_bucket = manifest_file_location.get("Bucket", "")
    manifest_key = manifest_file_location.get("Key", "")

    if not manifest_bucket or not manifest_key:
        raise ValueError(
            f"Unsupported source: DataSource {data_source_id} does not have "
            "S3Parameters.ManifestFileLocation. Cannot locate source files."
        )

    # ------------------------------------------------------------------
    # 4. Read the manifest JSON to get the actual S3 URIs
    # ------------------------------------------------------------------
    try:
        manifest_obj = s3.get_object(Bucket=manifest_bucket, Key=manifest_key)
        manifest = json.loads(manifest_obj["Body"].read())
    except botocore.exceptions.ClientError as e:
        raise RuntimeError(
            f"Could not read manifest s3://{manifest_bucket}/{manifest_key}: {e}"
        ) from e

    # Quick Sight manifest format:
    # {"fileLocations": [{"URIs": ["s3://bucket/key", ...]}, ...]}
    file_uris = []
    for loc in manifest.get("fileLocations", []):
        file_uris.extend(loc.get("URIs", []))
        file_uris.extend(loc.get("URIPrefixes", []))  # not standard but defensive

    # global_upload_settings carries format and delimiter
    global_settings = manifest.get("globalUploadSettings", {})
    file_format = global_settings.get("format", "CSV").upper()
    delimiter = global_settings.get("delimiter", ",")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        return {
            "status": "unsupported_source",
            "message": f"Invalid delimiter {delimiter!r}: must be a single character",
            "execution_id": execution_id,
        }
    contains_header = global_settings.get("containsHeader", "TRUE").upper() == "TRUE"

    if not file_uris:
        raise ValueError("Unsupported source: Manifest contains no file URIs.")

    # ------------------------------------------------------------------
    # 5. Read source files and write combined Parquet
    #    Use only stdlib + boto3 (no pandas — extract Lambda has no Layer)
    #    Write Parquet manually via lightweight approach.
    #    For simplicity, convert to CSV-in-memory first, then write as CSV
    #    wrapped in a minimal Parquet file using pyarrow if available.
    # ------------------------------------------------------------------
    all_rows = []
    header = None

    for uri in file_uris:
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "s3":
            logger.warning(json.dumps({"skip_uri": uri, "reason": "not s3 scheme"}))
            continue
        src_bucket = parsed.netloc
        src_key = parsed.path.lstrip("/")

        try:
            obj = s3.get_object(Bucket=src_bucket, Key=src_key)
            body = obj["Body"].read()
        except botocore.exceptions.ClientError as e:
            logger.warning(json.dumps({"skip_uri": uri, "error": str(e)}))
            continue

        if file_format in ("CSV", "TSV"):
            text = body.decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)
            if not rows:
                continue
            if contains_header:
                file_header = rows[0]
                if header is None:
                    header = file_header
                elif file_header != header:
                    logger.warning(json.dumps({
                        "schema_mismatch": uri,
                        "expected": header,
                        "got": file_header,
                    }))
                    continue
                data_rows = rows[1:]
            else:
                if header is None:
                    header = [f"col_{i}" for i in range(len(rows[0]))]
                data_rows = rows
            all_rows.extend(data_rows)
        elif file_format == "JSON":
            # Newline-delimited JSON
            for line in body.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(json.dumps({"skip_line": uri, "error": str(exc)}))
                    continue
                if header is None:
                    header = list(record.keys())
                all_rows.append([str(record.get(col, "")) for col in header])
        else:
            return {
                "status": "unsupported_source",
                "message": (
                    f"Unsupported manifest file format: {file_format}. "
                    "Supported formats: CSV, TSV, JSON."
                ),
                "execution_id": execution_id,
            }

    if header is None or not all_rows:
        return {
            "status": "unsupported_source",
            "message": "Source files contained no data.",
            "execution_id": execution_id,
        }

    # ------------------------------------------------------------------
    # 6. Write as Parquet using a minimal hand-built Parquet file
    #    The extract Lambda does NOT have the analytics Layer (no pyarrow).
    #    Write as CSV and use .csv extension; runner _read_input() reads CSV.
    # ------------------------------------------------------------------
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(all_rows)
    csv_bytes = output.getvalue().encode("utf-8")

    # Store as .csv in the inputs prefix — runner handles both .parquet and .csv
    input_key = f"inputs/{execution_id}/data.csv"
    s3.put_object(
        Bucket=compute_bucket,
        Key=input_key,
        Body=csv_bytes,
        ContentType="text/csv",
    )

    row_count = len(all_rows)
    input_s3_uri = f"s3://{compute_bucket}/{input_key}"

    logger.info(json.dumps({
        "step": "extract_complete",
        "execution_id": execution_id,
        "input_s3_uri": input_s3_uri,
        "row_count": row_count,
        "columns": header,
    }))

    return {
        "execution_id": execution_id,
        "input_s3_uri": input_s3_uri,
        "row_count": row_count,
        "columns": header,
    }


def _extract_from_claws_uri(event: dict, execution_id: str, compute_bucket: str, account_id: str) -> dict:
    """
    Resolve a claws:// URI to a Quick Sight dataset_id via the claws-resolver
    Lambda, then extract using the standard Quick Sight path.
    """
    source_uri = event.get("source_uri", "")
    source_id = source_uri[len("claws://"):]
    if not source_id:
        return {"status": "error", "error": "claws:// URI missing source_id", "execution_id": execution_id}

    if not CLAWS_RESOLVER_ARN:
        return {"status": "error", "error": "CLAWS_RESOLVER_ARN not configured", "execution_id": execution_id}

    try:
        resp = lambda_client.invoke(
            FunctionName=CLAWS_RESOLVER_ARN,
            InvocationType="RequestResponse",
            Payload=json.dumps({"source_id": source_id}).encode(),
        )
        result = json.loads(resp["Payload"].read())
    except Exception as exc:
        return {
            "status": "error",
            "error": f"clAWS resolver invocation failed: {exc}",
            "execution_id": execution_id,
        }

    if "error" in result:
        return {
            "status": "error",
            "error": f"clAWS resolver: {result['error']}",
            "execution_id": execution_id,
        }

    dataset_id = result["dataset_id"]
    modified_event = dict(event, dataset_id=dataset_id)
    modified_event.pop("source_uri", None)
    # Re-invoke handler logic with resolved dataset_id (tail-call via direct call)
    return handler(modified_event, None)


def _extract_from_s3_uri(source_uri: str, execution_id: str, compute_bucket: str) -> dict:
    """
    Download a file directly from an s3:// URI and store it in the compute
    bucket inputs prefix as CSV (for runner consumption).

    Supports CSV, TSV, JSON (newline-delimited), and Parquet.
    Parquet support requires pyarrow (not guaranteed in extract Lambda);
    falls back to raw copy with .parquet extension if unavailable.
    """
    parsed = urllib.parse.urlparse(source_uri)
    src_bucket = parsed.netloc
    src_key = parsed.path.lstrip("/")

    if not src_bucket or not src_key:
        return {
            "status": "unsupported_source",
            "error": f"Could not parse bucket/key from source_uri: {source_uri}",
            "execution_id": execution_id,
        }

    try:
        obj = s3.get_object(Bucket=src_bucket, Key=src_key)
        body = obj["Body"].read()
    except botocore.exceptions.ClientError as e:
        return {
            "status": "unsupported_source",
            "error": f"Failed to read {source_uri}: {e}",
            "execution_id": execution_id,
        }

    ext = src_key.lower().split(".")[-1] if "." in src_key else ""

    # Attempt Parquet → CSV conversion
    if ext in ("parquet", "parq"):
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415
            table = pq.read_table(io.BytesIO(body))
            header = table.schema.names
            all_rows = [
                [str(col[i].as_py()) for col in table.columns]
                for i in range(table.num_rows)
            ]
        except ImportError:
            # pyarrow not available — store raw Parquet, runner reads it directly
            input_key = f"inputs/{execution_id}/data.parquet"
            s3.put_object(Bucket=compute_bucket, Key=input_key,
                          Body=body, ContentType="application/octet-stream")
            input_s3_uri = f"s3://{compute_bucket}/{input_key}"
            logger.info(json.dumps({
                "step": "extract_complete",
                "execution_id": execution_id,
                "input_s3_uri": input_s3_uri,
                "source": "s3_direct",
                "format": "parquet",
            }))
            return {
                "execution_id": execution_id,
                "input_s3_uri": input_s3_uri,
                "row_count": -1,
                "columns": [],
            }
        except Exception as e:
            return {
                "status": "unsupported_source",
                "error": f"Failed to parse Parquet: {e}",
                "execution_id": execution_id,
            }
    elif ext in ("json", "jsonl", "ndjson"):
        header = None
        all_rows = []
        for line in body.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if header is None:
                header = list(record.keys())
            all_rows.append([str(record.get(col, "")) for col in header])
        if header is None:
            return {
                "status": "unsupported_source",
                "error": "JSON source file contained no valid records",
                "execution_id": execution_id,
            }
    else:
        # CSV / TSV
        delimiter = "\t" if ext in ("tsv", "tab") else ","
        text = body.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return {
                "status": "unsupported_source",
                "error": "Source file is empty",
                "execution_id": execution_id,
            }
        header = rows[0]
        all_rows = rows[1:]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(all_rows)
    csv_bytes = output.getvalue().encode("utf-8")

    input_key = f"inputs/{execution_id}/data.csv"
    s3.put_object(Bucket=compute_bucket, Key=input_key,
                  Body=csv_bytes, ContentType="text/csv")

    input_s3_uri = f"s3://{compute_bucket}/{input_key}"
    row_count = len(all_rows)

    logger.info(json.dumps({
        "step": "extract_complete",
        "execution_id": execution_id,
        "input_s3_uri": input_s3_uri,
        "row_count": row_count,
        "columns": header,
        "source": "s3_direct",
    }))

    return {
        "execution_id": execution_id,
        "input_s3_uri": input_s3_uri,
        "row_count": row_count,
        "columns": header,
    }
