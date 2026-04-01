"""
deliver — Step Functions Lambda

Registers compute results from S3 as a Quick Sight data source and dataset.
Reuses the same quicksight.create_data_source() + create_data_set() pattern
as open-data's dataset-loader.

Input (from Step Functions — full context):
  {
    "execution_id": str,
    "dataset_name": str,
    "user_arn": str,
    "profile": {"display_name": str, ...},
    "compute": {"result_s3_uri": str, ...},
    ...
  }

Output:
  {
    "status": "delivered",
    "dataset_id": str,
    "data_source_id": str,
    "result_dataset_name": str,
    "result_s3_uri": str,
    "quicksight_result": {...}
  }
"""

import json
import logging
import os
import time

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

quicksight = boto3.client("quicksight")
s3 = boto3.client("s3")


def _write_manifest(bucket: str, execution_id: str, result_s3_uri: str) -> str:
    """Write a QuickSight S3 manifest and return its URI."""
    manifest = {
        "fileLocations": [
            {"URIs": [result_s3_uri]},
        ],
        "globalUploadSettings": {
            "format": "PARQUET",
        },
    }
    manifest_key = f"manifests/{execution_id}/manifest.json"
    s3.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=json.dumps(manifest).encode(),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{manifest_key}"


def handler(event: dict, context) -> dict:
    logger.info(json.dumps({"event": {
        "execution_id": event.get("execution_id"),
        "profile_id": event.get("profile", {}).get("profile_id"),
    }}))

    execution_id = event["execution_id"]
    dataset_name = event.get("dataset_name") or f"Compute Result — {execution_id[:8]}"
    account_id = os.environ["QUICKSIGHT_ACCOUNT_ID"]
    _qs_region = os.environ.get("QUICKSIGHT_REGION", os.environ.get("AWS_REGION", "us-east-1"))
    compute_bucket = os.environ["COMPUTE_BUCKET"]

    # Get result location from compute step
    compute = event.get("compute") or {}
    result_s3_uri = compute.get("result_s3_uri")
    if not result_s3_uri:
        raise RuntimeError(
            "compute step did not return result_s3_uri — runner may have failed"
        )

    parts = result_s3_uri.replace("s3://", "").split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise RuntimeError(
            f"Invalid result_s3_uri '{result_s3_uri}': expected s3://bucket/key"
        )
    result_bucket, result_key = parts
    try:
        s3.head_object(Bucket=result_bucket, Key=result_key)
    except Exception as e:
        raise RuntimeError(f"Result file not found at {result_s3_uri}: {e}")

    # Write QuickSight manifest
    manifest_uri = _write_manifest(compute_bucket, execution_id, result_s3_uri)

    # Create QuickSight data source
    data_source_id = f"qs-compute-{execution_id[:8]}-source"
    dataset_id = f"qs-compute-{execution_id[:8]}-dataset"

    qs_result = {}
    try:
        ds_response = quicksight.create_data_source(
            AwsAccountId=account_id,
            DataSourceId=data_source_id,
            Name=f"{dataset_name} (source)",
            Type="S3",
            DataSourceParameters={
                "S3Parameters": {
                    "ManifestFileLocation": {
                        "Bucket": compute_bucket,
                        "Key": f"manifests/{execution_id}/manifest.json",
                    }
                }
            },
            Permissions=[
                {
                    "Principal": event.get("user_arn", f"arn:aws:iam::{account_id}:root"),
                    "Actions": [
                        "quicksight:DescribeDataSource",
                        "quicksight:DescribeDataSourcePermissions",
                        "quicksight:PassDataSource",
                    ],
                }
            ],
        )
        qs_result["data_source_status"] = ds_response.get("CreationStatus", "UNKNOWN")
        qs_result["data_source_id"] = data_source_id
        qs_result["data_source_created"] = True

        logger.info(json.dumps({"created_datasource": data_source_id}))

        # Wait for DataSource to become ready (S3 sources are typically < 2 seconds)
        for _ in range(10):
            status_resp = quicksight.describe_data_source(
                AwsAccountId=account_id, DataSourceId=data_source_id
            )
            ds_status = status_resp["DataSource"]["Status"]
            if ds_status == "CREATION_SUCCESSFUL":
                break
            if ds_status == "CREATION_FAILED":
                raise RuntimeError(f"DataSource creation failed: {ds_status}")
            time.sleep(1)
        else:
            raise RuntimeError("DataSource creation timed out after 10 seconds")

        # Create the QuickSight DataSet on top of the DataSource.
        # Use DIRECT_QUERY mode to avoid SPICE ingestion cost for ephemeral results.
        ds_resp = quicksight.create_data_set(
            AwsAccountId=account_id,
            DataSetId=dataset_id,
            Name=dataset_name,
            PhysicalTableMap={
                "result": {
                    "S3Source": {
                        "DataSourceArn": ds_response["Arn"],
                        "UploadSettings": {
                            "Format": "PARQUET",
                        },
                        "InputColumns": [],  # QuickSight auto-infers columns from Parquet
                    }
                }
            },
            ImportMode="DIRECT_QUERY",
            Permissions=[
                {
                    "Principal": event.get("user_arn", f"arn:aws:iam::{account_id}:root"),
                    "Actions": [
                        "quicksight:DescribeDataSet",
                        "quicksight:DescribeDataSetPermissions",
                        "quicksight:PassDataSet",
                        "quicksight:UpdateDataSet",
                        "quicksight:DeleteDataSet",
                    ],
                }
            ],
        )
        qs_result["dataset_status"] = ds_resp.get("CreationStatus", "UNKNOWN")
        qs_result["status"] = "created"

        logger.info(json.dumps({"created_dataset": dataset_id}))

    except Exception as exc:
        logger.error(f"QuickSight delivery failed: {exc}")
        qs_result["error"] = str(exc)
        qs_result["status"] = "failed"
        # Return manifest URI so the user can register manually
        return {
            "status": "manifest_ready",
            "execution_id": execution_id,
            "manifest_uri": manifest_uri,
            "result_s3_uri": result_s3_uri,
            "result_dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "quicksight_result": qs_result,
        }

    return {
        "status": "delivered",
        "dataset_id": dataset_id,
        "data_source_id": data_source_id,
        "result_dataset_name": dataset_name,
        "result_s3_uri": result_s3_uri,
        "manifest_uri": manifest_uri,
        "quicksight_result": qs_result,
    }
