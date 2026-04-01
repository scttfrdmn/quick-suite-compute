"""
transform.py — EMR Serverless Spark job for transform-spark profile

Large dataset join and transform using Apache Spark. Runs on EMR Serverless
when enable_emr=true is set in the CDK deployment context.

Arguments (passed via --arguments in EMR Serverless StartJobRun):
  --execution-id   str    Step Functions execution ID
  --input-s3       str    s3:// URI of primary input dataset (Parquet)
  --output-s3      str    s3:// URI prefix for output (Parquet)
  --additional-s3  str    JSON array of additional S3 URIs to join
  --join-keys      str    JSON array of column names to join on
  --join-type      str    inner | left | right | full (default: left)
  --select-cols    str    JSON array of columns to keep (empty = all)
  --filter-expr    str    SQL WHERE expression to filter rows (optional)

Output:
  Writes result Parquet to --output-s3/data.parquet and metadata to
  --output-s3/metadata.json.
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Quick Suite Compute Spark Transform")
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--input-s3", required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--additional-s3", default="[]")
    parser.add_argument("--join-keys", required=True)
    parser.add_argument("--join-type", default="left")
    parser.add_argument("--select-cols", default="[]")
    parser.add_argument("--filter-expr", default="")
    args = parser.parse_args()

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    import boto3
    import time

    execution_id = args.execution_id
    start_time = time.time()

    spark = (
        SparkSession.builder
        .appName(f"qs-compute-transform-{execution_id}")
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    additional_s3_paths = json.loads(args.additional_s3)
    join_keys = json.loads(args.join_keys)
    select_cols = json.loads(args.select_cols)
    filter_expr = args.filter_expr.strip()

    if not join_keys:
        raise ValueError("--join-keys must be a non-empty JSON array")

    # Read primary dataset
    primary_df = spark.read.parquet(args.input_s3)

    # Join additional datasets
    result_df = primary_df
    for i, s3_path in enumerate(additional_s3_paths):
        right_df = spark.read.parquet(s3_path)
        # Prefix right columns to avoid collision (except join keys)
        right_renamed = right_df
        for col in right_df.columns:
            if col not in join_keys:
                right_renamed = right_renamed.withColumnRenamed(col, f"src{i + 1}_{col}")
        result_df = result_df.join(right_renamed, on=join_keys, how=args.join_type)

    # Apply filter
    if filter_expr:
        result_df = result_df.filter(filter_expr)

    # Select columns
    if select_cols:
        result_df = result_df.select(select_cols)

    row_count = result_df.count()
    col_count = len(result_df.columns)
    elapsed = time.time() - start_time

    # Write output
    output_path = args.output_s3.rstrip("/")
    result_df.write.mode("overwrite").parquet(f"{output_path}/data.parquet")

    # Write metadata
    metadata = {
        "execution_id": execution_id,
        "profile_id": "transform-spark",
        "row_count": row_count,
        "column_count": col_count,
        "elapsed_seconds": elapsed,
        "input_s3": args.input_s3,
        "additional_sources": len(additional_s3_paths),
        "join_type": args.join_type,
    }

    s3_client = boto3.client("s3")
    bucket = output_path.split("/")[2]
    key = "/".join(output_path.split("/")[3:]) + "/metadata.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(metadata).encode(),
        ContentType="application/json",
    )

    print(json.dumps({"status": "success", **metadata}))
    spark.stop()


if __name__ == "__main__":
    main()
