# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-04-02

### Added
- **Issue 19 — Named result snapshots:** optional `result_label` input in `compute_run`; `record-spend` SFN step writes to new `qs-compute-snapshots` DynamoDB table (PK: `user_arn`, SK: `label`) when label is set; unlabeled runs are not written; new `compute_snapshots` AgentCore tool Lambda lists a user's snapshots sorted by `completed_at` desc; CDK: new table, new Lambda, registered as AgentCore target
- **Issue 20 — Compare two named snapshots:** new `compute_compare` AgentCore tool Lambda; inputs: `label_a`, `label_b`, `user_arn`; loads both result S3 paths, diffs row sets; returns `added_count`, `removed_count`, `unchanged_count`, `schema_diff`, `cost_delta_usd`, `duration_delta_seconds`; returns counts not full row data; handles schema mismatch gracefully
- **Issue 21 — Profile composition:** optional `chain_profile_id` in `compute_run`; validated at submission time; forwarded as `chain_profile` in SFN execution input for downstream chaining; `compute_status` surfaces `step: "profile_2"` and `total_cost_usd` from execution output; chain estimated cost = sum of both profile estimates
- **Issue 22 — Pre-submission cost estimate:** `compute_run` returns `estimated_cost_usd` and `estimated_duration_seconds` in the 200 response before SFN starts; estimate uses profile `cost_estimate` base + S3 `head_object` dataset size scaling (0.1x–10x relative to 10 MB baseline, clamped); fails open to profile-level estimate on S3 error; SFN starts regardless
- 37 new unit tests covering all four v0.6.0 features

## [0.5.0] - 2026-04-02

### Added
- Per-user concurrent job limit: `compute_run` rejects submissions when a user has ≥ `MAX_CONCURRENT_JOBS_PER_USER` (default 2) running executions; fails open on Step Functions errors; `MAX_CONCURRENT_JOBS_PER_USER` Lambda environment variable, configurable via CDK context
- `duration_seconds` field in runner Lambda output: elapsed wall-clock time passed through to `record-spend` and surfaced in `compute_status` response
- Per-user spend CloudWatch dashboard: job cost (USD/24h sum) and duration (p99) widgets per profile

### Fixed
- `runner/handler.py`: column parameter values validated against column names present in the extracted dataset; mismatched column names return a descriptive error before compute starts

## [0.4.3] - 2026-04-02

### Fixed
- `lambdas/extract/handler.py`: remove unused `import pyarrow as pa` (F401); only `pyarrow.parquet` is used
- CI `test` job: add `setup-node@v4` and `npm install -g aws-cdk` so `cdk synth` succeeds
- Lint: fix I001 import-sort order in `stacks/compute_stack.py` and test files; remove unused `os` and `pytest` imports

## [0.4.2] - 2026-04-01

### Fixed
- `lambdas/layer/Dockerfile`: changed pip install `--target` from `/asset-output/python` to `/asset/python` — CDK's `Code.from_docker_build()` copies the layer from `/asset` inside the container by default; the wrong path caused `docker cp` to fail with exit code 1 during stack synthesis and CDK stack tests
- `tests/test_stack.py`: updated DynamoDB table count assertion from 1 → 2 to reflect `SpendTable` + `HistoryTable` (added v0.4.0); renamed test to `test_dynamodb_tables_created`
- Integration tests (`test_integration_handlers.py`): replaced `sys.path` + `import handler as X` module-loading pattern with `importlib.util.spec_from_file_location` aliases (`_integ_check_budget`, `_integ_compute_run`, `_integ_compute_status`) to prevent handler collision when all three Lambda directories are on sys.path simultaneously
- Integration tests: fixed `TestComputeStatusHandler` status assertion to match handler's raw uppercase AWS status values (`"RUNNING"`, `"SUCCEEDED"`, `"FAILED"`) rather than lowercase strings

## [0.4.1] - 2026-04-01

### Added
- Unit tests for Step Functions handler chain: `extract` (9 tests), `runner` (7 tests), `deliver` (9 tests), `handle-failure` (6 tests); covers happy paths, error propagation, cost formula, polling timeout, and manifest-ready fallback
- Unit tests for `compute_history` (8 tests): limit clamping, missing/whitespace ARN, DynamoDB error handling, Decimal serialization
- Unit tests for `compute_cancel` (7 tests): ARN reconstruction, execution-not-found, already-complete, and generic exception paths
- `pandas>=2.0` and `pyarrow>=14` added to dev dependency group (required by `runner/handler.py` module-level imports)

### Fixed
- `compute_run`: validation order corrected — `profile_id` / `profiles` list is now checked before `user_arn`, so missing-profile-id and missing-dataset-id errors return before the ARN format check (resolves two failing unit tests from v0.4.0)

## [0.4.0] - 2026-04-01

### Added
- Budget 80% threshold alert in `check-budget` Lambda: publishes SNS notification when a job would push the user past 80% of their monthly budget for the first time; `threshold_alert_sent` field added to response
- CloudWatch spend metrics emitted from `record-spend` Lambda: `JobCost` (per ProfileId × UserArn) and `JobDuration` (per ProfileId) in `QuickSuiteCompute` namespace
- CloudWatch dashboard (`qs-compute-usage`): Job Cost (24h sum), Job Duration (p99), and State Machine execution widgets; `DashboardUrl` CloudFormation output added
- Job history DynamoDB table (`qs-compute-history`, 90-day TTL); `record-spend` Lambda writes a history item on every successful job completion
- `compute_history` AgentCore tool Lambda: returns recent jobs for a user (most recent first, configurable limit up to 20)
- `compute_cancel` AgentCore tool Lambda: stops a running Step Functions execution via `sfn.stop_execution()`; returns `{status: cancelled, job_id}` or descriptive error

## [0.3.0] - 2026-04-01

### Added
- `source_uri` parameter in `compute_run`: accepts `s3://bucket/key` for direct S3 input (bypasses Quick Sight dataset lookup) or `claws://` (returns clear "not yet implemented" error for v0.4.0)
- EMR Serverless integration in CDK stack (enabled via `--context enable_emr=true`): `CfnApplication` (Spark, emr-7.1.0), EMR job execution IAM role, `BucketDeployment` uploads `spark/transform.py`, `EmrServerlessStartJobRun` Step Functions task with dynamic argument passing; stub Fail state remains when disabled
- Parallel job execution in `compute_run`: optional `profiles` list (up to 10) starts one independent Step Functions execution per profile; returns `{status, count, jobs: [{job_id, execution_arn, profile_id}]}`; single-profile path unchanged
- Integration test suite (`tests/test_integration_handlers.py`) covering `check-budget`, `compute-run`, and `compute-status` handler chains using Substrate for DynamoDB and Step Functions

### Changed
- CI workflow CDK synth step uses `--no-asset-bundling` to skip Docker analytics layer build in CI

## [0.2.0] - 2026-04-01

### Added
- Post-deploy helper script (`scripts/post-deploy.sh`) — retrieves tool Lambda ARNs from CloudFormation and prints AgentCore Gateway registration commands for `compute_profiles`, `compute_run`, and `compute_status`
- AgentCore Gateway registration guide (`docs/agentcore-registration.md`) — console and CLI registration steps, invoke permission setup, verification with example payloads, and SNS notification subscription

## [0.1.0] - 2026-04-01

### Added
- Ten analysis profiles: K-Means Clustering, GLM Regression, Prophet Forecasting, Cohort Retention, Text Topic Modeling, Anomaly Detection (Isolation Forest), Spark Transform, Correlation Analysis, Geographic Enrichment (Census Bureau), Kaplan-Meier Survival Analysis
- Step Functions state machine workflow: CheckBudget → ExtractDataset → RouteCompute → Compute → DeliverResults → RecordSpend → NotifyUser
- Per-user monthly compute budget enforcement tracked in DynamoDB
- AgentCore Gateway Lambda targets: `compute_run`, `compute_status`, `compute_profiles`
- Lambda runner with profile dispatch — single function handles all nine Lambda-backed profiles
- EMR Serverless stub for Spark transform profile (returns `requires_emr` when not enabled)
- CDK stack with Lambda layers for scientific Python (scikit-learn, pandas, statsmodels, prophet, lifelines) and infrastructure wiring

[unreleased]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.4.3...v0.5.0
[0.4.2]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/scttfrdmn/quick-suite-compute/releases/tag/v0.1.0
