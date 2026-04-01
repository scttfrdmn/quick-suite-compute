# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/scttfrdmn/quick-suite-compute/releases/tag/v0.1.0
