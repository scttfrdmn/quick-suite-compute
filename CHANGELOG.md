# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/scttfrdmn/quick-suite-compute/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/scttfrdmn/quick-suite-compute/releases/tag/v0.1.0
