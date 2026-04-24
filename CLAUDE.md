# CLAUDE.md — Quick Suite Compute Extension (v0.18.0)

## What This Is

Ephemeral analytics compute for Amazon Quick Suite. University analytics
teams (IR, enrollment management, advancement, research offices) describe
what analysis they want in natural language through Quick Suite's chat
interface. The compute extension matches the request to a pre-built
analysis profile, runs it on ephemeral Lambda (or EMR Serverless for
Spark), and delivers results back as a Quick Sight dataset. The analyst
never sees an instance, a cluster, or a console.

Eight AgentCore Gateway Lambda targets + internal Lambdas:

| Tool | What It Does |
|------|-------------|
| `compute_profiles` | List available analysis types with inputs/outputs/cost |
| `compute_run` | Match intent to profile, validate params, check budget, start execution; returns `estimated_cost_usd` + `estimated_duration_seconds`; accepts `result_label` (snapshot) and `chain_profile_id` (profile chaining); pre-checks cross-stack Router spend table |
| `compute_status` | Poll job progress; SUCCEEDED includes `actual_cost_usd`, `duration_seconds`, `profile_id`, `summary`, `export_urls`; RUNNING includes `cost_usd_so_far`; shows `step: "profile_2"` for chained jobs |
| `compute_history` | List recent jobs for a user |
| `compute_cancel` | Abort a running job; ownership check via `sfn.describe_execution()` |
| `compute_snapshots` | List a user's named result snapshots sorted by completion time |
| `compute_compare` | Diff two named snapshots: added/removed/unchanged row counts + schema diff |
| `compute_schedule` | Create, list, delete scheduled compute jobs via EventBridge Scheduler (v0.13.0) |

**Async execution model.** `compute_run` returns a job ID immediately.
Quick Suite's agent calls `compute_status` to poll. Most Lambda-backed
jobs complete in 15-60 seconds. The agent says "I've started the
clustering job, I'll check back in a moment."

## Architecture

```
Quick Suite (Chat Agent)
    |  MCP Actions Integration
    v
AgentCore Gateway
    |  Lambda targets
    v
+------------------+     +-------------------------------------------+
| compute_profiles |     | compute_run                               |
| compute_status   |     |  1. Validate params against profile       |
| compute_history  |     |  2. Check budget (DynamoDB + Router spend) |
| compute_cancel   |     |  3. Check source_uri bucket allowlist     |
| compute_snapshots|     |  4. Start Step Functions execution        |
| compute_compare  |     |  5. Return execution ARN / job ID         |
| compute_schedule |     +-------------------+-----------------------+
| (read / control) |                         |
+------------------+                         v
                         +-------------------------------------------+
                         |      Step Functions State Machine          |
                         |                                           |
                         |  CheckBudget                              |
                         |      |                                    |
                         |  ExtractDataset (QS / S3 / claws:// URI) |
                         |      |                                    |
                         |  RouteCompute --> Lambda (41 profiles)    |
                         |                --> EMR Serverless (1)     |
                         |      |                                    |
                         |  DeliverResults (S3 -> QS + CSV/XLSX)    |
                         |      |                                    |
                         |  HasChainProfile? --yes--> PrepareChain   |
                         |      | no                  --> Compute #2 |
                         |      |                     --> Deliver #2  |
                         |      v                                    |
                         |  RecordSpend (DynamoDB + data registry)   |
                         |      |                                    |
                         |  NotifyUser (SNS)                         |
                         |      |                                    |
                         |  AuditLog (S3 NDJSON)                     |
                         +-------------------------------------------+
```

## 42 Analysis Profiles across 12 Categories

Each profile is a self-contained job definition in `config/profiles/*.json`.
Quick Suite's agent selects from this catalog. It cannot invent profiles or
modify parameters outside declared bounds.

### Profile Schema

```json
{
  "profile_id": "clustering-kmeans",
  "display_name": "K-Means Clustering",
  "description": "Segments records into k groups based on numeric features.",
  "category": "segmentation",
  "backend": "lambda",
  "entrypoint": "clustering.kmeans_handler",
  "parameters": {
    "k": {"type": "integer", "default": 5, "min": 2, "max": 20},
    "features": {"type": "column_list", "min_columns": 2},
    "standardize": {"type": "boolean", "default": true}
  },
  "input_requirements": {
    "min_rows": 50,
    "max_rows": 500000,
    "required_column_types": ["numeric"]
  },
  "output_schema": {
    "preserves_input_columns": true,
    "added_columns": [
      {"name": "cluster_id", "type": "integer"},
      {"name": "cluster_distance", "type": "float"}
    ]
  },
  "cost_estimate": {
    "typical_duration_seconds": 30,
    "typical_cost_usd": 0.001
  },
  "tags": ["clustering", "segmentation", "unsupervised", "enrollment", "survey"]
}
```

### All 42 Profiles

**Statistics (2):**
- `anova` — one-way/multi-factor ANOVA with Tukey HSD post-hoc
- `chi-square` — chi-square test of independence with Cramer's V

**Prediction/ML (3):**
- `regression-glm` — linear or logistic regression via statsmodels GLM; predictions + coefficients + diagnostics (R-squared, AUC, p-values)
- `regression-logistic` — dedicated logistic regression with ROC/AUC
- `classification-random-forest` — scikit-learn random forest classifier with feature importance

**Forecasting (3):**
- `forecast-prophet` — time series forecasting via Prophet; trend/seasonal decomposition + confidence intervals
- `change-detection` — detects structural breaks in time series data
- `seasonality-decompose` — STL decomposition into trend, seasonal, and residual components

**Clustering (1):**
- `clustering-kmeans` — scikit-learn KMeans; output includes cluster_id + cluster_distance

**Text (3):**
- `text-topics` — topic modeling via scikit-learn LDA/NMF on text columns
- `text-sentiment` — sentiment scoring on text data
- `text-similarity` — pairwise or query-based text similarity

**Anomaly (1):**
- `anomaly-isolation-forest` — scikit-learn IsolationForest; is_anomaly flag + anomaly_score

**Higher-Ed (9):**
- `cohort-flow` — Sankey-style flow analysis tracking student progression through milestones
- `dfwi-analysis` — D/F/W/Incomplete rate analysis by course, instructor, term
- `equity-gap` — equity gap analysis across demographic groups with statistical significance
- `peer-benchmark` — institutional benchmarking against peer cohort (see peer_cohort module)
- `retention-cohort` — semester-by-semester retention/persistence matrices for IPEDS reporting
- `survival-kaplan-meier` — time-to-event analysis via lifelines; survival curves + log-rank test + hazard ratios
- `intersectionality-equity` (v0.15.0) — cross-tabs outcome metric by 2+ demographic columns; Disparate Impact ratio (80% rule); cell suppression at n < `n_suppress`
- `assessment-irt` (v0.15.0) — 2PL Item Response Theory model via `girth`; item_parameters (difficulty, discrimination), person_abilities (theta, se), item_information; 503 if `girth` not installed; min 5 items / 100 respondents
- `financial-aid-effectiveness` (v0.18.0) — aid-band cohort tables, IRLS logistic regression for predicted persistence probability, unmet-need trend by aid year

**Geospatial (3):**
- `geo-enrich` — Census Bureau API enrichment; append ACS/TIGER demographics to lat/lon or FIPS
- `isochrone` — travel-time reachability polygons
- `spatial-aggregate` — aggregate metrics within geographic boundaries

**Exploration (1):**
- `explore-correlations` — correlation matrix + ranked feature importance; Pearson/Spearman/mutual information

**Research (10):**
- `grant-portfolio` — funding portfolio analysis across awards
- `network-coauthor` — co-authorship network analysis
- `causal-iv` (v0.16.0) — 2SLS instrumental variables via `linearmodels`; first-stage F-stat, weak instrument warning (F < 10), compliance rate, 95% CI; optional `peer_benchmark` annotation
- `causal-rd` (v0.16.0) — numpy-native regression discontinuity; IK bandwidth formula; 5-point bandwidth sensitivity; McCrary density manipulation test; fuzzy RD via IV
- `causal-did` (v0.16.0) — OLS difference-in-differences; parallel trends test; event study coefficients; placebo pre-period tests; staggered adoption via `csdid` (503 if absent)
- `grant-pipeline` (v0.16.0) — PI health scores (active*0.4 + continuity*0.4 + diversity*0.2); NCE risk flagging; sponsor timing analysis
- `provenance-graph` (v0.16.0) — W3C PROV-DM JSON-LD lineage from HistoryTable; Markdown summary; gap detection; `min_rows: 0` (queries DynamoDB, not input DataFrame)
- `power-analysis` (v0.17.0) — literature-informed sample size calculation via Router `extract` cross-reference on PubMed IDs; Cohen's d from pilot data or manual; scipy power curves; confound checklists; graceful degradation when Router unavailable
- `anomaly-hypothesis` (v0.17.0) — IsolationForest + Router `research` grounding for per-anomaly classification (instrument_error / known_noise / reported_effect / novel_candidate); domain z-thresholds; Router unavailable -> all novel_candidate
- `reproducibility-check` (v0.17.0) — re-execute analysis script in RestrictedPython sandbox against deposited data; compare outputs to `manuscript_results` with configurable tolerance; provenance-graph integration

**Ingest (3):**
- `ingest-netcdf` — convert NetCDF to Parquet for Quick Sight
- `ingest-pdf-extract` — extract tables/text from PDF documents
- `ingest-geojson` — convert GeoJSON features to tabular format

**Custom (2):**
- `custom-python` — user-provided Python code executed in RestrictedPython sandbox; `_SafePandasProxy` blocks URL-based `read_*` calls
- `custom-generated` — LLM-generated code gated by `_analyze_generated_code()` AST static analysis before execution

**Transform (1):**
- `transform-spark` (EMR Serverless) — large dataset join/transform via Spark; if EMR not enabled, returns `requires_emr` status

## Step Functions Workflow

```json
{
  "Comment": "Quick Suite Compute Job Lifecycle",
  "StartAt": "CheckBudget",
  "States": {
    "CheckBudget": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-check-budget",
      "Next": "BudgetDecision"
    },
    "BudgetDecision": {
      "Type": "Choice",
      "Choices": [{
        "Variable": "$.budget_ok",
        "BooleanEquals": false,
        "Next": "BudgetExceeded"
      }],
      "Default": "ExtractDataset"
    },
    "BudgetExceeded": {
      "Type": "Fail",
      "Cause": "Monthly compute budget exceeded",
      "Error": "BudgetExceeded"
    },
    "ExtractDataset": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-extract",
      "Comment": "QS dataset -> S3 Parquet; supports claws:// URIs via CLAWS_RESOLVER_ARN",
      "TimeoutSeconds": 300,
      "Next": "RouteCompute"
    },
    "RouteCompute": {
      "Type": "Choice",
      "Choices": [
        {"Variable": "$.profile.backend", "StringEquals": "lambda", "Next": "ComputeLambda"},
        {"Variable": "$.profile.backend", "StringEquals": "emr_serverless", "Next": "ComputeEMR"}
      ]
    },
    "ComputeLambda": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-runner",
      "TimeoutSeconds": 900,
      "Retry": [{"ErrorEquals": ["Lambda.TooManyRequestsException"], "MaxAttempts": 2}],
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "JobFailed"}],
      "Next": "DeliverResults"
    },
    "ComputeEMR": {
      "Type": "Task",
      "Resource": "arn:aws:states:::emr-serverless:startJobRun.sync",
      "TimeoutSeconds": 7200,
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "JobFailed"}],
      "Next": "DeliverResults"
    },
    "DeliverResults": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-deliver",
      "Comment": "S3 Parquet -> QS dataset + CSV/XLSX presigned URLs",
      "TimeoutSeconds": 120,
      "Next": "HasChainProfile"
    },
    "HasChainProfile": {
      "Type": "Choice",
      "Comment": "v0.13.0 job chaining: run second profile on output of first",
      "Choices": [{
        "Variable": "$.chain_profile_id",
        "IsPresent": true,
        "Next": "PrepareChainInput"
      }],
      "Default": "RecordSpend"
    },
    "PrepareChainInput": {
      "Type": "Pass",
      "Comment": "Rewire output of profile 1 as input for profile 2",
      "Next": "ComputeLambdaChain"
    },
    "ComputeLambdaChain": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-runner",
      "Next": "DeliverResultsChain"
    },
    "DeliverResultsChain": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-deliver",
      "Next": "RecordSpend"
    },
    "RecordSpend": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-record-spend",
      "Comment": "Updates spend table, writes snapshot if result_label set, registers result in data source registry",
      "Next": "NotifyUser"
    },
    "NotifyUser": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "...",
        "Subject": "Your analysis is ready",
        "Message.$": "States.Format('Your {} analysis is complete. Results are available as a new dataset in Quick Sight: {}', $.profile.display_name, $.result_dataset_name)"
      },
      "Next": "AuditLog"
    },
    "AuditLog": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-audit-log",
      "Comment": "Write audit record to S3; also called from JobFailed path",
      "End": true
    },
    "JobFailed": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-handle-failure",
      "Next": "NotifyFailure"
    },
    "NotifyFailure": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "...",
        "Subject": "Analysis job failed",
        "Message.$": "States.Format('Your {} analysis could not be completed. Error: {}', $.profile.display_name, $.error_message)"
      },
      "Next": "AuditLogFailure"
    },
    "AuditLogFailure": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-audit-log",
      "End": true
    }
  }
}
```

## Key Features

### Audit Log (v0.8.0)
Every terminal path (SUCCEEDED, FAILED) ends with `audit-log` Lambda writing
`s3://compute-results/audit/{year}/{month}/{job_id}.json`. Fields: `job_id`,
`profile_id`, `user_arn`, `dataset_uri`, `params`, `result_uri`, `cost_usd`,
`duration_seconds`, `status`, `timestamp`. URIs only -- no PII.

### VPC Support (v0.8.0)
`enable_vpc=true` CDK context flag places all SFN Lambda steps in an
isolated-subnet VPC with S3 Gateway endpoint.

### KMS Encryption (v0.8.0)
`enable_kms=true` CDK context flag encrypts HistoryTable and compute-results
bucket with customer-managed KMS keys.

### clAWS URI Support
`claws://roda-noaa-ghcn` in extract Lambda invokes `CLAWS_RESOLVER_ARN` Lambda
to resolve to `dataset_id`, then extracts via Quick Sight path. Wired via
`claws_resolver_arn` CDK context var.

### Named Snapshots (v0.6.0)
`qs-compute-snapshots` DynamoDB table (PK: `user_arn`, SK: `label`). Written
by `record-spend` when `result_label` is set. Read by `compute_snapshots` and
`compute_compare`.

### Router Spend Integration (v0.10.0)
`compute_run` reads `qs-router-spend` (the quick-suite-router's spend ledger)
via `ROUTER_SPEND_TABLE` env var (set from `router_spend_table_arn` CDK
context); blocks jobs if department cumulative spend + estimated cost exceeds
`MONTHLY_BUDGET_USD`; fails open on AWS errors.

### Job Chaining (v0.13.0)
`HasChainProfile` Choice + `PrepareChainInput` Pass states after DeliverResults.
Chain runs second compute+deliver pass with output of first as input.
`compute_status` shows `step: "profile_2"` for chained jobs.

### Scheduled Jobs (v0.13.0)
`compute_schedule` AgentCore tool: create/list/delete scheduled jobs via
EventBridge Scheduler. `qs-compute-schedules` DynamoDB table.
`compute-schedule-trigger` internal Lambda handles scheduled invocations.

### CSV/Excel Export (v0.18.0)
`deliver` Lambda converts Parquet to CSV + XLSX (1M row limit). Presigned S3
URLs (24h TTL) in `compute_status` SUCCEEDED response `export_urls` dict.
`openpyxl` added to Docker image.

### Results Write-Back to Data Registry (v0.18.0)
`record-spend` Lambda writes `compute-result-{job_id}` entries to
`qs-data-source-registry` after SUCCEEDED jobs. Discoverable via
`federated_search` and clAWS `discover`. Fail-open. `DATA_REGISTRY_TABLE`
env var from CDK context.

### Peer Cohort Matching (v0.16.0)
`peer_cohort` module: Carnegie class + enrollment +/-20% + control type +
Pell +/-10pt filtering; weighted scoring; 30-day DynamoDB cache in
`qs-compute-peer-cohort-cache` table. Used by `causal-iv` and
`peer-benchmark` profiles.

### Dashboard
Per-profile Cost (USD/24h), Duration (p99), and Cumulative Cost (30d SUM)
graph widgets generated from `config/profiles/*.json`.

## Security Hardening

### v0.14.0

- **compute-cancel ownership check:** requires `user_arn`; verifies against
  execution input via `sfn.describe_execution()`; same "not found" message on
  mismatch (no info leak)
- **Budget conditional write backstop:** `record-spend` uses
  `ConditionExpression` on `spend_usd <= :remaining`;
  `ConditionalCheckFailedException` fails open with warning log
- **source_uri bucket allowlist:** `COMPUTE_ALLOWED_BUCKETS` env var (CDK
  context `compute_allowed_buckets`); rejects s3:// buckets not in list;
  empty list = allow all (backward compat)
- **result_label validation:** `^[A-Za-z0-9_\-]{1,64}$` enforced in
  compute-run (400 error) and record-spend (silent skip)
- **Parameter max-length:** `MAX_STRING_PARAM_LENGTH = 256` guards on
  `column`, `column_list` items, `string_list` items
- **DynamoDB protection:** all 5 tables (SpendTable, SnapshotsTable,
  HistoryTable, SchedulesTable, PeerCohortCacheTable) have
  `deletion_protection=True`, `point_in_time_recovery=True`,
  `removal_policy=RETAIN`
- **Log retention:** `log_retention=logs.RetentionDays.THREE_MONTHS` (90 days)
  on all Lambda functions

### v0.13.0

- **Pandas sandbox proxy:** `_SafePandasProxy` blocks URL-based `read_*`
  method calls in RestrictedPython sandbox
- **AST static analysis:** `_analyze_generated_code()` gates LLM-generated
  code before execution in `custom-generated` profile
- **Scoped SFN execution role:** explicit `qs-compute-sfn-role` with
  `lambda:InvokeFunction` only on state machine Lambdas (no wildcard)

## Cost Controls

| Guard | Default | Configurable |
|-------|---------|--------------|
| Max wall time | 30 minutes | Per profile, max 4 hours |
| Max concurrent jobs per user | 2 | Per deployment |
| Max monthly spend per user | $50 | Per deployment |
| Cross-stack department budget | Router spend table | CDK context `router_spend_table_arn` |
| Auto-terminate on timeout | Always | Not configurable |
| Max input dataset size | 10 GB | Per profile |
| Source URI bucket allowlist | Allow all | CDK context `compute_allowed_buckets` |

Spend tracking: DynamoDB table keyed by user ARN, updated by RecordSpend
step on job completion. Conditional write backstop prevents overspend on
concurrent jobs.

## Per-Job Cost

| Profile | Typical Dataset | Duration | Cost |
|---------|----------------|----------|------|
| K-Means Clustering | 50K rows x 20 cols | 15 sec | ~$0.01 |
| Logistic Regression | 100K rows x 50 cols | 30 sec | ~$0.01 |
| Time Series Forecast | 5 years monthly | 45 sec | ~$0.01 |
| Text Topic Modeling | 10K documents | 2 min | ~$0.02 |
| Causal IV/RD/DiD | 50K rows | 30 sec | ~$0.01 |
| Power Analysis | N/A (Router cross-ref) | 10 sec | ~$0.01 |
| Spark Join (EMR) | 50M rows x 3 sources | 8 min | ~$0.50 |
| Geographic Enrichment | 200K addresses | 3 min | ~$0.05 |

Infrastructure idle cost: < $5/month.

## S3 Layout

```
s3://qs-compute-{account-id}-{region}/
+-- inputs/{execution-id}/data.parquet
+-- results/{execution-id}/
|   +-- data.parquet
|   +-- data.csv
|   +-- data.xlsx
|   +-- metadata.json
|   +-- diagnostics/
+-- audit/{year}/{month}/{job_id}.json
+-- spark/transform.py
```

Lifecycle: inputs/ deleted after 24 hours. results/ retained 90 days.
audit/ retained indefinitely.

## IAM (Four Roles)

1. **Tool Lambdas** (compute_run, compute_status, compute_profiles,
   compute_history, compute_cancel, compute_snapshots, compute_compare,
   compute_schedule):
   `states:StartExecution`, `states:DescribeExecution`, `states:StopExecution`,
   `dynamodb:GetItem` (profiles, spend, snapshots, schedules),
   `dynamodb:Query` (spend, history, schedules),
   `scheduler:CreateSchedule`, `scheduler:DeleteSchedule`, `scheduler:GetSchedule`

2. **Step Functions execution role** (`qs-compute-sfn-role`):
   `lambda:InvokeFunction` scoped to state machine Lambdas only (no wildcard).
   No direct S3, DynamoDB, or Quick Sight access.

3. **Runner / SFN Lambdas** (extract, runner, deliver, record-spend,
   check-budget, handle-failure, audit-log):
   `s3:GetObject` on inputs/, `s3:PutObject` on results/ and audit/.
   `quicksight:Create*`, `quicksight:UpdateDataSet` (deliver only).
   `dynamodb:PutItem` on spend/snapshots tables (record-spend).
   `dynamodb:PutItem` on data source registry (record-spend, write-back).
   `secretsmanager:GetSecretValue` (Runner, for Router API calls).
   `sns:Publish` (via Step Functions service integration).

4. **Schedule trigger Lambda** (`compute-schedule-trigger`):
   `states:StartExecution` on the compute state machine.
   `dynamodb:GetItem` on schedules table.

## DynamoDB Tables

| Table | PK | SK | Purpose |
|-------|----|----|---------|
| `qs-compute-spend` | `user_arn` | -- | Monthly spend tracking; conditional write backstop |
| `qs-compute-history` | `user_arn` | `job_id` | Job history; read by `compute_history` and `provenance-graph` |
| `qs-compute-snapshots` | `user_arn` | `label` | Named result snapshots |
| `qs-compute-schedules` | `user_arn` | `schedule_id` | Scheduled job definitions |
| `qs-compute-peer-cohort-cache` | `cache_key` | -- | 30-day peer cohort cache with TTL |

All tables: PAY_PER_REQUEST, deletion protection, PITR, RETAIN removal policy.

## Project Structure

```
quick-suite-compute/
+-- app.py
+-- cdk.json
+-- CLAUDE.md                            <- you are here
+-- CHANGELOG.md
+-- requirements.txt
+-- pyproject.toml
+-- Makefile
+-- trivy.yaml
+-- stacks/
|   +-- __init__.py
|   +-- compute_stack.py
+-- config/
|   +-- compute-tools.yaml               # AgentCore tool definitions
|   +-- profiles/                        # One JSON per profile (42 files)
|       +-- anova.json
|       +-- anomaly-hypothesis.json
|       +-- anomaly-isolation-forest.json
|       +-- assessment-irt.json
|       +-- causal-did.json
|       +-- causal-iv.json
|       +-- causal-rd.json
|       +-- change-detection.json
|       +-- chi-square.json
|       +-- classification-random-forest.json
|       +-- clustering-kmeans.json
|       +-- cohort-flow.json
|       +-- custom-generated.json
|       +-- custom-python.json
|       +-- dfwi-analysis.json
|       +-- equity-gap.json
|       +-- explore-correlations.json
|       +-- financial-aid-effectiveness.json
|       +-- forecast-prophet.json
|       +-- geo-enrich.json
|       +-- grant-pipeline.json
|       +-- grant-portfolio.json
|       +-- ingest-geojson.json
|       +-- ingest-netcdf.json
|       +-- ingest-pdf-extract.json
|       +-- intersectionality-equity.json
|       +-- isochrone.json
|       +-- network-coauthor.json
|       +-- peer-benchmark.json
|       +-- power-analysis.json
|       +-- provenance-graph.json
|       +-- regression-glm.json
|       +-- regression-logistic.json
|       +-- reproducibility-check.json
|       +-- retention-cohort.json
|       +-- seasonality-decompose.json
|       +-- spatial-aggregate.json
|       +-- survival-kaplan-meier.json
|       +-- text-sentiment.json
|       +-- text-similarity.json
|       +-- text-topics.json
|       +-- transform-spark.json
+-- lambdas/
|   +-- compute-profiles/handler.py      # AgentCore target: list profiles
|   +-- compute-run/handler.py           # AgentCore target: start job
|   +-- compute-status/handler.py        # AgentCore target: poll job
|   +-- compute-history/handler.py       # AgentCore target: list job history
|   +-- compute-cancel/handler.py        # AgentCore target: cancel job (ownership check)
|   +-- compute-snapshots/handler.py     # AgentCore target: list named snapshots
|   +-- compute-compare/handler.py       # AgentCore target: diff two snapshots
|   +-- compute-schedule/handler.py      # AgentCore target: scheduled jobs CRUD
|   +-- compute-schedule-trigger/handler.py  # Internal: EventBridge Scheduler target
|   +-- check-budget/handler.py          # SFN step: budget gate
|   +-- extract/handler.py              # SFN step: QS/S3/claws:// -> Parquet
|   +-- runner/handler.py               # SFN step: dispatch to profile handler
|   +-- deliver/handler.py              # SFN step: S3 -> QS dataset + CSV/XLSX
|   +-- record-spend/handler.py         # SFN step: update spend + snapshot + data registry
|   +-- handle-failure/handler.py       # SFN step: error handling
|   +-- audit-log/handler.py            # SFN step: write audit record on terminal state
|   +-- layer/                           # Lambda Layer build assets
|   +-- profiles/                        # Per-profile analysis code
|       +-- anomaly.py
|       +-- causal.py                    # causal-iv, causal-rd, causal-did
|       +-- clustering.py
|       +-- correlations.py
|       +-- custom.py                    # custom-python, custom-generated
|       +-- forecast.py
|       +-- geo_enrich.py
|       +-- geospatial.py               # isochrone, spatial-aggregate
|       +-- higher_ed.py                # cohort-flow, dfwi, equity-gap, peer-benchmark,
|       |                                # intersectionality-equity, assessment-irt,
|       |                                # financial-aid-effectiveness
|       +-- ingest.py                    # ingest-netcdf, ingest-pdf-extract, ingest-geojson
|       +-- ml.py                        # classification-random-forest
|       +-- peer_cohort.py               # peer cohort matching + cache
|       +-- regression.py
|       +-- research.py                  # grant-portfolio, network-coauthor, power-analysis,
|       |                                # anomaly-hypothesis, reproducibility-check
|       +-- retention.py
|       +-- statistics.py                # anova, chi-square
|       +-- survival.py
|       +-- text_analytics.py            # text-sentiment, text-similarity
|       +-- text_topics.py
|       +-- time_series.py              # change-detection, seasonality-decompose
+-- spark/
|   +-- transform.py                     # EMR Serverless job script
+-- tests/
|   +-- conftest.py
|   +-- test_profiles.py
|   +-- test_run.py
|   +-- test_stack.py
|   +-- test_sfn_handlers.py
|   +-- test_audit_log.py
|   +-- test_history_cancel.py
|   +-- test_v060_features.py
|   +-- test_integration_handlers.py
|   +-- test_security_hardening.py
|   +-- test_research_science.py
|   +-- test_v18_output.py
|   +-- e2e/                             # End-to-end tests
+-- scripts/
+-- docs/
+-- examples/
+-- gtm/
|   +-- demo-script.md
|   +-- university-use-cases.md
|   +-- cost-comparison.md
```

## Conventions

Follow the same patterns as router, data, and claws:

- **Python CDK.** Entry point: `app.py`.
- **Python 3.12 Lambdas.** `boto3` + stdlib for tool/infra Lambdas.
  Runner Lambdas need scikit-learn, pandas, statsmodels, prophet,
  lifelines, linearmodels, girth, openpyxl -- packaged as Lambda Layer(s)
  via Docker bundling in CDK.
- **AgentCore Lambda targets** for the eight tool Lambdas. Plain dict
  input, plain dict output. See top-level CLAUDE.md for the handler pattern.
- **Step Functions Lambdas** (extract, runner, deliver, etc.) use standard
  Lambda I/O -- they're invoked by Step Functions, not AgentCore.
- **Quick Sight API** via `boto3.client('quicksight')` -- that's the API
  namespace. User-facing text says "Quick Sight" or "Quick Suite."
- **DynamoDB** for spend tracking, history, snapshots, schedules, peer
  cohort cache (on-demand billing, PITR, deletion protection).
- **Structured JSON logging** at INFO level.
- **636 unit tests.** Substrate integration in CI.

## Project Tracking

Work is tracked in GitHub -- not in local files. Do not add TODO lists or task
tracking to CLAUDE.md or create TODO.md files.

- **Issues:** https://github.com/scttfrdmn/campus-compute/issues
- **Milestones:** https://github.com/scttfrdmn/campus-compute/milestones
- **Project board:** https://github.com/users/scttfrdmn/projects/46
- **Changelog:** CHANGELOG.md (keepachangelog format, semver 2.0)

To report a bug or propose a feature, open a GitHub Issue with the appropriate
label. All release planning happens via milestones.

## Design Decisions

1. **Async with polling.** `compute_run` returns immediately with a job ID.
   `compute_status` polls. This matches real analysis times (15s-8min)
   and lets the Quick Suite agent manage the conversation naturally.

2. **Lambda-first.** 41 of 42 profiles run on Lambda (scikit-learn,
   pandas, statsmodels, prophet, lifelines, linearmodels, girth, scipy).
   EMR Serverless only for transform-spark. If EMR is not enabled in the
   deployment, transform-spark returns `requires_emr` status -- same
   pattern as `requires_transform` in open-data.

3. **Single runner Lambda with profile dispatch.** One Lambda function
   (`runner/handler.py`) receives the profile_id and dispatches to the
   appropriate profile module. Simpler than 42 separate Lambda functions.
   The heavy dependencies are in a shared Layer built via Docker.

4. **Results as Quick Sight datasets.** The deliver Lambda uses the same
   `quicksight.create_data_source()` + `create_data_set()` pattern as
   the open-data dataset-loader. CSV + XLSX presigned URLs also provided
   for download.

5. **Budget enforcement in Step Functions.** CheckBudget is the first
   step. If the user's monthly spend exceeds the limit, the state
   machine fails immediately -- no compute is provisioned. Conditional
   write backstop in RecordSpend prevents concurrent overspend.

6. **Cross-stack spend awareness.** `compute_run` checks the Router's
   spend ledger before submitting jobs. A department that has blown its
   Router budget cannot start compute jobs either.

7. **Scoped SFN execution role.** The Step Functions state machine runs
   under `qs-compute-sfn-role` with `lambda:InvokeFunction` scoped only
   to the Lambdas in the state machine. No wildcard invoke permissions.

8. **Job chaining over multi-step profiles.** Rather than building
   complex multi-step profiles, `chain_profile_id` lets callers compose
   two profiles sequentially. The state machine handles the plumbing.

9. **Results feed back into the data catalog.** Completed job results
   are written to `qs-data-source-registry`, making them discoverable
   via `federated_search` and clAWS `discover` without manual registration.
