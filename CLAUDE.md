# CLAUDE.md — Quick Suite Compute Extension

## What This Is

Ephemeral analytics compute for Amazon Quick Suite. University analytics
teams (IR, enrollment management, advancement) describe what analysis they
want in natural language through Quick Suite's chat interface. The compute
extension matches the request to a pre-built analysis profile, runs it on
ephemeral Lambda (or EMR Serverless for Spark), and delivers results back
as a Quick Sight dataset. The analyst never sees an instance, a cluster,
or a console.

Eight AgentCore Gateway Lambda targets:

| Tool | What It Does |
|------|-------------|
| `compute_profiles` | List available analysis types with inputs/outputs/cost |
| `compute_run` | Match intent to profile, start execution, return job ID |
| `compute_status` | Poll job progress; return results when done |
| `compute_history` | List recent jobs for a user |
| `compute_cancel` | Abort a running job |
| `compute_snapshots` | List a user's named result snapshots (v0.6.0) |
| `compute_compare` | Diff two named snapshots by row set (v0.6.0) |
| `compute_schedule` | Create, list, delete scheduled compute jobs via EventBridge Scheduler (v0.13.0) |

**Async execution model.** `compute_run` returns a job ID immediately.
Quick Suite's agent calls `compute_status` to poll. Most Lambda-backed
jobs complete in 15–60 seconds. The agent says "I've started the
clustering job, I'll check back in a moment."

## Architecture

```
Quick Suite (Chat Agent)
    │  MCP Actions Integration
    ▼
AgentCore Gateway
    │  Lambda targets
    ▼
┌──────────────────┐     ┌───────────────────────────────────────┐
│ compute_profiles │     │ compute_run                           │
│ compute_status   │     │  1. Validate params against profile   │
│ (read-only)      │     │  2. Check budget (DynamoDB)           │
│                  │     │  3. Start Step Functions execution    │
└──────────────────┘     │  4. Return execution ARN / job ID     │
                         └───────────────────┬───────────────────┘
                                             │
                                             ▼
                         ┌───────────────────────────────────────┐
                         │      Step Functions State Machine      │
                         │                                       │
                         │  CheckBudget                          │
                         │      ↓                                │
                         │  ExtractDataset (QS → S3 Parquet)     │
                         │      ↓                                │
                         │  RouteCompute ──→ Lambda (9 profiles) │
                         │                ──→ EMR Serverless (1) │
                         │      ↓                                │
                         │  DeliverResults (S3 → QS dataset)     │
                         │      ↓                                │
                         │  RecordSpend (DynamoDB)                │
                         │      ↓                                │
                         │  NotifyUser (SNS)                     │
                         └───────────────────────────────────────┘
```

## 10 Analysis Profiles

Each profile is a self-contained job definition in DynamoDB. Quick Suite's
agent selects from this catalog. It cannot invent profiles or modify
parameters outside declared bounds.

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

### The 10 Profiles

#### 1. clustering-kmeans (Lambda)
Segment records into k groups. scikit-learn KMeans.
**University use case:** Segment incoming freshmen by application attributes
for targeted yield campaigns. Segment donors by giving patterns.
**Params:** k (2–20), feature columns, standardize flag.
**Output:** Original data + cluster_id + cluster_distance.

#### 2. regression-glm (Lambda)
Linear or logistic regression. statsmodels GLM.
**University use case:** Predict 4-year graduation probability from first-semester
GPA, credits attempted, demographics. Predict donor lapse risk.
**Params:** target column, feature columns, model type (linear/logistic),
include interactions flag.
**Output:** Predictions + coefficients table + diagnostics (R², AUC, p-values).

#### 3. forecast-prophet (Lambda)
Time series forecasting. Prophet.
**University use case:** Project enrollment headcount 3 years forward for
budget planning. Forecast research expenditure trends.
**Params:** date column, value column, forecast horizon, seasonality mode.
**Output:** Forecast values + confidence intervals + trend/seasonal decomposition.

#### 4. retention-cohort (Lambda)
Semester-by-semester retention/persistence matrices. pandas.
**University use case:** IPEDS reporting, accreditation self-studies.
**Params:** student ID, cohort column (entry term), event date, period grain.
**Output:** Cohort retention matrix + attrition rates + long-form table.

#### 5. text-topics (Lambda)
Topic modeling on text data. scikit-learn LDA/NMF.
**University use case:** Classify NSSE/course evaluation open-ended responses
into themes without manual coding.
**Params:** text column, num_topics (2–50), method (LDA/NMF),
min_doc_frequency, max_doc_frequency.
**Output:** Original data + dominant_topic + topic_probability + topic-term matrix.

#### 6. anomaly-isolation-forest (Lambda)
Anomaly detection. scikit-learn IsolationForest.
**University use case:** Flag unusual financial transactions. Detect anomalous
grade distributions.
**Params:** feature columns, contamination rate (0.01–0.20), return_scores.
**Output:** Original data + is_anomaly + anomaly_score.

#### 7. transform-spark (EMR Serverless)
Large dataset join and transform. Spark.
**University use case:** Join student records across SIS, LMS, financial aid,
housing for unified analytics. Cross-reference awards with publications.
**Params:** S3 paths for additional datasets, join keys, join type,
select/rename columns, filter expressions.
**Output:** Joined dataset registered in Quick Sight.
**Note:** This is the only profile that requires EMR Serverless. For initial
build, return `requires_emr: true` status if EMR is not enabled in the
deployment. Same pattern as `requires_transform` in open-data.

#### 8. explore-correlations (Lambda)
Correlation matrix and feature importance. pandas + scipy.
**University use case:** Identify which survey items predict student
satisfaction. Find leading indicators of donor engagement.
**Params:** target column (optional), feature columns,
method (pearson/spearman/mutual_info), top_n.
**Output:** Correlation matrix + ranked feature importance.

#### 9. geo-enrich (Lambda)
Geographic enrichment via Census Bureau API.
**University use case:** Append Census demographics to student addresses
for equity analysis. Map alumni density for regional event planning.
**Params:** lat/lon columns OR zip/FIPS column, enrichment source
(ACS 5-year, TIGER), variables to append.
**Output:** Original data + Census variables + geo identifiers.
**Note:** Census Bureau API is free, no key required for basic access.

#### 10. survival-kaplan-meier (Lambda)
Survival / time-to-event analysis. lifelines.
**University use case:** Time-to-degree by demographic group for equity
reporting. Time-to-tenure for provost office.
**Params:** duration column, event column, group column, confidence level.
**Output:** Survival curves + median survival times + log-rank test + hazard ratios.

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
      "TimeoutSeconds": 120,
      "Next": "RecordSpend"
    },
    "RecordSpend": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:qs-compute-record-spend",
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
      "End": true
    }
  }
}
```

## Cost Controls

| Guard | Default | Configurable |
|-------|---------|--------------|
| Max wall time | 30 minutes | Per profile, max 4 hours |
| Max concurrent jobs per user | 2 | Per deployment |
| Max monthly spend per user | $50 | Per deployment |
| Auto-terminate on timeout | Always | Not configurable |
| Max input dataset size | 10 GB | Per profile |

Spend tracking: DynamoDB table keyed by user ARN, updated by RecordSpend
step on job completion.

## Per-Job Cost

| Profile | Typical Dataset | Duration | Cost |
|---------|----------------|----------|------|
| K-Means Clustering | 50K rows × 20 cols | 15 sec | ~$0.01 |
| Logistic Regression | 100K rows × 50 cols | 30 sec | ~$0.01 |
| Time Series Forecast | 5 years monthly | 45 sec | ~$0.01 |
| Text Topic Modeling | 10K documents | 2 min | ~$0.02 |
| Spark Join (EMR) | 50M rows × 3 sources | 8 min | ~$0.50 |
| Geographic Enrichment | 200K addresses | 3 min | ~$0.05 |

Infrastructure idle cost: < $5/month.

## S3 Layout

```
s3://qs-compute-{account-id}-{region}/
├── inputs/{execution-id}/data.parquet
├── results/{execution-id}/
│   ├── data.parquet
│   ├── metadata.json
│   └── diagnostics/
└── spark/transform.py
```

Lifecycle: inputs/ deleted after 24 hours. results/ retained 90 days.

## IAM (Three Roles)

1. **Tool Lambdas** (compute_run, compute_status, compute_profiles):
   `states:StartExecution`, `states:DescribeExecution`, `dynamodb:GetItem`
   (profiles), `dynamodb:Query` (spend)

2. **Runner Lambdas** (inside Step Functions):
   `s3:GetObject` on inputs/, `s3:PutObject` on results/.
   No Quick Sight access. No network egress except S3 and Census API.

3. **Deliver Lambda**:
   `quicksight:Create*`, `quicksight:UpdateDataSet`, `s3:GetObject` on
   results/, `sns:Publish`

## Project Structure

```
quick-suite-compute/
├── app.py
├── cdk.json
├── CLAUDE.md                          ← you are here
├── requirements.txt
├── stacks/
│   ├── __init__.py
│   └── compute_stack.py
├── config/
│   ├── compute-tools.yaml             # AgentCore tool definitions
│   └── profiles/                      # One JSON per profile
│       ├── clustering-kmeans.json
│       ├── regression-glm.json
│       ├── forecast-prophet.json
│       ├── retention-cohort.json
│       ├── text-topics.json
│       ├── anomaly-isolation-forest.json
│       ├── transform-spark.json
│       ├── explore-correlations.json
│       ├── geo-enrich.json
│       └── survival-kaplan-meier.json
├── lambdas/
│   ├── compute-profiles/handler.py    # AgentCore target: list profiles
│   ├── compute-run/handler.py         # AgentCore target: start job
│   ├── compute-status/handler.py      # AgentCore target: poll job
│   ├── compute-history/handler.py     # AgentCore target: list job history
│   ├── compute-cancel/handler.py      # AgentCore target: cancel job
│   ├── compute-snapshots/handler.py   # AgentCore target: list named snapshots (v0.6.0)
│   ├── compute-compare/handler.py     # AgentCore target: diff two snapshots (v0.6.0)
│   ├── check-budget/handler.py        # Step Functions step
│   ├── extract/handler.py             # Step Functions: QS dataset → S3
│   ├── runner/handler.py              # Step Functions: dispatches to profile
│   ├── deliver/handler.py             # Step Functions: S3 → QS dataset
│   ├── record-spend/handler.py        # Step Functions: update spend + write snapshot
│   ├── handle-failure/handler.py      # Step Functions: error handling
│   ├── audit-log/handler.py           # Step Functions: write audit record on terminal state (v0.8.0)
│   └── profiles/                      # Per-profile analysis code
│       ├── clustering.py
│       ├── regression.py
│       ├── forecast.py
│       ├── retention.py
│       ├── text_topics.py
│       ├── anomaly.py
│       ├── correlations.py
│       ├── geo_enrich.py
│       └── survival.py
├── spark/
│   └── transform.py                   # EMR Serverless job script
├── tests/
│   ├── conftest.py
│   ├── test_profiles.py
│   ├── test_run.py
│   └── test_stack.py
└── gtm/
    ├── demo-script.md
    ├── university-use-cases.md
    └── cost-comparison.md
```

## Conventions

Follow the same patterns as model-router and open-data:

- **Python CDK.** Entry point: `app.py`.
- **Python 3.12 Lambdas.** `boto3` + stdlib for tool/infra Lambdas.
  Runner Lambdas need scikit-learn, pandas, statsmodels, prophet,
  lifelines — package as Lambda Layer(s) via Docker bundling in CDK.
- **AgentCore Lambda targets** for the three tool Lambdas (compute_profiles,
  compute_run, compute_status). Plain dict input, plain dict output.
  See top-level CLAUDE.md for the handler pattern.
- **Step Functions Lambdas** (extract, runner, deliver, etc.) use standard
  Lambda I/O — they're invoked by Step Functions, not AgentCore.
- **Quick Sight API** via `boto3.client('quicksight')` — that's the API
  namespace. User-facing text says "Quick Sight" or "Quick Suite."
- **DynamoDB** for profile catalog and spend tracking (on-demand billing).
- **Structured JSON logging** at INFO level.

## Project Tracking

Work is tracked in GitHub — not in local files. Do not add TODO lists or task
tracking to CLAUDE.md or create TODO.md files.

- **Issues:** https://github.com/scttfrdmn/quick-suite-compute/issues
- **Milestones:** https://github.com/scttfrdmn/quick-suite-compute/milestones
- **Project board:** https://github.com/users/scttfrdmn/projects/46
- **Changelog:** CHANGELOG.md (keepachangelog format, semver 2.0)

To report a bug or propose a feature, open a GitHub Issue with the appropriate
label. All release planning happens via milestones.

## Design Decisions

1. **Async with polling.** `compute_run` returns immediately with a job ID.
   `compute_status` polls. This matches real analysis times (15s–8min)
   and lets the Quick Suite agent manage the conversation naturally.

2. **Lambda-only first.** 9 of 10 profiles run on Lambda (scikit-learn,
   pandas, statsmodels, prophet, lifelines). EMR Serverless only for
   transform-spark. If EMR is not enabled in the deployment,
   transform-spark returns `requires_emr` status — same pattern as
   `requires_transform` in open-data.

3. **Single runner Lambda with profile dispatch.** One Lambda function
   (`runner/handler.py`) receives the profile_id and dispatches to the
   appropriate profile module. Simpler than 10 separate Lambda functions.
   The heavy dependencies (scikit-learn, etc.) are in a shared Layer.

4. **Results as Quick Sight datasets.** The deliver Lambda uses the same
   `quicksight.create_data_source()` + `create_data_set()` pattern as
   the open-data dataset-loader. Reuse that code or extract to a shared
   utility.

5. **Budget enforcement in Step Functions.** CheckBudget is the first
   step. If the user's monthly spend exceeds the limit, the state
   machine fails immediately — no compute is provisioned.
