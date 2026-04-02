# Quick Suite Compute

**Run statistical analysis on university data from a Quick Suite conversation — no infrastructure, no code, no waiting for a data engineer.**

A lot of the questions that matter most to a university can't be answered with a chart.
"Which students are most likely to not return next semester?" "What does enrollment look
like in three years if current trends hold?" "Which cluster of donors has the highest
major gift potential?" These require statistical models: regression, forecasting,
clustering, survival analysis.

Quick Suite can visualize data. It cannot run a k-means clustering job on 300,000 student
records, fit a Prophet forecast, or compute Kaplan-Meier survival curves. That's what
this extension does. An analyst describes what analysis they want in natural language, and
the Compute extension matches the request to the right pre-built profile, runs the job in
the background, and delivers the results back as a new Quick Sight dataset — usually in
under a minute.

## What Quick Suite Alone Can't Do Here

- Execute statistical models (regression, clustering, forecasting, topic modeling, survival analysis)
- Run analysis on data that lives in Quick Sight but hasn't had any computation applied to it
- Accept a `claws://` URI from the Data extension and use it as a job input
- Track monthly compute spend per user and enforce a budget ceiling
- Run large-scale Spark transformations (joining SIS + LMS + financial aid at 50M-row scale)
- Tell Quick Suite's agent when a job is done so it can synthesize the results

## What You Get

**Three tools** in Quick Suite's chat interface:

| Tool | What it does |
|------|-------------|
| `compute_profiles` | List available analysis types with their inputs, outputs, costs, and durations |
| `compute_run` | Validate the request, check the monthly budget, and start a job; returns a job ID immediately |
| `compute_status` | Check whether a job is running, succeeded, or failed; returns results and cost when done |

**Ten analysis profiles** — each a self-contained job definition that Quick Suite's agent
selects based on what the user is asking for:

| Profile | Method | University Use Case |
|---------|--------|---------------------|
| `clustering-kmeans` | K-Means | Segment incoming students for yield strategy; donor prospect grouping |
| `regression-glm` | GLM (linear/logistic) | Predict graduation probability; donor lapse risk |
| `forecast-prophet` | Prophet time series | Project enrollment 3 years forward; research expenditure trends — container image Lambda |
| `retention-cohort` | Cohort matrices | IPEDS retention reporting; accreditation self-studies |
| `text-topics` | LDA / NMF topic modeling | Analyze 40,000 course evaluation comments without manual coding |
| `anomaly-isolation-forest` | Isolation Forest | Flag at-risk students by LMS engagement; unusual financial transactions |
| `transform-spark` | Spark (EMR Serverless) | Join SIS + LMS + financial aid at scale |
| `explore-correlations` | Correlation + feature importance | Identify leading indicators of donor engagement |
| `geo-enrich` | Census Bureau ACS API | Append demographic variables to student or alumni addresses |
| `survival-kaplan-meier` | Kaplan-Meier | Time-to-degree by demographic group for equity reporting |

**Step Functions workflow** — every job follows the same path:
1. **CheckBudget** — verify the user hasn't exceeded their monthly spend limit
2. **ExtractDataset** — pull the Quick Sight dataset to S3 as Parquet (resolves `claws://` URIs)
3. **RouteCompute** — dispatch to Lambda (nine profiles) or EMR Serverless (Spark)
4. **DeliverResults** — write the output back to Quick Sight as a new dataset
5. **RecordSpend** — update DynamoDB with actual cost; emit CloudWatch metrics
6. **NotifyUser** — SNS notification with the result dataset name

The async model means `compute_run` returns a job ID immediately. Quick Suite's agent
polls `compute_status` and tells the user "your clustering job is running, I'll check
back in a moment" — which matches the real rhythm of analysis work.

## Example Conversations

*"Cluster this year's incoming applicants by academic preparation and financial need,
and tell me how the clusters differ."*

> Quick Suite calls `compute_run` with `profile_id=clustering-kmeans`, the incoming
> class dataset as the source URI, and k=5. The job runs in about 15 seconds. The agent
> calls `compute_status`, gets the result dataset name, and synthesizes a cluster
> summary using the Router extension.

*"Use the IPEDS enrollment data we loaded earlier and forecast fall headcount through
2028."*

> The `claws://roda-ipeds-fall-enrollment` URI resolves automatically via ClawsLookupTable
> (written when `roda_load` ran). Prophet runs a multiplicative seasonality forecast.
> Results land as a new Quick Sight dataset, ready to add to the provost's dashboard.

*"Flag any students in the 2025 cohort whose first-month LMS activity looks like
students who withdrew in prior years."*

> `compute_run` with `profile_id=anomaly-isolation-forest`, `contamination=0.05`.
> The output dataset adds `is_anomaly` and `anomaly_score` columns to every student record.

## Architecture

```
Quick Suite conversation
        │  MCP Actions
        ▼
AgentCore Gateway (Lambda targets)
        │
    ┌───┴──────────────────────────────────┐
    │   compute_run  compute_status        │
    │   compute_profiles                   │
    └───────────────┬──────────────────────┘
                    │  starts execution
                    ▼
        Step Functions State Machine
                    │
        ┌───────────▼───────────┐
        │     CheckBudget       │
        │     ExtractDataset    │  ←── resolves claws:// URIs
        │     RouteCompute      │
        │       ├─ Lambda       │  ←── 9 of 10 profiles
        │       └─ EMR Serverless│ ←── transform-spark only
        │     DeliverResults    │
        │     RecordSpend       │
        │     NotifyUser        │
        └───────────────────────┘
```

## Deploy

```bash
git clone https://github.com/scttfrdmn/quick-suite-compute.git
cd quick-suite-compute

uv sync   # or: pip install -r requirements.txt

uv run cdk synth    # validate — prints the synthesized CloudFormation
uv run cdk deploy   # deploy Lambda-backed profiles

# To also enable the Spark profile (requires EMR Serverless):
uv run cdk deploy --context enable_emr=true
```

**Lambda deployment model:** Most profiles deploy as standard zip-packaged Lambdas.
Two profiles are exceptions:

- **`forecast-prophet`** — Prophet requires C++ compilation (PyStan), which exceeds
  the 250 MB zip limit. This profile deploys as a **container image Lambda** (up to
  10 GB). The CDK stack handles this automatically, but your account must have ECR
  available and the initial deploy will push a Docker image. Memory is set to 6 GB
  by default; large time series datasets may still approach the 15-minute Lambda
  timeout ceiling.

- **`transform-spark`** — Runs on EMR Serverless, not Lambda. Requires
  `--context enable_emr=true` at deploy time.

After deploying, register the three tool Lambdas as AgentCore Gateway Lambda targets.
Get all ARNs from the CloudFormation output:

```bash
aws cloudformation describe-stacks \
  --stack-name QuickSuiteCompute \
  --query 'Stacks[0].Outputs[?OutputKey==`ToolArns`].OutputValue' \
  --output text
```

To connect the clAWS `claws://` URI resolution, set the `claws_resolver_arn` context
variable at deploy time:

```bash
uv run cdk deploy --context claws_resolver_arn=arn:aws:lambda:us-east-1:123456789012:function:qs-data-claws-resolver
```

## Deployment Options

```bash
uv run cdk deploy                                          # standard
uv run cdk deploy --context enable_emr=true               # enable transform-spark profile
uv run cdk deploy --context monthly_budget_usd=100        # raise per-user budget limit (default: $50)
uv run cdk deploy --context max_concurrent_jobs=5         # raise concurrent job limit (default: 2)
uv run cdk deploy --context claws_resolver_arn=arn:...    # enable claws:// URI resolution
```

## Cost

| Profile | Typical Input | Duration | Per-Job Cost |
|---------|--------------|----------|-------------|
| K-Means Clustering | 50K rows × 20 cols | ~15 sec | ~$0.01 |
| Logistic Regression | 100K rows × 50 cols | ~30 sec | ~$0.01 |
| Prophet Forecast | 5 years monthly data | ~45 sec | ~$0.01 |
| Text Topic Modeling | 10K documents | ~2 min | ~$0.02 |
| Geographic Enrichment | 8K addresses | ~15 min | ~$0.05 |
| Spark Join (EMR) | 50M rows × 3 datasets | ~8 min | ~$0.50 |

Infrastructure idle cost: < $5/month.

Monthly per-user budget: $50 by default (configurable). Users who hit the limit
receive a `budget_exceeded` status — no job starts, no cost is incurred.

## License

Apache-2.0 — Copyright 2026 Scott Friedman
