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

**Seven tools** in Quick Suite's chat interface:

| Tool | What it does |
|------|-------------|
| `compute_profiles` | List available analysis types with their inputs, outputs, costs, and durations |
| `compute_run` | Validate the request, check the monthly budget, and start a job; returns estimated cost and duration before starting |
| `compute_status` | Check whether a job is running, succeeded, or failed; returns results and cost when done |
| `compute_history` | List recent jobs for a user (most recent first) |
| `compute_cancel` | Abort a running job |
| `compute_snapshots` | List named result snapshots (set `result_label` in `compute_run` to create one) |
| `compute_compare` | Diff two named snapshots: added/removed/unchanged row counts and schema diff |

**31 analysis profiles** grouped by category:

**Statistics**
| Profile | Method | Use Case |
|---------|--------|---------|
| `anova` | One-way/N-way ANOVA | Course grade disparity by section; survey scale comparison |
| `chi-square` | Chi-square test | FAFSA completion rate by demographic; admission decision distribution |

**Prediction / ML**
| Profile | Method | Use Case |
|---------|--------|---------|
| `regression-glm` | GLM | Predict award amount; linear outcome modeling |
| `regression-logistic` | Logistic regression | Predict graduation probability; donor lapse risk |
| `classification-random-forest` | Random Forest | Multi-class classification; feature importance ranking |

**Forecasting**
| Profile | Method | Use Case |
|---------|--------|---------|
| `forecast-prophet` | Prophet time series | Project enrollment 3 years forward; research expenditure trends |
| `change-detection` | Ruptures change point | Detect structural breaks in enrollment or retention trends |
| `seasonality-decompose` | STL decomposition | Extract trend and seasonal components from time series |

**Clustering**
| Profile | Method | Use Case |
|---------|--------|---------|
| `clustering-kmeans` | K-Means | Segment incoming students for yield strategy; donor prospect grouping |

**Text Analytics**
| Profile | Method | Use Case |
|---------|--------|---------|
| `text-topics` | LDA / NMF topic modeling | Analyze 40,000 course evaluation comments without manual coding |
| `text-sentiment` | VADER | Score sentiment per page of policy documents or survey responses |
| `text-similarity` | TF-IDF cosine | Near-duplicate detection in survey responses or grant narratives |

**Anomaly Detection**
| Profile | Method | Use Case |
|---------|--------|---------|
| `anomaly-isolation-forest` | Isolation Forest | Flag at-risk students by LMS engagement; unusual financial transactions |

**Higher-Ed Specific**
| Profile | Method | Use Case |
|---------|--------|---------|
| `retention-cohort` | Cohort matrices | IPEDS retention reporting; accreditation self-studies |
| `cohort-flow` | Enrollment funnel | Application → admission → enrollment → persistence flow analysis |
| `dfwi-analysis` | D/F/W/I rates | Grade distribution equity across sections, instructors, demographics |
| `equity-gap` | Outcome disparity | Equity gap with effect sizes for accreditation and federal reporting |
| `peer-benchmark` | Rank comparison | Rank your institution against a peer set on any metric |
| `survival-kaplan-meier` | Kaplan-Meier | Time-to-degree by demographic group for equity reporting |

**Geospatial**
| Profile | Method | Use Case |
|---------|--------|---------|
| `geo-enrich` | Census Bureau ACS | Append demographic variables to student or alumni addresses |
| `isochrone` | Catchment area | Compute service area membership for facilities or events |
| `spatial-aggregate` | Point-in-polygon | Aggregate student addresses by census tract or ZIP code |

**Exploration**
| Profile | Method | Use Case |
|---------|--------|---------|
| `explore-correlations` | Correlation + feature importance | Identify leading indicators of donor engagement |

**Research**
| Profile | Method | Use Case |
|---------|--------|---------|
| `grant-portfolio` | Burn rate + NCE risk | Flag awards at risk before fiscal year close; PI-level rollup |
| `network-coauthor` | Graph centrality | Co-authorship network; identify collaboration bridge authors |

**Ingest** (source-file → tabular; no input data required)
| Profile | Method | Use Case |
|---------|--------|---------|
| `ingest-netcdf` | xarray flatten | Convert NetCDF4 climate or scientific data to tabular Parquet |
| `ingest-pdf-extract` | pypdf | Extract text page by page from PDF documents in S3 |
| `ingest-geojson` | shapely + WKT | Convert GeoJSON feature collections to a flat table |

**Custom**
| Profile | Method | Use Case |
|---------|--------|---------|
| `custom-python` | RestrictedPython sandbox | Run your own `transform(df)` script from S3 |
| `custom-generated` | LLM code generation | Describe what you want; the router writes and runs the script |

**Transform**
| Profile | Method | Use Case |
|---------|--------|---------|
| `transform-spark` | Spark (EMR Serverless) | Join SIS + LMS + financial aid at scale |

**Additional features:**

- **Chained profiles** — set `chain_profile_id` in `compute_run` to run two profiles in sequence in a single Step Functions execution; the first profile's output feeds directly into the second as input
- **Named snapshots** — set `result_label` to save a result under a memorable name; retrieve with `compute_snapshots`; diff two snapshots with `compute_compare`
- **Pre-submission cost estimate** — `compute_run` returns `estimated_cost_usd` and `estimated_duration_seconds` before starting the job
- **clAWS URI resolution** — `source_uri: "claws://roda-ipeds-fall-enrollment"` resolves to a Quick Sight dataset via `ClawsLookupTable`
- **Audit log** — every terminal job path writes an immutable record to `s3://compute-results/audit/{year}/{month}/{job_id}.json`
- **VPC isolation** — `enable_vpc=true` CDK context places all runner Lambdas in isolated subnets with S3 Gateway endpoint
- **KMS encryption** — `enable_kms=true` encrypts HistoryTable and results bucket with customer-managed keys

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

After deploying, register the tool Lambdas as AgentCore Gateway Lambda targets.
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
uv run cdk deploy                                             # standard
uv run cdk deploy --context enable_emr=true                  # enable transform-spark profile
uv run cdk deploy --context monthly_budget_usd=100           # raise per-user budget limit (default: $50)
uv run cdk deploy --context max_concurrent_jobs=5            # raise concurrent job limit (default: 2)
uv run cdk deploy --context claws_resolver_arn=arn:...       # enable claws:// URI resolution
uv run cdk deploy --context enable_vpc=true                  # place all runner Lambdas in isolated VPC
uv run cdk deploy --context enable_kms=true                  # KMS-encrypt HistoryTable and results bucket
uv run cdk deploy --context router_spend_table_arn=arn:...   # cross-stack spend ledger for budget pre-check
uv run cdk deploy --context router_invoke_arn=arn:...        # router Lambda ARN for custom-generated profile
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
