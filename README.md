# Quick Suite Compute

**Run statistical analysis on university data from any AgentCore-connected agent — no infrastructure, no code, no waiting for a data engineer.**

A lot of the questions that matter most to a university can't be answered with a chart.
"Which students are most likely to not return next semester?" "What does enrollment look
like in three years if current trends hold?" "Which cluster of donors has the highest
major gift potential?" "Is there a causal effect of this intervention, or just a
correlation?" These require statistical models: regression, forecasting, clustering,
survival analysis, causal inference.

Most agents can visualize data or generate text. They cannot run a k-means clustering job
on 300,000 student records, fit a Prophet forecast, compute Kaplan-Meier survival curves,
or run a difference-in-differences analysis. That's what this extension does. An analyst
describes what analysis they want in natural language, and the Compute extension matches
the request to the right pre-built profile, runs the job in the background, and delivers
the results back as a new Quick Sight dataset — usually in under a minute.

## What most agents can't do without this extension

- Execute statistical models (regression, clustering, forecasting, topic modeling, survival analysis, causal inference, IRT psychometrics)
- Run analysis on data that lives in Quick Sight but hasn't had any computation applied to it
- Accept a `claws://` URI from the Data extension and use it as a job input
- Track monthly compute spend per user and enforce a budget ceiling (including cross-stack Router spend)
- Chain two analysis profiles in a single job (e.g., ingest NetCDF → forecast)
- Run large-scale Spark transformations (joining SIS + LMS + financial aid at 50M-row scale)
- Schedule recurring analysis jobs
- Export results as CSV and Excel with presigned download URLs

## What You Get

**Eight tools** in Quick Suite's chat interface:

| Tool | What it does |
|------|-------------|
| `compute_profiles` | List available analysis types with their inputs, outputs, costs, and durations |
| `compute_run` | Validate the request, check the monthly budget, and start a job; returns estimated cost and duration before starting |
| `compute_status` | Check whether a job is running, succeeded, or failed; returns results, cost, and export URLs when done |
| `compute_history` | List recent jobs for a user (most recent first) |
| `compute_cancel` | Abort a running job (ownership-verified) |
| `compute_snapshots` | List named result snapshots (set `result_label` in `compute_run` to create one) |
| `compute_compare` | Diff two named snapshots: added/removed/unchanged row counts and schema diff |
| `compute_schedule` | Create, list, or delete scheduled compute jobs via EventBridge Scheduler |

**42 analysis profiles** grouped by category:

### Statistics
| Profile | Method | Use Case |
|---------|--------|---------|
| `anova` | One-way/N-way ANOVA | Course grade disparity by section; survey scale comparison |
| `chi-square` | Chi-square test | FAFSA completion rate by demographic; admission decision distribution |

### Prediction / ML
| Profile | Method | Use Case |
|---------|--------|---------|
| `regression-glm` | GLM | Predict award amount; linear outcome modeling |
| `regression-logistic` | Logistic regression | Predict graduation probability; donor lapse risk |
| `classification-random-forest` | Random Forest | Multi-class classification; feature importance ranking |

### Forecasting
| Profile | Method | Use Case |
|---------|--------|---------|
| `forecast-prophet` | Prophet time series | Project enrollment 3 years forward; research expenditure trends |
| `change-detection` | Ruptures change point | Detect structural breaks in enrollment or retention trends |
| `seasonality-decompose` | STL decomposition | Extract trend and seasonal components from time series |

### Clustering
| Profile | Method | Use Case |
|---------|--------|---------|
| `clustering-kmeans` | K-Means | Segment incoming students for yield strategy; donor prospect grouping |

### Text Analytics
| Profile | Method | Use Case |
|---------|--------|---------|
| `text-topics` | LDA / NMF topic modeling | Analyze 40,000 course evaluation comments without manual coding |
| `text-sentiment` | VADER | Score sentiment per page of policy documents or survey responses |
| `text-similarity` | TF-IDF cosine | Near-duplicate detection in survey responses or grant narratives |

### Anomaly Detection
| Profile | Method | Use Case |
|---------|--------|---------|
| `anomaly-isolation-forest` | Isolation Forest | Flag at-risk students by LMS engagement; unusual financial transactions |

### Higher-Ed Specific
| Profile | Method | Use Case |
|---------|--------|---------|
| `retention-cohort` | Cohort matrices | IPEDS retention reporting; accreditation self-studies |
| `cohort-flow` | Enrollment funnel | Application → admission → enrollment → persistence flow analysis |
| `dfwi-analysis` | D/F/W/I rates | Grade distribution equity across sections, instructors, demographics |
| `equity-gap` | Outcome disparity | Equity gap with effect sizes for accreditation and federal reporting |
| `peer-benchmark` | Rank comparison | Rank your institution against a peer set on any metric |
| `survival-kaplan-meier` | Kaplan-Meier | Time-to-degree by demographic group for equity reporting |
| `intersectionality-equity` | Cross-tab disparate impact | Two+ demographic dimensions; DI ratio (80% rule); small-n cell suppression |
| `assessment-irt` | 2PL IRT model | Item difficulty/discrimination; person abilities; item information curves |
| `financial-aid-effectiveness` | Aid-band cohort + IRLS logistic | Aid band persistence probability; unmet-need trends by year |

### Geospatial
| Profile | Method | Use Case |
|---------|--------|---------|
| `geo-enrich` | Census Bureau ACS | Append demographic variables to student or alumni addresses |
| `isochrone` | Catchment area | Compute service area membership for facilities or events |
| `spatial-aggregate` | Point-in-polygon | Aggregate student addresses by census tract or ZIP code |

### Exploration
| Profile | Method | Use Case |
|---------|--------|---------|
| `explore-correlations` | Correlation + feature importance | Identify leading indicators of donor engagement |

### Research
| Profile | Method | Use Case |
|---------|--------|---------|
| `grant-portfolio` | Burn rate + NCE risk | Flag awards at risk before fiscal year close; PI health scores |
| `network-coauthor` | Graph centrality | Co-authorship network; Louvain community detection; bridge authors |
| `causal-iv` | 2SLS instrumental variables | Causal effect estimation with instruments; first-stage F-stat; weak instrument warning |
| `causal-rd` | Regression discontinuity | IK bandwidth; 5-point sensitivity; McCrary manipulation test; fuzzy RD via IV |
| `causal-did` | Difference-in-differences | Parallel trends test; event study coefficients; placebo tests; staggered adoption |
| `grant-pipeline` | PI health scoring | Active/continuity/diversity scores; NCE risk (≤60 days, no overlap); sponsor timing |
| `provenance-graph` | W3C PROV-DM JSON-LD | Data lineage from HistoryTable; gap detection for broken chains |
| `power-analysis` | Literature-informed sample sizing | Cross-reference PubMed effect sizes via Router `extract`; scipy power curves n=2–100 |
| `anomaly-hypothesis` | IsolationForest + Router grounding | Per-anomaly classification (instrument error, known noise, reported effect, novel candidate) |
| `reproducibility-check` | RestrictedPython re-execution | Re-run analysis script against deposited data; compare with configurable tolerance |

### Ingest (source-file → tabular)
| Profile | Method | Use Case |
|---------|--------|---------|
| `ingest-netcdf` | xarray flatten | Convert NetCDF4 climate or scientific data to tabular Parquet |
| `ingest-pdf-extract` | pypdf | Extract text page by page from PDF documents in S3 |
| `ingest-geojson` | shapely + WKT | Convert GeoJSON feature collections to a flat table |

### Custom
| Profile | Method | Use Case |
|---------|--------|---------|
| `custom-python` | RestrictedPython sandbox | Run your own `transform(df)` script from S3 |
| `custom-generated` | LLM code generation | Describe what you want; the router writes and runs the script (AST-validated) |

### Transform
| Profile | Method | Use Case |
|---------|--------|---------|
| `transform-spark` | Spark (EMR Serverless) | Join SIS + LMS + financial aid at scale |

### Additional Features

- **Chained profiles** — set `chain_profile_id` to run two profiles in sequence; the first profile's output feeds the second as input
- **Named snapshots** — set `result_label` to save a result; retrieve with `compute_snapshots`; diff with `compute_compare`
- **CSV/Excel export** — SUCCEEDED jobs include presigned S3 URLs for CSV and XLSX downloads (24h TTL, 1M row limit)
- **Results write-back** — completed jobs auto-register in the Data source registry, discoverable via `federated_search` and clAWS `discover`
- **Pre-submission cost estimate** — `compute_run` returns `estimated_cost_usd` and `estimated_duration_seconds` before starting
- **Cross-stack budget enforcement** — checks both its own monthly budget and the Router spend ledger before submitting jobs
- **Peer cohort matching** — Carnegie class + enrollment ±20% + control type + Pell ±10pt filtering; weighted scoring; 30-day DynamoDB cache
- **clAWS URI resolution** — `source_uri: "claws://roda-ipeds-fall-enrollment"` resolves via ClawsLookupTable
- **Audit log** — every terminal job path writes an immutable record to `s3://compute-results/audit/{year}/{month}/{job_id}.json`
- **VPC isolation** — `enable_vpc=true` places all runner Lambdas in isolated subnets with S3 Gateway endpoint
- **KMS encryption** — `enable_kms=true` encrypts HistoryTable and results bucket with customer-managed keys
- **Scheduled jobs** — `compute_schedule` creates recurring jobs via EventBridge Scheduler with DynamoDB tracking

**Step Functions workflow** — every job follows the same path:
1. **CheckBudget** — verify the user hasn't exceeded their monthly spend limit
2. **ExtractDataset** — pull the Quick Sight dataset to S3 as Parquet (resolves `claws://` URIs)
3. **RouteCompute** — dispatch to Lambda or EMR Serverless
4. **DeliverResults** — write output to Quick Sight + generate CSV/XLSX exports
5. **RecordSpend** — update DynamoDB with actual cost; write snapshot if labeled; register in Data source registry
6. **NotifyUser** — SNS notification with the result dataset name
7. **AuditLog** — write immutable audit record to S3

Chained jobs add a second compute+deliver pass after the first completes.

## Example Conversations

*"Cluster this year's incoming applicants by academic preparation and financial need,
and tell me how the clusters differ."*

> Quick Suite calls `compute_run` with `profile_id=clustering-kmeans`, the incoming
> class dataset as the source URI, and k=5. The job runs in about 15 seconds. The agent
> calls `compute_status`, gets the result dataset name and CSV/XLSX download links, and
> synthesizes a cluster summary using the Router extension.

*"Run a difference-in-differences analysis on our student success initiative — did the
intervention group actually improve, or were they already trending better?"*

> `compute_run` with `profile_id=causal-did`. The output includes parallel trends
> pre-period tests, event study coefficients per time period, and placebo tests.

*"Based on the effect sizes in these three PubMed papers, what sample size do I need for
80% power?"*

> `compute_run` with `profile_id=power-analysis` and `pubmed_ids`. Router `extract` pulls
> reported effect sizes from the papers, scipy computes power curves from n=2 to 100, and
> a confound checklist is drawn from the methods sections.

## Architecture

```
Quick Suite conversation
        │  MCP Actions
        ▼
AgentCore Gateway (Lambda targets)
        │
    ┌───┴──────────────────────────────────┐
    │   compute_run   compute_status       │
    │   compute_profiles  compute_history  │
    │   compute_cancel  compute_snapshots  │
    │   compute_compare compute_schedule   │
    └───────────────┬──────────────────────┘
                    │  starts execution
                    ▼
        Step Functions State Machine
                    │
        ┌───────────▼───────────┐
        │     CheckBudget       │
        │     ExtractDataset    │  ←── resolves claws:// URIs
        │     RouteCompute      │
        │       ├─ Lambda       │  ←── 41 profiles
        │       └─ EMR Serverless│ ←── transform-spark
        │     DeliverResults    │  ←── CSV/XLSX export
        │     RecordSpend       │  ←── write-back to Data registry
        │     NotifyUser        │
        │     AuditLog          │
        │     [HasChainProfile] │  ←── optional second pass
        └───────────────────────┘
```

## Deploy

```bash
git clone https://github.com/scttfrdmn/campus-compute.git
cd campus-compute

uv sync   # or: pip install -r requirements.txt

uv run cdk synth    # validate
uv run cdk deploy   # deploy
```

## Deployment Options

```bash
uv run cdk deploy                                             # standard
uv run cdk deploy --context enable_emr=true                  # enable transform-spark profile
uv run cdk deploy --context monthly_budget_usd=100           # per-user budget limit (default: $50)
uv run cdk deploy --context max_concurrent_jobs=5            # concurrent job limit (default: 2)
uv run cdk deploy --context claws_resolver_arn=arn:...       # enable claws:// URI resolution
uv run cdk deploy --context enable_vpc=true                  # VPC isolation
uv run cdk deploy --context enable_kms=true                  # KMS encryption
uv run cdk deploy --context router_spend_table_arn=arn:...   # cross-stack Router spend check
uv run cdk deploy --context router_invoke_arn=arn:...        # Router Lambda for custom-generated + power-analysis
uv run cdk deploy --context compute_allowed_buckets=b1,b2    # restrict source_uri S3 buckets
uv run cdk deploy --context data_registry_table=arn:...      # results write-back to Data source registry
```

## Security

- **compute_cancel ownership check**: verifies `user_arn` against execution input; same error message on mismatch (no info leak)
- **Budget conditional write backstop**: `record-spend` uses `ConditionExpression` to prevent overspend race conditions
- **source_uri bucket allowlist**: `COMPUTE_ALLOWED_BUCKETS` restricts which S3 buckets can be used as job inputs
- **result_label validation**: `^[A-Za-z0-9_\-]{1,64}$` enforced in compute-run and record-spend
- **Parameter max-length**: 256-character guard on string parameters
- **DynamoDB protection**: all 4 tables have deletion protection, PITR, and RETAIN removal policy
- **Log retention**: 90-day retention on all Lambda log groups
- **Scoped SFN role**: explicit `qs-compute-sfn-role` with `lambda:InvokeFunction` only on state machine Lambdas
- **RestrictedPython sandbox**: `_SafePandasProxy` blocks URL-based reads; AST static analysis gates LLM-generated code

## Cost

| Profile | Typical Input | Duration | Per-Job Cost |
|---------|--------------|----------|-------------|
| K-Means Clustering | 50K rows × 20 cols | ~15 sec | ~$0.01 |
| Logistic Regression | 100K rows × 50 cols | ~30 sec | ~$0.01 |
| Prophet Forecast | 5 years monthly data | ~45 sec | ~$0.01 |
| Text Topic Modeling | 10K documents | ~2 min | ~$0.02 |
| Causal DiD | 200K rows × 3 periods | ~20 sec | ~$0.01 |
| Geographic Enrichment | 8K addresses | ~15 min | ~$0.05 |
| Spark Join (EMR) | 50M rows × 3 datasets | ~8 min | ~$0.50 |

Infrastructure idle cost: < $5/month.

Monthly per-user budget: $50 by default (configurable). Users who hit the limit
receive a `budget_exceeded` status — no job starts, no cost is incurred.

## License

Apache-2.0 — Copyright 2026 Scott Friedman
