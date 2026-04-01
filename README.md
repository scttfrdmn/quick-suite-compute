# Quick Suite Compute

Ephemeral analytics compute for [Amazon Quick Suite](https://aws.amazon.com/quick-suite/) via [Bedrock AgentCore Gateway](https://aws.amazon.com/bedrock/agentcore/).

University analytics teams describe what analysis they want in natural language through Quick Suite's chat interface. The compute extension matches the request to a pre-built profile, runs it on Lambda (or EMR Serverless for Spark), and delivers results back as a Quick Sight dataset — no infrastructure to manage.

## Architecture

```
Quick Suite (Chat Agent)
    │  MCP Actions Integration
    ▼
AgentCore Gateway
    │  Lambda targets
    ▼
compute_run ──────────────► Step Functions State Machine
compute_status               │
compute_profiles             ├─ CheckBudget (DynamoDB spend tracking)
                             ├─ ExtractDataset (Quick Sight → S3 Parquet)
                             ├─ RouteCompute ──► Lambda (9 profiles)
                             │               ──► EMR Serverless (Spark)
                             ├─ DeliverResults (S3 → Quick Sight dataset)
                             ├─ RecordSpend
                             └─ NotifyUser (SNS)
```

**Async model.** `compute_run` returns a job ID immediately. Quick Suite's agent polls `compute_status`. Most jobs complete in 15–60 seconds.

## Analysis Profiles

| Profile | Description | Typical University Use Case |
|---------|-------------|----------------------------|
| `clustering-kmeans` | K-Means segmentation | Segment incoming students for yield campaigns |
| `regression-glm` | Linear / logistic regression | Predict graduation probability from first-semester data |
| `forecast-prophet` | Time series forecasting | Project enrollment 3 years forward for budget planning |
| `retention-cohort` | Semester-by-semester retention matrix | IPEDS reporting, accreditation self-studies |
| `text-topics` | LDA / NMF topic modeling | Classify course evaluation open-ends into themes |
| `anomaly-isolation-forest` | Anomaly detection | Flag unusual financial transactions or grade distributions |
| `transform-spark` | Large-scale join and transform (EMR) | Join SIS + LMS + financial aid for unified analytics |
| `explore-correlations` | Correlation matrix + feature importance | Find leading indicators of donor engagement |
| `geo-enrich` | Census Bureau demographic enrichment | Append ACS variables to student addresses for equity analysis |
| `survival-kaplan-meier` | Time-to-event analysis | Time-to-degree by demographic group for equity reporting |

## Cost

| Profile | Typical Input | Duration | Cost |
|---------|--------------|----------|------|
| K-Means Clustering | 50K rows × 20 cols | 15 sec | ~$0.01 |
| Logistic Regression | 100K rows × 50 cols | 30 sec | ~$0.01 |
| Prophet Forecast | 5 years monthly | 45 sec | ~$0.01 |
| Text Topic Modeling | 10K documents | 2 min | ~$0.02 |
| Spark Join (EMR) | 50M rows × 3 sources | 8 min | ~$0.50 |
| Geographic Enrichment | 8K addresses | 15 min | ~$0.05 |

Infrastructure idle cost: < $5/month.

Monthly per-user compute budget: $50 (configurable).

## Deploy

```bash
# Install dependencies
uv sync   # or: pip install -r requirements.txt

# Validate
uv run cdk synth

# Deploy (Lambda-only profiles)
uv run cdk deploy

# Deploy with EMR Serverless (required for transform-spark)
uv run cdk deploy --context enable_emr=true
```

## Post-Deploy

Register the three AgentCore Gateway Lambda targets after deploying:

```bash
# Get Lambda ARNs from CloudFormation outputs
aws cloudformation describe-stacks \
  --stack-name QuickSuiteCompute \
  --query 'Stacks[0].Outputs[?OutputKey==`ToolArns`].OutputValue' \
  --output text
```

Register each ARN as an AgentCore Gateway Lambda target via the console or API.

## Project Tracking

Issues, milestones, and the project board are in GitHub — see [Issues](https://github.com/scttfrdmn/quick-suite-compute/issues).

## License

Apache-2.0 — Copyright 2026 Scott Friedman
