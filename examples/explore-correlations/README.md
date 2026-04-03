# Compute Example: Correlation Analysis

A self-contained quick-suite-compute example. Runs the `explore-correlations`
profile on a small CSV file and verifies the full compute lifecycle.

## What This Shows

- `compute_profiles` — list available profiles
- `compute_run` — submit a correlation analysis job using a direct S3 URI
- `compute_status` — poll until the Step Functions execution completes
- `compute_snapshots` — verify the named result snapshot was recorded

## Data

Uses the E2E test CSV already in the compute bucket
(`e2e-test/inputs/test_data.csv` — 35 rows, columns `x`, `y`, `z`).
Replace `source_uri` with any CSV, TSV, or JSON in an S3 bucket the
extract Lambda can read, or use a `claws://` URI for a registered
Quick Sight dataset.

## Prerequisites

- `QuickSuiteCompute` stack deployed
- `AWS_PROFILE` pointing to the deployment account (default region: us-west-2)
- The E2E test CSV uploaded: run `tests/e2e/` once, or upload manually:
  ```bash
  aws s3 cp /dev/stdin s3://qs-compute-942542972736-us-west-2/e2e-test/inputs/test_data.csv \
    --content-type text/csv <<'EOF'
  x,y,z
  1.0,2.0,3.0
  ...
  EOF
  ```

## Running

```bash
# Via the capstone scenario runner (set region override for us-west-2):
AWS_PROFILE=aws QS_SCENARIO_REGION=us-west-2 \
  python3 -m pytest tests/scenarios/ -v -m scenario -k explore-correlations

# Or directly against the compute E2E suite:
AWS_PROFILE=aws python3 -m pytest tests/e2e/ -v -m e2e -k correlations
```
