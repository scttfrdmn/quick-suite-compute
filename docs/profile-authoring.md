# Profile Authoring Guide

How to add a new analysis profile to quick-suite-compute without modifying any core code.

A profile is two things: a JSON definition that describes the analysis to the outside world,
and a Python handler function that implements it. CDK auto-discovers both at deploy time.

---

## Overview

Adding a profile requires three files and one deploy:

| File | What it does |
|---|---|
| `config/profiles/{profile_id}.json` | Declares parameters, I/O schema, cost estimate, and tags |
| `lambdas/profiles/{module}.py` | Implements the handler function |
| `tests/test_profiles.py` | Add profile ID to `EXPECTED_PROFILE_IDS` + handler unit tests |

Then `cdk deploy` uploads the merged `profiles.json` to S3. Tool Lambdas pick it up on next
invocation — no Lambda code change required.

---

## Step 1: Write the profile JSON

Create `config/profiles/{your-profile-id}.json`. All fields are required.

```json
{
  "profile_id": "your-profile-id",
  "display_name": "Human-Readable Name",
  "description": "One-to-three sentence description of what this analysis does and when to use it. This is what the Quick Suite agent reads to decide whether to call this profile.",
  "category": "statistics",
  "backend": "lambda",
  "entrypoint": "your_module.your_handler",
  "parameters": {
    "target": {
      "type": "column",
      "description": "The outcome column"
    },
    "features": {
      "type": "column_list",
      "min_columns": 1,
      "description": "Predictor columns"
    },
    "alpha": {
      "type": "float",
      "default": 0.05,
      "min": 0.001,
      "max": 0.2,
      "description": "Significance level"
    },
    "method": {
      "type": "enum",
      "values": ["pearson", "spearman"],
      "default": "pearson",
      "description": "Correlation method"
    }
  },
  "input_requirements": {
    "min_rows": 10,
    "max_rows": 500000,
    "required_column_types": ["numeric"],
    "column_parameters": ["target", "features"]
  },
  "output_schema": {
    "preserves_input_columns": true,
    "added_columns": [
      {"name": "predicted_value", "type": "float"}
    ],
    "extra_tables": ["diagnostics"]
  },
  "cost_estimate": {
    "typical_duration_seconds": 15,
    "typical_cost_usd": 0.001
  },
  "tags": ["statistics", "your-use-case", "higher-ed"]
}
```

### `category` values

Use one of the established categories for discoverability:

| Category | Use for |
|---|---|
| `prediction` | Regression, classification |
| `segmentation` | Clustering, grouping |
| `statistics` | Hypothesis tests (ANOVA, chi-square) |
| `higher-ed` | IR-specific analyses (equity gap, DFWI, cohort flow) |
| `time-series` | Forecasting, decomposition, change detection |
| `text` | Topic modeling, sentiment, similarity |
| `geospatial` | Spatial aggregation, enrichment |
| `research` | Grant analysis, citation networks |
| `ingest` | Format conversion profiles (NetCDF, GeoJSON, PDF) |
| `custom` | User-supplied or LLM-generated transforms |

### `backend` values

- `"lambda"` — runs in the runner Lambda (default; supports datasets up to ~500K rows)
- `"emr_serverless"` — runs on EMR Serverless (for Spark profiles, requires `enable_emr=true`)

### Parameter types

| Type | Validates as | Notes |
|---|---|---|
| `column` | String; must be a column name in the dataset | |
| `column_list` | List of strings; each must be a column name | `min_columns` enforced |
| `string` | Any string | |
| `string_list` | List of strings | |
| `integer` | Integer; `min`/`max` optional | |
| `float` | Float; `min`/`max` optional | |
| `boolean` | `true`/`false` | |
| `enum` | String; must be in `values` list (case-sensitive) | **Use lowercase values** |

**Important:** Enum values are validated case-sensitively. If you write `"values": ["lda", "nmf"]`,
the user must pass `"lda"` not `"LDA"`. Document this in the `description` field.

### `entrypoint` format

`"module_name.function_name"` — the runner does `importlib.import_module(module_name)` then
calls `getattr(module, function_name)`. Module name must match the filename in `lambdas/profiles/`
(without `.py`).

---

## Step 2: Write the handler function

Create or add to `lambdas/profiles/{module_name}.py`.

### Handler signature

```python
def your_handler(df: pd.DataFrame, parameters: dict) -> tuple[pd.DataFrame, dict]:
    """
    Brief docstring.

    Returns (result_df, diagnostics).
    result_df: the input DataFrame with any new columns appended.
    diagnostics: dict of analysis metadata (coefficients, test stats, etc.)
    """
    ...
    return result_df, diagnostics
```

### What `df` contains

The runner loads the input dataset from S3 (written by the `extract` Step Functions step) as
a pandas DataFrame. Columns are whatever was in the source data. Your handler should:
- Not assume column types — users pass arbitrary data
- Validate required columns exist and raise `ValueError` with a clear message if not
- Handle empty DataFrames gracefully (return early with a `"warning"` key in diagnostics)
- Drop or skip NaN rows as appropriate for your analysis

### What to return

`result_df` must have the same number of rows as `df` (or a documented subset). New columns are
appended. The deliver step writes this to S3 and registers it as a Quick Sight dataset.

`diagnostics` is a plain dict. It is stored in `metadata.json` alongside the results and
returned to the user via `compute_status`. Keep values JSON-serializable (no numpy types —
wrap with `float()`, `int()`, `str()` as needed).

### Dependencies

The runner is a Docker container. Available packages include:
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`
- `prophet`, `lifelines` (time-to-event)
- Standard library

If your profile needs a package not in this list, add it to
`lambdas/runner/requirements.txt` and rebuild the container image (`cdk deploy` handles this
automatically). Check the Dockerfile for the current full dependency list.

For packages with native binaries (e.g. `shapely`, `netCDF4`), test locally with the
container image — some packages need Linux-specific builds.

### Error handling

Raise `ValueError` for user-input errors (wrong column name, insufficient data, invalid
parameter combination). The runner catches `ValueError` and returns it as
`{"error": "...", "profile_id": "..."}` without failing the Step Functions execution.

Other exceptions propagate as Step Functions task failures, which trigger the
`HandleFailure` step and notify the user.

---

## Step 3: Register in tests

In `tests/test_profiles.py`, add your profile ID to `EXPECTED_PROFILE_IDS`:

```python
EXPECTED_PROFILE_IDS = {
    ...
    "your-profile-id",
}
```

The existing `TestProfileJsonSchema` class will automatically validate your JSON file.

Add handler unit tests — at minimum:
- Happy path: correct output shape and new columns present
- Error path: `ValueError` raised on bad input
- Edge case: empty DataFrame handled gracefully

```python
class TestYourHandler:
    def setup_method(self):
        from your_module import your_handler
        self.handler = your_handler

    def test_returns_expected_columns(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0] * 20, "y": [0, 1, 0] * 20})
        result, diag = self.handler(df, {"target": "y", "features": ["x"]})
        assert "predicted_value" in result.columns
        assert "some_diagnostic" in diag

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="target"):
            self.handler(df, {"target": "missing", "features": ["x"]})
```

Run the tests locally before deploying:

```bash
python3 -m pytest tests/test_profiles.py -v
```

---

## Step 4: Deploy

```bash
# From quick-suite-compute/:
cdk deploy

# CDK will:
# 1. Load all JSON files from config/profiles/
# 2. Merge into profiles.json
# 3. Upload to s3://{compute-bucket}/config/profiles.json
# 4. Update the runner container image if requirements.txt changed
```

Tool Lambdas cache profiles in memory per invocation. The cache is invalidated on the next
cold start, so changes propagate within a few minutes.

### Verify deployment

```bash
# Check your profile appears in the list
AWS_PROFILE=aws aws lambda invoke \
  --function-name qs-compute-compute-profiles \
  --region us-west-2 \
  --payload '{"tags":["your-tag"]}' \
  --cli-binary-format raw-in-base64-out /tmp/out.json && cat /tmp/out.json

# Smoke test compute_run
AWS_PROFILE=aws aws lambda invoke \
  --function-name qs-compute-compute-run \
  --region us-west-2 \
  --payload '{
    "profile_id": "your-profile-id",
    "source_uri": "s3://your-test-bucket/your-test-data.csv",
    "user_arn": "arn:aws:iam::ACCOUNT:user/your-user",
    "parameters": {"target": "your_column", "features": ["col1", "col2"]}
  }' \
  --cli-binary-format raw-in-base64-out /tmp/out.json && cat /tmp/out.json
```

---

## Ingest profiles (source-file → tabular)

Ingest profiles operate directly on a file in S3, not on an input DataFrame from a
Quick Sight dataset. They convert external file formats (NetCDF, GeoJSON, PDF, etc.) into
tabular Parquet that subsequent profiles or Quick Sight can use.

### Key differences from standard profiles

1. **Set `min_rows: 0`** in `input_requirements`. The runner's row-count guard runs before
   dispatch — if `min_rows` is 1 (the default), the runner blocks ingest profiles that
   receive an empty placeholder DataFrame before they have a chance to load their real data.

2. **Ignore `df`** — the handler receives an empty DataFrame. Read your data from
   `parameters["source_uri"]` using boto3:

   ```python
   def my_ingest_handler(df: pd.DataFrame, parameters: dict) -> tuple[pd.DataFrame, dict]:
       source_uri = parameters["source_uri"]          # e.g., "s3://bucket/file.nc"
       bucket, key = source_uri[5:].split("/", 1)
       body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
       # ... parse body, build result_df ...
       return result_df, diagnostics
   ```

3. **Set `preserves_input_columns: false`** — ingest profiles produce a new schema, not
   additions to the input.

### Profile JSON snippet for an ingest profile

```json
{
  "profile_id": "ingest-myformat",
  "category": "ingest",
  "backend": "lambda",
  "entrypoint": "ingest.myformat_handler",
  "input_requirements": {
    "min_rows": 0,
    "max_rows": 0
  },
  "output_schema": {
    "preserves_input_columns": false,
    "added_columns": []
  }
}
```

---

## Chaining two profiles (`chain_profile_id`)

A caller can chain two profiles by passing `chain_profile_id` to `compute_run`. The Step
Functions execution runs profile 1, then feeds its output DataFrame directly into profile 2
as input — no intermediate export or re-upload required.

This works especially well for ingest → analytics pipelines:

```
compute_run(
  profile_id: "ingest-netcdf",
  source_uri: "s3://my-bucket/climate.nc",
  parameters: {variables: ["tmax"]},
  chain_profile_id: "forecast-prophet",
  chain_parameters: {date_column: "time", value_column: "tmax", forecast_horizon: 24}
)
```

`compute_status` surfaces `step: "profile_2"` while the second profile is running and
includes `total_cost_usd` (sum of both profiles) in the SUCCEEDED response.

**Design constraint:** Both profiles must have compatible types. The first profile's
output DataFrame must satisfy the second profile's `min_rows` and column requirements.
The runner validates parameters for the chain profile at submission time.

---

## Worked example: adding a `factor-analysis` profile

**1. JSON:** `config/profiles/factor-analysis.json`

```json
{
  "profile_id": "factor-analysis",
  "display_name": "Factor Analysis",
  "description": "Reduces a set of survey or Likert-scale items to latent factors using scikit-learn FactorAnalysis. Returns factor loadings and per-record factor scores. Use to validate survey instruments or identify underlying constructs in NSSE/course evaluation data.",
  "category": "statistics",
  "backend": "lambda",
  "entrypoint": "statistics.factor_analysis_handler",
  "parameters": {
    "features": {"type": "column_list", "min_columns": 3, "description": "Survey item columns"},
    "n_factors": {"type": "integer", "default": 3, "min": 1, "max": 20, "description": "Number of latent factors to extract"},
    "rotation": {"type": "enum", "values": ["varimax", "none"], "default": "varimax", "description": "Factor rotation method"}
  },
  "input_requirements": {"min_rows": 50, "max_rows": 500000, "required_column_types": ["numeric"], "column_parameters": ["features"]},
  "output_schema": {
    "preserves_input_columns": true,
    "added_columns": [{"name": "factor_{n}", "type": "float"}],
    "extra_tables": ["loadings"]
  },
  "cost_estimate": {"typical_duration_seconds": 15, "typical_cost_usd": 0.001},
  "tags": ["factor-analysis", "survey", "dimensionality-reduction", "nsse", "course-evaluation"]
}
```

**2. Handler:** add `factor_analysis_handler` to `lambdas/profiles/statistics.py`

**3. Tests:** add `"factor-analysis"` to `EXPECTED_PROFILE_IDS` and a `TestFactorAnalysisHandler` class

**4. Deploy:** `cdk deploy`

Total time from idea to deployed: ~30 minutes.
