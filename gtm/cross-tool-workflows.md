# Cross-Tool Workflows — All Three Components Together

These workflows show Quick Suite's agent orchestrating all three extensions
in a single conversation: Open Data tools to find and load data, Compute
tools to analyze it, and the Model Router to interpret and communicate results.

The agent calls tools in sequence without the user managing the pipeline.
The user describes what they want; the agent figures out which tools to call
and in what order.

```
roda_search / s3_browse          → find data
roda_load / s3_load              → land it in Quick Sight
compute_run / compute_status     → analyze it
analyze / summarize / generate   → write it up
```

---

## University Administration

---

### 1. IPEDS Peer Benchmarking

**Who asks this:** IR director, accreditation liaison, provost office
**What they type:**
> "Compare our 6-year graduation rates to our IPEDS peer group and tell me
> where we stand."

**Agent pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `roda_search` | `{"query": "IPEDS graduation rates higher education"}` → finds NCES IPEDS dataset |
| 2 | `roda_load` | Loads IPEDS Graduation Rates survey component into Quick Sight |
| 3 | `s3_load` | Loads institutional enrollment/graduation records from institutional S3 |
| 4 | `compute_run` | `retention-cohort` on institutional data → computes comparable cohort metric |
| 5 | `compute_run` | `explore-correlations` → identifies which peer institution attributes (selectivity, size, Pell %) correlate with graduation rate gaps |
| 6 | `summarize` | Sends peer comparison table to Model Router → produces plain-language narrative |
| 7 | `generate` | "Draft the student success section of our HLC self-study based on these results" |

**What comes back:**
- Your 6-year rate vs. peer median, flagged by demographic group
- The 3 institutional characteristics most associated with the gap
- A draft paragraph suitable for the accreditation self-study

**Why it matters:** This workflow replaces 2–3 days of manual IPEDS data pulls,
Excel pivot tables, and a writing session. The agent does all of it.

---

### 2. Enrollment Forecast with Economic Context

**Who asks this:** Enrollment management VP, CFO
**What they type:**
> "Forecast our enrollment for the next 3 years and pull in state-level
> demographic and economic data to contextualize the projection."

**Agent pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `roda_search` | `{"query": "Census population projections college-age"}` → finds ACS projections |
| 2 | `roda_search` | `{"query": "BLS unemployment state level"}` → finds BLS LAUS data |
| 3 | `roda_load` | Loads both public datasets |
| 4 | `s3_load` | Loads 15 years of historical enrollment by term |
| 5 | `compute_run` | `forecast-prophet` on institutional enrollment (base projection) |
| 6 | `compute_run` | `explore-correlations` → correlates enrollment changes with unemployment rate and 18-year-old population in feeder states |
| 7 | `analyze` | Model Router synthesizes: "Your enrollment is more sensitive to state unemployment than peer institutions — here's why that matters for the 3-year forecast" |
| 8 | `generate` | Drafts executive summary for board presentation |

**What comes back:**
- Prophet forecast with confidence intervals
- External factor sensitivity analysis
- Board-ready narrative tying macro trends to enrollment projections

---

### 3. Advancement Prospect Research

**Who asks this:** Major gifts officer, advancement analytics director
**What they type:**
> "I need to identify which mid-level donors in our database are most likely
> to be capable of a major gift, and I want Census wealth indicators to
> supplement our internal data."

**Agent pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `s3_browse` | Lists available donor data sources in institutional S3 |
| 2 | `s3_preview` | Inspects schema of donor history file — confirms columns available |
| 3 | `s3_load` | Loads donor records into Quick Sight |
| 4 | `compute_run` | `geo-enrich` → appends ACS median income, wealth proxies to donor addresses |
| 5 | `compute_run` | `regression-glm` (logistic) → predicts `upgraded_to_major_gift` from enriched features |
| 6 | `compute_run` | `clustering-kmeans` → segments the high-probability group by giving history and wealth indicators |
| 7 | `summarize` | Model Router produces gift officer briefing: "Cluster A — 83 donors, suburban high-income ZIP codes, 10+ year giving history, never personally solicited — are your highest-priority major gift prospects" |

**What comes back:**
- Each donor scored for major gift probability
- Segments within the high-probability pool with different outreach strategies
- Gift officer briefing ready to share with development team

---

## Academic Research

---

### 4. Environmental Justice Study

**Discipline:** Public health, environmental science, sociology
**Who asks this:** PI or postdoc working on an NIH/EPA-funded study

**Researcher types:**
> "I have a dataset of childhood asthma hospitalizations in our study region.
> I want to enrich it with EPA air quality data and Census demographics, then
> model the relationship between pollution exposure and hospitalization rates
> by neighborhood income level."

**Agent pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `roda_search` | `{"query": "EPA air quality monitoring PM2.5 ozone", "quicksight_compatible": true}` |
| 2 | `roda_load` | Loads EPA AQS (Air Quality System) data for the study region/years |
| 3 | `s3_load` | Loads the researcher's de-identified hospitalization records from lab S3 |
| 4 | `compute_run` | `geo-enrich` → appends Census tract, median income, % poverty to each hospitalization record |
| 5 | `compute_run` | `explore-correlations` → correlates pm25_mean, o3_mean, pct_poverty with hospitalization_rate by tract |
| 6 | `compute_run` | `regression-glm` (linear) → models hospitalization_rate ~ pm25 + o3 + median_income + pct_poverty + interaction terms |
| 7 | `analyze` | Model Router interprets coefficients: "PM2.5 has a significant dose-response (β=0.34, p<0.001). The interaction term shows the effect is 2.1× larger in low-income tracts." |
| 8 | `generate` | "Draft the statistical methods and results sections for this analysis" |

**What comes back:**
- Correlation matrix identifying key exposure-outcome relationships
- Adjusted regression coefficients with confidence intervals
- Draft methods and results text in NIH grant / manuscript style

**Scientific value:** This is a complete preliminary analysis for an R01 or
R21 application — the kind that takes a postdoc 2–3 weeks to assemble manually.

---

### 5. Multi-Site Clinical Cohort Analysis

**Discipline:** Medicine, clinical research, epidemiology

**Researcher types:**
> "We have patient outcomes data from three hospital sites. Join them,
> check for site-level anomalies in the data, then run survival analysis
> comparing outcomes by treatment protocol and site."

**Agent pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `s3_browse` | Lists three site data directories in the research S3 bucket |
| 2 | `s3_preview` | Inspects schema at each site — confirms column alignment, flags discrepancies |
| 3 | `compute_run` | `transform-spark` → joins three site files on patient_id with site prefix to disambiguate IDs |
| 4 | `compute_run` | `anomaly-isolation-forest` → flags anomalous records (impossible values, duplicate IDs, outlier lab values) for data cleaning |
| 5 | `compute_run` | `survival-kaplan-meier` → KM curves by treatment_protocol, grouped by site |
| 6 | `analyze` | Model Router: "Site C shows significantly worse outcomes in the control arm (log-rank p=0.003). This may reflect patient population differences or protocol drift — here are the specific features that distinguish Site C patients." |
| 7 | `generate` | Drafts the data quality and statistical analysis sections |

**What comes back:**
- Cleaned, joined multi-site dataset in Quick Sight
- Data quality report with flagged anomalies to review
- Stratified survival curves with log-rank tests
- Draft methods section noting the site-level heterogeneity

---

### 6. Literature-Informed Survey Analysis

**Discipline:** Psychology, education research, behavioral economics

**Researcher types:**
> "I have responses from 1,200 participants on our new financial stress scale.
> Identify the latent topics, validate the subscale structure by checking
> correlations, and then write up a brief interpretation of what we found."

**Agent pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `s3_load` | Loads survey response file from lab S3 |
| 2 | `compute_run` | `text-topics` (NMF, 8 topics) on open-ended financial stress responses |
| 3 | `compute_run` | `explore-correlations` (Spearman) on all 24 Likert items vs. `financial_stress_total` outcome |
| 4 | `compute_run` | `clustering-kmeans` → clusters respondents by full item profile to identify latent subgroups |
| 5 | `research` | Model Router's research tool: "What does the literature say about the dimensionality of financial stress scales? Are 8 topics consistent with published factor structures?" |
| 6 | `summarize` | Synthesizes topic model output + correlation results + literature context |
| 7 | `generate` | Drafts instrument validation section for journal submission |

**What comes back:**
- Topic model with labeled themes and proportions
- Item-level correlations ranked by predictive validity
- Respondent segments (useful for norm-referencing)
- Literature context from the model router's research step
- Draft results section grounded in both the data and prior work

---

### 7. Ecological Change Detection

**Discipline:** Ecology, conservation biology, climate science

**Researcher types:**
> "I have 20 years of bird count data from our field sites. Pull in NOAA
> climate data for the same region, look for anomalous years in both datasets,
> then model how temperature and precipitation predict species richness."

**Agent pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `roda_search` | `{"query": "NOAA climate temperature precipitation historical"}` |
| 2 | `roda_load` | Loads NOAA Global Historical Climatology Network (GHCN) data |
| 3 | `s3_load` | Loads bird count records from researcher's S3 |
| 4 | `compute_run` | `anomaly-isolation-forest` on climate data → flags anomalous years (drought years, unusual frost events) |
| 5 | `compute_run` | `forecast-prophet` on species richness time series → separates trend from seasonal variation |
| 6 | `compute_run` | `explore-correlations` → correlates annual temperature anomaly, precip deficit with richness change |
| 7 | `compute_run` | `regression-glm` (linear) → models richness ~ temp_anomaly + precip_deficit + site_id + year |
| 8 | `analyze` | "Temperature anomaly explains 34% of variance in species richness (R²=0.34). The trend component from the forecast shows a 0.8 species/year decline independent of climate variation." |
| 9 | `generate` | Drafts results and discussion sections; flags the anomalous years for Methods |

**What comes back:**
- Flagged climate anomaly years (useful as covariates or exclusion criteria)
- Detrended species richness time series
- Regression model with effect sizes
- Draft manuscript sections

---

## Why This Architecture Enables These Workflows

Each workflow above is **impossible with any single tool** — it requires
finding external data, loading institutional or study data, running
statistical analysis, and producing human-readable output. Quick Suite's
agent orchestrates all four steps because all three extensions register as
MCP tools in the same AgentCore Gateway instance.

| Need | Component | Tools |
|------|-----------|-------|
| External public data | Open Data | `roda_search`, `roda_load` |
| Institutional / study data | Open Data | `s3_browse`, `s3_preview`, `s3_load` |
| Statistical analysis | Compute | `compute_profiles`, `compute_run`, `compute_status` |
| Interpretation & writing | Model Router | `analyze`, `summarize`, `generate`, `research` |

The agent decides the sequence. The user describes the goal.
