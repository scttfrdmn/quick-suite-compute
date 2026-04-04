# University Use Cases by Department

Quick Suite Compute maps specific analytical workflows to institutional
offices. Each use case maps to one or more compute profiles.

---

## Institutional Research (IR)

### IPEDS Retention Reporting
**Profile:** retention-cohort
**Workflow:** Load enrollment records → compute fall-to-fall retention by cohort → export to IPEDS format
**Time savings:** Eliminates 2–3 days of manual SAS/Excel work each cycle
**Output:** Cohort retention matrix, attrition rates, long-form table for submission

### Graduation Rate Equity Analysis
**Profile:** survival-kaplan-meier
**Workflow:** Time-to-degree by first-gen status, Pell eligibility, race/ethnicity
**Accreditation use:** Demonstrates student success outcomes by demographic group
**Output:** Survival curves per group, log-rank test, median time-to-degree

### Enrollment Trend Forecasting
**Profile:** forecast-prophet
**Workflow:** Historical headcount → 3-year projection → budget planning inputs
**Output:** Forecast + confidence intervals + trend/seasonal decomposition

---

## Enrollment Management

### Yield Campaign Segmentation
**Profile:** clustering-kmeans
**Workflow:** Admitted students → cluster by academic profile + geography + financials → personalized outreach
**Typical result:** 5–7 distinct segments with different yield strategies
**ROI:** 2–3% yield improvement on a class of 2,000 = 40–60 additional enrollments

### At-Risk Melt Prediction
**Profile:** regression-glm (logistic)
**Workflow:** Admitted students → predict summer melt probability → prioritize counselor outreach
**Features:** Financial aid package, housing commitment, orientation registration, email engagement
**Output:** Melt probability score per student + feature importance

### Freshman Profile Correlation Analysis
**Profile:** explore-correlations
**Workflow:** Which application attributes correlate with 4-year graduation?
**Output:** Ranked feature pairs + correlation matrix

---

## Advancement / Fundraising

### Donor Lapse Risk Scoring
**Profile:** regression-glm (logistic)
**Workflow:** Annual fund donors → predict lapse probability → prioritize personal outreach
**Features:** Years giving, last gift amount, engagement events, alumni class year
**Output:** Lapse probability per donor + 'high risk' segment for gift officers

### Donor Segment Discovery
**Profile:** clustering-kmeans
**Workflow:** Donor history → identify giving pattern clusters → tailored asks
**Typical segments:** Major gift prospects, loyal small donors, lapsed mid-level, event-engaged

### Alumni Geographic Mapping
**Profile:** geo-enrich
**Workflow:** Alumni addresses → append Census demographics → regional event planning
**Use case:** Identify high-density alumni ZIP codes for regional events
**Output:** Alumni data enriched with median income, population, education level

---

## Academic Affairs / Provost

### Course Evaluation Theme Analysis
**Profile:** text-topics
**Workflow:** NSSE or course evaluation open-ended responses → LDA topics → report to deans
**Input:** 5,000–50,000 text responses
**Output:** 8–12 dominant themes, proportion per department

### Faculty Time-to-Tenure Analysis
**Profile:** survival-kaplan-meier
**Workflow:** Faculty records → time-to-tenure decision → equity analysis by gender/department
**Reporting:** Provost office, AAUP benchmarking

---

## Financial Affairs

### Expense Anomaly Detection
**Profile:** anomaly-isolation-forest
**Workflow:** Procurement transactions → flag unusual patterns → send to internal audit
**Typical contamination rate:** 3–5%
**Output:** Transaction-level anomaly flag + anomaly score for triage

---

## Research Office (Administrative)

### Grant Portfolio Segmentation
**Profile:** clustering-kmeans
**Workflow:** Cluster active grants by PI department, award size, sponsor type, F&A rate, and years remaining
**Use case:** VPR office identifies which grant clusters are at renewal risk; targets pre-award support accordingly
**Output:** Grant segments with cluster profiles — e.g., "Cluster 3: large NIH R01s, 18 months remaining, low no-cost extension rate"

### Grant Expenditure Forecasting
**Profile:** forecast-prophet
**Workflow:** Monthly F&A expenditures per grant → project burn rate → alert sponsored programs when pace is off
**Use case:** Prevent unallowable cost accumulation; proactively identify grants that will under-spend
**Output:** Forecast vs. budget remaining by grant, flagged outliers

### PI Funding Success Correlation
**Profile:** explore-correlations
**Workflow:** PI attributes (years since PhD, prior award count, department, collaboration network size) vs. award rate
**Use case:** Identify which factors actually predict grant success at your institution for targeted faculty development
**Output:** Ranked feature importance against award_received outcome

### Anomalous Grant Expenditures
**Profile:** anomaly-isolation-forest
**Workflow:** Procurement and payroll transactions charged to sponsored accounts → flag statistical outliers
**Use case:** Pre-audit review; identify miscoded charges before sponsor or federal audit
**Output:** Transaction-level anomaly score; top flagged items for sponsored programs review

### Time-to-First-Award by Career Stage
**Profile:** survival-kaplan-meier
**Workflow:** Faculty start date, first submission date, first award date → survival curves by department and rank
**Use case:** NSF ADVANCE reporting; faculty equity analysis; identify where pre-award support is most needed
**Output:** Survival curves by group, log-rank test, median time-to-first-award

### Multi-Source Research Data Integration
**Profile:** transform-spark (requires EMR)
**Workflow:** Join grants system + HR + SIS for unified research personnel view
**Use case:** NSF HERD survey, NIH diversity supplement reporting, faculty productivity analysis
**Output:** Unified record with award history, appointment type, student mentorship counts

### Open-Ended Faculty Survey Analysis
**Profile:** text-topics (LDA/NMF)
**Workflow:** COACHE survey or faculty climate survey open-ended responses → topic discovery
**Use case:** Identify themes in faculty satisfaction/dissatisfaction without manual coding; present findings to Faculty Senate
**Output:** Dominant topics per response, proportions by department or rank, topic-term matrix

---

## Research Computing (Science-Facing)

### Grant Burn Rate and NCE Risk Flagging
**Profile:** grant-portfolio
**Workflow:** Sponsored program expenditure data → burn rate per award → flag awards >90% expended → PI-level rollup
**Use case:** Sponsored programs office pre-fiscal-year review; identify No-Cost Extension candidates; PI outreach prioritization
**Output:** Every transaction row enriched with `burn_rate`, `pct_expended`, `nce_risk`; PI-level summary in diagnostics

### Co-authorship Network and Collaboration Community Detection
**Profile:** network-coauthor
**Workflow:** Publication table with semicolon-separated author lists → weighted co-authorship graph → centrality + communities
**Use case:** Identify faculty who bridge research clusters (high betweenness centrality); map collaboration landscape for strategic partnerships
**Output:** One row per author-publication with `degree_centrality`, `betweenness_centrality`, `community_id`

### Climate Data Ingest → Forecast Pipeline
**Profile chain:** ingest-netcdf → forecast-prophet
**Workflow:** NetCDF4 file in S3 (NOAA GHCN, ERA5, campus station) → flatten to tabular → Prophet forecast
**Use case:** Infrastructure planning reports; NSF data management; campus sustainability tracking
**Output:** Flattened time series + 12–36 month forecast with confidence intervals

### PDF Document Ingest → Sentiment Analysis
**Profile chain:** ingest-pdf-extract → text-sentiment
**Workflow:** PDF document in S3 (accreditation study, NIH narrative, strategic plan) → text per page → VADER sentiment
**Use case:** Identify sections with negative tone before accreditation submission; trend analysis across annual reports
**Output:** One row per qualifying page with `page_number`, `text`, `sentiment`, `compound_score`

### GeoJSON Campus Data → Spatial Analysis
**Profile:** ingest-geojson
**Workflow:** GeoJSON feature collection in S3 (building footprints, campus zones, service area polygons) → flat table with WKT geometry
**Use case:** Join campus GIS data with space utilization or accessibility datasets in Quick Sight
**Output:** One row per feature with all property columns + `geometry_wkt`, optional bbox columns

---

## Custom Analysis

### Run a Validated Institutional Script
**Profile:** custom-python
**Workflow:** Upload a `transform(df)` Python script to S3 → compute_run with `profile_id: custom-python`, `script_uri`
**Use case:** Institutional analytics team has existing data cleaning or normalization scripts they want to run on Quick Suite data without re-architecting them
**Output:** Whatever `transform(df)` returns; script runs in a RestrictedPython sandbox (no network, no subprocess)
**Security:** Script compiled at AST level with RestrictedPython before execution; only pd, numpy, scipy, scikit-learn available

### Describe the Analysis in Plain Language
**Profile:** custom-generated
**Workflow:** Send a natural language `objective` to compute_run → router LLM generates a `transform(df)` script → script stored to S3 → executed in sandbox
**Use case:** Analyst wants a novel computation (rolling average with spike flag, cohort flow Sankey, custom equity metric) but doesn't have a script ready; the LLM writes it
**Audit trail:** Generated script stored at `s3://{bucket}/results/generated-scripts/{uuid}.py`; URI returned in diagnostics so IR team can review the generated code
