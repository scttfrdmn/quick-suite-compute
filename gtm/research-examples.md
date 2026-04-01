# Quick Suite Compute — Academic Research Examples

These examples are for **PIs, postdocs, and research teams** — not university
administration. The researcher has study data in their institutional S3 and
wants to find public reference data, analyze it, and get written output, all
in a single Quick Suite conversation.

Each example spans all three extensions: Open Data tools to find and load
public datasets, Compute tools to run the analysis, and the Model Router
to interpret results and draft text.

```
s3_load / roda_load     → load study + public reference data
compute_run             → run the analysis
analyze / generate      → interpret results, draft manuscript sections
```

---

## Life Sciences

### Clinical Trial Endpoint Analysis

**Discipline:** Medicine, public health, clinical research
**Data:** Patient follow-up records with time-on-study, event indicator, treatment arm

**Researcher types:**
> "Run Kaplan-Meier survival analysis comparing overall survival between
> treatment and control arms. Then pull in SEER incidence rates as a
> population reference and draft the results section."

**Pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `s3_load` | Loads de-identified patient follow-up records from lab S3 |
| 2 | `roda_search` | `{"query": "NCI SEER cancer incidence population-based"}` |
| 3 | `roda_load` | Loads SEER reference data for background incidence context |
| 4 | `compute_run` | `survival-kaplan-meier`: duration=days_on_study, event=event_occurred, group=treatment_arm |
| 5 | `analyze` | Model Router interprets KM output: median survival, hazard ratio, clinical significance |
| 6 | `generate` | "Draft the survival analysis results section, comparing our curves to SEER background rates" |

**Output:** KM survival curves with 95% CI, log-rank p-value, hazard ratio,
draft results paragraph citing the SEER comparison — directly citable.

---

### Gene Expression Cluster Discovery

**Discipline:** Molecular biology, bioinformatics, oncology
**Data:** Sample × gene expression matrix (normalized counts or log2 TPM)

**Researcher types:**
> "Cluster these 240 tumor samples by the 50 marker genes. Then search for
> any public TCGA expression datasets I can compare our clusters to, and
> write up what we found."

**Pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `s3_load` | Loads normalized expression matrix from lab S3 |
| 2 | `compute_run` | `clustering-kmeans`: k=4, features=marker genes, standardize=true |
| 3 | `roda_search` | `{"query": "TCGA cancer genome atlas RNA expression"}` |
| 4 | `roda_load` | Loads matched TCGA cohort for external validation |
| 5 | `compute_run` | `explore-correlations`: correlate cluster centroid profiles with TCGA subtypes |
| 6 | `analyze` | Model Router: "Your Cluster 2 aligns with TCGA Basal-like subtype (r=0.81). Cluster 4 has no clear TCGA equivalent — potential novel subtype." |
| 7 | `generate` | Drafts Figure 1 legend and molecular subtyping Methods paragraph |

**Output:** Cluster assignments per sample, TCGA alignment analysis,
draft methods and figure legend text.

---

### Epidemiological Risk Factor Analysis

**Discipline:** Epidemiology, public health
**Data:** Case-control or cohort study with exposures, outcomes, covariates

**Researcher types:**
> "Run logistic regression predicting disease_case from exposure and covariates.
> Then pull CDC behavioral risk factor data so I can compare our sample
> characteristics to the general population."

**Pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `s3_load` | Loads study dataset from research S3 |
| 2 | `roda_search` | `{"query": "CDC BRFSS behavioral risk factor surveillance"}` |
| 3 | `roda_load` | Loads BRFSS reference population data |
| 4 | `compute_run` | `regression-glm` (logistic): target=disease_case, features=exposure + covariates, include_interactions=true |
| 5 | `compute_run` | `explore-correlations`: compare demographic distributions — study sample vs. BRFSS population |
| 6 | `analyze` | "Your sample over-represents urban residents by 18% vs. BRFSS. Here's how that affects generalizability of the OR estimate." |
| 7 | `generate` | Drafts Table 2 (adjusted odds ratios) and the Limitations paragraph on sample representativeness |

**Output:** Adjusted ORs with CI and p-values, sample representativeness
comparison, draft Table 2 and Limitations text.

---

## Social Sciences

### Interview and Focus Group Analysis

**Discipline:** Qualitative sociology, anthropology, education research, public policy
**Data:** Transcribed interviews or focus groups, one row per respondent

**Researcher types:**
> "Identify the main themes in these 400 food insecurity interviews.
> Then pull in USDA food access data to map theme prevalence against
> local food environment, and write a summary for our policy brief."

**Pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `s3_load` | Loads interview transcripts from research S3 |
| 2 | `roda_search` | `{"query": "USDA food access research atlas food desert"}` |
| 3 | `roda_load` | Loads USDA Food Access Research Atlas |
| 4 | `compute_run` | `text-topics` (LDA, 10 topics): text_column=response |
| 5 | `compute_run` | `geo-enrich` → appends food desert flag and food access score to participant ZIP codes |
| 6 | `compute_run` | `explore-correlations` → correlates dominant topic with food_desert_flag and low_access_score |
| 7 | `analyze` | "Topic 3 ('transportation barriers') is 3.4× more prevalent among participants in food deserts. Topic 7 ('cost vs. nutrition') is evenly distributed regardless of food access." |
| 8 | `generate` | Drafts policy brief findings section; flags which themes are place-based vs. universal |

**Output:** Topics with labeled themes, food environment enrichment,
topic-by-geography crosstab, draft policy brief language.

---

### Survey Instrument Validation

**Discipline:** Psychology, education research, organizational behavior
**Data:** Likert-scale survey, one row per respondent

**Researcher types:**
> "Validate our new financial stress scale. Check item correlations,
> identify latent subgroups, and compare our sample's stress levels to
> published Census financial hardship indicators for the same demographics."

**Pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `s3_load` | Loads survey data from lab S3 |
| 2 | `roda_search` | `{"query": "Census American Community Survey financial hardship poverty"}` |
| 3 | `roda_load` | Loads ACS financial hardship variables for demographic comparison |
| 4 | `compute_run` | `explore-correlations` (Spearman): all 24 items vs. financial_stress_total |
| 5 | `compute_run` | `clustering-kmeans`: cluster respondents by full item profile, k=4 |
| 6 | `compute_run` | `geo-enrich` → appends Census tract poverty rate to respondent addresses |
| 7 | `analyze` | "Items 3, 7, and 14 form a tight cluster (r>0.7 pairwise) — candidate subscale. Your high-stress cluster (Cluster 3) maps to Census tracts with >20% poverty rate, suggesting ecological validity." |
| 8 | `generate` | Drafts the Measures section and construct validity discussion for journal submission |

**Output:** Item correlation ranking, respondent segments with Census
validation, draft Measures and Validity sections.

---

## Environmental & Earth Sciences

### Environmental Justice Analysis

**Discipline:** Public health, environmental science, environmental justice
**Data:** Community health outcome records (hospitalizations, disease rates)

**Researcher types:**
> "I have childhood asthma hospitalization rates by Census tract.
> Pull EPA air quality monitoring data, enrich with neighborhood demographics,
> and model the relationship between pollution exposure and hospitalization
> by income level."

**Pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `s3_load` | Loads hospitalization records from research S3 |
| 2 | `roda_search` | `{"query": "EPA air quality monitoring PM2.5 ozone annual"}` |
| 3 | `roda_load` | Loads EPA AQS annual summary data |
| 4 | `compute_run` | `geo-enrich` → appends median_income, pct_poverty, population to each tract |
| 5 | `compute_run` | `explore-correlations` → pm25_mean, o3_mean, pct_poverty vs. hospitalization_rate |
| 6 | `compute_run` | `regression-glm` (linear): hospitalization_rate ~ pm25 + o3 + median_income + pm25×median_income interaction |
| 7 | `analyze` | "PM2.5 effect is 2.1× larger in low-income tracts (interaction β=−0.18, p=0.002). This is consistent with differential exposure and reduced adaptive capacity." |
| 8 | `generate` | Drafts NIH-style Significance section and statistical methods for R01 application |

**Output:** Correlation matrix, regression coefficients with interaction
terms, draft Significance and Methods for grant application.

---

### Ecological Change Detection

**Discipline:** Ecology, conservation biology, climate science
**Data:** Long-term species count or population index records from field sites

**Researcher types:**
> "I have 20 years of bird count data. Pull NOAA climate data for the
> same region, flag anomalous years in both datasets, forecast the
> population trend, and draft the results."

**Pipeline:**

| Step | Tool | Action |
|------|------|--------|
| 1 | `s3_load` | Loads bird count records from field station S3 |
| 2 | `roda_search` | `{"query": "NOAA Global Historical Climatology Network temperature precipitation"}` |
| 3 | `roda_load` | Loads NOAA GHCN daily climate data for study region |
| 4 | `compute_run` | `anomaly-isolation-forest` on climate data → flags drought years, frost events |
| 5 | `compute_run` | `forecast-prophet` on species richness time series → separates trend from seasonal variation |
| 6 | `compute_run` | `regression-glm` (linear): richness ~ temp_anomaly + precip_deficit + anomalous_year_flag |
| 7 | `analyze` | "Trend component shows 0.8 species/decade decline (independent of climate variation). Temperature anomaly explains an additional 22% of inter-annual variance." |
| 8 | `generate` | Drafts Results section; flags anomalous years for Methods as potential confounds |

**Output:** Anomaly-flagged climate years, detrended population forecast,
regression model, draft Results and Methods.

---

## Notes on Research Use

**What this is good for:**
- Exploratory analysis and hypothesis generation
- Standard statistical endpoints (KM curves, logistic regression, correlations)
- Combining public reference datasets with study data for context or validation
- Large-scale qualitative coding
- Sensor / observational data QA/QC
- Draft methods, results, and grant narrative text

**What it is not a substitute for:**
- Confirmatory inference under a pre-registered analysis plan — use your
  statistical software of record for that
- Specialized methods (hierarchical models, Bayesian inference, SEM,
  phylogenetics, spatial autocorrelation) — none of the current profiles cover these
- Peer review of analytical choices — a biostatistician co-investigator is
  still appropriate for high-stakes clinical or policy research

**Data governance:** Study data never leaves your AWS account. No third-party
model training on your data. Appropriate for IRB-approved research with
de-identified or limited datasets under standard AWS BAA/DUA terms.
