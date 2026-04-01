# Cost Comparison: Quick Suite Compute vs. Alternatives

## Per-Analysis Cost

| Analysis Type | Quick Suite Compute | Data Scientist (in-house) | Vendor / Consulting |
|--------------|-------------------|--------------------------|-------------------|
| K-Means Clustering (50K rows) | ~$0.01 | $200–400 (2–4 hrs × $100/hr) | $2,000–5,000 |
| Logistic Regression (100K rows) | ~$0.01 | $200–400 | $2,000–5,000 |
| Time Series Forecast | ~$0.02 | $400–800 (4–8 hrs) | $5,000–10,000 |
| Text Topic Modeling (10K docs) | ~$0.02 | $800–1,600 (8–16 hrs) | $10,000–25,000 |
| Cohort Retention Matrix | ~$0.01 | $400–800 | $5,000+ |
| Anomaly Detection (transactions) | ~$0.01 | $200–400 | $3,000–8,000 |
| Geographic Enrichment (200K rows) | ~$0.05 | $600–1,200 | $8,000–15,000 |
| Survival Analysis | ~$0.01 | $400–800 | $5,000–10,000 |

**Monthly infrastructure cost at idle: < $5**

---

## Time-to-Insight Comparison

| Analysis Type | Quick Suite Compute | Traditional Workflow |
|--------------|-------------------|---------------------|
| One-off analysis | 1–3 minutes | 1–2 weeks (backlog + execution + review) |
| Repeated analysis (monthly IPEDS) | 1 minute | 2–3 days (manual re-run) |
| New dataset exploration | 5 minutes | 1 week (data access + code) |
| Board presentation prep | 30 minutes for 5 analyses | 4–6 weeks |

---

## Staffing Cost Context

A typical R1 university IR shop runs 3–5 FTEs at $70–120K salary + benefits.
The team's analytical bandwidth is consumed by:
- 40% recurring reporting (IPEDS, accreditation, state reports)
- 30% ad hoc requests from deans/provost
- 20% data cleaning and integration
- 10% strategic projects

Quick Suite Compute targets the 30% ad hoc tier — not replacing IR staff, but
eliminating the backlog so staff can focus on interpretation and strategy.

**Conservative estimate:** If Quick Suite Compute handles 50% of ad hoc requests
(~15% of total IR capacity), the freed analyst time is worth $30,000–60,000/year
at a mid-size university. Annual cost of Quick Suite for 50 power users: $24,000.

---

## Competitive Landscape

| Solution | Compute Cost | Data Scientist Required | Leaves AWS | Setup |
|----------|-------------|------------------------|-----------|-------|
| **Quick Suite Compute** | < $0.05/analysis | No | No | CDK deploy |
| Databricks (hosted) | $0.50–5.00/job | Yes (notebooks) | Yes | Weeks |
| SageMaker Studio | $0.10–1.00/job | Yes (Python) | No | Days |
| Vendor BI platform add-ons | $1–10/analysis | Varies | Yes | Months |
| In-house R/Python scripts | $0 compute | Yes | No | Years to build |
| External consulting | $200–500/analysis | No (but slow) | Yes | Per engagement |

---

## Total Cost of Ownership (3-Year Projection, 200-User Institution)

| Cost Category | Quick Suite Compute | Comparable Alternative |
|--------------|-------------------|----------------------|
| Infrastructure (3 yr) | $180 | $50,000–100,000 (Databricks/SageMaker) |
| Quick Suite licenses (200 power users × $40 × 36 mo) | $288,000 | — (already purchased) |
| Staff time saved (conservative) | ($180,000) | $0 |
| Analysis turnaround improvement | priceless | — |
| **Net 3-year value** | **+$180,000** | — |

*Quick Suite licenses are assumed pre-existing (purchased for model router + open data).*
*Infrastructure costs are AWS Lambda + Step Functions at stated workload.*
