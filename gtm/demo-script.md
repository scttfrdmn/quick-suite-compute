# Quick Suite Compute — Demo Script

Audience: IR director, enrollment management VP, or advancement analytics lead.
Time: 12–15 minutes. Live in Quick Suite chat interface.

---

## Opening (1 min)

"You already have your enrollment and donor data in Quick Suite. Today I want to show
you three things you can do in this chat window that would normally take a data
scientist a week: segment your incoming class, forecast headcount, and flag which
donors are about to lapse — all without writing a line of code."

---

## Demo 1: Segment incoming freshmen (K-Means Clustering)

**User types in Quick Suite:**
> "Segment our incoming freshman class by application attributes to help admissions
> target yield outreach."

**What the agent does:**
1. Calls `compute_profiles` → discovers clustering-kmeans
2. Calls `compute_run` with the enrollment dataset and `{"k": 5, "features": ["sat_score", "gpa", "distance_from_home", "application_round"]}`
3. Calls `compute_status` every 20 seconds (says "I've started the clustering, checking progress...")
4. When done: "Here are 5 student segments. Cluster 2 — high GPA, far from home — has the lowest yield historically. Want me to build a dashboard showing these segments?"

**Points to make:**
- No data engineer needed. No Python. No waiting.
- Results land as a new Quick Sight dataset — immediately usable in dashboards.
- Agent explains the clusters in plain language.

---

## Demo 2: Forecast enrollment headcount (Prophet)

**User types:**
> "Forecast our fall enrollment for the next three years based on historical trends."

**What the agent does:**
1. Calls `compute_profiles` with `{"category": "forecasting"}`
2. Calls `compute_run` with `{"profile_id": "forecast-prophet", "parameters": {"date_column": "term_date", "value_column": "headcount", "forecast_periods": 6, "seasonality_mode": "additive"}}`
3. Returns forecast with confidence intervals
4. "Based on your historical enrollment, I'm projecting 18,450 students this fall — with a 90% confidence interval of 17,800–19,100. The trend shows a slight acceleration in graduate enrollment."

**Points to make:**
- Budget planning conversations backed by real statistical models.
- Seasonal decomposition helps explain the "why" behind trends.
- Confidence intervals communicate uncertainty honestly.

---

## Demo 3: Identify donor lapse risk (Logistic Regression)

**User types:**
> "Which of our annual fund donors are at risk of lapsing this year?"

**What the agent does:**
1. Selects regression-glm with `model_type=logistic`
2. Target: `lapsed_next_year`, features: `["years_giving", "last_gift_amount", "engagement_events", "class_year"]`
3. Returns predictions + feature importance
4. "I've scored all 12,847 donors. The strongest predictor of lapse is fewer than 2 engagement events in the past year (2.4× higher risk). I've created a 'High Lapse Risk' segment with the top 800 donors to prioritize for personal outreach."

**Points to make:**
- Advancement teams can act on this today — no waiting for a data science request.
- Feature importance gives fundraisers talking points.
- Results are a Quick Sight dataset they can filter, export, and push to Salesforce.

---

## Demo 4: Research portfolio intelligence (Clustering + Forecasting)

Audience shift: VPR, associate provost for research, sponsored programs director.

**User types:**
> "Segment our active grant portfolio and show me which clusters are most at risk
> of under-spending in the next 12 months."

**What the agent does:**
1. Calls `compute_run` with clustering-kmeans on the grants dataset:
   `{"k": 4, "features": ["award_amount", "months_remaining", "pct_spent", "fa_rate", "co_investigators"]}`
2. While that runs (20–30 sec): "I've started the clustering. Let me check progress..."
3. When clustering returns: "I found 4 grant clusters. Cluster 1 — 47 NIH awards, large budgets, 18 months remaining — has an average spend rate 22% below target. These are your highest-risk grants for under-spending."
4. Follows up with Prophet on Cluster 1's monthly expenditures:
   `{"profile_id": "forecast-prophet", "parameters": {"date_column": "month", "value_column": "expenditure", "forecast_periods": 12}}`
5. "At the current trajectory, 31 of the 47 grants in this cluster will under-spend by more than 15%. The projected shortfall is $2.3M in recoverable F&A."

**Points to make:**
- Two analyses, one conversation, about 90 seconds of compute.
- The $2.3M F&A recovery figure is the number that gets a VPR's attention.
- Sponsored programs can now prioritize outreach to specific PIs — not "all grants."
- This runs every month; the agent can be asked "what changed since last month?"

---

## Close (1 min)

"Everything you just saw — three different analyses, different statistical methods,
different datasets — happened in one conversation, in under three minutes each,
with no code, no infrastructure, and no data science backlog.

The compute bill for all three demos: about four cents.

Want to talk about what analyses your team runs most often, and which ones would
make the biggest difference if they ran in minutes instead of weeks?"

---

## Objection Handling

| Objection | Response |
|-----------|----------|
| "We already have SAS/SPSS" | "This doesn't replace your tools — it puts the same capability in the conversation layer where your analyst is already working. No switching contexts." |
| "How do we trust the results?" | "The models are standard scikit-learn and statsmodels — same libraries your data scientists use. Diagnostics (R², AUC, p-values) are in the output." |
| "What about data governance?" | "Data never leaves your AWS account. The compute runs in your VPC, results land in your Quick Sight. No third-party model training on your data." |
| "We need custom models" | "This is a starting point. The architecture is open — your data science team can add profiles for your institution-specific models." |
