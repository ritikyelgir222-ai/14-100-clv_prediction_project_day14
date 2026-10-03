# SDLC Documentation: Customer Lifetime Value Prediction
### SaaS/Retail Industry | ML | Day 14 of the 100-Day Series
### Filled against the master 14-phase SDLC template, using the real project we built

Every number below comes from the actual pipeline run on the real UCI
Online Retail dataset — the fourth distinct project built on this same
data warehouse (after Days 5, 10, and 12).

---

## Phase 1: Discovery & Stakeholder Requirement Gathering
**Owner:** Business Analyst / Data Scientist | **Output:** Meeting notes, stakeholder map

- **Stakeholders identified:** VP of Customer Success (problem owner), Finance (budget approver for retention spend), retention/account management team (day-to-day users)
- **Discovery findings:**
  - Current process: retention budget spread evenly, or driven by whoever complains loudest, rather than by actual future value
  - Decision this project informs: which customers get proactive retention attention, ranked by predicted future value
  - Critical framing established immediately: this must be PREDICTIVE (using only early behavior), not descriptive (Day 5's RFM segmentation already covers "who has been valuable so far") — the whole point is prioritizing spend on customers whose future is still uncertain
- **Deliverable — Problem Statement:** "Predict each customer's value over the next several months using only their first few months of behavior, so retention budget can be prioritized before it's too late to act — not just measure who has already been valuable."

---

## Phase 2: Business Requirement Document (BRD)
**Owner:** Business Analyst | **Output:** BRD

- **Business objective:** maximize future revenue captured by prioritizing a capacity-limited retention team on the highest-predicted-value customers
- **Scope:** in-scope: customers with at least one purchase in the 6-month observation window; out-of-scope: brand-new customers with no observation-window history at all (nothing to predict from)
- **Success metrics / KPIs:** top-20% revenue capture (what share of actual future revenue comes from the top-predicted quintile) as the primary business metric, alongside MAE/RMSE as technical metrics
- **Assumptions & constraints:** single 6-month/6-month window from one retailer's one year of data; 28.7% of customers churn entirely, a real constraint on what "accuracy" can mean here
- **Sign-off:** N/A — self-directed portfolio project

---

## Phase 3: Functional & Technical Requirement Document (FRD/TRD)
**Owner:** Data Scientist / Tech Lead | **Output:** FRD/TRD

- **Functional requirements:** given a customer's observation-window summary, return a predicted future value AND its two components (return probability, spend-if-return) — not just one blended number, since the two components suggest different retention tactics
- **Non-functional requirements:** sub-second scoring (confirmed in testing)
- **Data sources:** a single static CSV in this exercise, the same file used for Days 5/10/12; a real system would compute observation-window features from a live transactions table on a rolling basis
- **Integration points:** none in this version — a production version would integrate with a CRM to surface predicted CLV alongside each customer record

---

## Phase 4: Project Planning
**Owner:** Project/Delivery Manager | **Output:** Project plan, risk register

- **Build sequence used:** data loading (defining the cutoff) → EDA (validating the observation/prediction split has real signal) → feature/target engineering (strict window separation) → stratified split → naive-vs-two-part modeling → API → dual-model monitoring → documentation
- **Real risks encountered during the build:**
  - Risk: the temptation to pick a modeling window that "looks good" after seeing results — resolved by choosing the cutoff (2011-06-01, the dataset's near-midpoint) BEFORE running any modeling code, based on structural reasoning (a clean 6/6 month split) rather than reverse-engineering a cutoff that flattered the results
  - Risk: accidentally computing a feature using post-cutoff data — mitigated by keeping `build_observation_features()` and `build_target()` as two structurally separate functions with clearly separate date filters, making this class of bug harder to introduce by accident, not just something to remember to check for

---

## Phase 5: Data Collection & Data Understanding
**Owner:** Data Engineer / Data Scientist | **Output:** Data dictionary, data quality report

- **Data inventory:** same 541,909-row Online Retail dataset used for Days 5, 10, and 12 — 2,718 customers with observation-window activity after the standard cleaning (missing CustomerID, cancellations, non-positive Quantity/UnitPrice — identical reasoning to Day 5)
- **Data quality report:** the SAME three cleaning issues as Day 5 apply identically here, since it's the same raw data
- **The central data-understanding finding, unique to this project:** 28.7% of customers who were active in the first 6 months generated exactly $0 in the following ~6 months — a real, substantial zero-inflation problem that directly shapes the entire modeling approach in Phase 8
- **Access & governance:** identical to Day 5 — public, already-anonymized dataset

---

## Phase 6: Exploratory Data Analysis (EDA)
**Owner:** Data Scientist | **Output:** `eda.py`, `outputs/eda_summary.png`

**Actual findings:**

- Churn rate: 28.7% (future_spend = $0)
- Future spend among returners: still heavily right-skewed ($7.50 to $195,015, median $833) — a SECOND statistical challenge stacked on top of the zero-inflation, not a separate issue
- Observation-window recency clearly differs by outcome: 83.8 days (churners) vs. 54.4 days (returners) — real, meaningful separation
- Observation-window frequency also differs: 1.45 orders (churners) vs. 3.25 (returners)
- Correlation between obs_monetary_log and future_spend among returners: 0.341 — real but moderate, an honestly-reported "not everything is predictable" finding

**Why this EDA had to answer a DIFFERENT question than every prior day's EDA:** Days 1, 2, 5, 7, 9, 11 all checked whether EXISTING features separated an ALREADY-KNOWN outcome. This EDA instead had to validate that EARLY (observation-window) behavior actually predicts a LATER (prediction-window) outcome — a genuinely forward-looking validation question, closest in spirit to Day 9's G1-bucket check but applied to a continuous, zero-inflated target instead of a binary one.

---

## Phase 7: Data Preprocessing & Feature Engineering
**Owner:** Data Scientist / ML Engineer | **Output:** `clean_and_engineer.py`, `split.py`

- **Cleaning:** identical to Day 5 (same dataset, same issues)
- **THE defining structural decision:** features and target are built by two separate functions, each filtering strictly on one side of the observation cutoff (`< cutoff` for features, `>= cutoff` for the target) — a code-organization choice made specifically to make temporal leakage structurally harder to introduce, not just something to remember to avoid
- **Feature engineering:** `obs_monetary_log` (same skew-handling as every prior monetary feature in this series); `tenure_days` and `recency_days` both anchored to the SAME reference point (the cutoff), mirroring Day 5's RFM snapshot-date discipline
- **Split strategy:** stratified on `will_return` (same category of reasoning as Day 7/9 — no further time axis was available, since the time dimension was already fully spent constructing the observation/prediction split itself)

---

## Phase 8: Model Development
**Owner:** Data Scientist / ML Engineer | **Output:** `train_model.py`, `outputs/experiment_log.csv`

**The defining methodological choice: building TWO fundamentally different modeling approaches, not just multiple model families within one approach** (a different kind of comparison than Day 11's unsupervised-vs-supervised split, but similar in spirit — comparing APPROACHES, not just algorithms):

1. **Naive** — a single regressor predicting `future_spend` directly
2. **Two-part** — classify `P(return)`, regress `log(spend)` among returners only, multiply

**Actual head-to-head (validation set):**

| Approach | RMSE | MAE | Top-20% Revenue Capture |
|---|---|---|---|
| Naive single regressor | 5979.46 | 1065.57 | **71.0%** |
| Two-part model | 6619.19 | **1059.55** | 70.4% |

**The most important honest finding in this project:** the two-part
model — the more methodologically sophisticated-SOUNDING approach —
does NOT cleanly win. It edges out the naive model on MAE but loses on
both RMSE and revenue capture. This is reported as a genuine,
disclosed judgment call (MAE was prioritized because it's less
dominated by extreme outliers), not smoothed into an uncomplicated
victory for the fancier-sounding method — continuing this series'
established pattern (Days 1, 7, 9, 11) of reporting close or mixed
model comparisons honestly rather than picking whichever framing tells
a cleaner story.

---

## Phase 9: Model Evaluation & Business Validation
**Owner:** Data Scientist + Business Stakeholder | **Output:** `outputs/business_validation.json`

- **Technical metrics (test set):** RMSE 1862.17, MAE 798.10
- **Business validation:** top-20% revenue capture of 63.11% on the test cohort ($533,687.56 actual total future revenue) — meaning a retention team focusing on the top-predicted fifth of customers would reach nearly two-thirds of all future revenue from this cohort
- **Bias/fairness check:** not applicable in the demographic sense — no demographic features exist in this transactional dataset (same as Day 5's use of it)
- **Explainability:** the return classifier's coefficients surfaced a genuinely counter-intuitive finding — `obs_avg_order_value` has a NEGATIVE coefficient (-1.08), meaning customers with larger average orders are LESS likely to return, plausibly because a large average order can reflect a one-off purchase (a gift, a bulk order) rather than habitual shopping. This is reported as a real, examined finding, not glossed over because it contradicts the "bigger spenders are better customers" intuition.

---

## Phase 10: MLOps & Deployment
**Owner:** ML Engineer | **Output:** `app.py`

- **Packaging:** FastAPI service (`/score`, `/health`, browser test form)
- **THE key design decision:** the API returns THREE numbers (P(return), spend-if-return, combined CLV), not one — because the two-part model's whole value is in that decomposition, and collapsing it to a single score would hide exactly the kind of nuance Phase 9 surfaced (verified directly: a test low-value profile showed low return probability (8.9%) but a substantial spend-if-return estimate ($621.89), a distinction a single blended score would have erased)
- **Verified with 2 constructed profiles:** a high-value customer scored $1,074 predicted CLV (91% return probability); a low-value/at-risk customer scored $55 predicted CLV, driven almost entirely by low return probability rather than low potential spend

---

## Phase 11: Testing
**Owner:** QA / Data Scientist | **Output:** manual test log

- **What was actually tested:** full pipeline run end-to-end; API tested via FastAPI's `TestClient` with two constructed profiles spanning the value spectrum, both producing sensible, correctly-decomposed results
- **What was NOT done:** no formal unit test suite; no test of alternative observation/prediction window lengths (e.g., 3-month/9-month) to check how sensitive the results are to that specific structural choice; no UAT with an actual customer success team

---

## Phase 12: Documentation & Handover
**Owner:** Data Scientist | **Output:** `README.md`, `PROJECT_DOCUMENTATION.md`, this file

- `README.md` leads with the honest naive-vs-two-part comparison table, since the "neither approach cleanly wins" finding is the single most important thing a reader of this project should understand before trusting either model
- The counter-intuitive `obs_avg_order_value` finding and the three-number API design are both documented explicitly, continuing this series' practice of surfacing genuine engineering and analytical judgment, not just final numbers

---

## Phase 13: Monitoring & Maintenance
**Owner:** MLOps / Data Scientist | **Output:** `monitor.py`

- **What's monitored:** the return classifier's AUC and the spend regressor's MAE, tracked SEPARATELY — a two-part model can degrade in either half independently, and monitoring only the combined prediction's error would obscure which half actually broke
- **Recommended cadence: roughly once per prediction-window length (~6 months)** — the longest feedback lag anywhere in this series, longer even than Day 7's loan-default maturation lag, since this model's ground truth isn't fully known until an entire future window has elapsed

---

## Phase 14: Project Closure & Delivery
**Owner:** N/A (self-directed project) | **Output:** this document, `PROJECT_DOCUMENTATION.md`

- **Closure against original objective:** a working CLV model was built and validated, achieving 63.11% top-20% revenue capture on real held-out data, WITH an honest disclosure that the "winning" two-part approach is a judgment call rather than a clean technical victory over the simpler naive alternative
- **Retrospective:**
  - What went well: choosing the observation/prediction cutoff based on structural reasoning BEFORE seeing any results, and keeping feature/target construction in separate functions with separate date filters, made the leakage discipline this entire project depends on much harder to accidentally violate
  - What to improve next time: test multiple observation/prediction window lengths to see how sensitive the results are to that specific choice; investigate the counter-intuitive `obs_avg_order_value` finding further with domain input, since a negative relationship between order size and return probability has real implications for how a retention team should interpret a customer's early large purchase
- **Handover:** packaged as a complete, runnable project (code + real data + trained models + documentation), consistent with every prior day in the series, and the fourth genuinely different technique built on the same underlying data warehouse

---

## Mapping back to the Day 14 LinkedIn post

| LinkedIn section | Pulled from phases | Real number used |
|---|---|---|
| Client & Problem | 1–2 | SaaS/Retail, prioritizing retention spend by future value |
| Requirements & Data | 3, 5 | Same Online Retail warehouse as Days 5/10/12, 6-month/6-month split |
| Approach | 6–8 | Zero-inflation EDA → naive vs. two-part model, honestly compared |
| Result & Business Impact | 9 | 63.11% top-20% revenue capture; the counter-intuitive order-size finding |
| Path to Production | 10–13 | Three-number scoring API + dual classifier/regressor monitoring |
