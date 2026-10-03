# Project Documentation: Customer Lifetime Value Prediction
### Technical Documentation & Handover — Phase 12 of the SDLC

Companion to `SDLC_DOCUMENTATION.md` and `README.md`.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Predict future customer value to prioritize retention spend |
| **Client context** | SaaS/Retail — reuses the Days 5/10/12 retail transaction warehouse |
| **Data source** | Online Retail dataset — [UCI: archive.ics.uci.edu/dataset/352](https://archive.ics.uci.edu/dataset/352/online+retail) |
| **Design** | Observation window (6 months) → features; Prediction window (~6 months) → target |
| **Winning approach** | Two-part model (return classifier × spend-if-return regressor) — a judgment call, not a clean win |
| **Test MAE** | $798.10 |
| **Business framing** | Rank customers by predicted future value; measure revenue capture at a fixed retention-team capacity |

---

## 2. Architecture

```
data/online_retail.csv (same file as Days 5, 10, 12)
        │
        ▼
data_loader.py ──────► loads data, defines OBSERVATION_CUTOFF (2011-06-01)
        │
        ▼
clean_and_engineer.py ─► build_observation_features() -- ONLY uses data
        │                BEFORE cutoff
        │                build_target() -- ONLY uses data AT/AFTER cutoff
        │                (kept as separate functions specifically to make
        │                 leakage structurally hard to introduce)
        ▼
   split.py ───────────► stratified customer-level split (on will_return)
        │
        ▼
train_model.py ───────► TWO PARALLEL APPROACHES, compared honestly:
        │                (1) naive single regressor on raw future_spend
        │                (2) two-part: classify P(return), regress
        │                    log(spend) on returners only, multiply
        ▼
outputs/return_classifier.joblib, spend_regressor.joblib, naive_model.joblib
        │
        ├──────────────► app.py ─── serves all 3 numbers (P(return),
        │                           spend-if-return, combined CLV)
        │
        └──────────────► monitor.py ─ monitors classifier AND regressor
                                       health SEPARATELY
```

---

## 3. Data Dictionary

**Observation-window features** (computed only from transactions before 2011-06-01):

| Feature | Formula | Rationale |
|---|---|---|
| `obs_frequency` | Distinct invoices in observation window | Purchase occasions, same reasoning as Day 5's RFM Frequency |
| `obs_monetary` / `obs_monetary_log` | Total spend in observation window (raw / log1p) | Same skew-handling as Day 5/7/12's monetary features |
| `obs_distinct_products` | Unique StockCodes purchased | Breadth of engagement |
| `obs_total_items` | Total quantity purchased | Volume of engagement |
| `tenure_days` | Days from first purchase to cutoff | How long they've been a customer, as of the decision point |
| `recency_days` | Days from last purchase to cutoff | How recently active, as of the decision point |
| `obs_avg_order_value` | obs_monetary / obs_frequency | Basket-size signal, independent of frequency |

**Target (computed only from transactions at/after 2011-06-01):**

| Target | Formula | Notes |
|---|---|---|
| `future_spend` | Total spend in prediction window | 0 for churned customers — a real, meaningful value, not missing data |
| `will_return` | 1 if future_spend > 0 | The classification half of the two-part model |

---

## 4. Key EDA Findings

| Finding | Detail |
|---|---|
| Churn rate | 28.7% of observation-window customers have $0 future spend |
| Future spend among returners | Still heavily right-skewed: $7.50 to $195,015, median $833 |
| Recency signal | Churners' observation-window recency averaged 83.8 days vs. 54.4 for returners — a real, meaningful gap |
| Frequency signal | Churners averaged 1.45 observation-window orders vs. 3.25 for returners |
| obs_monetary_log ↔ future_spend correlation (returners only) | 0.341 — real but moderate; not everything about future spend is predictable from early behavior |

---

## 5. Modeling Results

### Head-to-head comparison (validation set)

| Approach | RMSE | MAE | Top-20% Revenue Capture |
|---|---|---|---|
| Naive single regressor | 5979.46 | 1065.57 | **71.0%** |
| Two-part model | 6619.19 | **1059.55** | 70.4% |

**Selected: two-part model, by MAE** — but explicitly NOT a clean win.
It loses on both RMSE and revenue capture. MAE was chosen as the
primary criterion because it's less dominated by a handful of extreme
high-spend outliers than RMSE; a business prioritizing "never badly
underestimate our biggest whales" would reasonably pick the naive
model instead. This is disclosed as a judgment call in `train_model.py`,
not presented as an obvious technical victory.

### Test set (final, honest read)

| Metric | Value |
|---|---|
| Test RMSE | 1862.17 |
| Test MAE | 798.10 |
| Test top-20% revenue capture | 63.11% |
| Test cohort actual total future revenue | $533,687.56 |

### What drives each half of the model

**Return classifier** (top coefficients): `obs_avg_order_value` (-1.08,
i.e. surprisingly, larger average orders correlate with LOWER return
probability — plausibly one-off large purchases rather than habitual
shopping), `obs_total_items` (+0.55), `obs_distinct_products` (+0.46),
`recency_days` (-0.44, as expected).

**Spend-if-return regressor** (top importances): `obs_monetary_log`
dominates (0.60), consistent with the EDA correlation finding —
customers who already spend more tend to keep spending more, given
that they return at all.

---

## 6. API Reference

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser test form |
| `/docs` | GET | Interactive Swagger UI |
| `/health` | GET | Health check |
| `/score` | POST | Score a customer's observation-window summary |

**Response (three numbers, not one):**
```json
{
  "predicted_clv": 1074.09,
  "p_will_return": 0.9104,
  "expected_spend_if_return": 1179.75,
  "top_factors": ["obs_monetary_log", "obs_total_items", "obs_avg_order_value"]
}
```

Verified with two constructed profiles: a high-value customer (10
orders, $2,000 spend, recent) scored $1,074 predicted CLV with 91%
return probability; a low-value/at-risk customer (1 order, $30 spend,
150 days since last purchase) scored $55 predicted CLV — but with an
$621.89 expected-spend-if-return, illustrating exactly why the
three-number breakdown matters: this customer's low CLV is driven
almost entirely by low return probability (8.9%), not low potential
value if retained.

---

## 7. Monitoring & Maintenance Plan

`monitor.py` checks the return classifier's AUC and the spend
regressor's MAE **separately**, since a two-part model can degrade in
either half independently — a genuinely different monitoring shape
than any single-model project in this series.

**Recommended cadence: roughly once per prediction-window length (~6
months)** — the longest feedback lag in the series, longer even than
Day 7's loan-default maturation lag, since this model's ground truth
isn't fully known until an entire future window has played out.

---

## 8. Known Limitations

1. **Single 6-month/6-month window, one retailer, one year.** Not validated across different window lengths or a different business.
2. **The two-part "win" is a judgment call**, not a clean technical victory — it loses on RMSE and revenue capture, disclosed explicitly rather than hidden.
3. **The spend regressor is optimized in log-space on returners only**, which can underweight the very largest future spenders relative to a model optimized directly for total-dollar accuracy — a real trade-off given this project's MAE-based selection.
4. **The longest ground-truth lag in the series** — real-world monitoring can't validate this model's predictions until 6 months have elapsed, unlike Day 2's near-immediate fraud outcomes.
5. **No test of alternative window splits** (e.g., 3-month/9-month, or a rolling multi-window approach) — a single fixed cutoff was used throughout.

---

## 9. File Map

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load data, define the observation/prediction cutoff |
| `eda.py` | 6 | Zero-inflation and signal-validation checks |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Observation features, prediction-window target, leakage discipline |
| `split.py` | 7 | Stratified customer-level split |
| `train_model.py` | 8–9 | Naive vs. two-part model, honest comparison |
| `app.py` | 10 | FastAPI service, three-number response |
| `monitor.py` | 13 | Separate classifier/regressor health monitoring |
| `README.md` | 12 | Setup instructions, comparison to Days 5/10/12 |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation |
| `SDLC_DOCUMENTATION.md` | 1–14 | Full 14-phase narrative |
