# Customer Lifetime Value Prediction — Day 14 (SaaS/Retail, ML)

Full project code for Day 14 of the 100-day series. Reuses the same
real **Online Retail** dataset (UCI: archive.ics.uci.edu/dataset/352)
already used for Day 5 (Segmentation), Day 10 (Demand Forecasting), and
Day 12 (Market Basket Analysis) — continuing the deliberate pattern of
showing how one real data warehouse answers genuinely different
business questions with genuinely different techniques.

## What makes this day methodologically new

Days 5, 10, and 12 all described **past** behavior over a fixed window.
This project predicts **future** value from **early** behavior — the
first genuinely forward-looking task on this dataset:

- **Observation window**: Dec 2010–May 2011 (6 months) → features
- **Prediction window**: Jun–Dec 2011 (~6 months) → target (`future_spend`)

Every feature is computable using only observation-window data; the
target uses only prediction-window data. Mixing the two would be
temporal leakage — the entire premise is "predict what we don't yet
know from what we do," so this discipline isn't optional.

## The central modeling challenge: zero-inflation

**28.7% of customers churn entirely** (future_spend = $0), and among
the customers who *do* return, spend is itself heavily right-skewed
($7.50 to $195,015). A single regressor has to learn both "will they
come back" and "how much if so" as one blended number — a harder
problem than it needs to be.

This project builds and **honestly compares** two approaches:

| Approach | Val RMSE | Val MAE | Val Top-20% Revenue Capture |
|---|---|---|---|
| Naive single regressor | 5979.46 | 1065.57 | **71.0%** |
| Two-part model (classify return → regress spend-if-return) | 6619.19 | **1059.55** | 70.4% |

**Neither approach cleanly wins.** The two-part model was selected by
MAE, but explicitly loses on RMSE and revenue capture — reported as a
genuine judgment call in `train_model.py`, not smoothed into a clean
victory for the more sophisticated-sounding approach.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py          # Phase 5 — loads data, documents the observation/prediction split
python eda.py                   # Phase 6 — zero-inflation check, does obs data predict return?
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — observation features, prediction-window target
python train_model.py          # Phase 8-9 — naive vs. two-part model, honest comparison
python monitor.py              # Phase 13 — monitors BOTH halves of the two-part model separately
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/** — the response shows three numbers
(P(return), spend-if-return, combined CLV), not just one, since a
retention team needs to know *why* a customer scored the way they did.

## File map

| File | SDLC Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Loads data, defines the observation/prediction cutoff |
| `eda.py` | 6 | Zero-inflation check, validates observation features predict return |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Observation-window features, prediction-window target — strict leakage discipline |
| `split.py` | 7 | Stratified customer-level split (no further time axis available) |
| `train_model.py` | 8–9 | Naive regressor vs. two-part model, honestly compared |
| `app.py` | 10 | FastAPI service returning all three CLV components |
| `monitor.py` | 13 | Monitors classifier AND regressor health separately |

## Known limitations (stated honestly)

- Single 6-month/6-month window from one retailer, one year — not validated across multiple window-length choices.
- The two-part "winner" is a judgment call (MAE-optimized), not a clean win — it loses on RMSE and revenue capture.
- The spend regressor is trained in log-space on returners only, which can underweight the largest future spenders relative to a model optimized directly for total-dollar accuracy.
- Ground truth takes the full ~6-month prediction window to mature — the longest feedback lag of any project in this series, longer than Day 7's loan-default lag.
