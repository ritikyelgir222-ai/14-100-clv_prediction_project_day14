"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project reuses the same "Online Retail" dataset (UCI:
archive.ics.uci.edu/dataset/352) already used for Day 5 (Customer
Segmentation), Day 10 (Demand Forecasting), and Day 12 (Market Basket
Analysis) — 541,909 real transaction line-items from a UK-based online
gift retailer, Dec 2010-Dec 2011.

WHY THE SAME DATASET AGAIN, AND WHY THAT'S A FEATURE NOT A SHORTCUT
------------------------------------------------------------------------
This is deliberate, continuing the pattern from Days 10 and 12: the
same real transaction warehouse can answer four genuinely different
business questions with four genuinely different techniques —
segmentation (clustering), demand forecasting (time-series regression),
market basket analysis (association rule mining), and now customer
lifetime value (predictive regression). A real company's data warehouse
gets asked all four of these questions from the same underlying tables,
so showing that range on ONE real dataset is more realistic than a
fresh toy dataset per technique.

THE METHODOLOGICAL DIFFERENCE THAT MAKES THIS DAY GENUINELY NEW: every
prior use of this dataset (Day 5's RFM segmentation, Day 12's basket
rules) described PAST behavior over the WHOLE observation period. This
project instead splits each customer's timeline into two windows: an
EARLY "observation" window (what we'd actually know about a customer
today) and a LATER "prediction" window (what they're worth over the
following months) — and trains a model to predict the second from the
first. This is "predictive CLV," a fundamentally forward-looking task
that Day 5's purely descriptive RFM segmentation never attempted.

If you're following along on UCI/Kaggle: download "Online Retail.xlsx"
and place it at `data/online_retail.csv` (converted to CSV) — identical
schema to the file already included here.
"""

import pandas as pd

RAW_DATA_PATH = "data/online_retail.csv"

# WHY THIS SPECIFIC CUTOFF DATE: the dataset spans exactly 373 days
# (2010-12-01 to 2011-12-09). June 1, 2011 splits this into a clean
# ~6-month observation window and a ~6-month prediction window --
# roughly the midpoint, and long enough on both sides for meaningful
# RFM-style features (observation side) and a meaningful future-value
# target (prediction side). An uneven split (e.g. 2 months observation,
# 10 months prediction) would either starve the feature side of signal
# or the target side of a realistic forecasting horizon.
OBSERVATION_CUTOFF = pd.Timestamp("2011-06-01")


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", parse_dates=["InvoiceDate"], date_format="%m/%d/%y %H:%M")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    obs = df[df["InvoiceDate"] < OBSERVATION_CUTOFF]
    pred = df[df["InvoiceDate"] >= OBSERVATION_CUTOFF]

    print(f"Loaded {len(df)} transaction line-items, {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"\nObservation window (< {OBSERVATION_CUTOFF.date()}): {len(obs)} rows")
    print(f"Prediction window (>= {OBSERVATION_CUTOFF.date()}): {len(pred)} rows")

    obs_customers = set(obs.dropna(subset=["CustomerID"])["CustomerID"].unique())
    pred_customers = set(pred.dropna(subset=["CustomerID"])["CustomerID"].unique())
    returned = obs_customers & pred_customers

    print(f"\nCustomers active in observation window: {len(obs_customers)}")
    print(f"Of those, still active in prediction window (\"returned\"): {len(returned)} ({len(returned)/len(obs_customers):.1%})")
    print(f"Churned (zero spend in prediction window): {len(obs_customers) - len(returned)} ({1 - len(returned)/len(obs_customers):.1%})")
