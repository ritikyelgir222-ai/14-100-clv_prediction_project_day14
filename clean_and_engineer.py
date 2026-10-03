"""
Phases 5 (data quality) & 7 (feature engineering)
------------------------------------------------------
THE CORE STRUCTURAL DECISION FOR THIS PROJECT: every feature must be
computable using ONLY observation-window data (what we'd actually know
about a customer at decision time), and the target must be computed
ONLY from prediction-window data (what actually happened next). Mixing
the two windows for a feature would be a direct, textbook case of
temporal leakage -- more clear-cut than Day 9's G2 timing judgment call,
closer to Day 11's bright-line sub-label exclusion, but for a reason
unique to this project: the entire premise is "predict the future from
the past," so a feature that secretly encodes future information isn't
subtly optimistic, it's a different (and impossible) question entirely.
"""

import numpy as np
import pandas as pd

from data_loader import OBSERVATION_CUTOFF

# ---------------------------------------------------------------------------
# Rows excluded, and why -- same three data quality issues as Day 5's use
# of this dataset (missing CustomerID, cancellations, non-positive
# Quantity/UnitPrice), for the identical reasons documented there.
# ---------------------------------------------------------------------------


def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_observation_features(clean_df: pd.DataFrame, cutoff: pd.Timestamp = OBSERVATION_CUTOFF) -> pd.DataFrame:
    """
    Every column here is computable using ONLY transactions strictly
    before `cutoff` -- i.e. only information a business would actually
    have on the day it decides how much retention budget to spend on
    this customer.
    """
    obs = clean_df[clean_df["InvoiceDate"] < cutoff].copy()

    features = obs.groupby("CustomerID").agg(
        obs_frequency=("InvoiceNo", "nunique"),
        obs_monetary=("TotalPrice", "sum"),
        obs_first_purchase=("InvoiceDate", "min"),
        obs_last_purchase=("InvoiceDate", "max"),
        obs_distinct_products=("StockCode", "nunique"),
        obs_total_items=("Quantity", "sum"),
    ).reset_index()

    # WHY tenure_days AND recency_days ARE BOTH MEASURED RELATIVE TO THE
    # CUTOFF, NOT TO "TODAY": every observation-window feature has to be
    # anchored to the SAME reference point (the cutoff), so that a
    # customer's recency of 10 days means the same thing regardless of
    # exactly which day within the 6-month window they happened to be
    # observed -- this mirrors Day 5's RFM snapshot-date reasoning,
    # applied here to the observation cutoff instead of the dataset's
    # final date.
    features["tenure_days"] = (cutoff - features["obs_first_purchase"]).dt.days
    features["recency_days"] = (cutoff - features["obs_last_purchase"]).dt.days
    features["obs_avg_order_value"] = features["obs_monetary"] / features["obs_frequency"]

    features = features.drop(columns=["obs_first_purchase", "obs_last_purchase"])
    return features


def build_target(clean_df: pd.DataFrame, observation_customers: pd.Series, cutoff: pd.Timestamp = OBSERVATION_CUTOFF) -> pd.DataFrame:
    """
    WHY THE TARGET IS COMPUTED SEPARATELY, NOT MERGED UNTIL THE VERY END:
    keeping feature-building and target-building in two clearly separate
    functions, using clearly separate date filters (< cutoff vs. >=
    cutoff), makes it structurally hard to accidentally leak
    prediction-window data into a feature -- a deliberate code
    organization choice in service of the leakage discipline above, not
    just a style preference.
    """
    pred = clean_df[clean_df["InvoiceDate"] >= cutoff].copy()

    future_spend = pred.groupby("CustomerID")["TotalPrice"].sum()

    target_df = pd.DataFrame({"CustomerID": observation_customers})
    target_df["future_spend"] = target_df["CustomerID"].map(future_spend).fillna(0.0)
    # WHY fillna(0.0), NOT dropping non-returning customers: a customer
    # who churns (makes no purchase in the prediction window) has a
    # perfectly well-defined future value of $0 -- this is real,
    # meaningful information the model needs to learn from, not a
    # missing value to discard. Dropping these rows would silently
    # convert this into "predict spend among customers we already know
    # returned," a much easier and much less useful question than the
    # real business problem (which doesn't know in advance who'll churn).
    target_df["will_return"] = (target_df["future_spend"] > 0).astype(int)

    return target_df


def engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()

    # WHY log1p ON obs_monetary: same heavy right-skew reasoning as
    # every prior monetary feature in this series (Day 2's Amount, Day 5's
    # Monetary, Day 7's credit_amount).
    df["obs_monetary_log"] = np.log1p(df["obs_monetary"])
    return df


def get_feature_columns() -> list:
    return [
        "obs_frequency", "obs_monetary_log", "obs_distinct_products",
        "obs_total_items", "tenure_days", "recency_days", "obs_avg_order_value",
    ]


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_transactions(raw)

    obs_features = engineer_features(build_observation_features(cleaned))
    target = build_target(cleaned, obs_features["CustomerID"])

    full = obs_features.merge(target, on="CustomerID")
    feature_cols = get_feature_columns()

    print(f"Customers with observation-window features: {len(full)}")
    print(f"Feature columns: {feature_cols}")
    print(f"\nTarget summary (future_spend):")
    print(full["future_spend"].describe())
    print(f"\nReturn rate (will_return=1): {full['will_return'].mean():.1%}")

    full.to_csv("outputs/customer_clv_table.csv", index=False)
    print("\nSaved -> outputs/customer_clv_table.csv")
