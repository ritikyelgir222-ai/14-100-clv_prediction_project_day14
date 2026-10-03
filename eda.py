"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CHECKS: this EDA exists to answer two questions that
directly determine the modeling approach in Phase 8 -- (1) how severe is
the zero-inflation problem (do most customers actually churn, or is this
a minor edge case?), and (2) do observation-window RFM-style features
actually separate future high-value customers from future churners, or
is the whole premise of predictive CLV unsupported by this data?
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data
from clean_and_engineer import clean_transactions, build_observation_features, build_target, engineer_features


def run_eda(out_dir: str = "outputs"):
    raw = load_raw_data()
    cleaned = clean_transactions(raw)
    obs_features = engineer_features(build_observation_features(cleaned))
    target = build_target(cleaned, obs_features["CustomerID"])
    df = obs_features.merge(target, on="CustomerID")

    print("=== The zero-inflation problem ===")
    print(f"Churn rate (future_spend = 0): {(df['future_spend']==0).mean():.1%}")
    print(f"Future spend distribution among RETURNING customers only:")
    print(df[df["will_return"] == 1]["future_spend"].describe())
    # WHY WE SPLIT THIS VIEW: a naive describe() on the full target
    # mixes two very different populations (people who spend $0 because
    # they left, and people who spend anywhere from $1 to $195,015) --
    # looking at returners alone shows the actual future_spend
    # distribution is ALSO heavily right-skewed even after removing the
    # zero-inflation, meaning this target has two separate statistical
    # challenges stacked on top of each other, not just one.

    print("\n=== Do observation-window features predict WHO returns? ===")
    print(df.groupby("will_return")[["obs_frequency", "recency_days", "tenure_days"]].mean().round(2))
    # WHY THIS TABLE MATTERS MOST: if returners and churners looked
    # identical on observation-window features, there would be no
    # learnable signal for the classification half of the two-part model
    # this project builds -- checking this BEFORE building the model is
    # the same discipline as Day 9's G1-bucket check validating that an
    # early grade actually predicted final risk.

    print("\n=== Correlation between obs_monetary_log and future_spend (returners only) ===")
    returners = df[df["will_return"] == 1]
    corr = returners[["obs_monetary_log", "future_spend"]].corr().iloc[0, 1]
    print(f"Pearson correlation: {corr:.3f}")

    # ---- Chart ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    df["future_spend"].apply(lambda x: 0 if x == 0 else 1).value_counts().plot(
        kind="bar", ax=axes[0], color=["#C44E52", "#4C72B0"], title="Churned (0) vs. Returned (1)"
    )

    df.groupby("will_return")["recency_days"].mean().plot(
        kind="bar", ax=axes[1], color=["#C44E52", "#4C72B0"], title="Avg observation-window recency by outcome"
    )
    axes[1].set_ylabel("Days since last purchase (obs window)")

    axes[2].scatter(returners["obs_monetary_log"], returners["future_spend"].clip(upper=10000), alpha=0.3, s=10)
    axes[2].set_title("obs_monetary_log vs. future_spend (returners, clipped)")
    axes[2].set_xlabel("log(1 + observation-window spend)")
    axes[2].set_ylabel("Future spend ($, clipped at 10k for readability)")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    run_eda()
