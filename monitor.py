"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY THIS PROJECT NEEDS TWO SEPARATE MONITORING CHECKS, NOT ONE: a
two-part model can degrade in two independent ways -- the RETURN
classifier's AUC could decay (harder to tell who'll come back) while
the SPEND regressor stays accurate, or vice versa. Monitoring only the
combined prediction's error could mask which half actually broke,
making it much harder to know what to fix. This is a genuinely
different monitoring shape than any single-model project in this series.

WHY THE RIGHT CADENCE HERE IS TIED TO THE PREDICTION WINDOW LENGTH,
UNLIKE ANY PRIOR DAY: this model's ground truth (did they actually
return, how much did they actually spend) only becomes fully known
after an entire ~6-month prediction window has elapsed -- even longer
than Day 7's loan-maturation lag. Retraining more often than every few
months wouldn't add genuinely new ground truth to learn from.
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import roc_auc_score, mean_absolute_error

from data_loader import load_raw_data
from clean_and_engineer import clean_transactions, build_observation_features, build_target, engineer_features, get_feature_columns
from split import split_data

AUC_DROP_THRESHOLD = 0.05
MAE_INCREASE_THRESHOLD_PCT = 0.20


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    classifier = joblib.load("outputs/return_classifier.joblib")
    regressor = joblib.load("outputs/spend_regressor.joblib")
    scaler = joblib.load("outputs/scaler.joblib")
    config = joblib.load("outputs/model_config.joblib")
    feature_cols = config["feature_cols"]

    raw = load_raw_data()
    cleaned = clean_transactions(raw)
    obs_features = engineer_features(build_observation_features(cleaned))
    target = build_target(cleaned, obs_features["CustomerID"])
    full = obs_features.merge(target, on="CustomerID")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(full, feature_cols)

    print("=== Part A: Return classifier health ===")
    X_test_scaled = scaler.transform(X_test)
    p_return_test = classifier.predict_proba(X_test_scaled)[:, 1]
    current_auc = roc_auc_score(y_test["will_return"], p_return_test)
    print(f"Current AUC: {current_auc:.4f}")

    print("\n=== Part B: Spend regressor health (returners only) ===")
    returners_test = y_test["will_return"] == 1
    spend_pred = np.expm1(regressor.predict(X_test[returners_test]))
    current_mae = mean_absolute_error(y_test.loc[returners_test, "future_spend"], spend_pred)
    print(f"Current MAE (returners only): {current_mae:.2f}")

    print("\n=== Feature drift (PSI): key features, train vs. test ===")
    for feat in ["obs_monetary_log", "recency_days"]:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")

    with open("outputs/business_validation.json") as f:
        baseline = json.load(f)

    print("\n=== Retrain triggers ===")
    baseline_mae = baseline["test_mae"]
    print(f"NOTE: baseline_mae ({baseline_mae}) is the COMBINED model's MAE, not Part B's")
    print("returners-only MAE computed above -- these are different quantities measuring")
    print("different things, both worth tracking, and shouldn't be directly compared.")

    print(f"\n✅ Monitoring check complete. In production, compare current_auc and")
    print(f"current_mae against THEIR OWN previously-recorded baselines (not shown")
    print(f"here, since this run recomputes on the same historical test set rather")
    print(f"than genuinely new data -- see the cadence note below).")

    print("\n=== Recommended monitoring cadence ===")
    print("Roughly once per prediction-window length (~6 months here), the longest")
    print("cadence in the series so far -- longer even than Day 7's loan-maturation")
    print("lag. Ground truth for this model isn't fully known until an entire future")
    print("window has played out, so more frequent retraining wouldn't have new")
    print("outcomes to learn from.")


if __name__ == "__main__":
    run_monitoring_check()
