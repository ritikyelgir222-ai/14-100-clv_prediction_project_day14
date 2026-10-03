"""
Phase 10: MLOps & Deployment
--------------------------------
WHY THIS ENDPOINT RETURNS THREE NUMBERS, NOT JUST ONE: unlike every
prior day's single risk score, this project's winning model is a
TWO-PART combination -- serving only the final blended number would
hide exactly the kind of nuance Phase 9 surfaced (a customer with a
high combined CLV could get there via "certain to return, modest
spender" or "uncertain to return, huge spender if they do," and a
retention team would want to treat those very differently). Exposing
p_return and expected_spend_if_return separately, alongside the
combined estimate, lets a human make that judgment.

Run with:  uvicorn app:app --reload
"""

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Customer Lifetime Value Scoring API", version="1.0")

CLASSIFIER = joblib.load("outputs/return_classifier.joblib")
REGRESSOR = joblib.load("outputs/spend_regressor.joblib")
SCALER = joblib.load("outputs/scaler.joblib")
CONFIG = joblib.load("outputs/model_config.joblib")


class CustomerObservation(BaseModel):
    obs_frequency: int
    obs_monetary: float
    obs_distinct_products: int
    obs_total_items: int
    tenure_days: int
    recency_days: int


class CLVResponse(BaseModel):
    predicted_clv: float
    p_will_return: float
    expected_spend_if_return: float
    top_factors: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=CLVResponse)
def score_customer(obs: CustomerObservation):
    row = obs.model_dump()
    row["obs_monetary_log"] = float(np.log1p(row["obs_monetary"]))
    row["obs_avg_order_value"] = row["obs_monetary"] / row["obs_frequency"] if row["obs_frequency"] > 0 else 0.0

    X = pd.DataFrame([row])[CONFIG["feature_cols"]]
    X_scaled = SCALER.transform(X)

    p_return = float(CLASSIFIER.predict_proba(X_scaled)[0, 1])
    spend_if_return = float(np.expm1(REGRESSOR.predict(X)[0]))
    predicted_clv = p_return * spend_if_return

    importances = pd.Series(REGRESSOR.feature_importances_, index=CONFIG["feature_cols"])
    top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

    return CLVResponse(
        predicted_clv=round(predicted_clv, 2),
        p_will_return=round(p_return, 4),
        expected_spend_if_return=round(spend_if_return, 2),
        top_factors=top_factors,
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Customer Lifetime Value Scoring</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
        <h2>Customer Lifetime Value — Two-Part Scoring</h2>
        <p>Notice the response has THREE numbers, not one -- a high combined
           CLV can come from "certain to return, modest spend" or "risky but
           a big spender if retained," and those need different retention
           tactics. Full docs at <a href="/docs">/docs</a>.</p>
        <form id="clvForm">
            <label>Orders in first 6 months: <input name="obs_frequency" type="number" value="5"></label><br><br>
            <label>Total spend in first 6 months ($): <input name="obs_monetary" type="number" step="0.01" value="600"></label><br><br>
            <label>Distinct products bought: <input name="obs_distinct_products" type="number" value="12"></label><br><br>
            <label>Total items bought: <input name="obs_total_items" type="number" value="60"></label><br><br>
            <label>Tenure (days since first purchase): <input name="tenure_days" type="number" value="90"></label><br><br>
            <label>Recency (days since last purchase): <input name="recency_days" type="number" value="20"></label><br><br>
            <button type="submit">Score this customer</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("clvForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {
                obs_frequency: parseInt(form.get("obs_frequency")),
                obs_monetary: parseFloat(form.get("obs_monetary")),
                obs_distinct_products: parseInt(form.get("obs_distinct_products")),
                obs_total_items: parseInt(form.get("obs_total_items")),
                tenure_days: parseInt(form.get("tenure_days")),
                recency_days: parseInt(form.get("recency_days"))
            };
            const res = await fetch("/score", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            document.getElementById("result").innerText =
                "Predicted CLV: $" + data.predicted_clv +
                " | P(return): " + data.p_will_return +
                " | Spend if returns: $" + data.expected_spend_if_return +
                " | Top factors: " + data.top_factors.join(", ");
        });
        </script>
    </body>
    </html>
    """
