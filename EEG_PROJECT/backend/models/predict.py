# backend/models/predict.py

import random

def predict(features):
    """
    Simulated prediction function
    """
    print("🧠 Running AI prediction...")

    risk_score = round(random.uniform(0.1, 0.95), 2)

    if risk_score > 0.6:
        label = "High Risk"
    elif risk_score > 0.4:
        label = "Moderate Risk"
    else:
        label = "Low Risk"

    return {
        "risk_score": risk_score,
        "risk_level": label,
        "confidence": round(random.uniform(0.7, 0.99), 2),
        "explainability": {
            "delta_band": random.uniform(0, 1),
            "theta_band": random.uniform(0, 1),
            "alpha_band": random.uniform(0, 1),
            "beta_band": random.uniform(0, 1),
        }
    }