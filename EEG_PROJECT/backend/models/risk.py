# backend/models/risk.py

import numpy as np

def calculate_risk(prediction):
    """
    Calculates risk score from AI model output.

    prediction: dict or array-like
        If dict, expects 'risk_score' key (0-1 scale)
        If numeric array, fallback to mean

    Returns:
        float: risk score in 0-100 scale
    """
    if isinstance(prediction, dict) and "risk_score" in prediction:
        return prediction["risk_score"] * 100  # convert 0-1 to %
    else:
        return float(np.mean(prediction) * 100)