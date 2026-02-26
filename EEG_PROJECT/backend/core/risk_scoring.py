# core/risk_scoring.py

from config import RISK_THRESHOLD

def calculate_risk(probability):

    risk_percent = probability * 100

    if probability >= RISK_THRESHOLD:
        status = "High Risk"
    else:
        status = "Low Risk"

    return round(risk_percent, 2), status