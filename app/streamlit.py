import streamlit as st
import numpy as np
import pickle
import sys
import os

# Fix path to access src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import bandpass_filter, normalize_signal
from src.feature_extraction import extract_features
from src.risk_scoring import calculate_risk

st.title("🧠 EEG-Based Schizophrenia Risk Detection")

model_path = "models/trained_model.pkl"

if not os.path.exists(model_path):
    st.error("Model not found. Please run main.py first.")
else:
    model = pickle.load(open(model_path, "rb"))

    uploaded_file = st.file_uploader("Upload EEG Signal (.txt)")

    if uploaded_file is not None:
        signal = np.loadtxt(uploaded_file)

        st.subheader("Raw EEG Signal")
        st.line_chart(signal)

        # Preprocessing
        filtered = bandpass_filter(signal)
        normalized = normalize_signal(filtered)

        features = extract_features(normalized)

        risk = calculate_risk(model, features)

        st.subheader("Risk Score")
        st.write(f"Schizophrenia Risk: {risk:.2f}%")

        if risk < 30:
            st.success("Low Risk")
        elif risk < 60:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")