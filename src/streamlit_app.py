import streamlit as st
import numpy as np
import pickle
from src.preprocessing import bandpass_filter, normalize_signal
from src.feature_extraction import extract_features
from src.risk_scoring import calculate_risk

model = pickle.load(open("../models/trained_model.pkl", "rb"))

st.title("🧠 EEG-Based Schizophrenia Risk Detection")

uploaded_file = st.file_uploader("Upload EEG Signal (.txt)")

if uploaded_file:
    signal = np.loadtxt(uploaded_file)

    st.subheader("Raw EEG Signal")
    st.line_chart(signal)

    signal = bandpass_filter(signal)
    signal = normalize_signal(signal)

    features = extract_features(signal)

    risk = calculate_risk(model, features)

    st.subheader("Risk Score")
    st.write(f"Schizophrenia Risk: {risk:.2f}%")

    if risk < 30:
        st.success("Low Risk")
    elif risk < 60:
        st.warning("Moderate Risk")
    else:
        st.error("High Risk")