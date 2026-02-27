import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import run_pipeline  # ab import kaam karega

# --- UI Styling ---
st.markdown(
    """
    <style>
    .title {
        font-family: 'Comic Sans MS', sans-serif;
        font-size: 42px;
        color: #00FFAA;
        text-align: center;
    }
    .subtitle {
        font-family: 'Arial', sans-serif;
        font-size: 20px;
        color: #AAAAAA;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Title ---
st.markdown("<h1 class='title'>🧠⚡ EEG Schizophrenia Detection 🌌📊</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload EEG dataset or connect device to predict risk levels</p>", unsafe_allow_html=True)

st.write("---")

# --- EEG waveform / example plot ---
st.subheader("EEG Sample Waveform")
# Dummy waveform for display
t = np.linspace(0, 1, 256)
sample_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(t, sample_signal, color="#00FFAA")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_facecolor("#111518")
fig.patch.set_facecolor('#0e1117')
st.pyplot(fig)

st.write("---")

# --- Brain images ---
st.subheader("Brain Visualizations")
st.image([
    "assets/schizophrenia_brain1.png",
    "assets/schizophrenia_brain2.png"
], width=350)

st.write("---")

# --- Input mode ---
mode = st.radio("Select Input Mode:", ("Device", "Dataset"))

if mode == "Device":
    st.info("🔌 Connect EEG device to start acquisition (Device mode currently dummy).")
    source = None
else:
    source = st.file_uploader("Upload EEG dataset (CSV/XLSX)", type=['csv','xlsx'])
    if source:
        st.success(f"✅ File uploaded: {source.name}")

# --- Run Prediction ---
if st.button("Run Prediction"):

    if mode == "Dataset" and not source:
        st.error("Please upload a dataset file first!")
    else:
        with st.spinner("🧠 Running AI prediction..."):
            output = run_pipeline("dataset" if mode=="Dataset" else "device", source)
        
        st.success("Prediction Complete! ✅")
        st.metric("Risk Score", f"{output['risk_score']}%")
        st.json(output["prediction"])