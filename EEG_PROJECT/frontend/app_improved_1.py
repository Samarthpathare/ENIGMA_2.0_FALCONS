import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time

# ─── Step 1: Backend Path Configuration ─────────────────────────────
# Ensures the frontend can communicate with the backend pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))

try:
    from main import run_pipeline 
except ImportError:
    st.error("❌ Backend module not found. Please check your folder structure.")

# ─── Step 2: Page & Theme Configuration ──────────────────────────────
st.set_page_config(
    page_title="DeepBrain EEG | Schizophrenia Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deep Dark Theme with Neon Accents
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #00d4ff; font-family: 'Segoe UI', sans-serif; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .stButton>button { 
        width: 100%; border-radius: 20px; background: linear-gradient(45deg, #00d4ff, #005f73); 
        color: white; border: none; font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── Step 3: Header & Animated Waveform ──────────────────────────────
st.title("🧠 DeepBrain EEG Analyzer")
st.markdown("---")

# Placeholder for the animated plot
waveform_placeholder = st.empty()

# Animation loop for EEG Waveform
def animate_eeg():
    t = np.linspace(0, 2, 100)
    for i in range(50):  # Run for 50 frames
        fig, ax = plt.subplots(figsize=(12, 2))
        # Create shifting waves to simulate live data
        shift = i * 0.1
        y = np.sin(2 * np.pi * 5 * (t - shift)) + np.random.normal(0, 0.05, len(t))
        
        ax.plot(t, y, color="#00d4ff", linewidth=1.5)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.axis('off') # Clean look
        waveform_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.05)

# Trigger animation on load
if st.button("🔄 Preview Live Signal"):
    animate_eeg()

# ─── Step 4: Medical Imagery Section ────────────────────────────────
st.subheader("🧩 Neuro-Imaging Reference")
img_col1, img_col2, img_col3 = st.columns(3)

# Note: Ensure these images exist in your 'frontend/images/' folder
with img_col1:
    st.image("frontend/images/brain1.jpg", caption="Cortical Thickness Map", use_container_width=True)
with img_col2:
    st.image("frontend/images/brain2.jpg", caption="Neural Connectivity Scans", use_container_width=True)
with img_col3:
    st.image("frontend/images/brain3.jpg", caption="Ventricular Analysis", use_container_width=True)

# ─── Step 5: Sidebar Logic (Working Backend Integration) ──────────────
st.sidebar.header("🕹️ Control Panel")
input_mode = st.sidebar.selectbox("Analysis Mode", ["Dataset (CSV/XLSX)", "Device (Live)"])

if input_mode == "Dataset (CSV/XLSX)":
    uploaded_file = st.sidebar.file_uploader("Upload EEG Log", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Display data preview
        df_preview = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.write("### Raw Data Preview")
        st.dataframe(df_preview.head(5), use_container_width=True)

        if st.sidebar.button("🚀 Run AI Diagnosis"):
            with st.spinner("Analyzing Neural Patterns..."):
                try:
                    # Robust file handling from app.py
                    temp_path = os.path.join(os.getcwd(), uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Backend Pipeline Execution
                    output = run_pipeline("dataset", temp_path)
                    
                    # High-visibility results
                    st.markdown("---")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Risk Probability", f"{output['risk_score']:.2f}%")
                    with res_col2:
                        st.subheader("Model Interpretation")
                        st.json(output["prediction"])
                        
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")

else:
    st.sidebar.warning("🔌 Hardware connection required for Live Mode.")

# ─── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Developed for Clinical Research Support | EEG Schizophrenia Detection System")