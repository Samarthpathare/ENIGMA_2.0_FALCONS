import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# ─── Step 1: Fix Backend Path (Logic from app.py) ─────────────────────────────
# This ensures we look specifically in the 'backend' folder
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))

try:
    from main import run_pipeline 
except ImportError:
    st.error("❌ Could not find 'main.py' in the backend folder. Check your directory structure.")

# ─── Streamlit Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 EEG Schizophrenia Predictor 🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for theme (From app_improved.py) ───────────────────────────────
st.markdown(
    """
    <style>
    body { background-color: #121212; color: #FFFFFF; font-family: 'Arial', sans-serif; }
    h1 { font-family: 'Comic Sans MS', cursive; color: #00FFFF; font-size: 48px; }
    .stButton>button { background-color: #00FFFF; color: #121212; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── Title & Visuals ───────────────────────────────────────────────────────────
st.markdown("## 🧠 EEG Schizophrenia Predictor 🧬✨", unsafe_allow_html=True)

# EEG Waveform Visualization
st.markdown("### EEG Waveform Example 📈")
fig, ax = plt.subplots(figsize=(12, 3))
t = np.linspace(0, 1, 256)
for i in range(4): 
    ax.plot(t, np.sin(2 * np.pi * (i+1) * 5 * t) + np.random.rand(len(t))*0.1)
ax.set_facecolor('#121212')
fig.patch.set_facecolor('#121212')
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_color('white')
st.pyplot(fig)

# ─── Sidebar: Input Mode ──────────────────────────────────────────────────────
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Select input mode:", ["Device", "Dataset"])

# ─── Logic for "Device" Mode ──────────────────────────────────────────────────
if input_mode == "Device":
    st.info("🔌 Connect EEG device (Hardware mode currently dummy).")
    if st.sidebar.button("Run Prediction 🧠"):
        try:
            output = run_pipeline("device", None)
            st.success(f"✅ Prediction Complete! Risk Score: {output['risk_score']:.2f}%")
            st.json(output["prediction"])
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ─── Logic for "Dataset" Mode ─────────────────────────────────────────────────
else:
    uploaded_file = st.sidebar.file_uploader("Upload EEG dataset (CSV/XLSX):", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Preview data for the user
        if uploaded_file.name.endswith('.csv'):
            df_preview = pd.read_csv(uploaded_file)
        else:
            df_preview = pd.read_excel(uploaded_file)
        
        st.write("### Data Preview")
        st.dataframe(df_preview.head())
        
        if st.sidebar.button("Run Prediction 🧠"):
            with st.spinner("Running AI pipeline..."):
                try:
                    # Step 2: Fix File Saving (Logic from app.py)
                    # We save the raw buffer to avoid DataFrame corruption
                    temp_file_path = os.path.join(os.getcwd(), uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Step 3: Run Pipeline
                    output = run_pipeline("dataset", temp_file_path)
                    
                    # Step 4: Display Results
                    st.success("✅ Prediction Complete!")
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric(label="Risk Score (%)", value=f"{output['risk_score']:.2f}%")
                    with col_res2:
                        st.subheader("Model Details")
                        st.json(output["prediction"])
                        
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;'>Developed with ❤️ | EEG Schizophrenia Detection</p>", unsafe_allow_html=True)