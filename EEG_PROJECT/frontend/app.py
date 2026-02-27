# frontend/app.py
import streamlit as st
import os
import sys

# ─── Step 1: Add backend path ─────────────────────────────
# This allows us to import backend modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))

from main import run_pipeline  # import your backend pipeline function

# ─── Step 2: Streamlit page setup ───────────────────────
st.set_page_config(page_title="EEG Schizophrenia Detection", layout="wide")
st.title("🧠 EEG Schizophrenia Detection")

st.write(
    """
This tool predicts schizophrenia risk based on EEG data.
You can either connect a device (dummy mode) or upload a dataset (CSV/XLSX).
"""
)

# ─── Step 3: Input mode selection ───────────────────────
mode_option = st.radio("Select input mode:", ["Device", "Dataset"])

if mode_option == "Device":
    st.info("🔌 Connect EEG device to start acquisition (Device mode currently dummy).")
    if st.button("Run Prediction"):
        try:
            output = run_pipeline("device", None)
            st.success(f"Prediction done! Risk Score: {output['risk_score']:.2f}%")
            st.subheader("Prediction Details")
            st.json(output["prediction"])
        except Exception as e:
            st.error(f"❌ Error: {e}")

else:  # Dataset mode
    uploaded_file = st.file_uploader("Upload EEG dataset (CSV or XLSX):", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Save uploaded file to temporary path
        temp_file_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        if st.button("Run Prediction"):
            try:
                output = run_pipeline("dataset", temp_file_path)
                
                # ─── Step 4: Show features ─────────────────
                st.subheader("📊 Extracted Features")
                st.write(output["features"])
                
                # ─── Step 5: Show prediction ───────────────
                st.subheader("🧠 Prediction")
                st.json(output["prediction"])
                
                # ─── Step 6: Show risk score ───────────────
                st.subheader("📊 Risk Score")
                st.metric(label="Risk Score (%)", value=f"{output['risk_score']:.2f}%")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")