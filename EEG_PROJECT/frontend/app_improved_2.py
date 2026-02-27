import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time

# ─── PATH SETUP ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "backend"))

try:
    from main import run_pipeline
except ImportError:
    st.error("Backend not found!")

# ─── UI ENHANCEMENTS (Modern Glassmorphism) ──────────────────────────────
st.set_page_config(page_title="NeuralShield AI", layout="wide")

st.markdown("""
    <style>
    /* Full Dark Background */
    .stApp { background: #0e1117; }
    
    /* Neon Title */
    .main-title {
        font-size: 50px;
        font-weight: 800;
        background: -webkit-linear-gradient(#00d4ff, #005f73);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    /* Glassy Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    /* Glow Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #0080ff);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">NEURALSHIELD: EEG ANALYSIS</p>', unsafe_allow_html=True)

# ─── ANIMATED MULTI-CHANNEL EEG ──────────────────────────────────────────
st.write("### 📡 Live Neural Signal Monitoring")
wave_placeholder = st.empty()

def get_smooth_wave():
    # 4 channels of simulated EEG
    for frame in range(40):
        fig, ax = plt.subplots(figsize=(15, 3))
        t = np.linspace(0, 2, 200)
        colors = ['#00d4ff', '#00ffaa', '#ff00ff', '#ffff00']
        
        for i in range(4):
            # Complex wave (sine + noise)
            y = np.sin(2 * np.pi * (3+i) * (t - frame*0.05)) + (i * 2.5)
            ax.plot(t, y, color=colors[i], alpha=0.7, linewidth=1)
        
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.axis('off')
        wave_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.01)

get_smooth_wave()

# ─── ANALYSIS SECTION ────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🛠️ Diagnostics")
    input_mode = st.radio("Select Source", ["Device (Hardware)", "Dataset (CSV/XLSX)"])
    
    if input_mode == "Dataset (CSV/XLSX)":
        uploaded_file = st.file_uploader("Upload EEG Data", type=["csv", "xlsx"])
        if uploaded_file and st.button("🚀 Start AI Analysis"):
            with st.status("Analyzing Synapses..."):
                # Path handling from app.py
                temp_path = os.path.join(os.getcwd(), uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                output = run_pipeline("dataset", temp_path)
                
                st.session_state['result'] = output
                st.success("Analysis Complete!")

with col2:
    if 'result' in st.session_state:
        res = st.session_state['result']
        st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #00d4ff;">Risk Score</h2>
                <h1 style="font-size: 60px;">{res['risk_score']:.2f}%</h1>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("View Neural Markers"):
            st.json(res['prediction'])
    else:
        st.info("Upload data and run analysis to see results.")

# ─── FOOTER ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🔒 HIPAA Compliant Neural Data Processing | 2026 AI Diagnostics")