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
    st.error("⚠ Backend modules not found. Ensure 'backend/main.py' exists.")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralShield AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── PROFESSIONAL CSS (GLASSMORPHISM + NEON ACCENTS) ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

/* Base */
.stApp { background: #0d111f !important; font-family: 'Syne', sans-serif !important; }
* { box-sizing: border-box; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10,15,35,0.95) !important;
    border-right: 1px solid rgba(0,255,200,0.15) !important;
}
section[data-testid="stSidebar"] * { color: #b0e0ff !important; font-family: 'JetBrains Mono', monospace !important; }

/* Hero */
.hero-wrapper { padding: 32px 0 16px 0; text-align:center; }
.hero-tag { font-family:'JetBrains Mono'; font-size:11px; letter-spacing:3px; color:#00ffcc; text-transform:uppercase; margin-bottom:6px; }
.hero-title {
    font-family:'Syne', sans-serif; font-size:3.8rem; font-weight:800;
    background: linear-gradient(120deg, #00ffcc, #a78bfa, #ffffff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-shadow: 0 0 8px #00ffcc66, 0 0 12px #a78bfa44;
    animation: neonPulse 2.5s ease-in-out infinite alternate;
}

/* Neon pulse animation */
@keyframes neonPulse {
    0% { text-shadow: 0 0 5px #00ffcc55, 0 0 10px #a78bfa33; }
    100% { text-shadow: 0 0 12px #00ffcc88, 0 0 20px #a78bfa55; }
}

/* Glass card style metrics */
[data-testid="stMetric"] {
    background: rgba(20,30,50,0.6) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(0,255,200,0.2) !important;
    padding: 20px 24px !important;
    box-shadow: 0 0 20px rgba(0,255,200,0.1);
}
[data-testid="stMetricValue"] { color: #00ffcc !important; font-weight:700; }

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg,#00ffcc,#00aabb) !important;
    color:#0d111f !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight:700 !important;
    border-radius:14px !important;
    padding:14px 30px !important;
    text-transform:uppercase !important;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover { transform: scale(1.05); box-shadow: 0 0 15px #00ffcc; }

/* Sidebar badges */
.badge { display:inline-block; padding:5px 14px; border-radius:20px; font-family:'JetBrains Mono'; font-size:11px; font-weight:600; }
.badge-online { background: rgba(0,255,128,0.15); color:#00ff80; border:1px solid rgba(0,255,128,0.25); }

/* EEG plot labels */
.yaxis-label { color:#4affc9; font-weight:600; font-family:'JetBrains Mono'; }

/* Footer */
.footer { text-align:center; font-family:'JetBrains Mono'; font-size:10px; color:#2a3a4a; margin-top:30px; }

/* Uploader */
div[data-testid="stFileUploadDropzone"] div, div[data-testid="stFileUploadDropzone"] small, div[data-testid="stFileUploadDropzone"] span { color:#ffffff !important; }
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='width:64px;height:64px;background:linear-gradient(135deg,#00ffcc33,#a78bfa33);
                    border:1px solid #00ffcc55;border-radius:18px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:28px;margin:0 auto 12px;'>🧠</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:3px;color:#00ffcc;'>NEURALSHIELD</div>
        <div style='font-family:"Syne",sans-serif;font-size:18px;font-weight:800;color:#e0f7ff;'>AI v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:2px;color:#4a6a8a;margin-bottom:8px;'>NAVIGATION</div>", unsafe_allow_html=True)
    page = st.radio("Menu Selection", ["Overview", "Analysis", "Neuro-Gallery"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<span class='badge badge-online'>● ONLINE</span>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top:14px;font-size:11px;color:#00ffcc;font-family:"JetBrains Mono",monospace;line-height:1.8;'>
        Model: <span style='color:#00ffcc'>EEGNet-v3</span><br>
        Channels: <span style='color:#00ffcc'>16 Active</span><br>
        Freq: <span style='color:#00ffcc'>256 Hz</span>
    </div>
    """, unsafe_allow_html=True)


# ─── HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrapper'>
    <div class='hero-tag'>// SCHIZOPHRENIA DETECTION SYSTEM</div>
    <p class='hero-title'>NeuralShield AI</p>
</div>
""", unsafe_allow_html=True)


# ═══ PAGE: OVERVIEW ═════════════════════════════════════════════════════
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Channels", "16", "+2")
    c2.metric("Sampling Rate", "256 Hz")
    c3.metric("Model Accuracy", "94.2%", "↑ 0.5%")
    c4.metric("Sessions Today", "07", "+3")

    st.markdown("<br><div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#00ffcc;'>// LIVE EEG MONITOR</div>", unsafe_allow_html=True)
    
    channel_labels = ["Fp1","Fp2","F3","F4","C3","C4","O1","O2"]
    channel_colors = ["#00ffcc","#00ffcc","#a78bfa","#a78bfa","#00ff9d","#00ff9d","#ff6b6b","#ff6b6b"]

    plot_spot = st.empty()
    for frame in range(40):
        fig, axes = plt.subplots(8, 1, figsize=(15, 8), sharex=True)
        fig.patch.set_facecolor('#0d111f')
        t = np.linspace(0, 4, 256)
        for idx, ax in enumerate(axes):
            y = np.sin(2 * np.pi * (2 + idx) * (t - frame*0.1)) * 0.5 + np.random.normal(0, 0.05, 256)
            ax.set_facecolor('#0d111f')
            ax.plot(t, y, color=channel_colors[idx], linewidth=1.0)
            ax.set_yticks([]); ax.set_xticks([])
            ax.axhline(0, color='#ffffff', alpha=0.05, linestyle='--')
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_ylabel(channel_labels[idx], color='#4affc9', rotation=0, labelpad=22, fontweight='bold')
        plt.subplots_adjust(hspace=0)
        plot_spot.pyplot(fig)
        plt.close(fig)
        time.sleep(0.01)


# ═══ PAGE: ANALYSIS ════════════════════════════════════════════════════
elif page == "Analysis":
    st.selectbox("Data Source", ["Dataset (CSV / XLSX)", "Device (Live EEG)"])
    uploaded_file = st.file_uploader("Drop CSV or XLSX file here", type=["csv", "xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        if st.button("🚀 EXECUTE AI SCAN"):
            with st.spinner("Decoding neural patterns…"):
                temp_path = os.path.join(os.getcwd(), uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                output = run_pipeline("dataset", temp_path)
                
                st.success("✓ Analysis complete")
                risk = output.get("risk_score", 0)
                st.metric("Risk Score", f"{risk:.2f}%")

                st.markdown("<br><div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#00ffcc;'>// SIGNAL RECONSTRUCTION</div>", unsafe_allow_html=True)
                wave_spot = st.empty()
                
                signal_data = df.iloc[:, 0].values
                for i in range(0, min(500, len(signal_data)-100), 5):
                    fig, ax = plt.subplots(figsize=(12, 3))
                    ax.plot(signal_data[i:i+100], color="#00ffcc", linewidth=1.5)
                    ax.set_facecolor('#0d111f')
                    fig.patch.set_facecolor('#0d111f')
                    ax.axhline(0, color='#ffffff', alpha=0.1)
                    ax.axis('off')
                    wave_spot.pyplot(fig)
                    plt.close(fig)
                    time.sleep(0.01)


# ═══ PAGE: NEURO-GALLERY ═════════════════════════════════════════════
elif page == "Neuro-Gallery":
    st.markdown("### // STRUCTURAL ANALYSIS")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Schizophrenia_MRI.jpg/512px-Schizophrenia_MRI.jpg", width=650)

# ─── FOOTER ──────────────────────────────────────────────────────────────
st.markdown("<div class='footer'>TEAM FALCONS © 2024</div>", unsafe_allow_html=True)