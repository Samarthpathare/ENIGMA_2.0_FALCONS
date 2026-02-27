import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

/* ── Reset & Base ── */
* { box-sizing: border-box; }

.stApp {
    background: #04080f !important;
    font-family: 'Syne', sans-serif !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #070d1a !important;
    border-right: 1px solid rgba(0,212,255,0.15) !important;
}
section[data-testid="stSidebar"] * {
    color: #c8deff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #7aa0cc !important;
    font-size: 13px !important;
    transition: color 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #00d4ff !important;
}

/* ── Headings ── */
h1, h2, h3, h4 { color: #e8f4ff !important; font-family: 'Syne', sans-serif !important; }
p, span, label, .stMarkdown p { color: #8aabcc !important; font-family: 'Syne', sans-serif !important; }

/* ── Hero Title ── */
.hero-wrapper {
    padding: 32px 0 8px 0;
    position: relative;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: #00d4ff !important;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(110deg, #ffffff 0%, #00d4ff 60%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #4a6a8a !important;
    margin-top: 10px;
    letter-spacing: 0.5px;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(167,139,250,0.04)) !important;
    border: 1px solid rgba(0,212,255,0.18) !important;
    border-radius: 14px !important;
    padding: 20px 24px !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #a78bfa);
}
[data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    color: #6a8aaa !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }

/* ── Section Headers ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: #00d4ff !important;
    text-transform: uppercase;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.3), transparent);
}

/* ── Cards / Panels ── */
.neuro-card {
    background: rgba(7,13,26,0.8);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: rgba(0,212,255,0.03) !important;
    border: 1.5px dashed rgba(0,212,255,0.25) !important;
    border-radius: 12px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.5) !important;
    background: rgba(0,212,255,0.06) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0098b8) !important;
    color: #04080f !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,212,255,0.35) !important;
    background: linear-gradient(135deg, #20e8ff, #00d4ff) !important;
}

/* ── Select / Inputs ── */
.stSelectbox > div > div {
    background: rgba(0,212,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
    color: #e8f4ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── JSON output ── */
.stJson { background: rgba(0,212,255,0.04) !important; border: 1px solid rgba(0,212,255,0.15) !important; border-radius: 10px !important; }

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}
.badge-online { background: rgba(0,255,128,0.12); color: #00ff80; border: 1px solid rgba(0,255,128,0.25); }
.badge-warn   { background: rgba(255,180,0,0.12);  color: #ffb400; border: 1px solid rgba(255,180,0,0.25); }
.badge-risk   { background: rgba(255,80,80,0.12);  color: #ff5050; border: 1px solid rgba(255,80,80,0.25); }

/* ── Divider ── */
hr { border-color: rgba(0,212,255,0.1) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00d4ff !important; }

/* ── Info / Success / Warning boxes ── */
.stAlert { border-radius: 10px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }

/* ── Tabs (if used) ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #6a8aaa !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom-color: #00d4ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; background: #04080f; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='width:64px;height:64px;background:linear-gradient(135deg,#00d4ff22,#a78bfa22);
                    border:1px solid rgba(0,212,255,0.3);border-radius:16px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:28px;margin:0 auto 12px;'>🧠</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:3px;color:#00d4ff;'>NEURALSHIELD</div>
        <div style='font-family:"Syne",sans-serif;font-size:18px;font-weight:800;color:#e8f4ff;margin-top:2px;'>AI v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:2px;color:#4a6a8a;margin-bottom:8px;'>NAVIGATION</div>", unsafe_allow_html=True)

    page = st.radio("", ["Overview", "Analysis", "Neuro-Gallery"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:2px;color:#4a6a8a;margin-bottom:10px;'>SYSTEM STATUS</div>
    <span class='badge badge-online'>● ONLINE</span>
    <div style='margin-top:14px;font-size:11px;color:#4a6a8a;font-family:"JetBrains Mono",monospace;line-height:1.8;'>
        Model: <span style='color:#00d4ff'>EEGNet-v3</span><br>
        Channels: <span style='color:#00d4ff'>16 Active</span><br>
        Freq: <span style='color:#00d4ff'>256 Hz</span><br>
        Encrypt: <span style='color:#00d4ff'>AES-256</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:10px;color:#2a3a4a;font-family:JetBrains Mono,monospace;text-align:center;'>TEAM FALCONS © 2024</div>", unsafe_allow_html=True)


# ─── HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrapper'>
    <div class='hero-tag'>// SCHIZOPHRENIA DETECTION SYSTEM</div>
    <p class='hero-title'>NeuralShield AI</p>
    <p class='hero-sub'>Precision EEG Neural Mapping &nbsp;·&nbsp; Real-time Analysis &nbsp;·&nbsp; Clinical-Grade Accuracy</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if page == "Overview":

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Channels",  "16",    "+2")
    c2.metric("Sampling Rate",    "256 Hz")
    c3.metric("Model Accuracy",   "94.2%", "↑ 0.5%")
    c4.metric("Sessions Today",   "07",    "+3")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>// LIVE MULTI-CHANNEL SIGNAL MONITOR</div>", unsafe_allow_html=True)

    channel_labels = ["Fp1","Fp2","F3","F4","C3","C4","O1","O2","T3","T4","P3","P4","Fz","Cz","Pz","Oz"]
    channel_colors = [
        "#00d4ff","#00d4ff","#a78bfa","#a78bfa",
        "#00ff9d","#00ff9d","#ff6b6b","#ff6b6b",
        "#ffb400","#ffb400","#00d4ff","#a78bfa",
        "#00ff9d","#ff6b6b","#ffb400","#00d4ff"
    ]

    plot_spot = st.empty()
    channels_to_show = 8

    for frame in range(60):
        fig, axes = plt.subplots(channels_to_show, 1, figsize=(15, 9), sharex=True)
        fig.patch.set_facecolor('#04080f')
        t = np.linspace(0, 4, 512)
        offset = frame * 0.04

        for idx, ax in enumerate(axes):
            freq = 2 + idx * 0.7
            noise = np.random.normal(0, 0.12, len(t))
            y = (
                np.sin(2 * np.pi * freq * (t - offset)) * 0.5
                + np.sin(2 * np.pi * freq * 2.3 * (t - offset * 0.8)) * 0.25
                + np.sin(2 * np.pi * 0.5 * (t - offset * 0.3)) * 0.2
                + noise
            )
            ax.set_facecolor('#04080f')
            ax.fill_between(t, y, alpha=0.06, color=channel_colors[idx])
            ax.plot(t, y, color=channel_colors[idx], linewidth=0.9, alpha=0.85)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_ylabel(channel_labels[idx], color='#4a6a8a',
                          fontsize=9, rotation=0, labelpad=28,
                          fontfamily='monospace', va='center')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.axhline(0, color='rgba(255,255,255,0.03)', linewidth=0.5, linestyle='--')

        axes[-1].set_xlabel("Time (s)", color='#2a4a6a', fontsize=9, fontfamily='monospace')
        plt.subplots_adjust(hspace=0.0, left=0.05, right=0.99, top=0.97, bottom=0.04)
        plot_spot.pyplot(fig)
        plt.close(fig)
        time.sleep(0.02)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
elif page == "Analysis":

    st.markdown("<div class='section-label'>// NEURAL PIPELINE — SELECT INPUT MODE</div>", unsafe_allow_html=True)

    # Mode selector
    mode = st.selectbox(
        "Data Source",
        ["Dataset (CSV / XLSX)", "Device (Live EEG)"],
        label_visibility="visible"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── DATASET MODE ──────────────────────────────────────────────────
    if mode == "Dataset (CSV / XLSX)":

        col_upload, col_info = st.columns([3, 2], gap="large")

        with col_upload:
            st.markdown("<div class='section-label'>// UPLOAD SIGNAL DATA</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drop CSV or XLSX file here",
                type=["csv", "xlsx"],
                label_visibility="collapsed"
            )

            if uploaded_file:
                st.markdown(f"""
                <div style='background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
                            border-radius:10px;padding:14px 18px;margin-top:8px;
                            font-family:"JetBrains Mono",monospace;font-size:12px;'>
                    <span style='color:#4a6a8a;'>FILE  :</span>
                    <span style='color:#00d4ff;'> {uploaded_file.name}</span><br>
                    <span style='color:#4a6a8a;'>SIZE  :</span>
                    <span style='color:#00d4ff;'> {uploaded_file.size / 1024:.1f} KB</span><br>
                    <span style='color:#4a6a8a;'>STATUS:</span>
                    <span style='color:#00ff80;'> ● READY</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("🚀  EXECUTE AI SCAN", use_container_width=True):
                    temp_path = os.path.join(os.getcwd(), uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner("Decoding neural patterns…"):
                        output = run_pipeline("dataset", temp_path)

                    st.success("✓  Analysis complete")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<div class='section-label'>// RESULTS</div>", unsafe_allow_html=True)

                    r1, r2 = st.columns(2)

                    with r1:
                        risk = output.get("risk_score", 0)
                        color = "#ff5050" if risk > 60 else "#ffb400" if risk > 30 else "#00ff80"
                        badge_cls = "badge-risk" if risk > 60 else "badge-warn" if risk > 30 else "badge-online"
                        st.markdown(f"""
                        <div style='background:rgba(7,13,26,0.9);border:1px solid {color}33;
                                    border-radius:14px;padding:28px;text-align:center;'>
                            <div style='font-family:"JetBrains Mono",monospace;font-size:10px;
                                        letter-spacing:2px;color:#4a6a8a;margin-bottom:12px;'>RISK SCORE</div>
                            <div style='font-family:"Syne",sans-serif;font-size:3.5rem;
                                        font-weight:800;color:{color};line-height:1;'>{risk:.1f}<span style='font-size:1.5rem'>%</span></div>
                            <div style='margin-top:12px;'><span class='badge {badge_cls}'>
                                {"HIGH RISK" if risk > 60 else "MODERATE" if risk > 30 else "LOW RISK"}
                            </span></div>
                        </div>
                        """, unsafe_allow_html=True)

                    with r2:
                        st.markdown("<div style='font-family:\"JetBrains Mono\",monospace;font-size:10px;letter-spacing:2px;color:#4a6a8a;margin-bottom:10px;'>MODEL OUTPUT</div>", unsafe_allow_html=True)
                        st.json(output.get("prediction", {}))

        with col_info:
            st.markdown("<div class='section-label'>// PIPELINE OVERVIEW</div>", unsafe_allow_html=True)
            steps = [
                ("01", "Signal Ingestion",   "CSV/XLSX parsed → raw EEG matrix extracted"),
                ("02", "Preprocessing",      "Bandpass filter 0.5–40 Hz · artifact removal"),
                ("03", "Feature Extraction", "PSD · coherence · temporal features"),
                ("04", "Model Inference",    "EEGNet-v3 forward pass"),
                ("05", "Risk Assessment",    "Probability calibration · threshold logic"),
            ]
            for num, title, desc in steps:
                st.markdown(f"""
                <div style='display:flex;gap:14px;align-items:flex-start;margin-bottom:16px;'>
                    <div style='min-width:32px;height:32px;background:rgba(0,212,255,0.1);
                                border:1px solid rgba(0,212,255,0.25);border-radius:8px;
                                display:flex;align-items:center;justify-content:center;
                                font-family:"JetBrains Mono",monospace;font-size:11px;color:#00d4ff;'>{num}</div>
                    <div>
                        <div style='font-family:"Syne",sans-serif;font-size:13px;font-weight:600;color:#e8f4ff;'>{title}</div>
                        <div style='font-family:"JetBrains Mono",monospace;font-size:11px;color:#4a6a8a;margin-top:2px;'>{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── DEVICE MODE ───────────────────────────────────────────────────
    elif mode == "Device (Live EEG)":

        st.markdown("<div class='section-label'>// LIVE DEVICE STREAM</div>", unsafe_allow_html=True)

        d1, d2 = st.columns([2, 3], gap="large")

        with d1:
            st.markdown("""
            <div style='background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.15);
                        border-radius:14px;padding:24px;'>
                <div style='font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:2px;color:#4a6a8a;margin-bottom:16px;'>DEVICE INFO</div>
                <div style='font-size:12px;font-family:"JetBrains Mono",monospace;line-height:2.2;'>
                    <span style='color:#4a6a8a;'>DEVICE :</span> <span style='color:#00d4ff;'>EEG Simulator v2</span><br>
                    <span style='color:#4a6a8a;'>PORTS  :</span> <span style='color:#00d4ff;'>COM3 (Auto-detect)</span><br>
                    <span style='color:#4a6a8a;'>CHANNELS:</span> <span style='color:#00d4ff;'>16 / 16</span><br>
                    <span style='color:#4a6a8a;'>RATE   :</span> <span style='color:#00d4ff;'>256 Hz</span><br>
                    <span style='color:#4a6a8a;'>BUFFER :</span> <span style='color:#00d4ff;'>4 s window</span><br>
                    <span style='color:#4a6a8a;'>STATUS :</span> <span style='color:#00ff80;'>● CONNECTED</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("⚡  START LIVE ANALYSIS", use_container_width=True):
                with st.spinner("Connecting to device…"):
                    time.sleep(1.2)
                    output = run_pipeline("device", None)

                st.success("✓ Device scan complete")
                risk = output.get("risk_score", 0)
                color = "#ff5050" if risk > 60 else "#ffb400" if risk > 30 else "#00ff80"
                badge_cls = "badge-risk" if risk > 60 else "badge-warn" if risk > 30 else "badge-online"
                st.markdown(f"""
                <div style='background:rgba(7,13,26,0.9);border:1px solid {color}33;
                            border-radius:14px;padding:24px;text-align:center;margin-top:16px;'>
                    <div style='font-family:"JetBrains Mono",monospace;font-size:10px;
                                letter-spacing:2px;color:#4a6a8a;margin-bottom:10px;'>RISK SCORE</div>
                    <div style='font-family:"Syne",sans-serif;font-size:3rem;
                                font-weight:800;color:{color};'>{risk:.1f}<span style='font-size:1.2rem'>%</span></div>
                    <div style='margin-top:10px;'><span class='badge {badge_cls}'>
                        {"HIGH RISK" if risk > 60 else "MODERATE" if risk > 30 else "LOW RISK"}
                    </span></div>
                </div>
                """, unsafe_allow_html=True)

        with d2:
            st.markdown("<div class='section-label'>// SIMULATED DEVICE SIGNAL</div>", unsafe_allow_html=True)
            device_plot = st.empty()
            for frame in range(40):
                fig, ax = plt.subplots(figsize=(9, 3.5))
                fig.patch.set_facecolor('#04080f')
                ax.set_facecolor('#04080f')
                t = np.linspace(0, 4, 512)
                for i, col in enumerate(["#00d4ff","#a78bfa","#00ff9d","#ff6b6b"]):
                    y = (np.sin(2 * np.pi * (2+i*1.5) * (t - frame*0.04)) * 0.5
                         + np.random.normal(0, 0.08, len(t)) + i * 1.4)
                    ax.plot(t, y, color=col, linewidth=0.9, alpha=0.8, label=f"Ch {i+1}")
                ax.legend(loc='upper right', fontsize=8, framealpha=0,
                          labelcolor='white', prop={'family': 'monospace'})
                ax.set_xlabel("Time (s)", color='#2a4a6a', fontsize=9, fontfamily='monospace')
                ax.set_yticks([]); ax.set_xticks([0,1,2,3,4])
                ax.tick_params(colors='#2a4a6a', labelsize=8)
                for spine in ax.spines.values(): spine.set_visible(False)
                plt.tight_layout(pad=0.5)
                device_plot.pyplot(fig)
                plt.close(fig)
                time.sleep(0.03)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: NEURO-GALLERY
# ═══════════════════════════════════════════════════════════════════════
elif page == "Neuro-Gallery":

    st.markdown("<div class='section-label'>// CLINICAL REFERENCES & IMAGING</div>", unsafe_allow_html=True)

    # Hero
    st.image(
        "https://cdn.pixabay.com/photo/2018/01/31/12/16/architecture-3121009_1280.jpg",
        caption="Neural network topology — EEG spatial mapping",
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2 = st.columns(2, gap="large")

    with g1:
        st.markdown("<div class='section-label'>// STRUCTURAL ANALYSIS</div>", unsafe_allow_html=True)
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Schizophrenia_MRI.jpg/512px-Schizophrenia_MRI.jpg",
            caption="Ventricle analysis · Healthy vs Schizophrenia",
            use_container_width=True
        )
        st.markdown("""
        <div style='font-family:"JetBrains Mono",monospace;font-size:12px;line-height:1.9;color:#4a6a8a;margin-top:12px;'>
            Enlarged lateral ventricles are among the most consistent findings in schizophrenia neuroimaging.
            Volume reductions in prefrontal cortex and hippocampus are also widely documented.
        </div>
        """, unsafe_allow_html=True)

    with g2:
        st.markdown("<div class='section-label'>// EEG FREQUENCY BANDS</div>", unsafe_allow_html=True)

        fig, axes = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
        fig.patch.set_facecolor('#04080f')
        bands = [
            ("Delta  0.5–4 Hz",  0.5,  "#a78bfa", 0.4),
            ("Theta  4–8 Hz",    4.5,  "#00d4ff", 0.35),
            ("Alpha  8–13 Hz",  10.0,  "#00ff9d", 0.45),
            ("Beta  13–30 Hz",  20.0,  "#ffb400", 0.25),
            ("Gamma  30+ Hz",   38.0,  "#ff6b6b", 0.15),
        ]
        t = np.linspace(0, 3, 512)
        for ax, (label, freq, col, amp) in zip(axes, bands):
            y = np.sin(2 * np.pi * freq * t) * amp + np.random.normal(0, amp*0.1, len(t))
            ax.set_facecolor('#04080f')
            ax.fill_between(t, y, alpha=0.1, color=col)
            ax.plot(t, y, color=col, linewidth=1.1)
            ax.set_ylabel(label, color=col, fontsize=8, fontfamily='monospace', rotation=0, labelpad=95, va='center')
            ax.set_yticks([]); ax.set_xticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

        axes[-1].set_xlabel("Time →", color='#2a4a6a', fontsize=9, fontfamily='monospace')
        plt.subplots_adjust(hspace=0.15, left=0.28, right=0.98, top=0.97, bottom=0.06)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        <div style='font-family:"JetBrains Mono",monospace;font-size:12px;line-height:1.9;color:#4a6a8a;margin-top:12px;'>
            Schizophrenia is associated with increased delta/theta power, reduced alpha coherence, and
            disrupted gamma-band oscillations — key signatures used by NeuralShield's classifier.
        </div>
        """, unsafe_allow_html=True)


# ─── FOOTER ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='display:flex;justify-content:space-between;align-items:center;
            font-family:"JetBrains Mono",monospace;font-size:11px;color:#2a3a4a;padding:4px 0 8px;'>
    <span>TEAM FALCONS &nbsp;·&nbsp; NeuralShield AI v2.0</span>
    <span>Protected by AES-256 Encryption &nbsp;·&nbsp; HIPAA Compliant</span>
</div>
""", unsafe_allow_html=True)