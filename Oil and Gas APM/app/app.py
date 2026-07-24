import sys
import os

# Absolute path to the project root (Oil and Gas APM), regardless of
# where Streamlit Cloud sets its working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from logic.scheduling import calculate_maintenance_intervals, recommend_next_checkup

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Asset Predictive Maintenance", layout="wide")

NAVY = "#1f4e5f"
AMBER = "#edb564"
GOOD = "#3CA653"
BAD = "#B0392E"

FINAL_THRESHOLD = 0.30
EXCEL_PATH = os.path.join(BASE_DIR, "data", "asset_predictive_maintenance.xlsx")
DATA_PATH = os.path.join(BASE_DIR, "data", "model_ready_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "failure_risk_model.pkl")

FEATURE_COLS = [
    "vibration_mm_s", "temperature_c", "pressure_psi", "flow_rate_m3h", "rpm",
    "vibration_mm_s_roll7", "temperature_c_roll7", "pressure_psi_roll7",
    "flow_rate_m3h_roll7", "rpm_roll7",
    "vibration_mm_s_rate_of_change", "temperature_c_rate_of_change",
    "pressure_psi_rate_of_change", "flow_rate_m3h_rate_of_change", "rpm_rate_of_change",
]

# =========================================================
# GLOBAL STYLING
# =========================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .main {{
        background-color: #ffffff;
    }}

    div.stButton > button {{
        background-color: {AMBER};
        color: {NAVY};
        font-weight: 700;
        font-size: 16px;
        padding: 12px 32px;
        border-radius: 8px;
        border: none;
    }}
    div.stButton > button:hover {{
        background-color: #d99f4a;
        color: #ffffff;
    }}
    
    div[data-testid="column"] {{
        padding: 0 8px;
    }}

    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: {NAVY};
        color: #ffffff;
        text-align: center;
        padding: 12px;
        font-size: 13px;
    }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_dashboard():
    st.session_state.page = "dashboard"

def go_to_home():
    st.session_state.page = "home"

# =========================================================
# LOAD DATA AND MODEL
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_maintenance_and_asset_tables():
    maint_df = pd.read_excel(EXCEL_PATH, sheet_name="Maintenance_Log")
    maint_df["date"] = pd.to_datetime(maint_df["date"])
    asset_master_df = pd.read_excel(EXCEL_PATH, sheet_name="Asset_Master")
    return maint_df, asset_master_df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()
maint_df, asset_master_df = load_maintenance_and_asset_tables()

# =========================================================
# PAGE 1: LANDING PAGE
# =========================================================
if st.session_state.page == "home":
    st.markdown(f"""
        <div style="text-align:center; padding-top:80px;">
            <h1 style="color:{NAVY}; font-size:42px; font-weight:800;">
                Asset Predictive Maintenance
            </h1>
            <p style="color:#5C6B73; font-size:18px; font-style:italic;">
                Oil & Gas Fleet Health - Diagnosing the Present, Predicting the Future
            </p>
            <p style="color:#5C6B73; font-size:15px; max-width:600px; margin:20px auto;">
                This tool uses historical sensor data and machine learning to estimate
                the likelihood of failure for pumps and compressors across a fictional
                oil & gas fleet, before a breakdown happens.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Go to Asset Prediction →", on_click=go_to_dashboard, use_container_width=True)

    st.markdown(f"""
        <div style="text-align:center; margin-top:15px;">
            <a href="https://medium.com/@Nwokocha_Uchechi_Flora/oil-gas-asset-predictive-maintenance-part-2-teaching-a-model-to-see-whats-coming-23ccca8f98af" target="_blank"
               style="color:{NAVY}; font-size:14px; text-decoration:underline;">
                Read the full write-up on Medium
            </a>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="footer">
            Analysis by Nwokocha Uchechi Flora &nbsp;|&nbsp; Synthetic data for portfolio demonstration purposes
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 2: DASHBOARD
# =========================================================
else:
    # --- Header bar ---
    st.markdown(f"""
        <div style="background-color:{NAVY}; padding:20px; border-radius:8px; margin-bottom:20px;">
            <h1 style="color:#ffffff; margin:0;">Asset Risk Dashboard</h1>
            <p style="color:{AMBER}; margin:0;">Oil & Gas Fleet: Pumps & Compressors</p>
        </div>
    """, unsafe_allow_html=True)

    st.button("← Back to Home", on_click=go_to_home)

    # --- Asset selector ---
    asset_list = sorted(df["asset_id"].unique())
    selected_asset = st.selectbox("Select an Asset", asset_list)

    asset_df = df[df["asset_id"] == selected_asset].sort_values("timestamp")
    latest_row = asset_df.iloc[-1]

    # --- Risk prediction ---
    X_latest = latest_row[FEATURE_COLS].values.reshape(1, -1)
    risk_probability = model.predict_proba(X_latest)[0][1]
    is_elevated = risk_probability >= FINAL_THRESHOLD
    risk_label = "ELEVATED RISK" if is_elevated else "LOW RISK"

    # --- Scheduling logic ---
    avg_per_asset, overall_avg = calculate_maintenance_intervals(maint_df)
    dataset_reference_date = pd.to_datetime(df["timestamp"].max())

    rec_date, source, status = recommend_next_checkup(
        selected_asset, maint_df, asset_master_df, avg_per_asset, overall_avg,
        reference_date=dataset_reference_date
    )

    is_overdue = status is not None and "OVERDUE" in status
    status_color = BAD if is_overdue else GOOD
    status_label = status if status else "N/A"

    # =========================================================
    # THREE-CARD ROW: Risk Score | Maintenance Status | Next Checkup
    # =========================================================
    card1, card2, card3 = st.columns(3)

    with card1:
        st.markdown(f"""
            <div style="background-color:{NAVY}; border-radius:12px; padding:22px; text-align:center; height:150px; display:flex; flex-direction:column; justify-content:center;">
                <p style="color:#cfd9dd; font-size:14px; margin-bottom:6px;">FAILURE RISK SCORE</p>
                <p style="color:#ffffff; font-size:36px; font-weight:800; margin:0;">{risk_probability*100:.1f}%</p>
                <p style="color:{AMBER}; font-size:14px; font-weight:700; margin-top:4px;">{risk_label}</p>
            </div>
        """, unsafe_allow_html=True)

    with card2:
        st.markdown(f"""
            <div style="background-color:#f7f9fa; border:2px solid {status_color}; border-radius:12px; padding:22px; text-align:center; height:150px; display:flex; flex-direction:column; justify-content:center;">
                <p style="color:#5C6B73; font-size:14px; margin-bottom:6px;">MAINTENANCE STATUS</p>
                <p style="color:{status_color}; font-size:20px; font-weight:800; margin:0;">{status_label}</p>
            </div>
        """, unsafe_allow_html=True)

    with card3:
        st.markdown(f"""
            <div style="background-color:#f7f9fa; border:2px solid {AMBER}; border-radius:12px; padding:22px; text-align:center; height:150px; display:flex; flex-direction:column; justify-content:center;">
                <p style="color:#5C6B73; font-size:14px; margin-bottom:6px;">NEXT RECOMMENDED CHECKUP</p>
                <p style="color:{NAVY}; font-size:22px; font-weight:800; margin:0;">{rec_date.date() if rec_date else 'N/A'}</p>
            </div>
        """, unsafe_allow_html=True)

    st.caption(f"Scheduling basis: {source}" if source else "")

    col1, col2 = st.columns(2)
    col1.metric("Asset Type", latest_row["asset_type"])
    col2.metric("Last Reading Date", str(latest_row["timestamp"])[:10])

    # --- Explanation ---
    st.subheader("What's Driving This Score?")

    vibration_change = latest_row["vibration_mm_s_rate_of_change"]
    temp_change = latest_row["temperature_c_rate_of_change"]

    explanation_parts = []
    if vibration_change > 0.2:
        explanation_parts.append(f"vibration has climbed by {vibration_change:.2f} mm/s over the past week")
    if temp_change > 2:
        explanation_parts.append(f"temperature has risen by {temp_change:.1f}°C over the past week")

    if is_elevated:
        if explanation_parts:
            st.write("This asset was flagged because " + " and ".join(explanation_parts) + ".")
        else:
            st.write("This asset was flagged based on the model's overall pattern recognition across multiple sensors.")
    else:
        if explanation_parts:
            st.write(f"Some mild upward movement was observed ({' and '.join(explanation_parts)}), but overall readings remain within the model's low-risk range.")
        else:
            st.write("Readings remain stable with no meaningful upward trend, consistent with a low-risk status.")

    # --- Trend chart ---
    st.subheader("Recent Sensor Trend (Last 60 Days)")
    recent_data = asset_df.tail(60)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_data["timestamp"], y=recent_data["vibration_mm_s"],
        name="Vibration (mm/s)", line=dict(color=NAVY)
    ))
    fig.add_trace(go.Scatter(
        x=recent_data["timestamp"], y=recent_data["temperature_c"],
        name="Temperature (°C)", line=dict(color=AMBER), yaxis="y2"
    ))
    fig.update_layout(
        yaxis=dict(title="Vibration (mm/s)"),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)
