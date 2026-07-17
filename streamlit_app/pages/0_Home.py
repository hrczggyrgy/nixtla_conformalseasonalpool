import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG
from csp_universal_forecast import CSPConfig

# Ensure session state is initialized
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.set_page_config(
    page_title="CSP Universal Forecast — Home",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Mark as visited
st.session_state.has_visited = True

# Hero section
st.markdown("""
# Automated Forecasting for Any Time Series

Upload a CSV, auto-detect columns and frequency, generate forecasts with calibrated prediction intervals.
""")

# Quick action buttons
col1, col2 = st.columns(2, gap="large")

with col1:
    if st.button("📁 Upload my data", use_container_width=True, type="primary"):
        st.switch_page("streamlit_app/pages/1_Data_Upload.py")

with col2:
    if st.button("🧪 Try sample dataset", use_container_width=True):
        with st.spinner("Loading sample data..."):
            import pandas as pd
            sample_path = Path(__file__).resolve().parents[2] / "timeseries1.csv"
            df = pd.read_csv(sample_path)
            st.session_state.raw_df = df
            st.session_state.uploaded_filename = "timeseries1.csv (sample)"
            st.session_state.sample_data_loaded = True
            
            # Run auto-detection
            from streamlit_app.utils.data_loader import auto_detect_columns
            detected = auto_detect_columns(df)
            st.session_state.col_config = detected
            
            # Go to Column Config
            st.switch_page("streamlit_app/pages/3_Column_Config.py")

st.divider()

# Feature cards
st.markdown("### What this app does")

feat1, feat2, feat3 = st.columns(3)

with feat1:
    st.markdown("""
    **🎯 Auto-detects everything**  
    Date column, value column, series ID, frequency, seasonality — no config required
    """)

with feat2:
    st.markdown("""
    **📦 Handles any panel data**  
    Single series or multi-store, multi-SKU — automatically splits by ID column
    """)

with feat3:
    st.markdown("""
    **📊 Calibrated intervals**  
    Conformal prediction intervals with finite-sample coverage, not asymptotic approximations
    """)

st.divider()

# How it works
with st.expander("🔍 How it works (4 steps)"):
    st.markdown("""
    1. **Upload** — CSV, Excel, or Parquet with a date column and numeric target
    2. **Review** — Auto-detected columns & frequency shown for confirmation
    3. **Configure** — Horizon, confidence levels, outlier handling
    4. **Forecast** — CSP (ConformalSeasonalPool) + fallback to SeasonalNaive
    """)

with st.expander("📋 What kind of CSV works?"):
    st.markdown("""
    **Minimum:** Date column + numeric target column
    
    **Examples:**
    | date | sales | store_id |
    |------|-------|----------|
    | 2023-01-01 | 100 | A |
    | 2023-01-02 | 120 | A |
    | 2023-01-01 | 80 | B |
    
    | ds | y |
    |----|---|
    | 2023-01-01 | 100 |
    | 2023-01-02 | 120 |
    
    **Supported formats:** CSV, Excel (.xlsx, .xls), Parquet
    """)

with st.expander("🛡️ Trust & transparency"):
    st.markdown("""
    - Every auto-detection decision is shown with confidence level
    - Per-series status: OK / Fallback / Dropped (with reason)
    - All settings overrideable before forecasting
    - Results archived for before/after comparison
    """)

st.divider()
st.caption("CSP Universal Forecast — Built on Nixtla's ConformalSeasonalPool (statsforecast)")