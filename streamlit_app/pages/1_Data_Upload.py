import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from streamlit_app.utils.data_loader import load_dataframe, load_sample_data, get_dataframe_info
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG
from csp_universal_forecast import CSPConfig

# Ensure session state is initialized
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.title("Data Upload")
st.caption("Upload your time series data or use the built-in sample dataset.")

# Validation banner
has_data = st.session_state.raw_df is not None
if has_data:
    st.success("✅ Data loaded — ready for Column Configuration")
else:
    st.warning("⚠️ No data loaded yet — upload a file or load sample data")

# File uploader
st.subheader("Upload Data")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls", "parquet"],
    help="Supported formats: CSV, Excel (.xlsx, .xls), Parquet",
    disabled=has_data,  # Disable if already loaded
)

# Sample data button
st.subheader("Or Use Sample Data")
sample_disabled = has_data
if st.button("Load Sample Data (timeseries1.csv)", use_container_width=True, disabled=sample_disabled):
    with st.spinner("Loading sample data..."):
        df, meta = load_sample_data()
        st.session_state.raw_df = df
        st.session_state.uploaded_filename = "timeseries1.csv (sample)"
        st.session_state.sample_data_loaded = True
        st.success(f"Loaded sample data: {meta['rows']} rows x {meta['cols']} columns")
        st.rerun()

if sample_disabled:
    st.caption("Clear current data to load sample")

# Process uploaded file
if uploaded_file is not None and not has_data:
    with st.spinner("Loading file..."):
        df, meta = load_dataframe(uploaded_file)
        if df is not None:
            st.session_state.raw_df = df
            st.session_state.uploaded_filename = meta['filename']
            st.session_state.sample_data_loaded = False
            st.success(f"Loaded: {meta['filename']} ({meta['rows']:,} rows x {meta['cols']} cols, {meta['memory_mb']:.1f} MB)")
            st.rerun()

# Display current data
if has_data:
    df = st.session_state.raw_df
    
    st.divider()
    st.subheader("Data Preview")
    
    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    missing_pct = df.isnull().sum().sum() / df.size * 100
    col4.metric("Missing %", f"{missing_pct:.1f}%")
    
    tab1, tab2, tab3 = st.tabs(["Preview", "Info", "Column Types"])
    
    with tab1:
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(df):,} rows")
    
    with tab2:
        info = get_dataframe_info(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{info['shape'][0]:,}")
        col2.metric("Columns", info['shape'][1])
        col3.metric("Memory", f"{info['memory_mb']:.1f} MB")
        
        st.write("**Missing Values:**")
        missing_df = pd.DataFrame.from_dict(info['null_counts'], orient='index', columns=['Missing Count'])
        missing_df['Missing %'] = (missing_df['Missing Count'] / info['shape'][0] * 100).round(1)
        missing_only = missing_df[missing_df['Missing Count'] > 0]
        if len(missing_only) > 0:
            st.dataframe(missing_only, use_container_width=True)
        else:
            st.success("No missing values")
        
        # Duplicates check
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            st.warning(f"⚠️ {dup_count} duplicate row(s) found")
        else:
            st.success("No duplicate rows")
    
    with tab3:
        dtypes_df = pd.DataFrame.from_dict(info['dtypes'], orient='index', columns=['dtype'])
        st.dataframe(dtypes_df, use_container_width=True)
    
    # Quick stats for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        st.divider()
        st.subheader("Numeric Column Statistics")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    
    # Clear button
    st.divider()
    if st.button("Clear Data & Start Over", use_container_width=True):
        st.session_state.raw_df = None
        st.session_state.processed_df = None
        st.session_state.col_config = None
        st.session_state.uploaded_filename = None
        st.rerun()
    
    # Next step guidance
    st.info("➡️ Next: Go to **Column Configuration** to map your columns.")
else:
    st.info("Upload a file or load sample data to begin.")
    
    # Show expected format
    with st.expander("Expected Data Format"):
        st.markdown("""
        Your data should have at minimum:
        - **Date column**: Timestamps (any format pandas can parse)
        - **Value column**: Numeric target variable
        - **ID column** (optional): Series identifier for panel/multi-series data
        
        **Example formats:**
        
        | date | sales | store_id |
        |------|-------|----------|
        | 2023-01-01 | 100 | A |
        | 2023-01-02 | 120 | A |
        | 2023-01-01 | 80 | B |
        
        | ds | y |
        |----|---|
        | 2023-01-01 | 100 |
        | 2023-01-02 | 120 |
        
        Supported formats: CSV, Excel (.xlsx, .xls), Parquet
        """)