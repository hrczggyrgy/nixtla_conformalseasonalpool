import streamlit as st
import pandas as pd

from streamlit_app.utils.data_loader import load_dataframe, load_sample_data, get_dataframe_info

# Ensure session state is initialized
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    from csp_universal_forecast import CSPConfig
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.title("Data Upload")
st.caption("Upload your time series data or use the built-in sample dataset.")

# File uploader
st.subheader("Upload Data")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls", "parquet"],
    help="Supported formats: CSV, Excel (.xlsx, .xls), Parquet",
)

# Sample data button
st.subheader("Or Use Sample Data")
if st.button("Load Sample Data (timeseries1.csv)", use_container_width=True):
    with st.spinner("Loading sample data..."):
        df, meta = load_sample_data()
        st.session_state.raw_df = df
        st.session_state.sample_data_loaded = True
        st.success(f"Loaded sample data: {meta['rows']} rows x {meta['cols']} columns")
        st.rerun()

# Process uploaded file
if uploaded_file is not None:
    with st.spinner("Loading file..."):
        df, meta = load_dataframe(uploaded_file)
        if df is not None:
            st.session_state.raw_df = df
            st.session_state.sample_data_loaded = False
            st.success(f"Loaded: {meta['filename']} ({meta['rows']:,} rows x {meta['cols']} cols, {meta['memory_mb']:.1f} MB)")

# Display current data
if st.session_state.raw_df is not None:
    df = st.session_state.raw_df
    
    st.divider()
    st.subheader("Data Preview")
    
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
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
        if missing_df['Missing Count'].sum() == 0:
            st.success("No missing values")
    
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
        st.rerun()
    
    st.info("Next: Go to Column Configuration to map your columns.")
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