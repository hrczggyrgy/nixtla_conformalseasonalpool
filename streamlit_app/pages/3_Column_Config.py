import streamlit as st
import pandas as pd

from streamlit_app.utils.data_loader import auto_detect_columns, infer_frequency, prepare_long_dataframe, validate_column_selection
from csp_universal_forecast import CSPConfig

# Ensure session state is initialized
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.title("Column Configuration")
st.caption("Map your data columns to the required format: date, value, and optional series ID.")

if st.session_state.raw_df is None:
    st.warning("Please upload data first on the Data Upload page.")
    st.stop()

df = st.session_state.raw_df

# Auto-detect columns
with st.expander("Auto-Detection Results", expanded=True):
    detected = auto_detect_columns(df)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Date Column", detected["date_col"] or "Not detected")
    with c2:
        st.metric("Value Column", detected["value_col"] or "Not detected")
    with c3:
        st.metric("ID Column", detected["id_col"] or "Auto (single series)")

st.divider()

# Manual column selection
st.subheader("Column Mapping")

c1, c2, c3 = st.columns(3)

with c1:
    date_col = st.selectbox(
        "Date Column",
        df.columns.tolist(),
        index=df.columns.tolist().index(detected["date_col"]) if detected["date_col"] in df.columns else 0,
        key="col_date",
    )

with c2:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    value_col = st.selectbox(
        "Value Column",
        numeric_cols,
        index=numeric_cols.index(detected["value_col"]) if detected["value_col"] in numeric_cols else 0,
        key="col_value",
    )

with c3:
    id_options = ["(auto: single series)"] + [c for c in df.columns if c not in [date_col, value_col]]
    id_col = st.selectbox(
        "Series ID Column (optional)",
        id_options,
        index=id_options.index(detected["id_col"]) if detected["id_col"] in id_options else 0,
        key="col_id",
    )
    id_col = None if id_col == "(auto: single series)" else id_col

# Validation
issues = validate_column_selection(df, date_col, value_col, id_col)

if issues:
    for issue in issues:
        st.error(issue)
    st.stop()
else:
    st.success("Column mapping is valid")

# Frequency inference
freq = infer_frequency(df, date_col)
st.info(f"Inferred Frequency: **{freq}**")

# Preview processed data
st.divider()
st.subheader("Preview Processed Data")

if st.button("Prepare Data", type="primary", use_container_width=True):
    with st.spinner("Preparing data for forecasting..."):
        cfg = st.session_state.forecast_cfg
        processed = prepare_long_dataframe(df, date_col, value_col, id_col, freq, cfg)
        st.session_state.processed_df = processed
        st.session_state.col_config = {
            "date_col": date_col,
            "value_col": value_col,
            "id_col": id_col,
            "freq": freq,
        }
    st.success("Data prepared successfully!")
    st.rerun()

if st.session_state.processed_df is not None:
    pdf = st.session_state.processed_df
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(pdf):,}")
    c2.metric("Unique Series", pdf['unique_id'].nunique())
    c3.metric("Date Range", f"{pdf['ds'].min().date()} to {pdf['ds'].max().date()}")
    c4.metric("Avg Points/Series", f"{len(pdf) / pdf['unique_id'].nunique():.0f}")
    
    st.subheader("First 20 Rows (Long Format)")
    st.dataframe(pdf.head(20), use_container_width=True)
    
    # Series summary
    if pdf['unique_id'].nunique() > 1:
        st.subheader("Series Summary")
        summary = pdf.groupby('unique_id').agg(
            n_obs=('y', 'count'),
            mean=('y', 'mean'),
            std=('y', 'std'),
            min=('y', 'min'),
            max=('y', 'max'),
            start=('ds', 'min'),
            end=('ds', 'max'),
        ).reset_index()
        st.dataframe(summary, use_container_width=True)
    
    st.success("Ready for forecasting! Go to Forecast Configuration to set parameters.")