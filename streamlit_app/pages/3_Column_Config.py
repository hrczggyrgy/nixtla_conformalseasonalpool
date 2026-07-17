import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from streamlit_app.utils.data_loader import auto_detect_columns, infer_frequency, prepare_long_dataframe, validate_column_selection
from csp_universal_forecast import CSPConfig
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG, VALIDATION_RULES

# Ensure session state is initialized
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.title("Column Configuration")
st.caption("Review auto-detected column mappings and adjust if needed.")

if st.session_state.raw_df is None:
    st.warning("Please upload data first on the Data Upload page.")
    st.stop()

df = st.session_state.raw_df

# Auto-detect columns with confidence
detected = auto_detect_columns(df)

def get_detection_confidence(df, col_name, col_type):
    """Calculate confidence score for auto-detection."""
    if not col_name:
        return "low", "Not detected"
    
    if col_type == "date":
        try:
            parsed = pd.to_datetime(df[col_name], errors="coerce")
            success_rate = parsed.notna().mean()
            if success_rate >= 0.95:
                return "high", f"Parsed as datetime for {success_rate:.0%} of rows"
            elif success_rate >= 0.7:
                return "medium", f"Parsed as datetime for {success_rate:.0%} of rows"
            else:
                return "low", f"Only {success_rate:.0%} rows parsed as datetime"
        except Exception:
            return "low", "Failed to parse"
    
    elif col_type == "value":
        if pd.api.types.is_numeric_dtype(df[col_name]):
            missing = df[col_name].isna().mean()
            if missing < 0.05:
                return "high", "Numeric column with <5% missing"
            elif missing < 0.2:
                return "medium", f"Numeric with {missing:.0%} missing"
            else:
                return "low", f"Numeric but {missing:.0%} missing"
        return "low", "Not numeric"
    
    elif col_type == "id":
        nunique = df[col_name].nunique()
        total = len(df)
        if 1 < nunique < total * 0.8:
            return "high", f"Creates {nunique} series, {total//nunique} obs/series avg"
        elif nunique == total:
            return "low", "Unique per row (likely not an ID)"
        else:
            return "medium", f"Creates {nunique} series"
    
    return "low", "Unknown"

# Auto-detection results with confidence
st.subheader("Auto-Detection Results")

c1, c2, c3 = st.columns(3)

for col, (key, label, icon) in [
    (c1, ("date_col", "Date Column", "📅")),
    (c2, ("value_col", "Value Column", "🎯")),
    (c3, ("id_col", "Series ID", "🔑")),
]:
    col_name = detected[key[0]]
    confidence, reason = get_detection_confidence(df, col_name, key[0])
    
    confidence_colors = {"high": "#10b981", "medium": "#f59e0b", "low": "#ef4444"}
    confidence_labels = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}
    
    with col:
        st.metric(label[1], col_name or "Not detected")
        
        color = confidence_colors[confidence]
        st.markdown(
            f'<div style="padding: 8px; background: {color}20; border-left: 3px solid {color}; '
            f'border-radius: 4px; margin: 8px 0;">'
            f'<strong>{confidence_labels[confidence]} confidence</strong><br>'
            f'<small>{reason}</small></div>',
            unsafe_allow_html=True
        )
        
        if confidence == "low":
            st.warning("⚠️ Low confidence — please verify")

# Natural language summary
st.divider()
st.subheader("Dataset Summary")

# Generate natural language summary
def generate_summary(detected, df):
    parts = []
    
    # Series type
    if detected["id_col"]:
        n_series = df[detected["id_col"]].nunique()
        parts.append(f"a **multi-series panel** with **{n_series} series**")
    else:
        parts.append("a **single time series**")
    
    # Frequency hint
    freq = infer_frequency(df, detected["date_col"]) if detected["date_col"] else None
    if freq:
        freq_labels = {"D": "daily", "W": "weekly", "M": "monthly", "MS": "monthly", 
                       "Q": "quarterly", "QS": "quarterly", "A": "annual", "Y": "annual"}
        freq_label = freq_labels.get(freq.split("-")[0], freq)
        parts.append(f"**{freq_label}** frequency")
    
    # Length
    n_rows = len(df)
    parts.append(f"**{n_rows:,} rows**")
    
    # Seasonality hint
    if detected["value_col"] and detected["date_col"]:
        try:
            from csp_universal_forecast import _infer_season_length, _infer_freq
            from csp_universal_forecast import _prepare_long_df
            freq_inferred = _infer_freq(pd.to_datetime(df[detected["date_col"]], errors="coerce").dropna())
            season = _infer_season_length(freq_inferred, df[detected["value_col"]].values)
            if season > 1:
                parts.append(f"likely seasonality of **{season}** periods")
        except Exception:
            pass
    
    return "We detected " + ", ".join(parts) + "."

st.markdown(generate_summary(detected, df))

# Manual column selection
st.divider()
st.subheader("Column Mapping (Override if needed)")

c1, c2, c3 = st.columns(3)

with c1:
    date_col = st.selectbox(
        "Date Column",
        df.columns.tolist(),
        index=df.columns.tolist().index(detected["date_col"]) if detected["date_col"] in df.columns else 0,
        key="col_date",
        help="Column containing timestamps"
    )

with c2:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    value_col = st.selectbox(
        "Value Column",
        numeric_cols,
        index=numeric_cols.index(detected["value_col"]) if detected["value_col"] in numeric_cols else 0,
        key="col_value",
        help="Numeric target variable to forecast"
    )

with c3:
    id_options = ["(auto: single series)"] + [c for c in df.columns if c not in [date_col, value_col]]
    id_col = st.selectbox(
        "Series ID Column (optional)",
        id_options,
        index=id_options.index(detected["id_col"]) if detected["id_col"] in id_options else 0,
        key="col_id",
        help="Identifier for multi-series data; leave as '(auto)' for single series"
    )
    id_col = None if id_col == "(auto: single series)" else id_col

# Validation
issues = validate_column_selection(df, date_col, value_col, id_col)

if issues:
    st.error("Validation failed:")
    for issue in issues:
        st.markdown(f"- {issue}")
    st.stop()
else:
    st.success("✅ Column mapping is valid")

# Frequency inference
freq = infer_frequency(df, date_col)
st.info(f"📈 Inferred Frequency: **{freq}**")

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
        
        # Warning for short series
        min_obs = VALIDATION_RULES.get("min_obs_per_series", 10)
        short = summary[summary['n_obs'] < min_obs]
        if len(short) > 0:
            st.warning(f"⚠️ {len(short)} series have fewer than {min_obs} observations and may be dropped")
    
    st.success("Ready for forecasting! Go to **Forecast Configuration** to set parameters.")