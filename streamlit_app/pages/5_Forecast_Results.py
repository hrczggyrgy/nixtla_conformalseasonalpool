import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from streamlit_app.utils.forecasting import run_csp_with_config, get_model_name
from streamlit_app.utils.plotting import create_forecast_plot
from streamlit_app.utils.export import create_forecast_download

# Ensure session state is initialized
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    from csp_universal_forecast import CSPConfig
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.title("Forecast Results")
st.caption("Run CSP forecasting and visualize results with confidence intervals.")

if st.session_state.processed_df is None:
    st.warning("Please prepare data on the Column Config page first.")
    st.stop()

pdf = st.session_state.processed_df
cfg = st.session_state.forecast_cfg

# Series selector
if pdf['unique_id'].nunique() > 1:
    series_options = pdf['unique_id'].unique().tolist()
    selected_series = st.selectbox("Select Series", series_options, key="fcst_series_select")
else:
    selected_series = pdf['unique_id'].unique()[0]

# Run button
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    run_clicked = st.button("Run CSP Forecast", type="primary", use_container_width=True)
with col2:
    if st.session_state.csp_results is not None:
        st.success("CSP Results Available")
with col3:
    if st.button("Clear Results", use_container_width=True):
        st.session_state.csp_results = None
        st.rerun()

if run_clicked:
    with st.spinner("Running CSP forecast..."):
        try:
            result = run_csp_with_config(pdf, cfg)
            st.session_state.csp_results = {
                "forecast_df": result.forecast_df,
                "status": result.status,
                "model_name": result.model_name,
                "config": {
                    "h": cfg.h,
                    "levels": cfg.levels,
                    "min_obs_per_series": cfg.min_obs_per_series,
                    "outlier_clip": cfg.outlier_clip,
                    "outlier_iqr_mult": cfg.outlier_iqr_mult,
                    "random_seed": cfg.random_seed,
                },
            }
            st.success(f"CSP Forecast completed: {result.model_name}")
            st.rerun()
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.exception(e)

# Display results
if st.session_state.csp_results is not None:
    res = st.session_state.csp_results
    fcst_df = res["forecast_df"]
    status = res["status"]
    model_name = res["model_name"]
    config = res["config"]
    
    # Filter for selected series
    series_fcst = fcst_df[fcst_df['unique_id'] == selected_series].sort_values('ds')
    series_hist = pdf[pdf['unique_id'] == selected_series].sort_values('ds')
    
    # Status
    series_status = status.get(selected_series, "unknown")
    if series_status == "ok":
        st.success(f"Status: {series_status}")
    elif series_status.startswith("fallback"):
        st.warning(f"Status: {series_status}")
    else:
        st.error(f"Status: {series_status}")
    
    # Plot
    st.subheader(f"{model_name} Forecast - {selected_series}")
    
    fig = create_forecast_plot(
        forecast_df=series_fcst,
        history_df=series_hist,
        series_ids=[selected_series],
        model_name=model_name,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.subheader("Forecast Values")
    
    pred_col = get_model_name(fcst_df)
    display_cols = ['ds', pred_col]
    
    for level in config["levels"]:
        lo = f"{pred_col}-lo-{level}"
        hi = f"{pred_col}-hi-{level}"
        if lo in series_fcst.columns and hi in series_fcst.columns:
            display_cols.extend([lo, hi])
    
    st.dataframe(
        series_fcst[display_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    
    # Per-series status table
    with st.expander("Per-Series Status Details"):
        status_df = pd.DataFrame([
            {"Series ID": k, "Status": v} for k, v in status.items()
        ])
        st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    # Download
    st.divider()
    st.subheader("Download Results")
    
    download_files = create_forecast_download(
        forecast_df=fcst_df,
        status=status,
        config=config,
        model_name=model_name,
    )
    
    # Get the actual keys (they contain timestamps)
    forecast_key = [k for k in download_files if "_forecast_" in k and k.endswith(".csv")][0]
    status_key = [k for k in download_files if "_status_" in k and k.endswith(".csv")][0]
    excel_key = [k for k in download_files if "_results_" in k and k.endswith(".xlsx")][0]
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.download_button(
            "Forecast CSV",
            data=download_files[forecast_key],
            file_name=f"{model_name}_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    with c2:
        st.download_button(
            "Status CSV",
            data=download_files[status_key],
            file_name=f"{model_name}_status.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    with c3:
        st.download_button(
            "Full Excel",
            data=download_files[excel_key],
            file_name=f"{model_name}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

else:
    st.info("Click Run CSP Forecast to generate predictions.")
    
    # Show what will be forecasted
    st.subheader("Forecast Plan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Series", pdf['unique_id'].nunique())
    c2.metric("Horizon", cfg.h)
    c3.metric("Confidence Levels", ", ".join(map(str, cfg.levels)))