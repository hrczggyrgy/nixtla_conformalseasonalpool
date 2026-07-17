import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from streamlit_app.utils.forecasting import run_csp_with_config, run_seasonal_naive_with_config, compute_backtest_metrics
from streamlit_app.utils.plotting import plot_model_comparison, plot_metrics_comparison
from streamlit_app.utils.export import create_comparison_download

# Ensure session state is initialized
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    from csp_universal_forecast import CSPConfig
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.title("Model Comparison")
st.caption("Compare CSP vs SeasonalNaive forecasts and evaluate performance.")

if st.session_state.processed_df is None:
    st.warning("Please prepare data on the Column Config page first.")
    st.stop()

pdf = st.session_state.processed_df
cfg = st.session_state.forecast_cfg

# Series selector
if pdf['unique_id'].nunique() > 1:
    series_options = pdf['unique_id'].unique().tolist()
    selected_series = st.selectbox("Select Series for Comparison", series_options, key="cmp_series_select")
else:
    selected_series = pdf['unique_id'].unique()[0]

st.info(f"Series: **{selected_series}** | Observations: {len(pdf[pdf['unique_id'] == selected_series])}")

# Run comparison
run_col1, run_col2 = st.columns([2, 1])

with run_col1:
    if st.button("Run Model Comparison (CSP + SeasonalNaive)", type="primary", use_container_width=True):
        with st.spinner("Running CSP..."):
            try:
                csp_res = run_csp_with_config(pdf, cfg)
                st.session_state.csp_results = csp_res
                st.success("CSP completed")
            except Exception as e:
                st.error(f"CSP failed: {e}")
                st.session_state.csp_results = None
        
        with st.spinner("Running SeasonalNaive..."):
            try:
                sn_res = run_seasonal_naive_with_config(pdf, cfg)
                st.session_state.sn_results = sn_res
                st.success("SeasonalNaive completed")
            except Exception as e:
                st.error(f"SeasonalNaive failed: {e}")
                st.session_state.sn_results = None
        
        if st.session_state.csp_results and st.session_state.sn_results:
            # Compute metrics
            metrics = {}
            for model_name, res in [("CSP", st.session_state.csp_results), ("SeasonalNaive", st.session_state.sn_results)]:
                m = compute_backtest_metrics(
                    pdf, res.forecast_df, selected_series
                )
                if m:
                    metrics[model_name] = m
            
            if metrics:
                st.session_state.comparison_metrics = metrics
            
            st.rerun()

with run_col2:
    if st.session_state.csp_results and st.session_state.sn_results:
        if st.button("Re-run", use_container_width=True):
            st.session_state.csp_results = None
            st.session_state.sn_results = None
            st.session_state.comparison_metrics = None
            st.rerun()

# Display results
if st.session_state.csp_results and st.session_state.sn_results:
    csp_res = st.session_state.csp_results
    sn_res = st.session_state.sn_results
    
    st.divider()
    
    # Status comparison
    st.subheader("Model Status")
    
    c1, c2 = st.columns(2)
    with c1:
        csp_status = csp_res.status.get(selected_series, "unknown")
        if csp_status == "ok":
            st.success(f"CSP: {csp_status}")
        elif csp_status.startswith("fallback"):
            st.warning(f"CSP: {csp_status}")
        else:
            st.error(f"CSP: {csp_status}")
        st.caption(f"Model: {csp_res.model_name}")
    
    with c2:
        sn_status = sn_res.status.get(selected_series, "unknown")
        if sn_status == "ok":
            st.success(f"SeasonalNaive: {sn_status}")
        else:
            st.error(f"SeasonalNaive: {sn_status}")
        st.caption(f"Model: {sn_res.model_name}")
    
    # Comparison plot
    st.divider()
    st.subheader("Forecast Comparison")
    
    fig = plot_model_comparison(
        historical=pdf,
        csp_forecast=csp_res.forecast_df,
        sn_forecast=sn_res.forecast_df,
        series_id=selected_series,
        levels=cfg.levels,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics comparison
    if st.session_state.comparison_metrics:
        st.divider()
        st.subheader("Accuracy Metrics (Backtest)")
        
        metrics = st.session_state.comparison_metrics
        
        # Table
        metrics_df = pd.DataFrame(metrics).T.round(4)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Chart
        fig_metrics = plot_metrics_comparison(metrics)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Best model
        if "CSP" in metrics and "SeasonalNaive" in metrics:
            csp_mae = metrics["CSP"].get("MAE", float('inf'))
            sn_mae = metrics["SeasonalNaive"].get("MAE", float('inf'))
            
            if csp_mae < sn_mae:
                st.success(f"CSP wins (MAE: {csp_mae:.4f} vs {sn_mae:.4f})")
            elif sn_mae < csp_mae:
                st.success(f"SeasonalNaive wins (MAE: {sn_mae:.4f} vs {csp_mae:.4f})")
            else:
                st.info("Tie")
    
    # Forecast tables
    st.divider()
    st.subheader("Forecast Values")
    
    tab1, tab2 = st.tabs(["CSP", "SeasonalNaive"])
    
    with tab1:
        csp_fcst = csp_res.forecast_df[csp_res.forecast_df['unique_id'] == selected_series]
        pred_col = csp_res.model_name
        display_cols = ['ds', pred_col]
        for level in sorted(cfg.levels):
            lo = f"{pred_col}-lo-{level}"
            hi = f"{pred_col}-hi-{level}"
            if lo in csp_fcst.columns and hi in csp_fcst.columns:
                display_cols.extend([lo, hi])
        
        st.dataframe(csp_fcst[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
    
    with tab2:
        sn_fcst = sn_res.forecast_df[sn_res.forecast_df['unique_id'] == selected_series]
        pred_col = sn_res.model_name
        display_cols = ['ds', pred_col]
        for level in sorted(cfg.levels):
            lo = f"{pred_col}-lo-{level}"
            hi = f"{pred_col}-hi-{level}"
            if lo in sn_fcst.columns and hi in sn_fcst.columns:
                display_cols.extend([lo, hi])
        
        st.dataframe(sn_fcst[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
    
    # Download
    st.divider()
    st.subheader("Download Comparison Results")
    
    # Build config dict for export
    config_dict = {
        "h": cfg.h,
        "levels": cfg.levels,
        "min_obs_per_series": cfg.min_obs_per_series,
        "max_series_per_batch": cfg.max_series_per_batch,
        "outlier_clip": cfg.outlier_clip,
        "outlier_iqr_mult": cfg.outlier_iqr_mult,
        "random_seed": cfg.random_seed,
        "verbose": cfg.verbose,
    }
    
    download_files = create_comparison_download(
        csp_forecast=csp_res.forecast_df,
        sn_forecast=sn_res.forecast_df,
        csp_status=csp_res.status,
        sn_status=sn_res.status,
        config=config_dict,
    )
    
    # Get actual keys from download_files
    csv_key = [k for k in download_files if k.endswith("_forecasts_") and k.endswith(".csv")][0]
    excel_key = [k for k in download_files if k.endswith("_results_") and k.endswith(".xlsx")][0]
    
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Comparison CSV",
            data=download_files[csv_key],
            file_name=f"comparison_{selected_series}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Full Comparison Excel",
            data=download_files[excel_key],
            file_name=f"comparison_{selected_series}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

else:
    st.info("Click Run Model Comparison to generate forecasts from both models.")
    
    st.divider()
    st.subheader("What Will Be Compared")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**CSP (ConformalSeasonalPool)**")
        st.markdown("""
        - Training-free conformal prediction intervals
        - Data-driven seasonality detection
        - Robust to outliers (with IQR clipping)
        - Falls back to SeasonalNaive if needed
        """)
    
    with c2:
        st.markdown("**SeasonalNaive (Baseline)**")
        st.markdown("""
        - Simple seasonal naive benchmark
        - Repeats last seasonal cycle
        - Fast, interpretable baseline
        - No confidence intervals
        """)