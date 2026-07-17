import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from streamlit_app.utils.forecasting import (
    run_csp_with_config,
    run_seasonal_naive_with_config,
    run_auto_arima_with_config,
    run_auto_ets_with_config,
    run_auto_theta_with_config,
    run_all_models,
    compute_backtest_metrics,
    compute_interval_metrics,
)
from streamlit_app.utils.plotting import create_all_models_comparison_plot, create_metric_comparison_bar
from streamlit_app.utils.export import create_comparison_download
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG

# Ensure session state is initialized
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    from csp_universal_forecast import CSPConfig
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.title("Model Comparison")
st.caption("Compare CSP against multiple baselines — point forecasts & interval calibration.")

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

# Series-specific profile
st.divider()
st.subheader("📊 Series Profile")

series_data = pdf[pdf['unique_id'] == selected_series]['y'].dropna()
n_obs = len(series_data)
cv = series_data.std() / series_data.mean() if series_data.mean() != 0 else 0
nunique = series_data.nunique()

# Compute ACF strength for seasonality
acf_strength = 0
if n_obs > 20:
    try:
        from statsmodels.tsa.stattools import acf
        acf_vals = acf(series_data, nlags=min(20, n_obs // 4), fft=True)
        acf_strength = max(acf_vals[2:]) if len(acf_vals) > 2 else 0
    except Exception:
        acf_strength = 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{n_obs}")
c2.metric("CV (σ/μ)", f"{cv:.3f}")
c3.metric("Unique values", f"{nunique}")
c4.metric("ACF strength", f"{acf_strength:.3f}")

# Run comparison
st.divider()
st.subheader("🚀 Run Comparison")

# Timeout configuration
timeout = st.slider(
    "Model timeout (seconds per model)",
    min_value=30,
    max_value=300,
    value=st.session_state.get("model_timeout", 120),
    step=30,
    help="Maximum time to wait for each model. AutoARIMA can take 60-120s per series.",
    key="model_timeout",
)

# Model selection
st.markdown("**Model Selection**")
col1, col2 = st.columns(2)
with col1:
    run_fast = st.checkbox("Run fast models (CSP, SeasonalNaive, AutoETS, AutoTheta)", value=True, 
                          help="These models complete in seconds per series")
with col2:
    run_slow = st.checkbox("Run slow models (AutoARIMA)", value=False,
                          help="AutoARIMA can take 60-120s per series. For 10 series, this may take 10-20 minutes.")

MODEL_GROUPS = {
    "Tier 1 — Point Forecast Competitors": [
        ("CSP", "CSP", "ConformalSeasonalPool"),
        ("AutoETS", "AutoETS", "Exponential Smoothing"),
        ("AutoTheta", "AutoTheta", "Theta method"),
    ],
    "Tier 1 — Slow (Optional)": [
        ("AutoARIMA", "AutoARIMA", "Automatic ARIMA (slow)"),
    ],
    "Tier 2 — Interval Calibration Reference": [
        ("SeasonalNaive", "SeasonalNaive", "Seasonal Naive (CSP's point anchor)"),
    ],
}

if run_slow:
    ALL_MODELS = [m[0] for group in MODEL_GROUPS.values() for m in group]
else:
    # Only fast models
    fast_models = []
    for group in MODEL_GROUPS.values():
        for m in group:
            if m[0] != "AutoARIMA":
                fast_models.append(m[0])
    ALL_MODELS = fast_models

run_col1, run_col2 = st.columns([3, 1])

with run_col1:
    if st.button("Run All Models", type="primary", use_container_width=True):
        with st.spinner(f"Running all models (timeout: {timeout}s each)..."):
            results = run_all_models(pdf, cfg, timeout=timeout)
            
            # Store results
            for model_key, result in results.items():
                if result is not None:
                    st.session_state[f"{model_key.lower()}_results"] = result
                else:
                    st.session_state[f"{model_key.lower()}_results"] = None
            
            # Compute metrics for all models
            metrics = {}
            interval_metrics = {}
            for model_key in ALL_MODELS:
                res = st.session_state.get(f"{model_key.lower()}_results")
                if res:
                    m = compute_backtest_metrics(pdf, res.forecast_df, selected_series)
                    if m:
                        metrics[model_key] = m
                    
                    im = compute_interval_metrics(pdf, res.forecast_df, selected_series, level=95)
                    if im:
                        interval_metrics[model_key] = im
            
            if metrics:
                st.session_state.comparison_metrics = metrics
            if interval_metrics:
                st.session_state.comparison_interval_metrics = interval_metrics
            
            st.success("All models completed!")
            st.rerun()

with run_col2:
    if any(st.session_state.get(f"{m.lower()}_results") for m in ALL_MODELS):
        if st.button("Re-run All", use_container_width=True):
            for m in ALL_MODELS:
                st.session_state[f"{m.lower()}_results"] = None
            st.session_state.comparison_metrics = None
            st.session_state.comparison_interval_metrics = None
            st.rerun()

# Check if we have results
available_models = [m for m in ALL_MODELS if st.session_state.get(f"{m.lower()}_results") is not None]

if available_models:
    st.divider()
    
    # Status comparison
    st.subheader("Model Status")
    
    cols = st.columns(len(available_models))
    for i, model_key in enumerate(available_models):
        res = st.session_state[f"{model_key.lower()}_results"]
        with cols[i]:
            status = res.status.get(selected_series, "unknown")
            if status == "ok":
                st.success(f"{model_key}: {status}")
            elif status.startswith("fallback"):
                st.warning(f"{model_key}: {status}")
            else:
                st.error(f"{model_key}: {status}")
            st.caption(f"Model: {res.model_name}")
    
    # Tier 1: Point Forecast Comparison
    st.divider()
    st.subheader("🎯 Tier 1 — Point Forecast Accuracy (MAE / RMSE / MAPE)")
    
    tier1_models = [m[0] for m in MODEL_GROUPS["Tier 1 — Point Forecast Competitors"]]
    available_tier1 = [m for m in tier1_models if m in available_models]
    
    if st.session_state.get("comparison_metrics"):
        metrics = st.session_state.comparison_metrics
        
        # Filter to available tier1 models
        tier1_metrics = {k: v for k, v in metrics.items() if k in available_tier1}
        
        if tier1_metrics:
            metrics_df = pd.DataFrame(tier1_metrics).T.round(4)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Chart
            chart_rows = []
            for model_name, model_metrics in tier1_metrics.items():
                row = {"Series": selected_series, "Model": model_name}
                row.update(model_metrics)
                chart_rows.append(row)
            chart_df = pd.DataFrame(chart_rows)
            from streamlit_app.utils.plotting import create_metric_comparison_bar
            fig_metrics = create_metric_comparison_bar(chart_df, "MAE")
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Best model explanation
            if "CSP" in tier1_metrics:
                csp_mae = tier1_metrics["CSP"].get("MAE", float('inf'))
                others = {k: v.get("MAE", float('inf')) for k, v in tier1_metrics.items() if k != "CSP"}
                
                if others:
                    best_other = min(others, key=others.get)
                    best_other_mae = others[best_other]
                    
                    st.divider()
                    st.subheader("Why this model wins on point accuracy")
                    
                    if csp_mae < best_other_mae:
                        st.success(f"✅ **CSP wins** (MAE: {csp_mae:.4f} vs {best_other}: {best_other_mae:.4f})")
                        st.markdown("""
                        **Why CSP wins on point accuracy:**
                        - CSP's conformal intervals adapt to local volatility
                        - Better handling of seasonal transitions
                        - Calibrated coverage means fewer surprises in production
                        """)
                    elif best_other_mae < csp_mae:
                        st.warning(f"⚠️ **{best_other} wins** (MAE: {best_other_mae:.4f} vs CSP: {csp_mae:.4f})")
                        st.markdown(f"""
                        **Why {best_other} wins on point accuracy:**
                        - Extremely regular, repetitive pattern
                        - CSP's conformal intervals don't improve point accuracy
                        - Simpler model is sufficient and more interpretable
                        """)
                    else:
                        st.info("Tie — both models equivalent on this series")
    
    # Tier 2: Interval Calibration Comparison
    st.divider()
    st.subheader("📏 Tier 2 — Interval Calibration (Coverage / Pinball / CRPS)")
    
    if st.session_state.get("comparison_interval_metrics"):
        int_metrics = st.session_state.comparison_interval_metrics
        
        int_df = pd.DataFrame(int_metrics).T.round(4)
        st.dataframe(int_df, use_container_width=True)
        
        st.caption("""
        **Coverage**: Fraction of actuals within the prediction interval (target = confidence level)
        **Pinball Loss**: Quantile loss — lower is better
        **CRPS**: Continuous Ranked Probability Score — lower is better
        """)
        
        # Explanation
        if "CSP" in int_metrics and "SeasonalNaive" in int_metrics:
            csp_cov = int_metrics["CSP"].get("Coverage_95%", 0)
            sn_cov = int_metrics["SeasonalNaive"].get("Coverage_95%", 0)
            
            st.divider()
            st.subheader("Why interval calibration matters")
            
            if abs(csp_cov - 0.95) < abs(sn_cov - 0.95):
                st.success(f"✅ **CSP intervals better calibrated** (95% coverage: CSP={csp_cov:.1%}, SN={sn_cov:.1%})")
                st.markdown("""
                **Why CSP wins on calibration:**
                - Conformal prediction provides finite-sample coverage guarantees
                - Intervals adapt to local volatility and seasonality
                - SeasonalNaive intervals assume constant variance
                """)
            else:
                st.warning(f"⚠️ **SeasonalNaive intervals closer to nominal** (95% coverage: CSP={csp_cov:.1%}, SN={sn_cov:.1%})")
                st.markdown("""
                **Why SeasonalNaive calibration is competitive here:**
                - Very regular, stationary series
                - Constant variance assumption holds
                - Simpler method sufficient for calibration
                """)
    
    # Comparison plot - ALL models on one chart
    if available_models:
        st.divider()
        st.subheader("📈 Forecast Overlay — All Models")
        
        # Collect all available model forecasts
        model_forecasts = {}
        for model_key in available_models:
            res = st.session_state.get(f"{model_key.lower()}_results")
            if res:
                model_forecasts[model_key] = res.forecast_df
        
        from streamlit_app.utils.plotting import create_all_models_comparison_plot
        fig = create_all_models_comparison_plot(
            history_df=pdf,
            model_forecasts=model_forecasts,
            series_id=selected_series,
            history_len=100,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Forecast tables
    st.divider()
    st.subheader("Forecast Values")
    
    tabs = st.tabs([m for m in ALL_MODELS if m in available_models])
    for i, model_key in enumerate([m for m in ALL_MODELS if m in available_models]):
        with tabs[i]:
            res = st.session_state[f"{model_key.lower()}_results"]
            fcst = res.forecast_df[res.forecast_df['unique_id'] == selected_series]
            pred_col = res.model_name
            display_cols = ['ds', pred_col]
            for level in sorted(cfg.levels):
                lo = f"{pred_col}-lo-{level}"
                hi = f"{pred_col}-hi-{level}"
                if lo in fcst.columns and hi in fcst.columns:
                    display_cols.extend([lo, hi])
            
            st.dataframe(fcst[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
    
    # Download - using new fixed-key pattern
    st.divider()
    st.subheader("Download Comparison Results")
    
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
    
    # Build download with all available models
    csp_res = st.session_state.get("csp_results")
    sn_res = st.session_state.get("seasonalnaive_results")
    arima_res = st.session_state.get("autoarima_results")
    ets_res = st.session_state.get("autoets_results")
    theta_res = st.session_state.get("autotheta_results")
    
    download_files, download_names = create_comparison_download(
        csp_forecast=csp_res.forecast_df if csp_res else pd.DataFrame(),
        sn_forecast=sn_res.forecast_df if sn_res else pd.DataFrame(),
        csp_status=csp_res.status if csp_res else {},
        sn_status=sn_res.status if sn_res else {},
        config=config_dict,
    )
    
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Comparison CSV",
            data=download_files["comparison_forecasts_csv"],
            file_name=f"comparison_{selected_series}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Full Comparison Excel",
            data=download_files["comparison_results_xlsx"],
            file_name=f"comparison_{selected_series}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

else:
    st.info("Click **Run All Models** to generate forecasts from CSP, AutoARIMA, AutoETS, AutoTheta, and SeasonalNaive.")
    
    st.divider()
    st.subheader("What Will Be Compared")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tier 1 — Point Forecast Competitors**")
        for key, label, desc in MODEL_GROUPS["Tier 1 — Point Forecast Competitors"]:
            st.markdown(f"- **{label}** ({desc})")
    
    with c2:
        st.markdown("**Tier 2 — Interval Calibration Reference**")
        for key, label, desc in MODEL_GROUPS["Tier 2 — Interval Calibration Reference"]:
            st.markdown(f"- **{label}** ({desc})")