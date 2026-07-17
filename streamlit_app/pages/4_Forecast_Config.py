import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from csp_universal_forecast import CSPConfig
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG, VALIDATION_RULES

# Ensure session state is initialized
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.title("Forecast Configuration")
st.caption("Configure forecasting parameters for CSP and SeasonalNaive models.")

if st.session_state.processed_df is None:
    st.warning("Please configure columns first on the Column Config page.")
    st.stop()

pdf = st.session_state.processed_df

# Current config
cfg = st.session_state.forecast_cfg

st.subheader("Forecast Horizon & Confidence")

c1, c2 = st.columns(2)
with c1:
    h = st.number_input(
        "Forecast Horizon (h)",
        min_value=1,
        max_value=365,
        value=cfg.h,
        step=1,
        help="Number of future periods to forecast",
    )

with c2:
    levels = st.multiselect(
        "Confidence Levels (%)",
        options=[50, 60, 70, 80, 90, 95, 99],
        default=cfg.levels,
        help="Prediction interval confidence levels",
    )

st.divider()

st.subheader("Data Quality & Robustness")

c1, c2, c3 = st.columns(3)
with c1:
    min_obs = st.number_input(
        "Min Observations per Series",
        min_value=2,
        max_value=100,
        value=cfg.min_obs_per_series,
        step=1,
        help="Series with fewer observations will be dropped",
    )

with c2:
    outlier_clip = st.toggle(
        "Outlier Clipping (IQR)",
        value=cfg.outlier_clip,
        help="Clip outliers using IQR method before fitting",
    )

with c3:
    if outlier_clip:
        iqr_mult = st.slider(
            "IQR Multiplier",
            min_value=1.0,
            max_value=5.0,
            value=cfg.outlier_iqr_mult,
            step=0.5,
            help="Bounds: Q1 - k*IQR, Q3 + k*IQR",
        )
    else:
        iqr_mult = cfg.outlier_iqr_mult

st.divider()

st.subheader("Performance & Reproducibility")

c1, c2, c3 = st.columns(3)
with c1:
    batch_size = st.number_input(
        "Max Series per Batch",
        min_value=50,
        max_value=5000,
        value=cfg.max_series_per_batch,
        step=50,
        help="Batch size for large panels (memory management)",
    )

with c2:
    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=2**31 - 1,
        value=cfg.random_seed,
        step=1,
        help="For reproducible results",
    )

with c3:
    verbose = st.toggle("Verbose Logging", value=cfg.verbose)

st.divider()

# Save config with archive
if st.button("Save Configuration & Run Forecast", type="primary", use_container_width=True):
    from streamlit_app.components.app_shell import archive_current_results
    
    # Archive previous results if any
    archive_current_results()
    
    new_cfg = CSPConfig(
        h=h,
        levels=levels,
        min_obs_per_series=min_obs,
        max_series_per_batch=batch_size,
        outlier_clip=outlier_clip,
        outlier_iqr_mult=iqr_mult,
        random_seed=seed,
        verbose=verbose,
        date_col=None,
        value_col=None,
        id_col=None,
    )
    st.session_state.forecast_cfg = new_cfg
    st.session_state.last_run_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success("Configuration saved! Proceed to Forecast Results to run.")
    st.rerun()

# Display current config
st.divider()
st.subheader("Current Configuration")

config_dict = {
    "h": h,
    "levels": levels,
    "min_obs_per_series": min_obs,
    "max_series_per_batch": batch_size,
    "outlier_clip": outlier_clip,
    "outlier_iqr_mult": iqr_mult,
    "random_seed": seed,
    "verbose": verbose,
}
st.json(config_dict)

# Show data readiness
st.divider()
st.subheader("Data Readiness Check")

n_series = pdf['unique_id'].nunique()
total_obs = len(pdf)
avg_obs = total_obs / n_series if n_series > 0 else 0
short_series = (pdf.groupby('unique_id').size() < min_obs).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Series", n_series)
c2.metric("Total Observations", f"{total_obs:,}")
c3.metric("Avg Obs/Series", f"{avg_obs:.0f}")
c4.metric("Series < Min Obs", f"{short_series} Warning" if short_series > 0 else "0 OK")

if short_series > 0:
    st.warning(f"⚠️ {short_series} series will be dropped (less than {min_obs} observations)")

if n_series > batch_size:
    st.info(f"📦 {n_series} series will be processed in {n_series // batch_size + 1} batches")

# Estimated runtime
est_series_per_sec = 50  # rough estimate
est_time = n_series / est_series_per_sec
if est_time < 10:
    st.caption(f"⏱️ Estimated runtime: ~{est_time:.0f} seconds")
elif est_time < 60:
    st.caption(f"⏱️ Estimated runtime: ~{est_time:.0f} seconds")
else:
    st.caption(f"⏱️ Estimated runtime: ~{est_time/60:.1f} minutes")

st.divider()

# Quick actions
c1, c2 = st.columns(2)
with c1:
    if st.button("Go to Forecast Results", use_container_width=True):
        st.switch_page("pages/5_Forecast_Results.py")

with c2:
    if st.button("Go to Model Comparison", use_container_width=True):
        st.switch_page("pages/6_Model_Comparison.py")