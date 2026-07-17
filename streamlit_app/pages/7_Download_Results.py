import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from streamlit_app.utils.export import create_forecast_download, create_comparison_download, to_excel_bytes
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG, EXPORT_PRESETS

# Ensure session state is initialized
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    from csp_universal_forecast import CSPConfig
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.title("Download Results")
st.caption("Export forecasts, comparisons, and configuration in various formats.")

if st.session_state.csp_results is None and st.session_state.sn_results is None:
    st.warning("No results to download. Run forecasts first on the Forecast Results or Model Comparison pages.")
    st.stop()

# Check what's available
has_csp = st.session_state.csp_results is not None
has_sn = st.session_state.sn_results is not None
has_both = has_csp and has_sn

# --- EXPORT PRESETS ---
st.subheader("📦 Export Presets")

preset_options = []
for key, preset in EXPORT_PRESETS.items():
    preset_options.append({
        "key": key,
        "name": preset["name"],
        "description": preset["description"],
    })

selected_preset = st.radio(
    "Choose a preset bundle:",
    options=[p["key"] for p in preset_options],
    format_func=lambda k: next(p["name"] for p in preset_options if p["key"] == k),
    horizontal=True,
)

preset_info = next(p for p in preset_options if p["key"] == selected_preset)
st.caption(preset_info["description"])

# --- CUSTOM BUILD ---
st.divider()
st.subheader("🔧 Custom Build")

include = {}

if has_csp:
    st.markdown("**CSP Results**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        include["csp_forecast"] = st.checkbox("Forecasts", value=True, key="inc_csp_fcst")
    with c2:
        include["csp_status"] = st.checkbox("Status", value=True, key="inc_csp_status")
    with c3:
        include["csp_config"] = st.checkbox("Config", value=True, key="inc_csp_config")
    with c4:
        include["csp_model_card"] = st.checkbox("Model Card", value=False, key="inc_csp_card")

if has_sn:
    st.markdown("**SeasonalNaive Results**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        include["sn_forecast"] = st.checkbox("Forecasts", value=True, key="inc_sn_fcst")
    with c2:
        include["sn_status"] = st.checkbox("Status", value=True, key="inc_sn_status")
    with c3:
        include["sn_config"] = st.checkbox("Config", value=True, key="inc_sn_config")
    with c4:
        include["sn_model_card"] = st.checkbox("Model Card", value=False, key="inc_sn_card")

if has_both:
    st.markdown("**Comparison**")
    c1, c2, c3 = st.columns(3)
    with c1:
        include["comparison"] = st.checkbox("Forecast Comparison", value=True, key="inc_cmp")
    with c2:
        include["metrics"] = st.checkbox("Metrics", value=True, key="inc_metrics")
    with c3:
        include["status_cmp"] = st.checkbox("Status Comparison", value=True, key="inc_status_cmp")

st.markdown("**Raw & Processed Data**")
c1, c2, c3 = st.columns(3)
with c1:
    include["raw_data"] = st.checkbox("Raw Data (first 10k rows)", value=False, key="inc_raw")
with c2:
    include["processed_data"] = st.checkbox("Processed Data", value=False, key="inc_processed")
with c3:
    include["col_config"] = st.checkbox("Column Config", value=True, key="inc_col_config")

st.markdown("**Metadata**")
c1, c2 = st.columns(2)
with c1:
    include["metadata"] = st.checkbox("Run Metadata", value=True, key="inc_meta")
with c2:
    include["eda_summary"] = st.checkbox("EDA Summary", value=False, key="inc_eda")

# --- GENERATE ---
st.divider()

def build_export_package(include_dict):
    """Build the export package based on selections."""
    sheets = {}
    
    if has_csp and st.session_state.csp_results:
        res = st.session_state.csp_results
        if include_dict.get("csp_forecast"):
            sheets["CSP_Forecasts"] = res["forecast_df"]
        if include_dict.get("csp_status"):
            sheets["CSP_Status"] = pd.DataFrame([
                {"series_id": k, "status": v} for k, v in res["status"].items()
            ])
        if include_dict.get("csp_config"):
            sheets["CSP_Config"] = pd.DataFrame([res["config"]])
        if include_dict.get("csp_model_card"):
            # Generate model card
            card = generate_model_card(res)
            sheets["CSP_Model_Card"] = pd.DataFrame([{"Content": card}])
    
    if has_sn and st.session_state.sn_results:
        res = st.session_state.sn_results
        if include_dict.get("sn_forecast"):
            sheets["SN_Forecasts"] = res["forecast_df"]
        if include_dict.get("sn_status"):
            sheets["SN_Status"] = pd.DataFrame([
                {"series_id": k, "status": v} for k, v in res["status"].items()
            ])
        if include_dict.get("sn_config"):
            sheets["SN_Config"] = pd.DataFrame([res["config"]])
        if include_dict.get("sn_model_card"):
            card = generate_model_card(res)
            sheets["SN_Model_Card"] = pd.DataFrame([{"Content": card}])
    
    if has_both and include_dict.get("comparison"):
        csp_res = st.session_state.csp_results
        sn_res = st.session_state.sn_results
        
        csp_fcst = csp_res["forecast_df"]
        sn_fcst = sn_res["forecast_df"]
        
        csp_cols = [c for c in csp_fcst.columns if c not in ['unique_id', 'ds']]
        sn_cols = [c for c in sn_fcst.columns if c not in ['unique_id', 'ds']]
        
        csp_renamed = csp_fcst.rename(columns={c: f"CSP_{c}" for c in csp_cols})
        sn_renamed = sn_fcst.rename(columns={c: f"SN_{c}" for c in sn_cols})
        
        merged = pd.merge(csp_renamed, sn_renamed, on=['unique_id', 'ds'], how='outer')
        sheets["Comparison"] = merged.sort_values(['unique_id', 'ds'])
    
    if has_both and include_dict.get("metrics") and st.session_state.comparison_metrics:
        sheets["Metrics"] = pd.DataFrame(st.session_state.comparison_metrics).T
    
    if has_both and include_dict.get("status_cmp"):
        csp_res = st.session_state.csp_results
        sn_res = st.session_state.sn_results
        all_series = set(csp_res["status"].keys()) | set(sn_res["status"].keys())
        status_rows = []
        for s in sorted(all_series):
            status_rows.append({
                "Series ID": s,
                "CSP Status": csp_res["status"].get(s, "N/A"),
                "SeasonalNaive Status": sn_res["status"].get(s, "N/A"),
            })
        sheets["Status_Comparison"] = pd.DataFrame(status_rows)
    
    if include_dict.get("raw_data") and st.session_state.raw_df is not None:
        sheets["Original_Data"] = st.session_state.raw_df.head(10000)
    
    if include_dict.get("processed_data") and st.session_state.processed_df is not None:
        sheets["Processed_Data"] = st.session_state.processed_df
    
    if include_dict.get("col_config") and st.session_state.col_config:
        sheets["Column_Config"] = pd.DataFrame([st.session_state.col_config])
    
    if include_dict.get("eda_summary") and st.session_state.eda_summary:
        sheets["EDA_Summary"] = pd.DataFrame([st.session_state.eda_summary])
    
    if include_dict.get("metadata"):
        meta = {
            "Property": ["Generated", "Horizon", "Confidence Levels", "Min Obs/Series", 
                        "Outlier Clipping", "IQR Multiplier", "Random Seed"],
            "Value": [
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                st.session_state.forecast_cfg.h if hasattr(st.session_state.forecast_cfg, 'h') else st.session_state.forecast_cfg.get('h'),
                ", ".join(map(str, st.session_state.forecast_cfg.levels)) if hasattr(st.session_state.forecast_cfg, 'levels') else str(st.session_state.forecast_cfg.get('levels')),
                st.session_state.forecast_cfg.min_obs_per_series if hasattr(st.session_state.forecast_cfg, 'min_obs_per_series') else st.session_state.forecast_cfg.get('min_obs_per_series'),
                st.session_state.forecast_cfg.outlier_clip if hasattr(st.session_state.forecast_cfg, 'outlier_clip') else st.session_state.forecast_cfg.get('outlier_clip'),
                st.session_state.forecast_cfg.outlier_iqr_mult if hasattr(st.session_state.forecast_cfg, 'outlier_iqr_mult') else st.session_state.forecast_cfg.get('outlier_iqr_mult'),
                st.session_state.forecast_cfg.random_seed if hasattr(st.session_state.forecast_cfg, 'random_seed') else st.session_state.forecast_cfg.get('random_seed'),
            ],
        }
        sheets["Metadata"] = pd.DataFrame(meta)
    
    return to_excel_bytes(sheets)


def generate_model_card(res):
    """Generate a markdown model card."""
    model_name = res["model_name"]
    status = res["status"]
    config = res["config"]
    
    ok = sum(1 for v in status.values() if v == "ok")
    fb = sum(1 for v in status.values() if v.startswith("fallback"))
    dropped = sum(1 for v in status.values() if v.startswith("dropped"))
    
    card = f"""# {model_name} Model Card

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- **Horizon (h):** {config.get('h', 'N/A')}
- **Confidence Levels:** {', '.join(map(str, config.get('levels', [])))}
- **Min Observations/Series:** {config.get('min_obs_per_series', 'N/A')}
- **Outlier Clipping (IQR):** {config.get('outlier_clip', 'N/A')}
- **IQR Multiplier:** {config.get('outlier_iqr_mult', 'N/A')}
- **Random Seed:** {config.get('random_seed', 'N/A')}

## Series Status
| Series ID | Status |
|-----------|--------|
"""
    
    for sid, stat in sorted(status.items()):
        card += f"| {sid} | {stat} |\n"
    
    card += f"""
## Summary
- **OK:** {ok}
- **Fallback:** {fb}
- **Dropped:** {dropped}

## Forecast Preview (first 5 series)
```markdown
"""
    
    fcst = res["forecast_df"]
    preview = fcst.groupby('unique_id').head(3).to_markdown(index=False)
    card += preview
    card += "\n```"
    
    return card


if st.button("📥 Generate & Download Excel", use_container_width=True, type="primary"):
    with st.spinner("Building export package..."):
        sheets = build_export_package(include)
        st.session_state.export_bytes = sheets
        st.session_state.export_filename = f"csp_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    st.success("Package ready!")

if st.session_state.get("export_bytes"):
    st.download_button(
        "💾 Download Excel Package",
        data=st.session_state.export_bytes,
        file_name=st.session_state.export_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )

# --- EXPORT SUMMARY ---
st.divider()
st.subheader("📋 Export Summary")

if has_csp:
    csp_ok = sum(1 for v in st.session_state.csp_results["status"].values() if v == "ok")
    csp_fb = sum(1 for v in st.session_state.csp_results["status"].values() if v.startswith("fallback"))
    st.write(f"**CSP:** {csp_ok} OK, {csp_fb} fallback")

if has_sn:
    sn_ok = sum(1 for v in st.session_state.sn_results["status"].values() if v == "ok")
    st.write(f"**SeasonalNaive:** {sn_ok} OK")

st.write(f"**Horizon:** {st.session_state.forecast_cfg.h if hasattr(st.session_state.forecast_cfg, 'h') else st.session_state.forecast_cfg.get('h')}")
levels = st.session_state.forecast_cfg.levels if hasattr(st.session_state.forecast_cfg, 'levels') else st.session_state.forecast_cfg.get('levels')
st.write(f"**Confidence Levels:** {', '.join(map(str, levels))}")

st.divider()
st.caption("Tip: Use the Complete Package for sharing results with colleagues or for audit trails.")