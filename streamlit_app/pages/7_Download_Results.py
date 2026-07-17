import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from streamlit_app.utils.export import create_forecast_download, create_comparison_download, to_excel_bytes

# Ensure session state is initialized
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG
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

st.subheader("Available Downloads")

if has_csp:
    with st.expander("CSP Forecast Results", expanded=True):
        res = st.session_state.csp_results
        fcst_df = res["forecast_df"]
        status = res["status"]
        model_name = res["model_name"]
        config = res["config"]
        
        st.write(f"Series: {fcst_df['unique_id'].nunique()} | Forecasts: {len(fcst_df):,} rows")
        
        download_files = create_forecast_download(fcst_df, status, config, model_name)
        
        cols = st.columns(4)
        for i, (fname, fdata) in enumerate(download_files.items()):
            with cols[i % 4]:
                st.download_button(
                    f"Download {fname}",
                    data=fdata,
                    file_name=fname,
                    mime="text/csv" if fname.endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

if has_sn:
    with st.expander("SeasonalNaive Forecast Results", expanded=True):
        res = st.session_state.sn_results
        fcst_df = res["forecast_df"]
        status = res["status"]
        model_name = res["model_name"]
        config = res["config"]
        
        st.write(f"Series: {fcst_df['unique_id'].nunique()} | Forecasts: {len(fcst_df):,} rows")
        
        download_files = create_forecast_download(fcst_df, status, config, model_name)
        
        cols = st.columns(4)
        for i, (fname, fdata) in enumerate(download_files.items()):
            with cols[i % 4]:
                st.download_button(
                    f"Download {fname}",
                    data=fdata,
                    file_name=fname,
                    mime="text/csv" if fname.endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

if has_both:
    st.divider()
    with st.expander("Model Comparison Results", expanded=True):
        csp_res = st.session_state.csp_results
        sn_res = st.session_state.sn_results
        
        download_files = create_comparison_download(
            csp_forecast=csp_res["forecast_df"],
            sn_forecast=sn_res["forecast_df"],
            config=csp_res["config"],
        )
        
        cols = st.columns(3)
        for i, (fname, fdata) in enumerate(download_files.items()):
            with cols[i % 3]:
                st.download_button(
                    f"Download {fname}",
                    data=fdata,
                    file_name=fname,
                    mime="text/csv" if fname.endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        
        if st.session_state.comparison_metrics:
            metrics_df = pd.DataFrame(st.session_state.comparison_metrics).T.round(4)
            st.download_button(
                "Download Metrics CSV",
                data=metrics_df.to_csv().encode('utf-8'),
                file_name="model_comparison_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

# Complete package
if has_csp or has_sn:
    st.divider()
    st.subheader("Complete Package")
    
    if st.button("Prepare Complete Export", use_container_width=True):
        sheets = {}
        
        if has_csp:
            csp_res = st.session_state.csp_results
            sheets["CSP_Forecasts"] = csp_res["forecast_df"]
            sheets["CSP_Status"] = pd.DataFrame([
                {"series_id": k, "status": v} for k, v in csp_res["status"].items()
            ])
            sheets["CSP_Config"] = pd.DataFrame([csp_res["config"]])
        
        if has_sn:
            sn_res = st.session_state.sn_results
            sheets["SN_Forecasts"] = sn_res["forecast_df"]
            sheets["SN_Status"] = pd.DataFrame([
                {"series_id": k, "status": v} for k, v in sn_res["status"].items()
            ])
            sheets["SN_Config"] = pd.DataFrame([sn_res["config"]])
        
        if has_both:
            csp_fcst = csp_res["forecast_df"]
            sn_fcst = sn_res["forecast_df"]
            
            csp_cols = [c for c in csp_fcst.columns if c not in ['unique_id', 'ds']]
            sn_cols = [c for c in sn_fcst.columns if c not in ['unique_id', 'ds']]
            
            csp_renamed = csp_fcst.rename(columns={c: f"CSP_{c}" for c in csp_cols})
            sn_renamed = sn_fcst.rename(columns={c: f"SN_{c}" for c in sn_cols})
            
            merged = pd.merge(csp_renamed, sn_renamed, on=['unique_id', 'ds'], how='outer')
            sheets["Comparison"] = merged.sort_values(['unique_id', 'ds'])
            
            if st.session_state.comparison_metrics:
                sheets["Metrics"] = pd.DataFrame(st.session_state.comparison_metrics).T
        
        if st.session_state.raw_df is not None:
            sheets["Original_Data"] = st.session_state.raw_df.head(10000)
        
        if st.session_state.processed_df is not None:
            sheets["Processed_Data"] = st.session_state.processed_df
        
        if st.session_state.col_config:
            sheets["Column_Config"] = pd.DataFrame([st.session_state.col_config])
        
        excel_bytes = to_excel_bytes(sheets)
        
        st.download_button(
            "Download Complete Package (Excel)",
            data=excel_bytes,
            file_name=f"csp_forecast_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )
        st.success("Complete package ready for download!")

st.divider()
st.caption("Tip: Use the complete package for sharing results with colleagues or for audit trails.")