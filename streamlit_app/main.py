import streamlit as st

st.set_page_config(
    page_title="CSP Universal Forecast",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG

def init_session_state():
    for key in SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None
    if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
        from csp_universal_forecast import CSPConfig
        st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

def clear_results():
    st.session_state.csp_results = None
    st.session_state.sn_results = None
    st.session_state.comparison_metrics = None

def reset_app():
    for key in SESSION_KEYS:
        st.session_state[key] = None
    from csp_universal_forecast import CSPConfig
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

init_session_state()

pages = {
    "Data Upload": [
        st.Page("streamlit_app/pages/1_Data_Upload.py", title="Data Upload"),
    ],
    "Exploratory Analysis": [
        st.Page("streamlit_app/pages/2_EDA.py", title="EDA"),
    ],
    "Configuration": [
        st.Page("streamlit_app/pages/3_Column_Config.py", title="Column Config"),
        st.Page("streamlit_app/pages/4_Forecast_Config.py", title="Forecast Config"),
    ],
    "Results": [
        st.Page("streamlit_app/pages/5_Forecast_Results.py", title="Forecast Results"),
        st.Page("streamlit_app/pages/6_Model_Comparison.py", title="Model Comparison"),
        st.Page("streamlit_app/pages/7_Download_Results.py", title="Download"),
    ],
}

pg = st.navigation(pages)
pg.run()

with st.sidebar:
    st.divider()
    st.caption("CSP Universal Forecast v2")
    if st.button("Reset App", use_container_width=True):
        reset_app()
        st.rerun()
    
    if st.session_state.raw_df is not None:
        st.caption(f"Data: {st.session_state.raw_df.shape[0]} rows x {st.session_state.raw_df.shape[1]} cols")
    if st.session_state.processed_df is not None:
        n_series = st.session_state.processed_df["unique_id"].nunique()
        st.caption(f"Series: {n_series}")
    if st.session_state.csp_results is not None:
        st.caption("CSP Forecast: Ready")
    if st.session_state.sn_results is not None:
        st.caption("SeasonalNaive: Ready")