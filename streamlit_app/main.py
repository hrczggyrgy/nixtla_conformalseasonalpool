import sys
from pathlib import Path

# Add project root to sys.path for Streamlit Cloud
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="CSP Universal Forecast",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG
from streamlit_app.components.app_shell import render_top_bar, reset_workflow

def init_session_state():
    for key in SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None
    if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
        from csp_universal_forecast import CSPConfig
        st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

init_session_state()

# Render top bar
render_top_bar()

# Navigation pages
pages = {
    "Home": [
        st.Page("pages/0_Home.py", title="Home"),
    ],
    "Data Upload": [
        st.Page("pages/1_Data_Upload.py", title="Data Upload"),
    ],
    "Exploratory Analysis": [
        st.Page("pages/2_EDA.py", title="EDA"),
    ],
    "Configuration": [
        st.Page("pages/3_Column_Config.py", title="Column Config"),
        st.Page("pages/4_Forecast_Config.py", title="Forecast Config"),
    ],
    "Results": [
        st.Page("pages/5_Forecast_Results.py", title="Forecast Results"),
        st.Page("pages/6_Model_Comparison.py", title="Model Comparison"),
        st.Page("pages/7_Download_Results.py", title="Download Results"),
    ],
}

pg = st.navigation(pages)

# Run the page
pg.run()

# Sidebar footer
with st.sidebar:
    st.divider()
    st.caption("CSP Universal Forecast v2")