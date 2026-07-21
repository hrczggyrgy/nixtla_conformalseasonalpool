"""App shell components: top bar, sidebar stepper, progress tracking."""
import streamlit as st
from streamlit_app.config import EXPORT_PRESETS, VALIDATION_RULES

STEPS = [
    {"id": "home", "label": "Home", "icon": "🏠", "page": "0_Home.py"},
    {"id": "upload", "label": "Upload", "icon": "📤", "page": "1_Data_Upload.py"},
    {"id": "eda", "label": "Explore", "icon": "🔍", "page": "2_EDA.py"},
    {"id": "columns", "label": "Columns", "icon": "🔗", "page": "3_Column_Config.py"},
    {"id": "forecast", "label": "Forecast", "icon": "⚙️", "page": "4_Forecast_Config.py"},
    {"id": "results", "label": "Results", "icon": "📈", "page": "5_Forecast_Results.py"},
    {"id": "compare", "label": "Compare", "icon": "⚖️", "page": "6_Model_Comparison.py"},
    {"id": "download", "label": "Download", "icon": "💾", "page": "7_Download_Results.py"},
]

STEP_ORDER = [s["id"] for s in STEPS]

def get_step_status(step_id: str) -> str:
    """Determine status of a step: not_started, ready, done, needs_attention."""
    if step_id == "home":
        return "done"
    
    if step_id == "upload":
        if st.session_state.raw_df is not None:
            return "done"
        return "not_started"
    
    if step_id == "eda":
        if st.session_state.raw_df is not None:
            if st.session_state.eda_summary:
                return "done"
            return "ready"
        return "not_started"
    
    if step_id == "columns":
        if st.session_state.col_config:
            return "done"
        if st.session_state.raw_df is not None:
            return "ready"
        return "not_started"
    
    if step_id == "forecast":
        if st.session_state.forecast_cfg:
            if st.session_state.csp_results or st.session_state.sn_results:
                return "done"
            return "ready"
        return "not_started"
    
    if step_id == "results":
        if st.session_state.csp_results or st.session_state.sn_results:
            return "done"
        return "not_started"
    
    if step_id == "compare":
        if st.session_state.csp_results and st.session_state.sn_results:
            return "done"
        if st.session_state.csp_results or st.session_state.sn_results:
            return "ready"
        return "not_started"
    
    if step_id == "download":
        if st.session_state.csp_results or st.session_state.sn_results:
            return "done"
        return "not_started"
    
    return "not_started"


def render_top_bar():
    """Render top bar with app title, dataset info, and reset button."""
    cols = st.columns([3, 2, 1, 1])

    with cols[0]:
        st.markdown("## Automated Forecasting")
        st.caption(
            "Powered by Conformal Seasonal Pool (CSP), a model introduced by "
            "Valery Manokhin and implemented in Nixtla's StatsForecast."
        )
    
    with cols[1]:
        if st.session_state.raw_df is not None:
            n_series = 1
            if st.session_state.processed_df is not None:
                n_series = st.session_state.processed_df["unique_id"].nunique()
            elif st.session_state.col_config and st.session_state.col_config.get("id_col"):
                n_series = st.session_state.raw_df[st.session_state.col_config["id_col"]].nunique()
            
            st.caption(
                f"📁 **{st.session_state.uploaded_filename or 'Data'}**  •  "
                f"{len(st.session_state.raw_df):,} rows  •  "
                f"{n_series} series  •  "
                f"{st.session_state.raw_df.shape[1]} cols"
            )
        else:
            st.caption("No data loaded")
    
    with cols[2]:
        if st.session_state.raw_df is not None:
            if st.button("🔄 Reset", use_container_width=True, help="Clear all data and start over"):
                reset_workflow()
                st.rerun()
    
    with cols[3]:
        if st.session_state.raw_df is not None:
            if st.button("💾 Download", use_container_width=True):
                st.switch_page("pages/7_Download_Results.py")


def render_sidebar_stepper(current_step: str = None):
    """Render sidebar with stepper showing progress through workflow."""
    with st.sidebar:
        st.markdown("### Workflow Progress")
        
        for i, step in enumerate(STEPS):
            status = get_step_status(step["id"])
            
            status_icons = {
                "done": "✅",
                "ready": "▶️",
                "needs_attention": "⚠️",
                "not_started": "⏳",
            }
            
            status_colors = {
                "done": "#10b981",
                "ready": "#3b82f6",
                "needs_attention": "#f59e0b",
                "not_started": "#9ca3af",
            }
            
            is_current = (current_step == step["id"]) or (status == "ready" and all(
                get_step_status(s["id"]) != "not_started" for s in STEPS[:i]
            ))
            
            icon = status_icons[status]
            color = status_colors[status]
            
            label = f"{icon} {step['icon']} {step['label']}"
            
            if is_current and status in ("ready", "not_started"):
                st.markdown(
                    f'<div style="padding: 8px 12px; background: {color}20; '
                    f'border-left: 3px solid {color}; border-radius: 4px; margin: 4px 0;">'
                    f'<strong>{label}</strong></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="padding: 8px 12px; color: {color}; margin: 4px 0;">'
                    f'{label}</div>',
                    unsafe_allow_html=True
                )
        
        st.divider()
        
        if st.session_state.raw_df is not None:
            st.markdown("### Dataset Summary")
            st.caption(f"**File:** {st.session_state.uploaded_filename}")
            st.caption(f"**Rows:** {len(st.session_state.raw_df):,}")
            st.caption(f"**Columns:** {st.session_state.raw_df.shape[1]}")
            
            if st.session_state.processed_df is not None:
                n_series = st.session_state.processed_df["unique_id"].nunique()
                st.caption(f"**Series:** {n_series}")
                date_range = f"{st.session_state.processed_df['ds'].min().date()} → {st.session_state.processed_df['ds'].max().date()}"
                st.caption(f"**Date range:** {date_range}")
            
            if st.session_state.col_config:
                cc = st.session_state.col_config
                st.caption(f"**Date col:** {cc.get('date_col', '—')}")
                st.caption(f"**Value col:** {cc.get('value_col', '—')}")
                st.caption(f"**ID col:** {cc.get('id_col') or '(single series)'}")
            
            if st.session_state.forecast_cfg:
                cfg = st.session_state.forecast_cfg
                levels = ", ".join(f"{l}%" for l in cfg.levels) if hasattr(cfg, 'levels') else f"{cfg.get('levels', [80, 95])}"
                st.caption(f"**Horizon:** {cfg.h if hasattr(cfg, 'h') else cfg.get('h', 14)}")
                st.caption(f"**Confidence:** {levels}")
            
            if st.session_state.last_run_timestamp:
                st.caption(f"**Last run:** {st.session_state.last_run_timestamp}")


def reset_workflow():
    """Clear all session state and reset to initial state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def archive_current_results():
    """Move current results to archive for comparison."""
    if st.session_state.get("csp_results") or st.session_state.get("sn_results"):
        st.session_state.last_results = {
            "csp_results": st.session_state.get("csp_results"),
            "sn_results": st.session_state.get("sn_results"),
            "comparison_metrics": st.session_state.get("comparison_metrics"),
            "comparison_interval_metrics": st.session_state.get("comparison_interval_metrics"),
            "col_config": st.session_state.get("col_config"),
            "forecast_cfg": st.session_state.get("forecast_cfg"),
            "timestamp": st.session_state.get("last_run_timestamp"),
        }


def get_step_page_map():
    """Return mapping of step IDs to page paths."""
    return {s["id"]: f"pages/{s['page']}" for s in STEPS}


def validate_step(step_id: str) -> tuple[bool, list[str], list[str]]:
    """Validate if a step can be completed. Returns (can_proceed, errors, warnings)."""
    errors = []
    warnings = []
    
    if step_id == "upload":
        if st.session_state.raw_df is None:
            errors.append("No data uploaded")
        else:
            df = st.session_state.raw_df
            if len(df) < VALIDATION_RULES["min_rows"]:
                warnings.append(f"Only {len(df)} rows (minimum {VALIDATION_RULES['min_rows']})")
            if df.isnull().sum().sum() / df.size > VALIDATION_RULES["max_missing_pct"] / 100:
                warnings.append("High percentage of missing values")
    
    elif step_id == "columns":
        if not st.session_state.col_config:
            errors.append("Column configuration not set")
        else:
            cc = st.session_state.col_config
            if cc.get("date_col") not in st.session_state.raw_df.columns:
                errors.append(f"Date column '{cc['date_col']}' not found")
            if cc.get("value_col") not in st.session_state.raw_df.columns:
                errors.append(f"Value column '{cc['value_col']}' not found")
    
    elif step_id == "forecast":
        if not st.session_state.forecast_cfg:
            errors.append("Forecast configuration not set")
    
    elif step_id in ("results", "compare", "download"):
        if not st.session_state.csp_results and not st.session_state.sn_results:
            errors.append("No forecast results available")
    
    return len(errors) == 0, errors, warnings
