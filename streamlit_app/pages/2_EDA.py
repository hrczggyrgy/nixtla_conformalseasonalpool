import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import probplot
from scipy.signal import periodogram, find_peaks
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Ensure session state is initialized
from streamlit_app.config import SESSION_KEYS, DEFAULT_CSP_CONFIG
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None
if "forecast_cfg" not in st.session_state or st.session_state.forecast_cfg is None:
    from csp_universal_forecast import CSPConfig
    st.session_state.forecast_cfg = CSPConfig(**DEFAULT_CSP_CONFIG)

st.title("Exploratory Data Analysis")
st.caption("Diagnose data quality & forecastability before configuring.")

if st.session_state.raw_df is None:
    st.warning("Please upload data first on the Data Upload page.")
    st.stop()

df = st.session_state.raw_df

# Column selection (inline, not in expander)
c1, c2, c3 = st.columns(3)

with c1:
    date_col = st.selectbox("Date Column", df.columns.tolist(), index=0, key="eda_date_col")

with c2:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    value_col = st.selectbox(
        "Value Column",
        numeric_cols,
        index=0 if numeric_cols else None,
        key="eda_value_col",
    )

with c3:
    id_options = ["(none)"] + [c for c in df.columns if c not in [date_col, value_col]]
    id_col = st.selectbox("Series ID Column (optional)", id_options, index=0, key="eda_id_col")
    id_col = None if id_col == "(none)" else id_col

if not value_col:
    st.error("No numeric column available for analysis.")
    st.stop()

# Prepare series
if id_col:
    series_ids = df[id_col].unique()
    selected_id = st.selectbox("Select Series", series_ids, key="eda_series_select")
    series_df = df[df[id_col] == selected_id].sort_values(date_col)
else:
    series_df = df.sort_values(date_col)
    selected_id = "Series"

series_data = series_df[value_col].dropna()

# --- DIAGNOSTIC CHECKLIST (Main View) ---
st.divider()
st.subheader("📋 Data Quality Checklist")

# Run diagnostics
checks = []

# 1. Date parsing
if pd.api.types.is_datetime64_any_dtype(series_df[date_col]):
    checks.append(("✅", "Date column parsed", f"{pd.to_datetime(series_df[date_col], errors='coerce').notna().mean():.0%} valid", "success"))
else:
    try:
        parsed = pd.to_datetime(series_df[date_col], errors="coerce")
        rate = parsed.notna().mean()
        if rate > 0.9:
            checks.append(("✅", "Date column parsed", f"{rate:.0%} valid", "success"))
        elif rate > 0.7:
            checks.append(("⚠️", "Date column parsed", f"{rate:.0%} valid — some rows will be dropped", "warning"))
        else:
            checks.append(("❌", "Date column issues", f"Only {rate:.0%} parseable", "error"))
    except Exception:
        checks.append(("❌", "Date column unparseable", "Cannot convert to datetime", "error"))

# 2. Duplicate timestamps
if id_col:
    dup = series_df.duplicated(subset=[date_col]).sum()
else:
    dup = series_df.duplicated(subset=[date_col]).sum()
if dup == 0:
    checks.append(("✅", "No duplicate timestamps", "", "success"))
elif dup < len(series_df) * 0.01:
    checks.append(("⚠️", "Duplicate timestamps", f"{dup} duplicates (will be averaged)", "warning"))
else:
    checks.append(("❌", "Many duplicate timestamps", f"{dup} duplicates — review data source", "error"))

# 3. Gaps
date_diffs = pd.to_datetime(series_df[date_col]).sort_values().diff().dropna()
if len(date_diffs) > 0:
    expected = date_diffs.mode()[0]
    gaps = date_diffs[date_diffs > expected * 1.5]
    if len(gaps) == 0:
        checks.append(("✅", "Regular time index", "No significant gaps", "success"))
    elif len(gaps) < len(series_df) * 0.05:
        checks.append(("⚠️", "Minor gaps", f"{len(gaps)} gaps detected (will be interpolated)", "warning"))
    else:
        checks.append(("❌", "Irregular time index", f"{len(gaps)} large gaps", "error"))
else:
    checks.append(("❌", "Single timestamp", "Cannot infer frequency", "error"))

# 4. Missing values
missing_pct = series_data.isna().mean()
if missing_pct == 0:
    checks.append(("✅", "No missing values", "", "success"))
elif missing_pct < 0.05:
    checks.append(("⚠️", "Few missing values", f"{missing_pct:.1%} — will be interpolated", "warning"))
elif missing_pct < 0.2:
    checks.append(("⚠️", "Moderate missing", f"{missing_pct:.1%} — may affect quality", "warning"))
else:
    checks.append(("❌", "High missing rate", f"{missing_pct:.1%} — consider imputation", "error"))

# 5. Short series
min_obs = 10
if len(series_data) < min_obs:
    checks.append(("❌", "Too short for forecasting", f"{len(series_data)} observations (min {min_obs})", "error"))
else:
    checks.append(("✅", "Sufficient length", f"{len(series_data)} observations", "success"))

# 6. Constant/near-constant
if series_data.nunique() <= 1:
    checks.append(("❌", "Constant series", "CSP may produce zero-width intervals", "error"))
elif series_data.nunique() < 5:
    checks.append(("⚠️", "Near-constant", f"Only {series_data.nunique()} unique values", "warning"))
else:
    checks.append(("✅", "Variable series", f"{series_data.nunique()} unique values", "success"))

# 7. Outliers (IQR)
q1, q3 = series_data.quantile([0.25, 0.75])
iqr = q3 - q1
if iqr > 0:
    outliers = ((series_data < q1 - 3 * iqr) | (series_data > q3 + 3 * iqr)).sum()
    if outliers == 0:
        checks.append(("✅", "No extreme outliers", "", "success"))
    elif outliers < len(series_data) * 0.02:
        checks.append(("⚠️", "Few outliers", f"{outliers} extreme (will be clipped if enabled)", "warning"))
    else:
        checks.append(("❌", "Many outliers", f"{outliers} extreme ({outliers/len(series_data):.1%})", "error"))
else:
    checks.append(("✅", "IQR check N/A", "Constant series", "success"))

# Display checklist
for icon, label, detail, status in checks:
    colors = {"success": "#10b981", "warning": "#f59e0b", "error": "#ef4444"}
    color = colors[status]
    st.markdown(
        f'<div style="display: flex; align-items: center; padding: 10px 14px; '
        f'margin: 6px 0; background: {color}15; border-left: 4px solid {color}; '
        f'border-radius: 6px;">'
        f'<span style="font-size: 1.2em; margin-right: 12px;">{icon}</span>'
        f'<div><strong>{label}</strong>'
        f'{f"<br><small>{detail}</small>" if detail else ""}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# Overall readiness
error_count = sum(1 for _, _, _, s in checks if s == "error")
warning_count = sum(1 for _, _, _, s in checks if s == "warning")

if error_count > 0:
    st.error(f"🚫 **Not ready for forecasting** — {error_count} critical issue(s)")
elif warning_count > 0:
    st.warning(f"⚠️ **Forecastable with warnings** — {warning_count} issue(s) to review")
else:
    st.success("✅ **Ready for forecasting** — no critical issues detected")

# Natural language summary
st.divider()
st.subheader("📝 Dataset Summary")

def generate_nl_summary(checks, series_data, selected_id, id_col):
    parts = []
    
    # Series type
    if id_col:
        n_series = df[id_col].nunique()
        parts.append(f"This is a **multi-series panel** with **{n_series} series** (currently viewing: {selected_id})")
    else:
        parts.append("This is a **single time series**")
    
    # Length
    parts.append(f"**{len(series_data)} observations**")
    
    # Frequency
    try:
        from csp_universal_forecast import _infer_freq
        freq = _infer_freq(pd.to_datetime(series_df[date_col], errors="coerce").dropna())
        freq_labels = {"D": "daily", "W": "weekly", "M": "monthly", "MS": "monthly", 
                       "Q": "quarterly", "QS": "quarterly", "A": "annual", "Y": "annual"}
        freq_label = freq_labels.get(freq.split("-")[0], freq)
        parts.append(f"appears **{freq_label}**")
    except Exception:
        parts.append("frequency unclear")
    
    # Issues
    issues = [c[1] for c in checks if c[3] in ("error", "warning")]
    if issues:
        parts.append(f"**Issues:** {', '.join(issues)}")
    
    # Forecastability
    if any(c[3] == "error" for c in checks):
        parts.append("⚠️ **May need data cleaning before forecasting**")
    elif warning_count > 0:
        parts.append("✅ **Forecastable with minor caveats**")
    else:
        parts.append("✅ **Clean — ready for forecasting**")
    
    return "  •  ".join(parts)

st.markdown(generate_nl_summary(checks, series_data, selected_id, id_col))

# --- ADVANCED DIAGNOSTICS (Collapsed by default) ---
with st.expander("🔬 Advanced Diagnostics", expanded=False):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Time Series", "Distribution", "Autocorrelation", "Seasonality", "Summary"
    ])

    with tab1:
        st.subheader(f"Time Series: {selected_id}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series_df[date_col],
            y=series_df[value_col],
            mode="lines+markers",
            name=selected_id,
            line=dict(color="#1f77b4", width=1.5),
            marker=dict(size=3),
            hovertemplate="Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
        ))
        
        fig.update_layout(
            title=f"{selected_id} - {value_col}",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode="x unified",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gap detection
        if date_col and pd.api.types.is_datetime64_any_dtype(series_df[date_col]):
            date_diffs = series_df[date_col].diff().dropna()
            if len(date_diffs) > 0:
                expected = date_diffs.mode()[0]
                gaps = date_diffs[date_diffs > expected * 1.5]
                if len(gaps) > 0:
                    st.warning(f"{len(gaps)} potential gaps detected (expected freq: {expected})")
                else:
                    st.success("No significant gaps in time index")

    with tab2:
        st.subheader("Distribution Analysis")
        
        c1, c2 = st.columns(2)
        
        with c1:
            fig_hist = px.histogram(
                series_data, nbins=50, marginal="box",
                title="Histogram with Box Plot",
                template="plotly_white",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with c2:
            fig_qq = go.Figure()
            (osm, osr), (slope, intercept, r) = probplot(series_data, dist="norm", plot=None)
            fig_qq.add_trace(go.Scatter(
                x=osm, y=osr, mode="markers", name="Data",
                marker=dict(color="#1f77b4"),
            ))
            fig_qq.add_trace(go.Scatter(
                x=osm, y=slope*osm+intercept, mode="lines", name="Normal Fit",
                line=dict(color="red", dash="dash"),
            ))
            fig_qq.update_layout(title="Q-Q Plot (Normal)", template="plotly_white", height=400)
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # Statistics
        stats_df = pd.DataFrame({
            "Statistic": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Skew", "Kurtosis"],
            "Value": [
                len(series_data),
                series_data.mean(),
                series_data.std(),
                series_data.min(),
                series_data.quantile(0.25),
                series_data.median(),
                series_data.quantile(0.75),
                series_data.max(),
                series_data.skew(),
                series_data.kurtosis(),
            ],
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Autocorrelation Analysis")
        
        max_lag = st.slider("Max Lag", 10, 100, 40, key="eda_max_lag")
        
        if len(series_data) > max_lag * 2:
            acf_vals = acf(series_data, nlags=max_lag, fft=True)
            pacf_vals = pacf(series_data, nlags=max_lag, method="ols")
            
            conf_int = 1.96 / np.sqrt(len(series_data))
            
            c1, c2 = st.columns(2)
            
            with c1:
                fig_acf = go.Figure()
                fig_acf.add_trace(go.Bar(
                    x=list(range(max_lag + 1)), y=acf_vals,
                    name="ACF", marker_color="#1f77b4",
                ))
                fig_acf.add_hline(y=conf_int, line_dash="dash", line_color="red")
                fig_acf.add_hline(y=-conf_int, line_dash="dash", line_color="red")
                fig_acf.update_layout(title="ACF", template="plotly_white", height=400)
                st.plotly_chart(fig_acf, use_container_width=True)
            
            with c2:
                fig_pacf = go.Figure()
                fig_pacf.add_trace(go.Bar(
                    x=list(range(max_lag + 1)), y=pacf_vals,
                    name="PACF", marker_color="#ff7f0e",
                ))
                fig_pacf.add_hline(y=conf_int, line_dash="dash", line_color="red")
                fig_pacf.add_hline(y=-conf_int, line_dash="dash", line_color="red")
                fig_pacf.update_layout(title="PACF", template="plotly_white", height=400)
                st.plotly_chart(fig_pacf, use_container_width=True)
            
            # Significant lags
            sig_lags = [i for i, v in enumerate(acf_vals) if abs(v) > conf_int and i > 0]
            if sig_lags:
                st.info(f"Significant ACF lags (95% CI): {sig_lags[:10]}...")
        else:
            st.warning("Series too short for autocorrelation analysis.")

    with tab4:
        st.subheader("Seasonality Detection")
        
        # Periodogram
        fs = 1.0
        freqs, psd = periodogram(series_data.values, fs=fs)
        periods = 1 / freqs[1:]
        psd_vals = psd[1:]
        
        peaks, _ = find_peaks(psd_vals, height=np.max(psd_vals) * 0.1)
        peak_periods = periods[peaks]
        peak_powers = psd_vals[peaks]
        
        fig_per = go.Figure()
        fig_per.add_trace(go.Scatter(
            x=periods, y=psd_vals, mode="lines", name="PSD",
            line=dict(color="#1f77b4"),
        ))
        if len(peaks) > 0:
            fig_per.add_trace(go.Scatter(
                x=peak_periods, y=peak_powers, mode="markers", name="Peaks",
                marker=dict(color="red", size=10),
            ))
        fig_per.update_layout(
            title="Periodogram", xaxis_title="Period", yaxis_title="Power",
            template="plotly_white", height=400,
        )
        st.plotly_chart(fig_per, use_container_width=True)
        
        if len(peak_periods) > 0:
            top_peaks = sorted(zip(peak_periods, peak_powers), key=lambda x: x[1], reverse=True)[:5]
            st.write("**Top Detected Periods:**")
            for p, pw in top_peaks:
                st.write(f"- Period: {p:.1f} (power: {pw:.2f})")
        
        # STL Decomposition
        st.divider()
        st.subheader("STL Decomposition")
        
        if len(series_data) > 20:
            period_guess = st.number_input(
                "Seasonal Period", min_value=2, max_value=len(series_data)//2,
                value=int(peak_periods[0]) if len(peak_periods) > 0 else 7,
            )
            
            if st.button("Run STL Decomposition"):
                with st.spinner("Computing STL..."):
                    decomp = seasonal_decompose(
                        series_data, model="additive", period=period_guess, extrapolate_trend="freq"
                    )
                    
                    from plotly.subplots import make_subplots
                    fig_stl = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                    )
                    
                    components = [
                        ("Observed", decomp.observed, "#1f77b4"),
                        ("Trend", decomp.trend, "#ff7f0e"),
                        ("Seasonal", decomp.seasonal, "#2ca02c"),
                        ("Residual", decomp.resid, "#d62728"),
                    ]
                    
                    for i, (name, data, color) in enumerate(components, 1):
                        # Handle both pandas Series/DataFrame and numpy arrays
                        if hasattr(data, 'index'):
                            x_vals = data.index
                        else:
                            x_vals = range(len(data))
                        
                        if hasattr(data, 'values'):
                            y_vals = data.values
                        else:
                            y_vals = data
                        
                        fig_stl.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            name=name,
                            line=dict(color=color, width=1.5),
                        ), row=i, col=1)
                    
                    fig_stl.update_layout(
                        title=f"STL Decomposition (Period={period_guess})",
                        template="plotly_white", height=700, showlegend=False,
                    )
                    st.plotly_chart(fig_stl, use_container_width=True)
        else:
            st.warning("Series too short for STL decomposition.")

    with tab5:
        st.subheader("Data Summary")
        
        # Stationarity test
        if len(series_data) > 10:
            adf_result = adfuller(series_data.dropna())
            st.write("**Augmented Dickey-Fuller Test (Stationarity):**")
            c1, c2 = st.columns(2)
            c1.metric("ADF Statistic", f"{adf_result[0]:.4f}")
            c2.metric("p-value", f"{adf_result[1]:.4f}")
            
            if adf_result[1] < 0.05:
                st.success("Series is likely stationary (p < 0.05)")
            else:
                st.warning("Series may be non-stationary (p >= 0.05)")
            
            with st.expander("Critical Values"):
                for key, val in adf_result[4].items():
                    st.write(f"{key}: {val:.4f}")
        
        # Multi-series info
        if id_col and id_col in df.columns:
            st.divider()
            st.subheader("Multi-Series Overview")
            
            series_stats = []
            for sid in df[id_col].unique():
                s = df[df[id_col] == sid][value_col].dropna()
                if len(s) > 0:
                    series_stats.append({
                        "Series ID": sid,
                        "Count": len(s),
                        "Mean": s.mean(),
                        "Std": s.std(),
                        "Min": s.min(),
                        "Max": s.max(),
                        "Start": df[df[id_col] == sid][date_col].min(),
                        "End": df[df[id_col] == sid][date_col].max(),
                    })
            
            if series_stats:
                st.dataframe(pd.DataFrame(series_stats), use_container_width=True)

# Store EDA summary
st.session_state.eda_summary = {
    "date_col": date_col,
    "value_col": value_col,
    "id_col": id_col,
    "n_series": len(df[id_col].unique()) if id_col else 1,
    "total_obs": len(df),
    "checks_passed": sum(1 for c in checks if c[3] == "success"),
    "checks_warnings": sum(1 for c in checks if c[3] == "warning"),
    "checks_errors": sum(1 for c in checks if c[3] == "error"),
}