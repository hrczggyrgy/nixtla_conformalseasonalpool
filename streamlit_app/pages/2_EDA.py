import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import probplot
from scipy.signal import periodogram, find_peaks
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Exploratory Data Analysis")
st.caption("Explore your time series data before forecasting.")

if st.session_state.raw_df is None:
    st.warning("Please upload data first on the Data Upload page.")
    st.stop()

df = st.session_state.raw_df

# Column selection
with st.expander("EDA Column Selection", expanded=True):
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

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Time Series",
    "Distribution",
    "Autocorrelation",
    "Seasonality",
    "Summary",
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
                
                fig_stl = go.Figure()
                components = [
                    ("Observed", decomp.observed, "#1f77b4"),
                    ("Trend", decomp.trend, "#ff7f0e"),
                    ("Seasonal", decomp.seasonal, "#2ca02c"),
                    ("Residual", decomp.resid, "#d62728"),
                ]
                
                for i, (name, data, color) in enumerate(components, 1):
                    fig_stl.add_trace(go.Scatter(
                        x=series_data.index if hasattr(series_data.index, '__iter__') else range(len(data)),
                        y=data.values if hasattr(data, 'values') else data,
                        mode="lines", name=name, line=dict(color=color, width=1.5),
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
}