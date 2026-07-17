"""Plotting utilities for Streamlit app."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict


def create_forecast_plot(
    forecast_df: pd.DataFrame,
    history_df: pd.DataFrame,
    series_ids: List[str],
    model_name: str = "CSP",
    history_col: str = "y",
    show_history: int = 100,
) -> go.Figure:
    """Create forecast plot with history and confidence intervals."""
    fig = go.Figure()
    
    # Filter to requested series
    hist = history_df[history_df["unique_id"].isin(series_ids)]
    fcst = forecast_df[forecast_df["unique_id"].isin(series_ids)]
    
    # Determine model column
    model_col = model_name
    if model_name not in fcst.columns:
        candidates = [c for c in fcst.columns if c not in ["unique_id", "ds"] and not c.endswith(("-lo-", "-hi-"))]
        model_col = candidates[0] if candidates else "CSP"
    
    # Find confidence interval columns
    lo_cols = [c for c in fcst.columns if c.startswith(f"{model_col}-lo-")]
    hi_cols = [c for c in fcst.columns if c.startswith(f"{model_col}-hi-")]
    levels = sorted(set(c.split("-")[-1] for c in lo_cols))
    
    colors = {80: ("rgba(31, 119, 180, 0.2)", "rgba(31, 119, 180, 0.3)"),
              95: ("rgba(255, 127, 14, 0.2)", "rgba(255, 127, 14, 0.3)")}
    
    for sid in series_ids:
        sid_hist = hist[hist["unique_id"] == sid].sort_values("ds")
        sid_fcst = fcst[fcst["unique_id"] == sid].sort_values("ds")
        
        if sid_hist.empty and sid_fcst.empty:
            continue
        
        # History
        hist_tail = sid_hist.tail(show_history) if len(sid_hist) > show_history else sid_hist
        if not hist_tail.empty:
            fig.add_trace(go.Scatter(
                x=hist_tail["ds"],
                y=hist_tail[history_col],
                mode="lines+markers",
                name=f"{sid} (history)",
                line=dict(color="#1f77b4", width=1.5),
                marker=dict(size=3),
                hovertemplate="Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            ))
        
        # Forecast
        if not sid_fcst.empty:
            fig.add_trace(go.Scatter(
                x=sid_fcst["ds"],
                y=sid_fcst[model_col],
                mode="lines+markers",
                name=f"{sid} ({model_col})",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                marker=dict(size=4),
                hovertemplate="Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>",
            ))
            
            # Confidence intervals
            for level in levels:
                lo_col = f"{model_col}-lo-{level}"
                hi_col = f"{model_col}-hi-{level}"
                if lo_col in sid_fcst.columns and hi_col in sid_fcst.columns:
                    fill_color = colors.get(level, ("rgba(128,128,128,0.2)", "rgba(128,128,128,0.3)"))
                    fig.add_trace(go.Scatter(
                        x=sid_fcst["ds"].tolist() + sid_fcst["ds"].tolist()[::-1],
                        y=sid_fcst[lo_col].tolist() + sid_fcst[hi_col].tolist()[::-1],
                        fill="toself",
                        fillcolor=fill_color[0],
                        line=dict(color="rgba(255,255,255,0)"),
                        name=f"{sid} {level}% CI",
                        hoverinfo="skip",
                        showlegend=True,
                    ))
    
    fig.update_layout(
        title=f"{model_name} Forecast",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_all_models_comparison_plot(
    history_df: pd.DataFrame,
    model_forecasts: Dict[str, pd.DataFrame],
    series_id: str,
    history_col: str = "y",
    history_len: int = 100,
) -> go.Figure:
    """Create single plot comparing all models on one chart."""
    hist = history_df[history_df["unique_id"] == series_id].sort_values("ds")
    
    # Model colors
    model_colors = {
        "CSP": "#ff7f0e",
        "AutoARIMA": "#1f77b4",
        "AutoETS": "#2ca02c",
        "AutoTheta": "#d62728",
        "SeasonalNaive": "#9467bd",
    }
    
    fig = go.Figure()
    
    # History
    if not hist.empty:
        hist_tail = hist.tail(history_len)
        fig.add_trace(go.Scatter(
            x=hist_tail["ds"],
            y=hist_tail[history_col],
            mode="lines+markers",
            name="History",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            hovertemplate="Date: %{x}<br>Actual: %{y:.2f}<extra></extra>",
        ))
    
    # Each model's forecast
    for model_key, forecast_df in model_forecasts.items():
        if forecast_df is None or forecast_df.empty:
            continue
            
        fcst = forecast_df[forecast_df["unique_id"] == series_id].sort_values("ds")
        if fcst.empty:
            continue
            
        # Find model column
        model_col = model_key
        if model_key not in fcst.columns:
            candidates = [c for c in fcst.columns if c not in ["unique_id", "ds"] and not c.endswith(("-lo-", "-hi-"))]
            model_col = candidates[0] if candidates else model_key
        
        if model_col not in fcst.columns:
            continue
        
        color = model_colors.get(model_key, "#888888")
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=fcst["ds"],
            y=fcst[model_col],
            mode="lines+markers",
            name=model_key,
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=5),
            hovertemplate=f"Date: %{{x}}<br>{model_key}: %{{y:.2f}}<extra></extra>",
        ))
        
        # Confidence intervals (only for models that have them)
        lo_cols = [c for c in fcst.columns if c.startswith(f"{model_col}-lo-")]
        hi_cols = [c for c in fcst.columns if c.startswith(f"{model_col}-hi-")]
        levels = sorted(set(c.split("-")[-1] for c in lo_cols))
        
        for level in levels:
            lo_col = f"{model_col}-lo-{level}"
            hi_col = f"{model_col}-hi-{level}"
            if lo_col in fcst.columns and hi_col in fcst.columns:
                fig.add_trace(go.Scatter(
                    x=fcst["ds"].tolist() + fcst["ds"].tolist()[::-1],
                    y=fcst[lo_col].tolist() + fcst[hi_col].tolist()[::-1],
                    fill="toself",
                    fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{model_key} {level}% CI",
                    showlegend=False,
                    hoverinfo="skip",
                ))
    
    fig.update_layout(
        title=f"All Models Comparison — Series: {series_id}",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_comparison_plot(
    history_df: pd.DataFrame,
    csp_forecast: pd.DataFrame,
    sn_forecast: pd.DataFrame,
    series_id: str,
    model_name: str = "y",
    history_len: int = 100,
) -> go.Figure:
    """Create comparison plot for CSP vs SeasonalNaive."""
    hist = history_df[history_df["unique_id"] == series_id].sort_values("ds")
    csp_f = csp_forecast[csp_forecast["unique_id"] == series_id].sort_values("ds")
    sn_f = sn_forecast[sn_forecast["unique_id"] == series_id].sort_values("ds")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("CSP Forecast", "SeasonalNaive Forecast"),
        shared_xaxes=True,
        vertical_spacing=0.1,
    )
    
    def add_forecast_traces(fig, forecast_df, model_name, row, color):
        if forecast_df.empty:
            return
        
        model_col = model_name
        if model_name not in forecast_df.columns:
            candidates = [c for c in forecast_df.columns if c not in ["unique_id", "ds"] and not c.endswith(("-lo-", "-hi-"))]
            model_col = candidates[0] if candidates else model_name
        
        lo_cols = [c for c in forecast_df.columns if c.startswith(f"{model_col}-lo-")]
        hi_cols = [c for c in forecast_df.columns if c.startswith(f"{model_col}-hi-")]
        levels = sorted(set(c.split("-")[-1] for c in lo_cols))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df[model_col],
            mode="lines+markers",
            name=model_name,
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=4),
        ), row=row, col=1)
        
        # CI
        for level in levels:
            lo = f"{model_col}-lo-{level}"
            hi = f"{model_col}-hi-{level}"
            if lo in forecast_df.columns and hi in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"].tolist() + forecast_df["ds"].tolist()[::-1],
                    y=forecast_df[lo].tolist() + forecast_df[hi].tolist()[::-1],
                    fill="toself",
                    fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{model_name} {level}% CI",
                    showlegend=row == 1,
                ), row=row, col=1)
    
    # History (top plot)
    if not hist.empty:
        hist_tail = hist.tail(history_len)
        fig.add_trace(go.Scatter(
            x=hist_tail["ds"],
            y=hist_tail[model_name],
            mode="lines+markers",
            name="History",
            line=dict(color="#1f77b4", width=1.5),
            marker=dict(size=3),
        ), row=1, col=1)
    
    add_forecast_traces(fig, csp_f, "CSP", 1, "#ff7f0e")
    add_forecast_traces(fig, sn_f, "SeasonalNaive", 2, "#2ca02c")
    
    # History (bottom plot)
    if not hist.empty:
        hist_tail = hist.tail(history_len)
        fig.add_trace(go.Scatter(
            x=hist_tail["ds"],
            y=hist_tail[model_name],
            mode="lines+markers",
            name="History",
            line=dict(color="#1f77b4", width=1.5),
            marker=dict(size=3),
            showlegend=False,
        ), row=2, col=1)
    
    fig.update_layout(
        title=f"Series: {series_id} - Model Comparison",
        template="plotly_white",
        height=700,
        hovermode="x unified",
    )
    return fig


"""Plotting utilities for Streamlit app."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict


def create_all_models_comparison_plot(
    history_df: pd.DataFrame,
    model_forecasts: Dict[str, pd.DataFrame],
    series_id: str,
    model_name: str = "y",
    history_len: int = 100,
) -> go.Figure:
    """Create a single overlay plot showing all models' forecasts."""
    hist = history_df[history_df["unique_id"] == series_id].sort_values("ds")
    
    # Model colors
    model_colors = {
        "CSP": "#ff7f0e",
        "AutoARIMA": "#1f77b4",
        "AutoETS": "#2ca02c",
        "AutoTheta": "#d62728",
        "SeasonalNaive": "#9467bd",
    }
    
    fig = go.Figure()
    
    # History
    if not hist.empty:
        hist_tail = hist.tail(history_len)
        fig.add_trace(go.Scatter(
            x=hist_tail["ds"],
            y=hist_tail["y"],
            mode="lines+markers",
            name="History",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            hovertemplate="Date: %{x}<br>Actual: %{y:.2f}<extra></extra>",
        ))
    
    # Each model's forecast
    for model_key, forecast_df in model_forecasts.items():
        fcst = forecast_df[forecast_df["unique_id"] == series_id].sort_values("ds")
        if fcst.empty:
            continue
            
        color = model_colors.get(model_key, "#9467bd")
        
        # Find model column
        model_col = model_key
        if model_key not in fcst.columns:
            candidates = [c for c in fcst.columns if c not in ["unique_id", "ds"] and not c.endswith(("-lo-", "-hi-"))]
            model_col = candidates[0] if candidates else model_key
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=fcst["ds"],
            y=fcst[model_col],
            mode="lines+markers",
            name=f"{model_key} Forecast",
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=5),
            hovertemplate="Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>",
        ))
        
        # Confidence intervals (only for first level to avoid clutter)
        lo_cols = [c for c in fcst.columns if c.startswith(f"{model_col}-lo-")]
        hi_cols = [c for c in fcst.columns if c.startswith(f"{model_col}-hi-")]
        levels = sorted(set(c.split("-")[-1] for c in lo_cols))
        
        # Show only 95% CI
        if "95" in levels:
            level = "95"
            lo_col = f"{model_col}-lo-{level}"
            hi_col = f"{model_col}-hi-{level}"
            if lo_col in fcst.columns and hi_col in fcst.columns:
                fig.add_trace(go.Scatter(
                    x=fcst["ds"].tolist() + fcst["ds"].tolist()[::-1],
                    y=fcst[lo_col].tolist() + fcst[hi_col].tolist()[::-1],
                    fill="toself",
                    fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{model_key} {level}% CI",
                    hoverinfo="skip",
                    showlegend=True,
                ))
    
    fig.update_layout(
        title=f"All Models Forecast Overlay — {series_id}",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_histogram_with_box(data: pd.Series) -> go.Figure:
    """Create histogram with marginal box plot."""
    import plotly.express as px
    
    fig = px.histogram(
        data,
        nbins=50,
        marginal="box",
        template="plotly_white",
        title="Distribution with Box Plot",
    )
    return fig


def create_qq_plot(data: pd.Series) -> go.Figure:
    """Create Q-Q plot."""
    from scipy.stats import probplot
    
    (osm, osr), (slope, intercept, r) = probplot(data.dropna(), dist="norm", plot=None)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers", name="Data",
        marker=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=osm, y=slope * osm + intercept, mode="lines", name="Normal Fit",
        line=dict(color="red", dash="dash"),
    ))
    fig.update_layout(
        title="Q-Q Plot (Normal)",
        template="plotly_white",
        height=400,
    )
    return fig


def create_acf_pacf_plot(acf_vals, pacf_vals, conf_int: float) -> go.Figure:
    """Create ACF and PACF plots."""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ACF", "PACF"),
    )
    
    lags = list(range(len(acf_vals)))
    
    fig.add_trace(go.Bar(
        x=lags, y=acf_vals, name="ACF", marker_color="#1f77b4",
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=lags, y=pacf_vals, name="PACF", marker_color="#ff7f0e",
    ), row=1, col=2)
    
    # Confidence intervals
    for row in [1, 2]:
        fig.add_hline(y=conf_int, line_dash="dash", line_color="red", row=row, col=1 if row == 1 else 2)
        fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=row, col=1 if row == 1 else 2)
    
    fig.update_layout(
        title="Autocorrelation Analysis",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig


def create_periodogram_plot(periods, psd, peak_periods=None, peak_powers=None) -> go.Figure:
    """Create periodogram plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=periods, y=psd, mode="lines", name="PSD",
        line=dict(color="#1f77b4"),
    ))
    
    if peak_periods is not None and len(peak_periods) > 0:
        fig.add_trace(go.Scatter(
            x=peak_periods, y=peak_powers, mode="markers", name="Peaks",
            marker=dict(color="red", size=10),
        ))
    
    fig.update_layout(
        title="Periodogram",
        xaxis_title="Period",
        yaxis_title="Power",
        template="plotly_white",
        height=400,
    )
    return fig


def create_stl_plot(decomp, period: int, dates=None) -> go.Figure:
    """Create STL decomposition plot."""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        shared_xaxes=True,
        vertical_spacing=0.05,
    )
    
    x_vals = dates if dates is not None else range(len(decomp.observed))
    
    components = [
        ("Observed", decomp.observed, "#1f77b4"),
        ("Trend", decomp.trend, "#ff7f0e"),
        ("Seasonal", decomp.seasonal, "#2ca02c"),
        ("Residual", decomp.resid, "#d62728"),
    ]
    
    for i, (name, data, color) in enumerate(components, 1):
        y_vals = data.values if hasattr(data, "values") else data
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines", name=name,
            line=dict(color=color, width=1.5),
        ), row=i, col=1)
    
    fig.update_layout(
        title=f"STL Decomposition (Period={period})",
        template="plotly_white",
        height=700,
        showlegend=False,
    )
    return fig


def create_forecast_histogram(forecast_df: pd.DataFrame, value_cols: List[str]) -> go.Figure:
    """Create histogram of forecast values by model."""
    fig = go.Figure()
    
    colors = {"CSP": "#ff7f0e", "SeasonalNaive": "#2ca02c"}
    
    for col in value_cols:
        if col in forecast_df.columns:
            model_name = col.replace("CSP-", "").replace("Adaptive-", "")
            fig.add_trace(go.Bar(
                x=forecast_df[col],
                name=model_name,
                marker_color=colors.get(model_name, "#1f77b4"),
                opacity=0.7,
                histnorm="probability density",
            ))
    
    fig.update_layout(
        title="Forecast Distribution Comparison",
        xaxis_title="Forecast Value",
        yaxis_title="Density",
        template="plotly_white",
        barmode="overlay",
    )
    return fig


def create_metric_comparison_bar(metrics_df: pd.DataFrame, metric_col: str = "MAE") -> go.Figure:
    """Create bar chart comparing model metrics."""
    import plotly.express as px
    
    fig = px.bar(
        metrics_df,
        x="Series",
        y=metric_col,
        color="Model",
        barmode="group",
        template="plotly_white",
        title=f"{metric_col} by Series and Model",
    )
    return fig