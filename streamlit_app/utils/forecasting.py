"""Forecasting utilities for Streamlit app."""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from csp_universal_forecast import CSPConfig, run_csp_forecast


@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame
    status: Dict[str, str]
    model_name: str


def run_csp_with_config(
    df: pd.DataFrame,
    cfg: CSPConfig,
) -> ForecastResult:
    """Run CSP forecast with the given config."""
    forecast_df, status, _ = run_csp_forecast(df, cfg=cfg)

    model_col = [c for c in forecast_df.columns if c not in ["unique_id", "ds"] and "-lo-" not in c and "-hi-" not in c]
    model_name = model_col[0] if model_col else "CSP"

    return ForecastResult(
        forecast_df=forecast_df,
        status=status,
        model_name=model_name,
    )


def get_model_name(forecast_df: pd.DataFrame) -> str:
    """Extract model name from forecast columns."""
    for c in forecast_df.columns:
        if c not in ["unique_id", "ds"] and "-lo-" not in c and "-hi-" not in c:
            return c
    return "CSP"


def run_model_with_config(
    df: pd.DataFrame,
    cfg: CSPConfig,
    model_class,
    model_name: str,
    **model_kwargs,
) -> ForecastResult:
    """Run a StatsForecast model with the given config."""
    from statsforecast import StatsForecast
    from csp_universal_forecast import _infer_season_length

    freq = pd.infer_freq(pd.to_datetime(df["ds"]).sort_values().drop_duplicates()) or "D"
    y_all = df["y"].to_numpy()
    season_length = _infer_season_length(freq, y_all)

    model = model_class(season_length=season_length, **model_kwargs)
    sf = StatsForecast(models=[model], freq=freq, n_jobs=-1)
    forecasts = sf.forecast(df=df, h=cfg.h, level=cfg.levels)

    model_col = [c for c in forecasts.columns if c not in ["unique_id", "ds"] and "-lo-" not in c and "-hi-" not in c]
    name = model_col[0] if model_col else model_name

    return ForecastResult(
        forecast_df=forecasts,
        status={uid: "ok" for uid in df["unique_id"].unique()},
        model_name=name,
    )


def run_seasonal_naive_with_config(
    df: pd.DataFrame,
    cfg: CSPConfig,
) -> ForecastResult:
    """Run SeasonalNaive forecast with the given config."""
    from statsforecast.models import SeasonalNaive
    return run_model_with_config(df, cfg, SeasonalNaive, "SeasonalNaive")


def run_auto_arima_with_config(
    df: pd.DataFrame,
    cfg: CSPConfig,
) -> ForecastResult:
    """Run AutoARIMA forecast with the given config."""
    from statsforecast.models import AutoARIMA
    return run_model_with_config(df, cfg, AutoARIMA, "AutoARIMA")


def run_auto_ets_with_config(
    df: pd.DataFrame,
    cfg: CSPConfig,
) -> ForecastResult:
    """Run AutoETS forecast with the given config."""
    from statsforecast.models import AutoETS
    return run_model_with_config(df, cfg, AutoETS, "AutoETS")


def run_auto_theta_with_config(
    df: pd.DataFrame,
    cfg: CSPConfig,
) -> ForecastResult:
    """Run AutoTheta forecast with the given config."""
    from statsforecast.models import AutoTheta
    return run_model_with_config(df, cfg, AutoTheta, "AutoTheta")


def compute_backtest_metrics(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    series_id: str,
) -> Optional[Dict[str, float]]:
    """Compute backtest metrics for a single series."""
    hist = historical_df[historical_df["unique_id"] == series_id].sort_values("ds")
    fcst = forecast_df[forecast_df["unique_id"] == series_id].sort_values("ds")

    if hist.empty or fcst.empty:
        return None

    model_name = get_model_name(forecast_df)
    pred_col = model_name

    if pred_col not in fcst.columns:
        return None

    y_true = hist["y"].values[-len(fcst):]  # Align with forecast horizon
    y_pred = fcst[pred_col].values

    if len(y_true) != len(y_pred):
        return None

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def compute_interval_metrics(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    series_id: str,
    level: int = 95,
) -> Optional[Dict[str, float]]:
    """Compute interval calibration metrics (coverage, pinball loss, CRPS)."""
    hist = historical_df[historical_df["unique_id"] == series_id].sort_values("ds")
    fcst = forecast_df[forecast_df["unique_id"] == series_id].sort_values("ds")

    if hist.empty or fcst.empty:
        return None

    model_name = get_model_name(forecast_df)
    lo_col = f"{model_name}-lo-{level}"
    hi_col = f"{model_name}-hi-{level}"

    if lo_col not in fcst.columns or hi_col not in fcst.columns:
        return None

    y_true = hist["y"].values[-len(fcst):]
    lo = fcst[lo_col].values
    hi = fcst[hi_col].values
    pred = fcst[get_model_name(forecast_df)].values

    if len(y_true) != len(lo):
        return None

    # Coverage
    covered = np.sum((y_true >= lo) & (y_true <= hi))
    coverage = covered / len(y_true)

    # Pinball loss (quantile loss)
    alpha = level / 100.0
    pinball = np.mean(np.where(y_true < lo, (1 - alpha) * (lo - y_true), alpha * (y_true - hi)))

    # Approximate CRPS (using uniform distribution within interval)
    crps = np.mean((hi - lo) / 4 + (lo - y_true)**2 / (hi - lo) * (y_true < lo) + (y_true - hi)**2 / (hi - lo) * (y_true > hi))

    return {
        f"Coverage_{level}%": coverage,
        f"PinballLoss_{level}%": float(pinball),
        f"CRPS_{level}%": float(crps),
    }


def run_all_models(
    df: pd.DataFrame,
    cfg: CSPConfig,
) -> Dict[str, ForecastResult]:
    """Run all models: CSP, SeasonalNaive, AutoARIMA, AutoETS, AutoTheta."""
    from statsforecast.models import SeasonalNaive, AutoARIMA, AutoETS, AutoTheta

    results = {}

    # CSP
    try:
        csp_result = run_csp_with_config(df, cfg)
        results["CSP"] = csp_result
    except Exception as e:
        results["CSP"] = None

    # SeasonalNaive (reference anchor)
    try:
        results["SeasonalNaive"] = run_seasonal_naive_with_config(df, cfg)
    except Exception as e:
        results["SeasonalNaive"] = None

    # AutoARIMA
    try:
        results["AutoARIMA"] = run_auto_arima_with_config(df, cfg)
    except Exception as e:
        results["AutoARIMA"] = None

    # AutoETS
    try:
        results["AutoETS"] = run_auto_ets_with_config(df, cfg)
    except Exception as e:
        results["AutoETS"] = None

    # AutoTheta
    try:
        results["AutoTheta"] = run_auto_theta_with_config(df, cfg)
    except Exception as e:
        results["AutoTheta"] = None

    return results