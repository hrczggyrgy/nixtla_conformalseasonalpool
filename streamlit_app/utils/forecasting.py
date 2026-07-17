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


def run_seasonal_naive_with_config(
    df: pd.DataFrame,
    cfg: CSPConfig,
) -> ForecastResult:
    """Run SeasonalNaive forecast with the given config."""
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive
    from csp_universal_forecast import _infer_season_length

    freq = pd.infer_freq(pd.to_datetime(df["ds"]).sort_values().drop_duplicates()) or "D"
    y_all = df["y"].to_numpy()
    season_length = _infer_season_length(freq, y_all)

    sn_model = SeasonalNaive(season_length=season_length)
    sf = StatsForecast(models=[sn_model], freq=freq, n_jobs=-1)
    sn_forecasts = sf.forecast(df=df, h=cfg.h, level=cfg.levels)

    model_col = [c for c in sn_forecasts.columns if c not in ["unique_id", "ds"] and "-lo-" not in c and "-hi-" not in c]
    model_name = model_col[0] if model_col else "SeasonalNaive"

    return ForecastResult(
        forecast_df=sn_forecasts,
        status={uid: "ok" for uid in df["unique_id"].unique()},
        model_name=model_name,
    )


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


def run_model_comparison(
    df: pd.DataFrame,
    cfg: CSPConfig,
) -> Dict[str, pd.DataFrame]:
    """Run CSP and SeasonalNaive for comparison."""
    from statsforecast import StatsForecast
    from statsforecast.models import ConformalSeasonalPool, SeasonalNaive
    from csp_universal_forecast import _infer_season_length

    # CSP
    try:
        csp_cfg = CSPConfig(
            h=cfg.h,
            levels=cfg.levels,
            min_obs_per_series=cfg.min_obs_per_series,
            max_series_per_batch=cfg.max_series_per_batch,
            outlier_clip=cfg.outlier_clip,
            outlier_iqr_mult=cfg.outlier_iqr_mult,
            random_seed=cfg.random_seed,
            verbose=cfg.verbose,
        )
        csp_result = run_csp_with_config(df, csp_cfg)
        csp_forecasts = csp_result.forecast_df
    except Exception as e:
        csp_forecasts = pd.DataFrame()

    # SeasonalNaive
    try:
        freq = pd.infer_freq(pd.to_datetime(df["ds"]).sort_values().drop_duplicates()) or "D"
        y_all = df["y"].to_numpy()
        season_length = _infer_season_length(freq, y_all)

        sn_model = SeasonalNaive(season_length=season_length)
        sf = StatsForecast(models=[sn_model], freq=freq, n_jobs=-1)
        sn_forecasts = sf.forecast(df=df, h=cfg.h, level=cfg.levels)
    except Exception as e:
        sn_forecasts = pd.DataFrame()

    return {
        "CSP": csp_forecasts,
        "SeasonalNaive": sn_forecasts,
    }