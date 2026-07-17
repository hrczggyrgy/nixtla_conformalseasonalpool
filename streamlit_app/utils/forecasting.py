"""Forecasting utilities for Streamlit app."""
import pandas as pd
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
    
    model_col = [c for c in forecast_df.columns if c not in ["unique_id", "ds"] and not c.endswith(("-lo-", "-hi-"))]
    model_name = model_col[0] if model_col else "CSP"
    
    return ForecastResult(
        forecast_df=forecast_df,
        status=status,
        model_name=model_name,
    )


def get_model_name(forecast_df: pd.DataFrame) -> str:
    """Extract model name from forecast columns."""
    for c in forecast_df.columns:
        if c not in ["unique_id", "ds"] and not c.endswith(("-lo-", "-hi-")):
            return c
    return "CSP"


def run_model_comparison(
    df: pd.DataFrame,
    cfg: CSPConfig,
) -> Dict[str, pd.DataFrame]:
    """Run CSP and SeasonalNaive for comparison."""
    from statsforecast import StatsForecast
    from statsforecast.models import ConformalSeasonalPool, SeasonalNaive
    
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
        from csp_universal_forecast import _infer_season_length
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