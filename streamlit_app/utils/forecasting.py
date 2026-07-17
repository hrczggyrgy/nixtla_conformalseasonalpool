"""Forecasting utilities for Streamlit app."""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from csp_universal_forecast import CSPConfig, run_csp_forecast
import concurrent.futures
import time


@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame
    status: Dict[str, str]
    model_name: str
    config: Dict = field(default_factory=dict)


def run_csp_with_config(
    df: pd.DataFrame,
    cfg: CSPConfig,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> Optional["ForecastResult"]:
    """Run CSP forecast with the given config and optional column overrides."""
    cfg_copy = CSPConfig(
        h=cfg.h,
        levels=cfg.levels,
        min_obs_per_series=cfg.min_obs_per_series,
        max_series_per_batch=cfg.max_series_per_batch,
        outlier_clip=cfg.outlier_clip,
        outlier_iqr_mult=cfg.outlier_iqr_mult,
        random_seed=cfg.random_seed,
        verbose=cfg.verbose,
        date_col=date_col,
        value_col=value_col,
        id_col=id_col,
    )

    try:
        forecast_df, status, _ = run_csp_forecast(df, cfg=cfg_copy)
    except Exception:
        return None

    model_col = [c for c in forecast_df.columns if c not in ["unique_id", "ds"] and "-lo-" not in c and "-hi-" not in c]
    model_name = model_col[0] if model_col else "CSP"

    return ForecastResult(
        forecast_df=forecast_df,
        status=status,
        model_name=model_name,
        config={
            "h": cfg.h,
            "levels": cfg.levels,
            "min_obs_per_series": cfg.min_obs_per_series,
            "max_series_per_batch": cfg.max_series_per_batch,
            "outlier_clip": cfg.outlier_clip,
            "outlier_iqr_mult": cfg.outlier_iqr_mult,
            "random_seed": cfg.random_seed,
            "verbose": cfg.verbose,
        },
    )


def get_model_name(forecast_df: pd.DataFrame) -> str:
    """Extract model name from forecast columns."""
    for c in forecast_df.columns:
        if c not in ["unique_id", "ds"] and "-lo-" not in c and "-hi-" not in c:
            return c
    return "CSP"


def _run_single_model(
    df: pd.DataFrame,
    cfg: "CSPConfig",
    model_class,
    model_name: str,
    timeout: int = 120,
) -> Optional["ForecastResult"]:
    """Run a single model with timeout."""
    from statsforecast import StatsForecast
    from csp_universal_forecast import _infer_season_length
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    def _run():
        freq = pd.infer_freq(pd.to_datetime(df["ds"]).sort_values().drop_duplicates()) or "D"
        y_all = df["y"].to_numpy()
        season_length = _infer_season_length(freq, y_all)

        model = model_class(season_length=season_length)
        sf = StatsForecast(models=[model], freq=freq, n_jobs=-1)
        forecasts = sf.forecast(df=df, h=cfg.h, level=cfg.levels)

        model_col = [c for c in forecasts.columns if c not in ["unique_id", "ds"] and "-lo-" not in c and "-hi-" not in c]
        name = model_col[0] if model_col else model_name

        return ForecastResult(
            forecast_df=forecasts,
            status={uid: "ok" for uid in df["unique_id"].unique()},
            model_name=name,
            config={
                "h": cfg.h,
                "levels": cfg.levels,
                "min_obs_per_series": cfg.min_obs_per_series,
                "max_series_per_batch": cfg.max_series_per_batch,
                "outlier_clip": cfg.outlier_clip,
                "outlier_iqr_mult": cfg.outlier_iqr_mult,
                "random_seed": cfg.random_seed,
                "verbose": cfg.verbose,
            },
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return None
        except Exception:
            return None


def run_model_with_config(
    df: pd.DataFrame,
    cfg: "CSPConfig",
    model_class,
    model_name: str,
    timeout: int = 120,
    **model_kwargs,
) -> Optional["ForecastResult"]:
    """Run a StatsForecast model with the given config and timeout."""
    return _run_single_model(df, cfg, model_class, model_name, timeout=timeout)


def run_seasonal_naive_with_config(
    df: pd.DataFrame,
    cfg: "CSPConfig",
    timeout: int = 60,
) -> Optional["ForecastResult"]:
    """Run SeasonalNaive forecast with the given config."""
    from statsforecast.models import SeasonalNaive
    return _run_single_model(df, cfg, SeasonalNaive, "SeasonalNaive", timeout=timeout)


def run_auto_arima_with_config(
    df: pd.DataFrame,
    cfg: "CSPConfig",
    timeout: int = 300,
) -> Optional["ForecastResult"]:
    """Run AutoARIMA forecast with the given config."""
    from statsforecast.models import AutoARIMA
    return _run_single_model(df, cfg, AutoARIMA, "AutoARIMA", timeout=timeout)


def run_auto_ets_with_config(
    df: pd.DataFrame,
    cfg: "CSPConfig",
    timeout: int = 60,
) -> Optional["ForecastResult"]:
    """Run AutoETS forecast with the given config."""
    from statsforecast.models import AutoETS
    return _run_single_model(df, cfg, AutoETS, "AutoETS", timeout=timeout)


def run_auto_theta_with_config(
    df: pd.DataFrame,
    cfg: "CSPConfig",
    timeout: int = 60,
) -> Optional["ForecastResult"]:
    """Run AutoTheta forecast with the given config."""
    from statsforecast.models import AutoTheta
    return _run_single_model(df, cfg, AutoTheta, "AutoTheta", timeout=timeout)


def run_all_models(
    df: pd.DataFrame,
    cfg: "CSPConfig",
    timeout: int = 120,
    include_slow: bool = False,
) -> Dict[str, Optional["ForecastResult"]]:
    """Run all models: CSP, SeasonalNaive, AutoETS, AutoTheta (AutoARIMA is optional/slow)."""
    from statsforecast.models import SeasonalNaive, AutoETS, AutoTheta

    results = {}

    # CSP
    try:
        csp_result = run_csp_with_config(df, cfg)
        results["CSP"] = csp_result
    except Exception:
        results["CSP"] = None

    # Fast models: SeasonalNaive, AutoETS, AutoTheta (run sequentially to avoid GIL contention)
    fast_models = [
        ("SeasonalNaive", SeasonalNaive, "SeasonalNaive"),
        ("AutoETS", AutoETS, "AutoETS"),
        ("AutoTheta", AutoTheta, "AutoTheta"),
    ]

    for name, model_class, name in fast_models:
        try:
            results[name] = _run_single_model(df, cfg, model_class, name, timeout=60)
        except Exception:
            results[name] = None

    # Slow model: AutoARIMA (optional)
    if include_slow:
        from statsforecast.models import AutoARIMA
        try:
            results["AutoARIMA"] = _run_single_model(df, cfg, AutoARIMA, "AutoARIMA", timeout=300)
        except Exception:
            results["AutoARIMA"] = None
    else:
        results["AutoARIMA"] = None

    return results


def get_model_name(forecast_df: pd.DataFrame) -> str:
    """Extract model name from forecast columns."""
    for c in forecast_df.columns:
        if c not in ["unique_id", "ds"] and "-lo-" not in c and "-hi-" not in c:
            return c
    return "CSP"


def compute_backtest_metrics(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    series_id: str,
) -> Optional[Dict[str, float]]:
    """Compute backtest metrics for a single series using naive last-window method."""
    hist = historical_df[historical_df["unique_id"] == series_id].sort_values("ds")
    fcst = forecast_df[forecast_df["unique_id"] == series_id].sort_values("ds")

    if hist.empty or fcst.empty:
        return None

    model_name = get_model_name(forecast_df)
    pred_col = model_name

    if pred_col not in fcst.columns:
        return None

    h = len(fcst)
    y_true = hist["y"].values[-h:] if len(hist) >= h else hist["y"].values
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

    h = len(fcst)
    y_true = hist["y"].values[-h:] if len(hist) >= h else hist["y"].values
    lo = fcst[lo_col].values
    hi = fcst[hi_col].values
    pred = fcst[model_name].values

    if len(y_true) != len(lo):
        return None

    # Coverage
    covered = np.sum((y_true >= lo) & (y_true <= hi))
    coverage = covered / len(y_true)

    # Pinball loss (quantile loss)
    alpha = level / 100.0
    pinball = np.mean(np.where(y_true < lo, (1 - alpha) * (lo - y_true), alpha * (y_true - hi)))

    # CRPS approximation (uniform distribution within interval)
    crps = np.mean(
        (hi - lo) / 4
        + (lo - y_true) ** 2 / (hi - lo) * (y_true < lo)
        + (y_true - hi) ** 2 / (hi - lo) * (y_true > hi)
    )

    return {
        f"Coverage_{level}%": coverage,
        f"PinballLoss_{level}%": float(pinball),
        f"CRPS_{level}%": float(crps),
    }