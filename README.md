# CSP Universal Forecast

Autonomous, robust, and universal time-series forecasting script built on
Nixtla's **ConformalSeasonalPool (CSP)** — a training-free forecasting model
with natively conformal (calibrated, distribution-free) prediction intervals.

Point it at **any** DataFrame that has a date column and a numeric value
column — regardless of column names, frequency, or whether it's a single
series or a multi-series panel — and it will auto-detect everything, clean
the data, infer seasonality, fit the model, and return forecasts with
prediction intervals.

---

## Features

- 🔎 **Auto-detects** date column, target column, and series-id column (no config needed)
- 📅 **Infers sampling frequency** (hourly, daily, weekly, monthly, quarterly, yearly)
- 🔁 **Data-driven seasonality detection** via autocorrelation (ACF), not just frequency lookup
- 🧹 **Cleans data**: parses dates, drops bad rows, deduplicates, fills gaps, interpolates
- 🎯 **Outlier-resistant** preprocessing (IQR clipping) before fitting
- 🧩 **Handles single or multi-series (panel) data** automatically
- 🛟 **Automatic fallback** to `SeasonalNaive` if CSP fails on a given series/batch
- 📊 **Per-series status report** (`ok`, `fallback`, `dropped:<reason>`)
- 📦 **Batched fitting** for very large panels (memory-safe)
- ✅ **Built-in self-tests** to catch regressions
- 🧪 Fully typed, validated config via a single `CSPConfig` dataclass

---

## Requirements

```
statsforecast>=2.1.1
pandas>=2.2.3
numpy>=1.26.4
```

Install:

```bash
pip install -r requirements.txt
```

---

## Files

| File | Description |
|---|---|
| `csp_universal_forecast_v2.py` | Main script — hardened, production-ready version |
| `requirements.txt` | Python dependencies |

---

## Quick Start

```python
import pandas as pd
from csp_universal_forecast_v2 import run_csp_forecast, CSPConfig

df = pd.read_csv("your_data.csv")  # any column names, any order

forecast_df, status, sf_model = run_csp_forecast(df, h=14, levels=[80, 95])

print(forecast_df.head())
print(status)  # per-series outcome: "ok" / "fallback:SeasonalNaive" / "dropped:..."
```

That's it. No need to specify the date column, target column, frequency, or
seasonal period — they are all inferred automatically.

---

## Configuration

All parameters can be passed either as keyword arguments or via a `CSPConfig`
object for stricter typing and validation:

```python
from csp_universal_forecast_v2 import CSPConfig, run_csp_forecast

cfg = CSPConfig(
    date_col=None,            # auto-detected if None
    value_col=None,           # auto-detected if None
    id_col=None,               # auto-detected if None (assumes single series)
    h=14,                       # forecast horizon
    levels=[80, 95],            # prediction interval confidence levels
    min_obs_per_series=10,      # series shorter than this are dropped
    max_series_per_batch=500,   # batch size for large panels
    outlier_clip=True,          # enable IQR-based outlier clipping
    outlier_iqr_mult=3.0,       # IQR multiplier for clipping bounds
    random_seed=42,
    verbose=True,
)

forecast_df, status, sf_model = run_csp_forecast(df, cfg=cfg)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `date_col` | `str` or `None` | `None` | Name of the date column. Auto-detected if not provided. |
| `value_col` | `str` or `None` | `None` | Name of the numeric target column. Auto-detected if not provided. |
| `id_col` | `str` or `None` | `None` | Name of the series-id column for panel data. Auto-detected if present. |
| `h` | `int` | `14` | Forecast horizon (number of future periods). |
| `levels` | `List[int]` | `[80, 95]` | Confidence levels (%) for prediction intervals. |
| `min_obs_per_series` | `int` | `10` | Minimum observations required to keep a series. |
| `max_series_per_batch` | `int` | `500` | Max series fitted per batch (memory control). |
| `outlier_clip` | `bool` | `True` | Whether to clip outliers via IQR before fitting. |
| `outlier_iqr_mult` | `float` | `3.0` | IQR multiplier defining outlier clipping bounds. |
| `random_seed` | `int` | `42` | Seed for reproducibility. |
| `verbose` | `bool` | `True` | Enable INFO-level logging. |

---

## Output

### `forecast_df`
A `pandas.DataFrame` with columns:

- `unique_id` — series identifier
- `ds` — forecast timestamp
- `CSP-Adaptive` (or fallback model name) — point forecast
- `CSP-Adaptive-lo-80`, `CSP-Adaptive-hi-80`, ... — prediction interval bounds per requested level

### `status`
A `Dict[str, str]` mapping each `unique_id` to one of:

- `"ok"` — CSP fit and forecasted successfully
- `"fallback:SeasonalNaive"` — CSP failed; fallback model used instead
- `"dropped:too_short"` — series had fewer than `min_obs_per_series` observations
- `"dropped:error(...)"` — both CSP and fallback failed

### `sf_model`
The last fitted `StatsForecast` instance, useful for further inspection or
plotting via `sf_model.plot(...)`.

---

## Example: Multi-Series Panel

```python
import pandas as pd
import numpy as np
from csp_universal_forecast_v2 import run_csp_forecast

rng = pd.date_range("2023-01-01", periods=400, freq="D")

df = pd.concat([
    pd.DataFrame({
        "order_date": rng,
        "units_sold": 100 + 10 * np.sin(2 * np.pi * np.arange(400) / 7) + np.random.normal(0, 2, 400),
        "store_id": "Store_A",
    }),
    pd.DataFrame({
        "order_date": rng,
        "units_sold": 80 + 8 * np.sin(2 * np.pi * np.arange(400) / 7) + np.random.normal(0, 2, 400),
        "store_id": "Store_B",
    }),
])

forecast_df, status, sf_model = run_csp_forecast(df, h=14)
print(forecast_df.head())
```

---

## Running Self-Tests

The script includes built-in self-tests covering normal series, too-short
series, constant series, and multi-series panels:

```bash
python csp_universal_forecast_v2.py
```

Expected output ends with:

```
INFO | All self-tests passed.
```

---

## How It Works (Under the Hood)

1. **Column detection** — dtype checks, then name-keyword heuristics, then brute-force parsing.
2. **Frequency inference** — `pandas.infer_freq`, with a median-timestamp-gap fallback for irregular data.
3. **Seasonality inference** — autocorrelation analysis on the actual values; falls back to a frequency-based default (7 for daily, 12 for monthly, etc.) if no clear seasonal signal is found.
4. **Cleaning** — drops unparseable dates, deduplicates timestamps (mean aggregation), reindexes onto a complete regular grid, interpolates gaps, clips outliers.
5. **Fitting** — instantiates `ConformalSeasonalPool(season_length=...)` inside a `StatsForecast` pipeline, batched for large panels.
6. **Fallback** — if CSP raises an exception for a batch, retries with `SeasonalNaive` before giving up.
7. **Reporting** — returns forecasts plus a transparent per-series status dictionary.

---

## Known Limitations

- CSP does not provide a formal finite-sample conformal coverage guarantee for arbitrary time series — validate empirical coverage via backtesting before production use.
- Assumes all series in a single DataFrame share the same underlying frequency; mixed-frequency panels should be split beforehand.
- Forecast horizons are sampled independently per step (not jointly coherent trajectories), consistent with CSP's design.

---

## License

Use freely within your own projects. No warranty provided — validate model
performance on your own data before deploying to production.