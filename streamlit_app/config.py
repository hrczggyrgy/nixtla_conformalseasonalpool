SESSION_KEYS = {
    "raw_df",
    "processed_df",
    "col_config",
    "forecast_cfg",
    "csp_results",
    "sn_results",
    "comparison_metrics",
    "eda_summary",
    "sample_data_loaded",
}

DEFAULT_CSP_CONFIG = {
    "h": 14,
    "levels": [80, 95],
    "min_obs_per_series": 10,
    "max_series_per_batch": 500,
    "outlier_clip": True,
    "outlier_iqr_mult": 3.0,
    "random_seed": 42,
    "verbose": True,
}

CONFIDENCE_LEVEL_OPTIONS = [80, 90, 95, 99]

import os
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "timeseries1.csv")

PAGE_TITLE = "CSP Universal Forecast"
PAGE_ICON = ""
LAYOUT = "wide"

SIDEBAR_STATE = "expanded"