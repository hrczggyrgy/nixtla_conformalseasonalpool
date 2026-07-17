import os

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
    "uploaded_filename",
    "last_results",
    "last_run_timestamp",
    "validation_errors",
    "validation_warnings",
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

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "timeseries1.csv")

PAGE_TITLE = "Automated Forecasting"
PAGE_ICON = ""
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

EXPORT_PRESETS = {
    "minimal": {
        "name": "Minimal (CSV only)",
        "description": "Forecasts CSV + Status CSV",
        "sheets": ["Forecasts", "Status"],
    },
    "standard": {
        "name": "Standard (Excel)",
        "description": "Forecasts, Status, Config, Metadata",
        "sheets": ["Forecasts", "Status", "Config", "Metadata"],
    },
    "complete": {
        "name": "Complete Package",
        "description": "All results + original data + comparison",
        "sheets": [
            "Forecasts", "Status", "Config", "Metadata",
            "Original_Data", "Processed_Data", "Column_Config"
        ],
    },
    "comparison": {
        "name": "Model Comparison",
        "description": "CSP vs SeasonalNaive forecasts, metrics, status",
        "sheets": [
            "CSP_Forecasts", "SN_Forecasts", "Comparison",
            "Metrics", "Status_Comparison", "Config"
        ],
    },
}

VALIDATION_RULES = {
    "min_rows": 10,
    "max_missing_pct": 50,
    "min_obs_per_series": 2,
    "date_parse_threshold": 0.8,
    "value_numeric_threshold": 0.9,
}