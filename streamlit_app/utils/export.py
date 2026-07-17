"""Export utilities for Streamlit app."""
import io
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_forecast_download(
    forecast_df: pd.DataFrame,
    status: Dict[str, str],
    config: Dict,
    model_name: str,
) -> Tuple[Dict[str, bytes], Dict[str, str]]:
    """Create downloadable files for forecast results.

    Returns:
        Tuple of (files_dict, filenames_dict) where:
        - files_dict: fixed keys -> file bytes
        - filenames_dict: fixed keys -> display filenames
    """
    timestamp = _timestamp()

    # Forecast CSV
    forecast_csv = forecast_df.to_csv(index=False).encode("utf-8")

    # Status CSV
    status_df = pd.DataFrame([
        {"Series ID": k, "Status": v} for k, v in status.items()
    ])
    status_csv = status_df.to_csv(index=False).encode("utf-8")

    # Config CSV
    config_df = pd.DataFrame([config])
    config_csv = config_df.to_csv(index=False).encode("utf-8")

    # Excel with multiple sheets
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="Forecasts", index=False)
        status_df.to_excel(writer, sheet_name="Status", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)

        # Add metadata sheet
        meta_df = pd.DataFrame({
            "Property": ["Model", "Generated", "Horizon", "Confidence Levels",
                        "Min Obs/Series", "Outlier Clipping", "IQR Multiplier", "Random Seed"],
            "Value": [model_name, timestamp, config.get("h", ""),
                     ", ".join(map(str, config.get("levels", []))),
                     config.get("min_obs_per_series", ""),
                     config.get("outlier_clip", ""),
                     config.get("outlier_iqr_mult", ""),
                     config.get("random_seed", "")],
        })
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)

    excel_buffer.seek(0)
    excel_bytes = excel_buffer.read()

    files = {
        "forecast_csv": forecast_csv,
        "status_csv": status_csv,
        "config_csv": config_csv,
        "results_xlsx": excel_bytes,
    }
    filenames = {
        "forecast_csv": f"{model_name}_forecast_{timestamp}.csv",
        "status_csv": f"{model_name}_status_{timestamp}.csv",
        "config_csv": f"{model_name}_config_{timestamp}.csv",
        "results_xlsx": f"{model_name}_results_{timestamp}.xlsx",
    }
    return files, filenames


def create_comparison_download(
    csp_forecast: pd.DataFrame,
    sn_forecast: pd.DataFrame,
    csp_status: Dict[str, str],
    sn_status: Dict[str, str],
    config: Dict,
) -> Tuple[Dict[str, bytes], Dict[str, str]]:
    """Create downloadable files for model comparison."""
    timestamp = _timestamp()

    # Merge forecasts
    all_forecasts = pd.concat([csp_forecast, sn_forecast], ignore_index=True)

    # Status comparison
    all_series = set(csp_status.keys()) | set(sn_status.keys())
    status_rows = []
    for s in sorted(all_series):
        status_rows.append({
            "Series ID": s,
            "CSP Status": csp_status.get(s, "N/A"),
            "SeasonalNaive Status": sn_status.get(s, "N/A"),
        })
    status_df = pd.DataFrame(status_rows)

    # Config
    config_df = pd.DataFrame([config])

    # Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        all_forecasts.to_excel(writer, sheet_name="All Forecasts", index=False)
        status_df.to_excel(writer, sheet_name="Status Comparison", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)

        meta_df = pd.DataFrame({
            "Property": ["Generated", "Horizon", "Confidence Levels", "Min Obs/Series"],
            "Value": [timestamp, config.get("h", ""),
                     ", ".join(map(str, config.get("levels", []))),
                     config.get("min_obs_per_series", "")],
        })
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)

    excel_buffer.seek(0)
    excel_bytes = excel_buffer.read()

    files = {
        "comparison_forecasts_csv": all_forecasts.to_csv(index=False).encode("utf-8"),
        "comparison_status_csv": status_df.to_csv(index=False).encode("utf-8"),
        "comparison_results_xlsx": excel_bytes,
    }
    filenames = {
        "comparison_forecasts_csv": f"comparison_forecasts_{timestamp}.csv",
        "comparison_status_csv": f"comparison_status_{timestamp}.csv",
        "comparison_results_xlsx": f"comparison_results_{timestamp}.xlsx",
    }
    return files, filenames


def create_eda_download(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    col_config: Dict,
    eda_summary: Dict,
) -> Tuple[Dict[str, bytes], Dict[str, str]]:
    """Create downloadable files for EDA results."""
    timestamp = _timestamp()

    # Raw data
    raw_csv = raw_df.to_csv(index=False).encode("utf-8")

    # Processed data
    proc_csv = processed_df.to_csv(index=False).encode("utf-8")

    # Config
    config_df = pd.DataFrame([col_config])
    config_csv = config_df.to_csv(index=False).encode("utf-8")

    # EDA summary
    summary_df = pd.DataFrame([eda_summary])
    summary_csv = summary_df.to_csv(index=False).encode("utf-8")

    # Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw Data", index=False)
        processed_df.to_excel(writer, sheet_name="Processed Data", index=False)
        config_df.to_excel(writer, sheet_name="Column Config", index=False)
        summary_df.to_excel(writer, sheet_name="EDA Summary", index=False)

    excel_buffer.seek(0)
    excel_bytes = excel_buffer.read()

    files = {
        "raw_csv": raw_csv,
        "processed_csv": proc_csv,
        "config_csv": config_csv,
        "summary_csv": summary_csv,
        "eda_full_xlsx": excel_bytes,
    }
    filenames = {
        "raw_csv": f"raw_data_{timestamp}.csv",
        "processed_csv": f"processed_data_{timestamp}.csv",
        "config_csv": f"column_config_{timestamp}.csv",
        "summary_csv": f"eda_summary_{timestamp}.csv",
        "eda_full_xlsx": f"eda_full_{timestamp}.xlsx",
    }
    return files, filenames


def create_model_card(
    model_name: str,
    forecast_df: pd.DataFrame,
    status: Dict[str, str],
    config: Dict,
    metrics: Optional[Dict] = None,
) -> Tuple[Dict[str, bytes], Dict[str, str]]:
    """Create a model card with all relevant info."""
    timestamp = _timestamp()

    # Main report
    report = f"""# {model_name} Model Card

**Generated:** {timestamp}

## Configuration
- **Horizon (h):** {config.get('h', 'N/A')}
- **Confidence Levels:** {', '.join(map(str, config.get('levels', [])))}
- **Min Observations/Series:** {config.get('min_obs_per_series', 'N/A')}
- **Outlier Clipping (IQR):** {config.get('outlier_clip', 'N/A')}
- **IQR Multiplier:** {config.get('outlier_iqr_mult', 'N/A')}
- **Random Seed:** {config.get('random_seed', 'N/A')}

## Series Status
| Series ID | Status |
|-----------|--------|
"""
    
    for sid, stat in sorted(status.items()):
        report += f"| {sid} | {stat} |\n"
    
    if metrics:
        report += "\n## Performance Metrics\n"
        for metric, value in metrics.items():
            report += f"- **{metric}:** {value:.4f}\n"
    
    report += f"\n## Forecast Preview (first 5 series)\n"
    
    # Add preview of forecast
    preview = forecast_df.head(10).to_markdown(index=False)
    report += f"\n```markdown\n{preview}\n```\n"
    
    report_bytes = report.encode("utf-8")
    
    files = {f"{model_name}_model_card": report_bytes}
    filenames = {f"{model_name}_model_card": f"{model_name}_model_card_{timestamp}.md"}
    return files, filenames


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Convert dict of sheet_name -> DataFrame to Excel bytes."""
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    excel_buffer.seek(0)
    return excel_buffer.read()