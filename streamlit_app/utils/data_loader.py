"""Data loading utilities for Streamlit app."""
import pandas as pd
import io
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


def load_dataframe(file) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Load CSV/Excel/Parquet file into DataFrame."""
    filename = getattr(file, "name", "uploaded_file")
    
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(file)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    except Exception as e:
        raise ValueError(f"Failed to load {filename}: {e}")
    
    meta = {
        "filename": filename,
        "rows": len(df),
        "cols": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    return df, meta


def load_sample_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the built-in sample dataset."""
    sample_path = Path(__file__).resolve().parents[2] / "timeseries1.csv"
    df = pd.read_csv(sample_path)
    
    meta = {
        "filename": "timeseries1.csv (sample)",
        "rows": len(df),
        "cols": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    return df, meta


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive DataFrame information."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Auto-detect date, value, and ID columns."""
    from csp_universal_forecast import (
        _detect_date_column,
        _detect_value_column,
        _detect_id_column,
    )
    
    try:
        date_col = _detect_date_column(df)
    except Exception:
        date_col = None
    
    try:
        id_col = _detect_id_column(df, date_col, None) if date_col else None
    except Exception:
        id_col = None
    
    try:
        value_col = _detect_value_column(df, date_col, id_col) if date_col else None
    except Exception:
        value_col = None
    
    return {
        "date_col": date_col,
        "value_col": value_col,
        "id_col": id_col,
    }


def infer_frequency(df: pd.DataFrame, date_col: str) -> str:
    """Infer time series frequency."""
    from csp_universal_forecast import _infer_freq
    
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    return _infer_freq(dates)


def prepare_long_dataframe(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    id_col: Optional[str],
    freq: str,
    cfg,
) -> pd.DataFrame:
    """Prepare long-format DataFrame for StatsForecast."""
    from csp_universal_forecast import _prepare_long_df
    
    return _prepare_long_df(df, date_col, value_col, id_col, freq, cfg)


def validate_column_selection(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    id_col: Optional[str],
) -> list:
    """Validate column selection and return list of issues."""
    issues = []
    
    if date_col not in df.columns:
        issues.append(f"Date column '{date_col}' not found")
    
    if value_col not in df.columns:
        issues.append(f"Value column '{value_col}' not found")
    
    if id_col and id_col not in df.columns:
        issues.append(f"ID column '{id_col}' not found")
    
    if value_col in df.columns and not pd.api.types.is_numeric_dtype(df[value_col]):
        issues.append(f"Value column '{value_col}' is not numeric")
    
    return issues