from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from stock_analysis.stock_search import resolve_stock_input


REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def normalize_ticker(ticker: str, exchange: str | None = None) -> str:
    """Normalize common Korean ticker shortcuts for Yahoo Finance."""
    original = ticker.strip()
    value = original.upper()
    if not value:
        raise ValueError("Ticker is empty.")

    if re.fullmatch(r"\d{6}", value):
        if not exchange:
            resolved = resolve_stock_input(original, exchange_hint=None)
            if re.fullmatch(r"[0-9A-Z]{6}\.(KS|KQ)", resolved):
                return resolved
        suffix_map = {
            None: ".KS",
            "KRX": ".KS",
            "KS": ".KS",
            "KOSPI": ".KS",
            "KQ": ".KQ",
            "KOSDAQ": ".KQ",
        }
        suffix = suffix_map.get(exchange.upper() if exchange else None)
        if suffix is None:
            allowed = ", ".join(key for key in suffix_map if key)
            raise ValueError(f"Unsupported exchange '{exchange}'. Use one of: {allowed}.")
        return f"{value}{suffix}"

    if re.fullmatch(r"[0-9A-Z]{6}", value) and any(char.isalpha() for char in value):
        if exchange:
            suffix_map = {
                "KRX": ".KS",
                "KS": ".KS",
                "KOSPI": ".KS",
                "KQ": ".KQ",
                "KOSDAQ": ".KQ",
            }
            suffix = suffix_map.get(exchange.upper())
            if suffix is None:
                allowed = ", ".join(suffix_map)
                raise ValueError(f"Unsupported exchange '{exchange}'. Use one of: {allowed}.")
            return f"{value}{suffix}"
        return resolve_stock_input(original, exchange_hint=exchange)

    if re.fullmatch(r"[0-9A-Z]{6}\.(KS|KQ)", value):
        return value

    if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", value):
        return value

    return resolve_stock_input(original, exchange_hint=exchange)


def benchmark_symbol_for_ticker(ticker: str, exchange: str | None = None) -> str:
    normalized_exchange = exchange.upper() if exchange else None
    upper_ticker = ticker.strip().upper()

    if normalized_exchange in {"KS", "KOSPI", "KRX"} or upper_ticker.endswith(".KS"):
        return "^KS11"
    if normalized_exchange in {"KQ", "KOSDAQ"} or upper_ticker.endswith(".KQ"):
        return "^KQ11"
    return "SPY"


def load_history(
    ticker: str,
    *,
    period: str = "5y",
    interval: str = "1d",
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load OHLCV history from CSV or Yahoo Finance."""
    if csv_path:
        return _load_csv(csv_path)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required to download market data. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    frame = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    frame = _flatten_columns(frame)
    return _clean_history(frame, ticker)


def _load_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    frame = pd.read_csv(path)
    date_column = next((col for col in frame.columns if col.lower() in {"date", "datetime"}), None)
    if date_column:
        frame[date_column] = pd.to_datetime(frame[date_column])
        frame = frame.set_index(date_column)
    else:
        frame.index = pd.to_datetime(frame.index)

    return _clean_history(_flatten_columns(frame), str(path))


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [str(col[0]) for col in frame.columns.to_flat_index()]
    return frame


def _clean_history(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"No price data was returned for {label}.")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for {label}: {', '.join(missing)}")

    clean = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(subset=list(REQUIRED_COLUMNS))
    clean = clean[clean["Volume"] > 0]
    clean = clean.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]

    if len(clean) < 260:
        raise ValueError(
            f"Only {len(clean)} valid rows are available for {label}. "
            "At least about one trading year is recommended."
        )

    return clean
