from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "price_vs_sma_5",
    "price_vs_sma_20",
    "price_vs_sma_60",
    "sma_5_vs_20",
    "sma_20_vs_60",
    "volatility_5",
    "volatility_20",
    "volume_change_1d",
    "volume_vs_sma_20",
    "intraday_range",
    "close_position",
    "gap_open",
    "rsi_14",
]


def build_feature_table(history: pd.DataFrame) -> pd.DataFrame:
    """Create technical features using only information known at each close."""
    open_price = history["Open"].astype(float)
    high = history["High"].astype(float)
    low = history["Low"].astype(float)
    close = history["Close"].astype(float)
    volume = history["Volume"].astype(float)

    returns = close.pct_change()
    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    sma_60 = close.rolling(60).mean()
    volume_sma_20 = volume.rolling(20).mean()
    daily_range = (high - low).replace(0, np.nan)

    features = pd.DataFrame(index=history.index)
    features["return_1d"] = close.pct_change(1)
    features["return_2d"] = close.pct_change(2)
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["price_vs_sma_5"] = close / sma_5 - 1
    features["price_vs_sma_20"] = close / sma_20 - 1
    features["price_vs_sma_60"] = close / sma_60 - 1
    features["sma_5_vs_20"] = sma_5 / sma_20 - 1
    features["sma_20_vs_60"] = sma_20 / sma_60 - 1
    features["volatility_5"] = returns.rolling(5).std()
    features["volatility_20"] = returns.rolling(20).std()
    features["volume_change_1d"] = volume.pct_change(1)
    features["volume_vs_sma_20"] = volume / volume_sma_20 - 1
    features["intraday_range"] = (high - low) / close
    features["close_position"] = (close - low) / daily_range
    features["gap_open"] = open_price / close.shift(1) - 1
    features["rsi_14"] = _rsi(close, 14) / 100

    return features.replace([np.inf, -np.inf], np.nan)


def build_training_frame(history: pd.DataFrame) -> pd.DataFrame:
    features = build_feature_table(history)
    close = history["Close"].astype(float)
    next_close = close.shift(-1)

    frame = features.copy()
    frame["target_up"] = np.where(next_close > close, 1.0, 0.0)
    frame.loc[next_close.isna(), "target_up"] = np.nan
    return frame.dropna(subset=FEATURE_COLUMNS + ["target_up"])


def latest_feature_row(history: pd.DataFrame) -> pd.Series:
    features = build_feature_table(history).dropna(subset=FEATURE_COLUMNS)
    if features.empty:
        raise ValueError("Not enough recent rows to build a latest feature row.")
    return features.iloc[-1].loc[FEATURE_COLUMNS]


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50)
