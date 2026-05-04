from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from stock_analysis.data import benchmark_symbol_for_ticker, load_history, normalize_ticker
from stock_analysis.features import FEATURE_COLUMNS, build_training_frame, latest_feature_row
from stock_analysis.model import Metrics, train_direction_model


@dataclass
class Prediction:
    ticker: str
    latest_date: str
    latest_close: float
    probability_up: float
    probability_down: float
    signal: str
    threshold: float
    metrics: Metrics

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["metrics"] = asdict(self.metrics)
        return payload


def predict_next_day(
    ticker: str,
    *,
    exchange: str | None = None,
    period: str = "5y",
    interval: str = "1d",
    csv_path: str | Path | None = None,
    test_size: float = 0.2,
    threshold: float = 0.5,
) -> Prediction:
    symbol = normalize_ticker(ticker, exchange)
    history = load_history(symbol, period=period, interval=interval, csv_path=csv_path)

    benchmark_history = None
    if not csv_path:
        benchmark_symbol = benchmark_symbol_for_ticker(symbol, exchange)
        if benchmark_symbol != symbol:
            try:
                benchmark_history = load_history(benchmark_symbol, period=period, interval=interval)
            except Exception:
                benchmark_history = None

    training_frame = build_training_frame(history, benchmark_history=benchmark_history)
    latest_row = latest_feature_row(history, benchmark_history=benchmark_history)

    result = train_direction_model(
        training_frame,
        FEATURE_COLUMNS,
        test_size=test_size,
        threshold=threshold,
    )
    probability_up = float(result.model.predict_proba(latest_row.to_numpy(dtype=float))[0])
    signal = "UP" if probability_up >= threshold else "DOWN"

    latest_date = history.index[-1]
    return Prediction(
        ticker=symbol,
        latest_date=str(getattr(latest_date, "date", lambda: latest_date)()),
        latest_close=float(history["Close"].iloc[-1]),
        probability_up=probability_up,
        probability_down=1 - probability_up,
        signal=signal,
        threshold=threshold,
        metrics=result.metrics,
    )
