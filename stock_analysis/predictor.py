from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from stock_analysis.data import benchmark_symbol_for_ticker, load_history, normalize_ticker
from stock_analysis.features import FEATURE_COLUMNS, build_training_frame, latest_feature_row
from stock_analysis.model import Metrics, ModelComparison, train_direction_model


@dataclass
class BacktestSummary:
    rows: int
    signal_count: int
    hit_rate: float
    average_return: float
    cumulative_strategy_return: float
    cumulative_buy_hold_return: float


@dataclass
class BacktestResult:
    summary: BacktestSummary
    chart_rows: list[dict[str, float | str]]
    recent_rows: list[dict[str, float | int | str]]


@dataclass
class Prediction:
    ticker: str
    latest_date: str
    latest_close: float
    probability_up: float
    probability_down: float
    signal: str
    threshold: float
    threshold_source: str
    model_name: str
    model_label: str
    model_selection_basis: str
    compared_models: list[ModelComparison]
    metrics: Metrics
    backtest: BacktestResult | None = None

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
    compute_walk_forward_metrics: bool = False,
    optimize_threshold: bool = False,
    compare_tree_model: bool = False,
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
        compute_walk_forward_metrics=compute_walk_forward_metrics,
        optimize_threshold=optimize_threshold,
        compare_tree_model=compare_tree_model,
    )
    probability_up = float(result.model.predict_proba(latest_row.to_numpy(dtype=float))[0])
    threshold_to_use = (
        float(result.metrics.recommended_threshold)
        if optimize_threshold and result.metrics.recommended_threshold is not None
        else float(threshold)
    )
    signal = "UP" if probability_up >= threshold_to_use else "DOWN"
    backtest = build_backtest_result(
        training_frame=training_frame,
        walk_forward_start_index=result.walk_forward_start_index,
        walk_forward_actuals=result.walk_forward_actuals,
        walk_forward_probabilities=result.walk_forward_probabilities,
        threshold=threshold_to_use,
    )

    latest_date = history.index[-1]
    return Prediction(
        ticker=symbol,
        latest_date=str(getattr(latest_date, "date", lambda: latest_date)()),
        latest_close=float(history["Close"].iloc[-1]),
        probability_up=probability_up,
        probability_down=1 - probability_up,
        signal=signal,
        threshold=threshold_to_use,
        threshold_source="recommended" if optimize_threshold else "manual",
        model_name=result.model_name,
        model_label=result.model_label,
        model_selection_basis=result.model_selection_basis,
        compared_models=result.compared_models,
        metrics=result.metrics,
        backtest=backtest,
    )


def build_backtest_result(
    *,
    training_frame,
    walk_forward_start_index: int | None,
    walk_forward_actuals,
    walk_forward_probabilities,
    threshold: float,
) -> BacktestResult | None:
    if (
        walk_forward_start_index is None
        or walk_forward_actuals is None
        or walk_forward_probabilities is None
        or len(walk_forward_actuals) == 0
    ):
        return None

    backtest_frame = training_frame.iloc[int(walk_forward_start_index) :].copy()
    expected_rows = min(len(backtest_frame), len(walk_forward_actuals), len(walk_forward_probabilities))
    if expected_rows == 0:
        return None

    backtest_frame = backtest_frame.iloc[:expected_rows].copy()
    probabilities = walk_forward_probabilities[:expected_rows].astype(float)
    actuals = walk_forward_actuals[:expected_rows].astype(int)
    predicted = (probabilities >= threshold).astype(int)

    backtest_frame["probability_up"] = probabilities
    backtest_frame["actual_up"] = actuals
    backtest_frame["predicted_up"] = predicted
    backtest_frame["signal"] = np.where(predicted == 1, "UP", "DOWN")
    backtest_frame["strategy_return"] = np.where(
        predicted == 1,
        backtest_frame["next_day_return"].astype(float),
        0.0,
    )
    backtest_frame["buy_hold_return"] = backtest_frame["next_day_return"].astype(float)
    backtest_frame["cumulative_strategy_return"] = (1 + backtest_frame["strategy_return"]).cumprod() - 1
    backtest_frame["cumulative_buy_hold_return"] = (1 + backtest_frame["buy_hold_return"]).cumprod() - 1

    signaled = backtest_frame[backtest_frame["predicted_up"] == 1]
    hit_rate = float((signaled["actual_up"] == 1).mean()) if not signaled.empty else 0.0
    average_return = float(signaled["next_day_return"].mean()) if not signaled.empty else 0.0

    summary = BacktestSummary(
        rows=int(len(backtest_frame)),
        signal_count=int(len(signaled)),
        hit_rate=hit_rate,
        average_return=average_return,
        cumulative_strategy_return=float(backtest_frame["cumulative_strategy_return"].iloc[-1]),
        cumulative_buy_hold_return=float(backtest_frame["cumulative_buy_hold_return"].iloc[-1]),
    )

    chart_rows = [
        {
            "date": str(getattr(index, "date", lambda: index)()),
            "strategy_return": float(row["cumulative_strategy_return"]),
            "buy_hold_return": float(row["cumulative_buy_hold_return"]),
        }
        for index, row in backtest_frame.iterrows()
    ]
    recent_rows = [
        {
            "date": str(getattr(index, "date", lambda: index)()),
            "probability_up": float(row["probability_up"]),
            "signal": str(row["signal"]),
            "actual": "UP" if int(row["actual_up"]) == 1 else "DOWN",
            "next_day_return": float(row["next_day_return"]),
            "strategy_return": float(row["strategy_return"]),
        }
        for index, row in backtest_frame.tail(15).iterrows()
    ]
    return BacktestResult(summary=summary, chart_rows=chart_rows, recent_rows=recent_rows)
