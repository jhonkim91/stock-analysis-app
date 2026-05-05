from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stock_analysis.predictor import Prediction, predict_next_day


@dataclass
class WatchlistItem:
    ticker: str
    exchange: str | None = None
    name: str = ""
    enabled: bool = True


@dataclass
class AutoRunSummary:
    run_at: str
    watchlist_path: str
    output_dir: str
    csv_path: str
    json_path: str
    total: int
    succeeded: int
    failed: int


def read_watchlist(path: str | Path) -> list[WatchlistItem]:
    watchlist_path = Path(path)
    if not watchlist_path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {watchlist_path}")

    with watchlist_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Watchlist file is empty: {watchlist_path}")

        fieldnames = {name.strip().lower(): name for name in reader.fieldnames}
        if "ticker" not in fieldnames:
            raise ValueError("Watchlist must include a 'ticker' column.")

        items: list[WatchlistItem] = []
        for row in reader:
            ticker = _get_cell(row, fieldnames, "ticker")
            if not ticker:
                continue

            enabled = _parse_bool(_get_cell(row, fieldnames, "enabled"), default=True)
            items.append(
                WatchlistItem(
                    ticker=ticker,
                    exchange=_get_cell(row, fieldnames, "exchange") or None,
                    name=_get_cell(row, fieldnames, "name"),
                    enabled=enabled,
                )
            )

    enabled_items = [item for item in items if item.enabled]
    if not enabled_items:
        raise ValueError("No enabled tickers were found in the watchlist.")
    return enabled_items


def run_watchlist(
    *,
    watchlist_path: str | Path = "watchlist.csv",
    output_dir: str | Path = "outputs",
    period: str = "5y",
    interval: str = "1d",
    test_size: float = 0.2,
    threshold: float = 0.5,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    retries: int = 1,
    fail_fast: bool = False,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    run_dir: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> AutoRunSummary:
    items = read_watchlist(watchlist_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")
        items = items[:limit]

    run_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_label = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    destination = Path(run_dir) if run_dir else Path(output_dir) / run_label
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        row = _run_item(
            item,
            run_at=run_at,
            period=period,
            interval=interval,
            test_size=test_size,
            threshold=threshold,
            retries=retries,
        )
        rows.append(row)
        if progress:
            progress(index, len(items), row)

        if row["status"] == "failed" and fail_fast:
            break
        if sleep_seconds > 0 and index < len(items):
            time.sleep(sleep_seconds)

    rows = _sort_rows(rows)
    csv_path = destination / "predictions.csv"
    json_path = destination / "predictions.json"
    _write_csv(csv_path, rows)
    payload = {
        "run_at": run_at,
        "watchlist_path": str(Path(watchlist_path)),
        "period": period,
        "interval": interval,
        "test_size": test_size,
        "threshold": threshold,
        "rows": rows,
    }
    if metadata:
        payload["metadata"] = metadata
    _write_json(json_path, payload)

    succeeded = sum(1 for row in rows if row["status"] == "success")
    failed = len(rows) - succeeded
    return AutoRunSummary(
        run_at=run_at,
        watchlist_path=str(Path(watchlist_path)),
        output_dir=str(destination),
        csv_path=str(csv_path),
        json_path=str(json_path),
        total=len(rows),
        succeeded=succeeded,
        failed=failed,
    )


def _run_item(
    item: WatchlistItem,
    *,
    run_at: str,
    period: str,
    interval: str,
    test_size: float,
    threshold: float,
    retries: int,
) -> dict[str, Any]:
    attempts = max(1, retries + 1)
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            prediction = predict_next_day(
                item.ticker,
                exchange=item.exchange,
                period=period,
                interval=interval,
                test_size=test_size,
                threshold=threshold,
            )
            return _prediction_row(item, prediction, run_at=run_at, attempt=attempt)
        except Exception as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(min(2 * attempt, 10))

    return {
        "run_at": run_at,
        "status": "failed",
        "name": item.name,
        "ticker_input": item.ticker,
        "exchange": item.exchange or "",
        "ticker": "",
        "latest_date": "",
        "latest_close": "",
        "signal": "",
        "probability_up": "",
        "probability_down": "",
        "accuracy": "",
        "baseline_accuracy": "",
        "precision_up": "",
        "recall_up": "",
        "train_rows": "",
        "test_rows": "",
        "attempt": attempts,
        "error": last_error,
    }


def _prediction_row(
    item: WatchlistItem,
    prediction: Prediction,
    *,
    run_at: str,
    attempt: int,
) -> dict[str, Any]:
    metrics = prediction.metrics
    backtest = prediction.backtest.summary if prediction.backtest is not None else None
    return {
        "run_at": run_at,
        "status": "success",
        "name": item.name,
        "ticker_input": item.ticker,
        "exchange": item.exchange or "",
        "ticker": prediction.ticker,
        "latest_date": prediction.latest_date,
        "latest_close": round(prediction.latest_close, 6),
        "signal": prediction.signal,
        "probability_up": round(prediction.probability_up, 6),
        "probability_down": round(prediction.probability_down, 6),
        "threshold": round(prediction.threshold, 4),
        "threshold_source": prediction.threshold_source,
        "model_name": prediction.model_name,
        "model_label": prediction.model_label,
        "model_selection_basis": prediction.model_selection_basis,
        "accuracy": round(metrics.accuracy, 6),
        "baseline_accuracy": round(metrics.baseline_accuracy, 6),
        "accuracy_edge": round(metrics.edge_vs_baseline, 6),
        "balanced_accuracy": round(metrics.balanced_accuracy, 6),
        "precision_up": round(metrics.precision_up, 6),
        "recall_up": round(metrics.recall_up, 6),
        "walk_forward_accuracy": ""
        if metrics.walk_forward_accuracy is None
        else round(metrics.walk_forward_accuracy, 6),
        "walk_forward_edge": ""
        if metrics.walk_forward_edge_vs_baseline is None
        else round(metrics.walk_forward_edge_vs_baseline, 6),
        "recommended_threshold": ""
        if metrics.recommended_threshold is None
        else round(metrics.recommended_threshold, 4),
        "train_rows": metrics.train_rows,
        "test_rows": metrics.test_rows,
        "backtest_rows": "" if backtest is None else backtest.rows,
        "backtest_signal_count": "" if backtest is None else backtest.signal_count,
        "backtest_hit_rate": "" if backtest is None else round(backtest.hit_rate, 6),
        "backtest_average_return": "" if backtest is None else round(backtest.average_return, 6),
        "backtest_cumulative_strategy_return": ""
        if backtest is None
        else round(backtest.cumulative_strategy_return, 6),
        "backtest_cumulative_buy_hold_return": ""
        if backtest is None
        else round(backtest.cumulative_buy_hold_return, 6),
        "attempt": attempt,
        "error": "",
    }


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        if row["status"] != "success":
            return (1, -1.0)
        return (0, -float(row["probability_up"]))

    return sorted(rows, key=sort_key)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_at",
        "status",
        "name",
        "ticker_input",
        "exchange",
        "ticker",
        "latest_date",
        "latest_close",
        "signal",
        "probability_up",
        "probability_down",
        "threshold",
        "threshold_source",
        "model_name",
        "model_label",
        "model_selection_basis",
        "accuracy",
        "baseline_accuracy",
        "accuracy_edge",
        "balanced_accuracy",
        "precision_up",
        "recall_up",
        "walk_forward_accuracy",
        "walk_forward_edge",
        "recommended_threshold",
        "train_rows",
        "test_rows",
        "backtest_rows",
        "backtest_signal_count",
        "backtest_hit_rate",
        "backtest_average_return",
        "backtest_cumulative_strategy_return",
        "backtest_cumulative_buy_hold_return",
        "attempt",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_cell(row: dict[str, str], fieldnames: dict[str, str], name: str) -> str:
    source_name = fieldnames.get(name)
    if not source_name:
        return ""
    return (row.get(source_name) or "").strip()


def _parse_bool(value: str, *, default: bool) -> bool:
    if not value:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default
