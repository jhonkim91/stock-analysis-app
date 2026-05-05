from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stock_analysis.auto_runner import AutoRunSummary, run_watchlist
from stock_analysis.market_cap import (
    MarketCapItem,
    get_top_market_cap,
    write_market_cap_universe,
    write_market_cap_watchlist,
)


@dataclass
class TopCandidateSummary:
    run_at: str
    output_dir: str
    universe_path: str
    watchlist_path: str
    predictions_path: str
    top_csv_path: str
    top_json_path: str
    universe_count: int
    evaluated_count: int
    succeeded: int
    failed: int
    top_count: int


def run_top_market_cap_screen(
    *,
    output_dir: str | Path = "outputs/top_market_cap",
    rank_limit: int = 300,
    run_limit: int | None = None,
    top: int = 10,
    market: str = "ALL",
    source: str = "naver",
    date: str | None = None,
    exclude_preferred: bool = False,
    period: str = "5y",
    interval: str = "1d",
    test_size: float = 0.2,
    threshold: float = 0.5,
    sleep_seconds: float = 0.0,
    retries: int = 1,
    fail_fast: bool = False,
    min_price: float | None = None,
    min_accuracy_edge: float | None = None,
    max_per_market: int | None = None,
    progress=None,
) -> TopCandidateSummary:
    if top <= 0:
        raise ValueError("top must be greater than 0.")

    universe = get_top_market_cap(
        limit=rank_limit,
        market=market,
        source=source,
        date=date,
        exclude_preferred=exclude_preferred,
    )
    if not universe:
        raise ValueError("No market-cap universe rows were collected.")

    selected_universe = universe[:run_limit] if run_limit else universe
    run_label = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(output_dir) / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    universe_path = run_dir / "market_cap_universe.csv"
    watchlist_path = run_dir / "market_cap_watchlist.csv"
    write_market_cap_universe(universe_path, universe)
    write_market_cap_watchlist(watchlist_path, selected_universe)

    auto_summary = run_watchlist(
        watchlist_path=watchlist_path,
        output_dir=output_dir,
        period=period,
        interval=interval,
        test_size=test_size,
        threshold=threshold,
        sleep_seconds=sleep_seconds,
        retries=retries,
        fail_fast=fail_fast,
        progress=progress,
        run_dir=run_dir,
        metadata={
            "screen": "top_market_cap",
            "rank_limit": rank_limit,
            "run_limit": run_limit,
            "top": top,
            "market": market,
            "source": source,
            "date": date,
            "exclude_preferred": exclude_preferred,
            "min_price": min_price,
            "min_accuracy_edge": min_accuracy_edge,
            "max_per_market": max_per_market,
        },
    )

    prediction_rows = _read_csv_rows(auto_summary.csv_path)
    enriched_rows = _merge_market_cap_metadata(prediction_rows, universe)
    _write_csv(Path(auto_summary.csv_path), enriched_rows)

    top_rows = _select_top_rows(
        enriched_rows,
        top=top,
        min_price=min_price,
        min_accuracy_edge=min_accuracy_edge,
        max_per_market=max_per_market,
    )
    for selection_rank, row in enumerate(top_rows, start=1):
        row["selection_rank"] = selection_rank

    top_csv_path = run_dir / f"top{top}.csv"
    top_json_path = run_dir / f"top{top}.json"
    _write_csv(top_csv_path, top_rows)
    _write_json(
        top_json_path,
        {
            "run_at": auto_summary.run_at,
            "screen": "top_market_cap",
            "rank_limit": rank_limit,
            "run_limit": run_limit,
            "top": top,
            "market": market,
            "source": source,
            "date": date,
            "exclude_preferred": exclude_preferred,
            "rows": top_rows,
        },
    )

    _rewrite_prediction_json(auto_summary, enriched_rows)

    return TopCandidateSummary(
        run_at=auto_summary.run_at,
        output_dir=auto_summary.output_dir,
        universe_path=str(universe_path),
        watchlist_path=str(watchlist_path),
        predictions_path=auto_summary.csv_path,
        top_csv_path=str(top_csv_path),
        top_json_path=str(top_json_path),
        universe_count=len(universe),
        evaluated_count=len(selected_universe),
        succeeded=auto_summary.succeeded,
        failed=auto_summary.failed,
        top_count=len(top_rows),
    )


def _merge_market_cap_metadata(
    prediction_rows: list[dict[str, str]],
    universe: list[MarketCapItem],
) -> list[dict[str, Any]]:
    metadata = {(item.ticker, item.exchange): item for item in universe}
    enriched: list[dict[str, Any]] = []

    for row in prediction_rows:
        key = (row.get("ticker_input", ""), row.get("exchange", ""))
        item = metadata.get(key)
        merged: dict[str, Any] = dict(row)
        if item:
            accuracy = _safe_float(row.get("accuracy"))
            baseline_accuracy = _safe_float(row.get("baseline_accuracy"))
            merged.update(
                {
                    "market_cap_rank": item.rank,
                    "market": item.market,
                    "market_cap": item.market_cap,
                    "market_cap_source": item.source,
                    "market_cap_source_date": item.source_date,
                    "screen_price": item.price or "",
                    "accuracy_edge": ""
                    if accuracy is None or baseline_accuracy is None
                    else round(accuracy - baseline_accuracy, 6),
                }
            )
        else:
            merged.update(
                {
                    "market_cap_rank": "",
                    "market": "",
                    "market_cap": "",
                    "market_cap_source": "",
                    "market_cap_source_date": "",
                    "screen_price": "",
                    "accuracy_edge": "",
                }
            )
        enriched.append(merged)

    return enriched


def _select_top_rows(
    rows: list[dict[str, Any]],
    *,
    top: int,
    min_price: float | None,
    min_accuracy_edge: float | None,
    max_per_market: int | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "success" or row.get("probability_up") in {"", None}:
            continue
        price = _safe_float(row.get("screen_price")) or _safe_float(row.get("latest_close"))
        accuracy_edge = _safe_float(row.get("accuracy_edge"))
        if min_price is not None and price is not None and price < min_price:
            continue
        if min_accuracy_edge is not None and accuracy_edge is not None and accuracy_edge < min_accuracy_edge:
            continue
        filtered.append(row)

    filtered.sort(
        key=lambda row: (
            -(_safe_float(row.get("probability_up")) or 0.0),
            -(_safe_float(row.get("accuracy_edge")) or -999.0),
            -(_safe_float(row.get("market_cap")) or 0.0),
        )
    )

    selected: list[dict[str, Any]] = []
    market_counts: dict[str, int] = {}
    for row in filtered:
        market = str(row.get("market") or "UNKNOWN")
        if max_per_market is not None and market_counts.get(market, 0) >= max_per_market:
            continue
        selected.append(row)
        market_counts[market] = market_counts.get(market, 0) + 1
        if len(selected) >= top:
            break
    return selected


def _rewrite_prediction_json(summary: AutoRunSummary, rows: list[dict[str, Any]]) -> None:
    json_path = Path(summary.json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["rows"] = rows
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "selection_rank",
        "market_cap_rank",
        "market",
        "market_cap",
        "screen_price",
        "market_cap_source",
        "market_cap_source_date",
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
        "accuracy",
        "accuracy_edge",
        "baseline_accuracy",
        "precision_up",
        "recall_up",
        "train_rows",
        "test_rows",
        "attempt",
        "error",
    ]
    seen = set()
    names: list[str] = []
    for name in preferred:
        if any(name in row for row in rows):
            names.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _safe_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
