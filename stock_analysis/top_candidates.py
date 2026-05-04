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
        },
    )

    prediction_rows = _read_csv_rows(auto_summary.csv_path)
    enriched_rows = _merge_market_cap_metadata(prediction_rows, universe)
    _write_csv(Path(auto_summary.csv_path), enriched_rows)

    top_rows = [
        row
        for row in enriched_rows
        if row.get("status") == "success" and row.get("probability_up") not in {"", None}
    ][:top]
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
            merged.update(
                {
                    "market_cap_rank": item.rank,
                    "market": item.market,
                    "market_cap": item.market_cap,
                    "market_cap_source": item.source,
                    "market_cap_source_date": item.source_date,
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
                }
            )
        enriched.append(merged)

    return enriched


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
