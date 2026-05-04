from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stock_analysis.market_cap import (
    MarketCapItem,
    get_top_market_cap,
    write_market_cap_universe,
    write_market_cap_watchlist,
)
from stock_analysis.valuation import ValuationResult, calculate_target_price


@dataclass
class ValuationScreenSummary:
    run_at: str
    output_dir: str
    universe_path: str
    watchlist_path: str
    valuations_csv_path: str
    valuations_json_path: str
    top_csv_path: str
    top_json_path: str
    universe_count: int
    evaluated_count: int
    succeeded: int
    failed: int
    top_count: int


def run_valuation_screen(
    *,
    output_dir: str | Path = "outputs/target_market_cap",
    rank_limit: int = 300,
    run_limit: int | None = None,
    top: int = 10,
    market: str = "ALL",
    source: str = "naver",
    date: str | None = None,
    exclude_preferred: bool = True,
    target_pe: float | None = None,
    target_pbr: float | None = None,
    growth: float | None = None,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.02,
    years: int = 5,
    sleep_seconds: float = 0.0,
    retries: int = 1,
    fail_fast: bool = False,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> ValuationScreenSummary:
    if rank_limit <= 0:
        raise ValueError("rank_limit must be greater than 0.")
    if top <= 0:
        raise ValueError("top must be greater than 0.")
    if run_limit is not None and run_limit <= 0:
        raise ValueError("run_limit must be greater than 0.")

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
    run_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_label = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(output_dir) / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    universe_path = run_dir / "market_cap_universe.csv"
    watchlist_path = run_dir / "market_cap_watchlist.csv"
    valuations_csv_path = run_dir / "valuations.csv"
    valuations_json_path = run_dir / "valuations.json"
    top_csv_path = run_dir / f"top{top}.csv"
    top_json_path = run_dir / f"top{top}.json"

    write_market_cap_universe(universe_path, universe)
    write_market_cap_watchlist(watchlist_path, selected_universe)

    rows: list[dict[str, Any]] = []
    valuation_payloads: list[dict[str, Any]] = []
    for index, item in enumerate(selected_universe, start=1):
        row, payload = _run_item(
            item,
            run_at=run_at,
            target_pe=target_pe,
            target_pbr=target_pbr,
            growth=growth,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            years=years,
            retries=retries,
        )
        rows.append(row)
        if payload:
            valuation_payloads.append(payload)
        if progress:
            progress(index, len(selected_universe), row)

        if row["status"] == "failed" and fail_fast:
            break
        if sleep_seconds > 0 and index < len(selected_universe):
            time.sleep(sleep_seconds)

    rows = _sort_rows(rows)
    top_rows = [row for row in rows if row["status"] == "success"][:top]
    for selection_rank, row in enumerate(top_rows, start=1):
        row["selection_rank"] = selection_rank

    _write_csv(valuations_csv_path, rows)
    _write_json(
        valuations_json_path,
        {
            "run_at": run_at,
            "screen": "valuation_market_cap",
            "rank_limit": rank_limit,
            "run_limit": run_limit,
            "top": top,
            "market": market,
            "source": source,
            "date": date,
            "exclude_preferred": exclude_preferred,
            "assumptions": {
                "target_pe": target_pe,
                "target_pbr": target_pbr,
                "growth": growth,
                "discount_rate": discount_rate,
                "terminal_growth": terminal_growth,
                "years": years,
            },
            "rows": rows,
            "valuation_details": valuation_payloads,
        },
    )
    _write_csv(top_csv_path, top_rows)
    _write_json(
        top_json_path,
        {
            "run_at": run_at,
            "screen": "valuation_market_cap",
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

    succeeded = sum(1 for row in rows if row["status"] == "success")
    failed = len(rows) - succeeded
    return ValuationScreenSummary(
        run_at=run_at,
        output_dir=str(run_dir),
        universe_path=str(universe_path),
        watchlist_path=str(watchlist_path),
        valuations_csv_path=str(valuations_csv_path),
        valuations_json_path=str(valuations_json_path),
        top_csv_path=str(top_csv_path),
        top_json_path=str(top_json_path),
        universe_count=len(universe),
        evaluated_count=len(rows),
        succeeded=succeeded,
        failed=failed,
        top_count=len(top_rows),
    )


def _run_item(
    item: MarketCapItem,
    *,
    run_at: str,
    target_pe: float | None,
    target_pbr: float | None,
    growth: float | None,
    discount_rate: float,
    terminal_growth: float,
    years: int,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    attempts = max(1, retries + 1)
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            result = calculate_target_price(
                item.ticker,
                exchange=item.exchange,
                target_pe=target_pe,
                target_pbr=target_pbr,
                growth=growth,
                discount_rate=discount_rate,
                terminal_growth=terminal_growth,
                years=years,
            )
            return _success_row(item, result, run_at=run_at, attempt=attempt), result.to_dict()
        except Exception as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(min(2 * attempt, 10))

    return _failed_row(item, run_at=run_at, attempt=attempts, error=last_error), None


def _success_row(
    item: MarketCapItem,
    result: ValuationResult,
    *,
    run_at: str,
    attempt: int,
) -> dict[str, Any]:
    snapshot = result.snapshot
    assumptions = result.assumptions
    method_targets = {method.name: method.target_price for method in result.methods}
    methods_used = ",".join(method.name for method in result.methods)

    return {
        "selection_rank": "",
        "run_at": run_at,
        "status": "success",
        "market_cap_rank": item.rank,
        "market": item.market,
        "market_cap": item.market_cap,
        "market_cap_source": item.source,
        "market_cap_source_date": item.source_date,
        "name": item.name,
        "ticker_input": item.ticker,
        "exchange": item.exchange,
        "ticker": result.ticker,
        "currency": snapshot.currency,
        "financial_currency": snapshot.financial_currency,
        "statement_date": snapshot.statement_date,
        "current_price": round(result.current_price, 6),
        "target_price": round(result.target_price, 6),
        "upside": round(result.upside, 6),
        "upside_percent": round(result.upside * 100, 4),
        "methods_used": methods_used,
        "per_target": _round_or_blank(method_targets.get("PER")),
        "pbr_target": _round_or_blank(method_targets.get("PBR")),
        "dcf_target": _round_or_blank(method_targets.get("DCF")),
        "eps": _round_or_blank(snapshot.eps),
        "bps": _round_or_blank(snapshot.book_value_per_share),
        "fcf_per_share": _round_or_blank(snapshot.free_cash_flow_per_share),
        "roe": _round_or_blank(snapshot.roe),
        "revenue_growth": _round_or_blank(snapshot.revenue_growth),
        "net_income_growth": _round_or_blank(snapshot.net_income_growth),
        "free_cash_flow_growth": _round_or_blank(snapshot.free_cash_flow_growth),
        "target_pe": round(assumptions.target_pe, 6),
        "target_pbr": round(assumptions.target_pbr, 6),
        "dcf_growth": round(assumptions.dcf_growth, 6),
        "discount_rate": round(assumptions.discount_rate, 6),
        "terminal_growth": round(assumptions.terminal_growth, 6),
        "years": assumptions.years,
        "attempt": attempt,
        "error": "",
    }


def _failed_row(
    item: MarketCapItem,
    *,
    run_at: str,
    attempt: int,
    error: str,
) -> dict[str, Any]:
    return {
        "selection_rank": "",
        "run_at": run_at,
        "status": "failed",
        "market_cap_rank": item.rank,
        "market": item.market,
        "market_cap": item.market_cap,
        "market_cap_source": item.source,
        "market_cap_source_date": item.source_date,
        "name": item.name,
        "ticker_input": item.ticker,
        "exchange": item.exchange,
        "ticker": "",
        "currency": "",
        "financial_currency": "",
        "statement_date": "",
        "current_price": "",
        "target_price": "",
        "upside": "",
        "upside_percent": "",
        "methods_used": "",
        "per_target": "",
        "pbr_target": "",
        "dcf_target": "",
        "eps": "",
        "bps": "",
        "fcf_per_share": "",
        "roe": "",
        "revenue_growth": "",
        "net_income_growth": "",
        "free_cash_flow_growth": "",
        "target_pe": "",
        "target_pbr": "",
        "dcf_growth": "",
        "discount_rate": "",
        "terminal_growth": "",
        "years": "",
        "attempt": attempt,
        "error": error,
    }


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        if row["status"] != "success":
            return (1, -1.0)
        return (0, -float(row["upside"]))

    return sorted(rows, key=sort_key)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _round_or_blank(value: float | None) -> float | str:
    if value is None:
        return ""
    return round(float(value), 6)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "selection_rank",
        "run_at",
        "status",
        "market_cap_rank",
        "market",
        "market_cap",
        "market_cap_source",
        "market_cap_source_date",
        "name",
        "ticker_input",
        "exchange",
        "ticker",
        "currency",
        "financial_currency",
        "statement_date",
        "current_price",
        "target_price",
        "upside",
        "upside_percent",
        "methods_used",
        "per_target",
        "pbr_target",
        "dcf_target",
        "eps",
        "bps",
        "fcf_per_share",
        "roe",
        "revenue_growth",
        "net_income_growth",
        "free_cash_flow_growth",
        "target_pe",
        "target_pbr",
        "dcf_growth",
        "discount_rate",
        "terminal_growth",
        "years",
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
