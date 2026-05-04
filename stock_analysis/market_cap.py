from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup


@dataclass
class MarketCapItem:
    rank: int
    ticker: str
    exchange: str
    market: str
    name: str
    market_cap: int
    price: int | None = None
    shares: int | None = None
    source: str = ""
    source_date: str = ""

    def to_watchlist_row(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "exchange": self.exchange,
            "name": self.name,
            "enabled": 1,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_top_market_cap(
    *,
    limit: int = 300,
    market: str = "ALL",
    source: str = "auto",
    date: str | None = None,
    exclude_preferred: bool = False,
) -> list[MarketCapItem]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0.")

    normalized_market = market.upper()
    if normalized_market not in {"ALL", "KOSPI", "KOSDAQ"}:
        raise ValueError("market must be one of: ALL, KOSPI, KOSDAQ.")

    normalized_source = source.lower()
    if normalized_source not in {"auto", "pykrx", "naver"}:
        raise ValueError("source must be one of: auto, pykrx, naver.")

    errors: list[str] = []
    if normalized_source in {"auto", "pykrx"}:
        try:
            items = _fetch_with_pykrx(
                limit=limit,
                market=normalized_market,
                date=date,
                exclude_preferred=exclude_preferred,
            )
            if items:
                return items
        except Exception as exc:
            errors.append(f"pykrx: {exc}")
            if normalized_source == "pykrx":
                raise RuntimeError(errors[-1]) from exc

    try:
        return _fetch_with_naver(
            limit=limit,
            market=normalized_market,
            exclude_preferred=exclude_preferred,
        )
    except Exception as exc:
        errors.append(f"naver: {exc}")
        raise RuntimeError("Failed to fetch market-cap universe. " + " | ".join(errors)) from exc


def write_market_cap_watchlist(path: str | Path, items: list[MarketCapItem]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["ticker", "exchange", "name", "enabled"])
        writer.writeheader()
        writer.writerows(item.to_watchlist_row() for item in items)


def write_market_cap_universe(path: str | Path, items: list[MarketCapItem]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "rank",
            "ticker",
            "exchange",
            "market",
            "name",
            "market_cap",
            "price",
            "shares",
            "source",
            "source_date",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(item.to_dict() for item in items)


def _fetch_with_pykrx(
    *,
    limit: int,
    market: str,
    date: str | None,
    exclude_preferred: bool,
) -> list[MarketCapItem]:
    try:
        from pykrx import stock
    except ImportError as exc:
        raise RuntimeError("pykrx is not installed. Run: python -m pip install -r requirements.txt") from exc

    source_date = _normalize_date(date)
    markets = ["KOSPI", "KOSDAQ"] if market == "ALL" else [market]
    items: list[MarketCapItem] = []

    for current_market in markets:
        frame = stock.get_market_cap_by_ticker(source_date, market=current_market)
        if frame.empty:
            continue

        frame = frame.reset_index().rename(columns={"티커": "ticker", "index": "ticker"})
        if "시가총액" not in frame.columns:
            raise RuntimeError("pykrx response did not include market-cap data.")

        for _, row in frame.iterrows():
            ticker = str(row["ticker"]).zfill(6)
            name = stock.get_market_ticker_name(ticker)
            if exclude_preferred and _looks_like_preferred(name):
                continue

            items.append(
                MarketCapItem(
                    rank=0,
                    ticker=ticker,
                    exchange="KS" if current_market == "KOSPI" else "KQ",
                    market=current_market,
                    name=name,
                    market_cap=int(row["시가총액"]),
                    price=_safe_int(row.get("종가")),
                    shares=_safe_int(row.get("상장주식수")),
                    source="pykrx",
                    source_date=source_date,
                )
            )

    return _rank_items(items, limit)


def _fetch_with_naver(
    *,
    limit: int,
    market: str,
    exclude_preferred: bool,
) -> list[MarketCapItem]:
    markets = ["KOSPI", "KOSDAQ"] if market == "ALL" else [market]
    items: list[MarketCapItem] = []

    for current_market in markets:
        per_market_limit = limit if market != "ALL" else limit
        items.extend(_fetch_naver_market(current_market, per_market_limit, exclude_preferred))

    return _rank_items(items, limit)


def _fetch_naver_market(
    market: str,
    limit: int,
    exclude_preferred: bool,
) -> list[MarketCapItem]:
    sosok = "0" if market == "KOSPI" else "1"
    exchange = "KS" if market == "KOSPI" else "KQ"
    pages = max(1, (limit + 49) // 50)
    items: list[MarketCapItem] = []

    for page in range(1, pages + 1):
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        response.raise_for_status()
        response.encoding = "euc-kr"

        page_items = _parse_naver_market_page(
            response.text,
            market=market,
            exchange=exchange,
            source_date=datetime.now().strftime("%Y%m%d"),
            exclude_preferred=exclude_preferred,
        )
        if not page_items:
            break

        items.extend(page_items)
        if len(items) >= limit:
            break

    return items[:limit]


def _parse_naver_market_page(
    html: str,
    *,
    market: str,
    exchange: str,
    source_date: str,
    exclude_preferred: bool,
) -> list[MarketCapItem]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[MarketCapItem] = []

    for link in soup.select("a.tltle"):
        href = link.get("href", "")
        code_match = re.search(r"code=(\d{6})", href)
        row = link.find_parent("tr")
        if not code_match or row is None:
            continue

        columns = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(columns) < 8:
            continue

        name = link.get_text(strip=True)
        if exclude_preferred and _looks_like_preferred(name):
            continue

        items.append(
            MarketCapItem(
                rank=_parse_int(columns[0]) or 0,
                ticker=code_match.group(1),
                exchange=exchange,
                market=market,
                name=name,
                market_cap=(_parse_int(columns[6]) or 0) * 100_000_000,
                price=_parse_int(columns[2]),
                shares=_parse_int(columns[7]),
                source="naver",
                source_date=source_date,
            )
        )

    return items


def _rank_items(items: list[MarketCapItem], limit: int) -> list[MarketCapItem]:
    filtered = [item for item in items if item.market_cap > 0]
    filtered.sort(key=lambda item: item.market_cap, reverse=True)
    ranked = filtered[:limit]
    for index, item in enumerate(ranked, start=1):
        item.rank = index
    return ranked


def _normalize_date(date: str | None) -> str:
    if date:
        return re.sub(r"\D", "", date)

    # KRX data may be unavailable before the daily close, so try the previous day first.
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _looks_like_preferred(name: str) -> bool:
    return name.endswith("우") or "우B" in name or "우C" in name or "우선주" in name
