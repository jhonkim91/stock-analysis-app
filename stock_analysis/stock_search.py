from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import requests
from bs4 import BeautifulSoup


USER_AGENT = "Mozilla/5.0"
KRX_CORP_LIST_URL = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"


@dataclass(frozen=True)
class StockMatch:
    ticker: str
    exchange: str
    name: str
    symbol: str
    source: str


def resolve_stock_input(query: str, exchange_hint: str | None = None) -> str:
    value = query.strip()
    if not value:
        raise ValueError("Ticker is empty.")

    if _is_hard_symbol(value):
        return value.upper()

    if _is_soft_symbol_candidate(value):
        matches = search_stock_candidates(value, exchange_hint=exchange_hint, limit=10)
        if matches:
            return matches[0].symbol
        return value.upper()

    matches = search_stock_candidates(value, exchange_hint=exchange_hint, limit=10)
    if not matches:
        raise ValueError(f"No stock match was found for '{query}'.")

    best = matches[0]
    return best.symbol


def search_stock_candidates(
    query: str,
    *,
    exchange_hint: str | None = None,
    limit: int = 8,
) -> list[StockMatch]:
    value = query.strip()
    if not value:
        return []

    normalized_query = _normalize_text(value)
    matches: list[tuple[int, StockMatch]] = []

    for item in _load_korean_name_index():
        if exchange_hint and item.exchange != _normalize_exchange_hint(exchange_hint):
            continue

        score = _match_score(normalized_query, item)
        if score is not None:
            matches.append((score, item))

    matches.sort(key=lambda pair: (pair[0], pair[1].name, pair[1].ticker))
    local_matches = [item for _, item in matches]

    if len(local_matches) >= limit:
        return local_matches[:limit]

    remote_matches = _search_yfinance_candidates(value, exchange_hint=exchange_hint, limit=limit)
    merged = _dedupe_matches(local_matches + remote_matches)
    return merged[:limit]


def _is_hard_symbol(value: str) -> bool:
    upper = value.upper()
    if re.fullmatch(r"\d{6}", upper):
        return True
    if re.fullmatch(r"[0-9A-Z]{6}\.(KS|KQ)", upper):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", upper) and ("." in upper or "-" in upper or any(char.isdigit() for char in upper)):
        return True
    return False


def _is_soft_symbol_candidate(value: str) -> bool:
    upper = value.upper()
    return bool(re.fullmatch(r"[A-Z]{1,5}", upper))


def _match_score(query: str, item: StockMatch) -> int | None:
    item_name = _normalize_text(item.name)
    item_ticker = item.ticker
    item_symbol = item.symbol.upper()

    if query == item_name:
        return 0
    if query == item_ticker:
        return 1
    if query == item_symbol:
        return 2
    if item_name.startswith(query):
        return 3
    if query in item_name:
        return 4
    return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def _normalize_exchange_hint(value: str) -> str:
    hint = value.strip().upper()
    if hint in {"KRX", "KS", "KOSPI"}:
        return "KS"
    if hint in {"KQ", "KOSDAQ"}:
        return "KQ"
    return hint


def _dedupe_matches(items: list[StockMatch]) -> list[StockMatch]:
    seen: set[tuple[str, str]] = set()
    unique: list[StockMatch] = []
    for item in items:
        key = (item.symbol.upper(), item.name.upper())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _search_yfinance_candidates(
    query: str,
    *,
    exchange_hint: str | None,
    limit: int,
) -> list[StockMatch]:
    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        search = yf.Search(query)
        quotes = getattr(search, "quotes", []) or []
    except Exception:
        return []

    exchange_filter = _normalize_exchange_hint(exchange_hint) if exchange_hint else None
    results: list[StockMatch] = []
    for quote in quotes:
        symbol = str(quote.get("symbol") or "").upper()
        name = str(quote.get("shortname") or quote.get("longname") or symbol).strip()
        if not symbol or not name:
            continue

        if symbol.endswith(".KS"):
            exchange = "KS"
        elif symbol.endswith(".KQ"):
            exchange = "KQ"
        else:
            exchange = str(quote.get("exchange") or quote.get("exchDisp") or "").upper()

        if exchange_filter and exchange not in {exchange_filter, "KSC", "KOE"}:
            if not (exchange_filter == "KS" and symbol.endswith(".KS")) and not (
                exchange_filter == "KQ" and symbol.endswith(".KQ")
            ):
                continue

        ticker = symbol.split(".")[0]
        results.append(
            StockMatch(
                ticker=ticker,
                exchange="KS" if symbol.endswith(".KS") else "KQ" if symbol.endswith(".KQ") else exchange,
                name=name,
                symbol=symbol,
                source="yfinance",
            )
        )
        if len(results) >= limit:
            break

    return results


@lru_cache(maxsize=1)
def _load_korean_name_index() -> tuple[StockMatch, ...]:
    try:
        return _load_korean_name_index_from_krx()
    except Exception:
        pass

    items: list[StockMatch] = []
    seen: set[str] = set()
    for market, exchange, sosok in (("KOSPI", "KS", "0"), ("KOSDAQ", "KQ", "1")):
        for page in range(1, 45):
            page_items = _fetch_naver_search_page(market=market, exchange=exchange, sosok=sosok, page=page)
            if not page_items:
                break

            new_count = 0
            for item in page_items:
                if item.symbol in seen:
                    continue
                items.append(item)
                seen.add(item.symbol)
                new_count += 1

            if new_count == 0:
                break

    return tuple(items)


def _load_korean_name_index_from_krx() -> tuple[StockMatch, ...]:
    response = requests.get(KRX_CORP_LIST_URL, headers={"User-Agent": USER_AGENT}, timeout=20)
    response.raise_for_status()
    response.encoding = "euc-kr"
    soup = BeautifulSoup(response.text, "html.parser")

    rows = soup.select("table tr")
    if not rows:
        raise ValueError("KRX corp list returned no rows.")

    items: list[StockMatch] = []
    for row in rows[1:]:
        columns = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(columns) < 3:
            continue

        name, market_label, ticker = columns[0], columns[1], columns[2]
        ticker = ticker.upper()
        if not re.fullmatch(r"[0-9A-Z]{6}", ticker):
            continue

        normalized_market = _normalize_text(market_label)
        if normalized_market in {"유가", "코스피", "KOSPI"}:
            exchange = "KS"
        elif normalized_market in {"코스닥", "KOSDAQ"}:
            exchange = "KQ"
        else:
            continue

        items.append(
            StockMatch(
                ticker=ticker,
                exchange=exchange,
                name=name,
                symbol=f"{ticker}.{exchange}",
                source="krx-corp-list",
            )
        )

    if not items:
        raise ValueError("KRX corp list did not contain KOSPI/KOSDAQ matches.")
    return tuple(items)


def _fetch_naver_search_page(
    *,
    market: str,
    exchange: str,
    sosok: str,
    page: int,
) -> list[StockMatch]:
    url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    response.raise_for_status()
    response.encoding = "euc-kr"
    soup = BeautifulSoup(response.text, "html.parser")

    items: list[StockMatch] = []
    for link in soup.select("a.tltle"):
        href = link.get("href", "")
        match = re.search(r"code=(\d{6})", href)
        if not match:
            continue

        ticker = match.group(1)
        symbol = f"{ticker}.{exchange}"
        name = link.get_text(strip=True)
        items.append(
            StockMatch(
                ticker=ticker,
                exchange=exchange,
                name=name,
                symbol=symbol,
                source=f"naver-{market.lower()}",
            )
        )

    return items
