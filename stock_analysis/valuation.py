from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stock_analysis.data import load_history, normalize_ticker


@dataclass
class FinancialSnapshot:
    ticker: str
    name: str
    currency: str
    financial_currency: str
    current_price: float
    market_cap: float | None
    shares_outstanding: float | None
    statement_date: str
    revenue: float | None
    net_income: float | None
    free_cash_flow: float | None
    operating_cash_flow: float | None
    capital_expenditure: float | None
    total_equity: float | None
    total_debt: float | None
    cash: float | None
    eps: float | None
    book_value_per_share: float | None
    free_cash_flow_per_share: float | None
    roe: float | None
    revenue_growth: float | None
    net_income_growth: float | None
    free_cash_flow_growth: float | None
    trailing_pe: float | None
    forward_pe: float | None
    price_to_book: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValuationMethod:
    name: str
    target_price: float
    weight: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValuationAssumptions:
    target_pe: float
    target_pbr: float
    dcf_growth: float
    discount_rate: float
    terminal_growth: float
    years: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValuationResult:
    ticker: str
    run_at: str
    snapshot: FinancialSnapshot
    assumptions: ValuationAssumptions
    methods: list[ValuationMethod]
    target_price: float
    current_price: float
    upside: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "run_at": self.run_at,
            "snapshot": self.snapshot.to_dict(),
            "assumptions": self.assumptions.to_dict(),
            "methods": [method.to_dict() for method in self.methods],
            "target_price": self.target_price,
            "current_price": self.current_price,
            "upside": self.upside,
        }


def calculate_target_price(
    ticker: str,
    *,
    exchange: str | None = None,
    target_pe: float | None = None,
    target_pbr: float | None = None,
    growth: float | None = None,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.02,
    years: int = 5,
) -> ValuationResult:
    symbol = normalize_ticker(ticker, exchange)
    snapshot = fetch_financial_snapshot(symbol)
    assumptions = _build_assumptions(
        snapshot,
        target_pe=target_pe,
        target_pbr=target_pbr,
        growth=growth,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        years=years,
    )
    methods = _build_methods(snapshot, assumptions)

    if not methods:
        raise ValueError(
            f"Not enough financial data was available to value {symbol}. "
            "Try another ticker or provide more assumptions."
        )

    target = _weighted_average(methods)
    return ValuationResult(
        ticker=symbol,
        run_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        snapshot=snapshot,
        assumptions=assumptions,
        methods=methods,
        target_price=target,
        current_price=snapshot.current_price,
        upside=(target / snapshot.current_price) - 1,
    )


def save_valuation_json(path: str | Path, result: ValuationResult) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_financial_snapshot(ticker: str) -> FinancialSnapshot:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required to download financial data. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    stock = yf.Ticker(ticker)
    info = _safe_info(stock)
    income = _safe_frame(lambda: stock.income_stmt)
    quarterly_income = _safe_frame(lambda: stock.quarterly_income_stmt)
    balance = _safe_frame(lambda: stock.balance_sheet)
    cashflow = _safe_frame(lambda: stock.cashflow)
    quarterly_cashflow = _safe_frame(lambda: stock.quarterly_cashflow)

    current_price = _first_number(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        _fast_info_value(stock, "last_price"),
        _latest_close(ticker),
    )
    if current_price is None or current_price <= 0:
        raise ValueError(f"Could not determine current price for {ticker}.")

    shares = _first_number(
        info.get("sharesOutstanding"),
        _latest_value(balance, ["Ordinary Shares Number", "Share Issued"]),
    )
    market_cap = _first_number(info.get("marketCap"), current_price * shares if shares else None)

    revenue = _ttm_or_latest(
        quarterly_income,
        income,
        ["Total Revenue", "Operating Revenue"],
    )
    net_income = _ttm_or_latest(
        quarterly_income,
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income From Continuing Operation Net Minority Interest",
        ],
    )
    free_cash_flow = _ttm_or_latest(quarterly_cashflow, cashflow, ["Free Cash Flow"])
    operating_cash_flow = _ttm_or_latest(
        quarterly_cashflow,
        cashflow,
        ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    )
    capex = _ttm_or_latest(quarterly_cashflow, cashflow, ["Capital Expenditure"])
    if free_cash_flow is None and operating_cash_flow is not None and capex is not None:
        free_cash_flow = operating_cash_flow + capex

    total_equity = _latest_value(
        balance,
        ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
    )
    total_debt = _latest_value(balance, ["Total Debt"])
    cash = _latest_value(
        balance,
        [
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash Financial",
        ],
    )

    eps = _first_number(
        info.get("trailingEps"),
        net_income / shares if net_income is not None and shares else None,
    )
    book_value_per_share = _first_number(
        info.get("bookValue"),
        total_equity / shares if total_equity is not None and shares else None,
    )
    free_cash_flow_per_share = free_cash_flow / shares if free_cash_flow is not None and shares else None
    roe = net_income / total_equity if net_income is not None and total_equity and total_equity > 0 else None

    return FinancialSnapshot(
        ticker=ticker,
        name=str(info.get("shortName") or info.get("longName") or ticker),
        currency=str(info.get("currency") or ""),
        financial_currency=str(info.get("financialCurrency") or info.get("currency") or ""),
        current_price=float(current_price),
        market_cap=_clean_number(market_cap),
        shares_outstanding=_clean_number(shares),
        statement_date=_statement_date(income, balance, cashflow),
        revenue=_clean_number(revenue),
        net_income=_clean_number(net_income),
        free_cash_flow=_clean_number(free_cash_flow),
        operating_cash_flow=_clean_number(operating_cash_flow),
        capital_expenditure=_clean_number(capex),
        total_equity=_clean_number(total_equity),
        total_debt=_clean_number(total_debt),
        cash=_clean_number(cash),
        eps=_clean_number(eps),
        book_value_per_share=_clean_number(book_value_per_share),
        free_cash_flow_per_share=_clean_number(free_cash_flow_per_share),
        roe=_clean_number(roe),
        revenue_growth=_clean_number(_growth(income, ["Total Revenue", "Operating Revenue"])),
        net_income_growth=_clean_number(
            _growth(
                income,
                [
                    "Net Income",
                    "Net Income Common Stockholders",
                    "Net Income From Continuing Operation Net Minority Interest",
                ],
            )
        ),
        free_cash_flow_growth=_clean_number(_growth(cashflow, ["Free Cash Flow"])),
        trailing_pe=_clean_number(info.get("trailingPE")),
        forward_pe=_clean_number(info.get("forwardPE")),
        price_to_book=_clean_number(info.get("priceToBook")),
    )


def _build_assumptions(
    snapshot: FinancialSnapshot,
    *,
    target_pe: float | None,
    target_pbr: float | None,
    growth: float | None,
    discount_rate: float,
    terminal_growth: float,
    years: int,
) -> ValuationAssumptions:
    if discount_rate <= terminal_growth:
        raise ValueError("discount_rate must be greater than terminal_growth.")
    if years <= 0:
        raise ValueError("years must be greater than 0.")

    dcf_growth = _clamp(
        growth if growth is not None else _median_available(
            snapshot.revenue_growth,
            snapshot.net_income_growth,
            snapshot.free_cash_flow_growth,
        ),
        -0.05,
        0.15,
        default=0.03,
    )

    fair_pe = target_pe if target_pe is not None else _auto_target_pe(snapshot, dcf_growth)
    fair_pbr = target_pbr if target_pbr is not None else _auto_target_pbr(snapshot)

    return ValuationAssumptions(
        target_pe=float(fair_pe),
        target_pbr=float(fair_pbr),
        dcf_growth=float(dcf_growth),
        discount_rate=float(discount_rate),
        terminal_growth=float(terminal_growth),
        years=int(years),
    )


def _build_methods(
    snapshot: FinancialSnapshot,
    assumptions: ValuationAssumptions,
) -> list[ValuationMethod]:
    methods: list[ValuationMethod] = []

    if snapshot.eps is not None and snapshot.eps > 0 and assumptions.target_pe > 0:
        target = snapshot.eps * assumptions.target_pe
        methods.append(
            ValuationMethod(
                name="PER",
                target_price=target,
                weight=0.45,
                detail=f"EPS {snapshot.eps:.4f} x target PER {assumptions.target_pe:.2f}",
            )
        )

    if (
        snapshot.book_value_per_share is not None
        and snapshot.book_value_per_share > 0
        and assumptions.target_pbr > 0
    ):
        target = snapshot.book_value_per_share * assumptions.target_pbr
        methods.append(
            ValuationMethod(
                name="PBR",
                target_price=target,
                weight=0.15,
                detail=f"BPS {snapshot.book_value_per_share:.4f} x target PBR {assumptions.target_pbr:.2f}",
            )
        )

    if (
        snapshot.free_cash_flow is not None
        and snapshot.free_cash_flow > 0
        and snapshot.shares_outstanding is not None
        and snapshot.shares_outstanding > 0
    ):
        target = _dcf_equity_value_per_share(snapshot, assumptions)
        if target > 0 and math.isfinite(target):
            methods.append(
                ValuationMethod(
                    name="DCF",
                    target_price=target,
                    weight=0.40,
                    detail=(
                        f"FCF growth {assumptions.dcf_growth:.2%}, "
                        f"discount {assumptions.discount_rate:.2%}, "
                        f"terminal {assumptions.terminal_growth:.2%}"
                    ),
                )
            )

    return _normalize_method_weights(methods)


def _dcf_equity_value_per_share(
    snapshot: FinancialSnapshot,
    assumptions: ValuationAssumptions,
) -> float:
    assert snapshot.free_cash_flow is not None
    assert snapshot.shares_outstanding is not None

    discount = assumptions.discount_rate
    growth = assumptions.dcf_growth
    terminal_growth = assumptions.terminal_growth

    present_value = 0.0
    fcf = snapshot.free_cash_flow
    for year in range(1, assumptions.years + 1):
        fcf *= 1 + growth
        present_value += fcf / ((1 + discount) ** year)

    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount - terminal_growth)
    present_value += terminal_value / ((1 + discount) ** assumptions.years)

    net_cash = (snapshot.cash or 0.0) - (snapshot.total_debt or 0.0)
    equity_value = present_value + net_cash
    return equity_value / snapshot.shares_outstanding


def _weighted_average(methods: list[ValuationMethod]) -> float:
    weight_sum = sum(method.weight for method in methods)
    if weight_sum <= 0:
        raise ValueError("No valid valuation weights were produced.")
    return sum(method.target_price * method.weight for method in methods) / weight_sum


def _normalize_method_weights(methods: list[ValuationMethod]) -> list[ValuationMethod]:
    total = sum(method.weight for method in methods)
    if total <= 0:
        return methods
    return [
        ValuationMethod(
            name=method.name,
            target_price=method.target_price,
            weight=method.weight / total,
            detail=method.detail,
        )
        for method in methods
    ]


def _safe_info(stock: Any) -> dict[str, Any]:
    try:
        return dict(stock.get_info())
    except Exception:
        return {}


def _safe_frame(loader) -> pd.DataFrame:
    try:
        frame = loader()
    except Exception:
        return pd.DataFrame()
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _fast_info_value(stock: Any, name: str) -> float | None:
    try:
        return _clean_number(getattr(stock.fast_info, name))
    except Exception:
        return None


def _latest_close(ticker: str) -> float | None:
    try:
        history = load_history(ticker, period="1y", interval="1d")
    except Exception:
        return None
    return _clean_number(history["Close"].iloc[-1])


def _ttm_or_latest(
    quarterly: pd.DataFrame,
    annual: pd.DataFrame,
    aliases: list[str],
) -> float | None:
    values = _series_values(quarterly, aliases)
    if len(values) >= 4:
        return sum(values[:4])
    return _latest_value(annual, aliases)


def _latest_value(frame: pd.DataFrame, aliases: list[str]) -> float | None:
    values = _series_values(frame, aliases)
    return values[0] if values else None


def _series_values(frame: pd.DataFrame, aliases: list[str]) -> list[float]:
    if frame.empty:
        return []

    for alias in aliases:
        if alias in frame.index:
            values: list[float] = []
            for value in frame.loc[alias].dropna().tolist():
                clean = _clean_number(value)
                if clean is not None:
                    values.append(clean)
            return values
    return []


def _growth(frame: pd.DataFrame, aliases: list[str]) -> float | None:
    values = _series_values(frame, aliases)
    if len(values) < 2:
        return None
    current, previous = values[0], values[1]
    if previous == 0:
        return None
    return (current - previous) / abs(previous)


def _statement_date(*frames: pd.DataFrame) -> str:
    dates: list[str] = []
    for frame in frames:
        if frame.empty:
            continue
        for column in frame.columns:
            if hasattr(column, "date"):
                dates.append(str(column.date()))
            else:
                dates.append(str(column))
            break
    return dates[0] if dates else ""


def _first_number(*values: Any) -> float | None:
    for value in values:
        clean = _clean_number(value)
        if clean is not None:
            return clean
    return None


def _clean_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _median_available(*values: float | None) -> float | None:
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2


def _clamp(value: float | None, low: float, high: float, *, default: float) -> float:
    if value is None or not math.isfinite(value):
        return default
    return min(max(value, low), high)


def _auto_target_pe(snapshot: FinancialSnapshot, growth: float) -> float:
    roe_bonus = _clamp(snapshot.roe, 0.0, 0.25, default=0.08) * 20
    growth_bonus = max(growth, 0.0) * 60
    return _clamp(10 + roe_bonus + growth_bonus, 8, 30, default=15)


def _auto_target_pbr(snapshot: FinancialSnapshot) -> float:
    roe = _clamp(snapshot.roe, 0.0, 0.30, default=0.08)
    return _clamp(0.6 + roe * 10, 0.5, 5.0, default=1.2)
