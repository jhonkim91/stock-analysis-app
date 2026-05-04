from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests


BASE_SITE_URL = "https://kospi-fear-greed-index.co.kr/"
VALUE_URL = (
    "https://raw.githubusercontent.com/immanuelk1m/"
    "kospi-feargreedindex/refs/heads/main/assets/js/json/value.json"
)
INDEX_URL = (
    "https://raw.githubusercontent.com/immanuelk1m/"
    "kospi-feargreedindex/main/assets/js/json/index.json"
)
FACTOR_STATUS_URL = (
    "https://raw.githubusercontent.com/immanuelk1m/"
    "kospi-feargreedindex/refs/heads/main/assets/js/json/factor_status.json"
)


STATUS_MAP = {
    "1": "극도의 공포",
    "2": "공포",
    "3": "중립",
    "4": "탐욕",
    "5": "극도의 탐욕",
}


FACTOR_LABELS = {
    "ema_spread_scaled": "EMA Spread",
    "mcclenllan_scaled": "McClellan",
    "p_c_ema_scaled": "Put/Call EMA",
    "vix_ema_spread_scaled": "VIX EMA Spread",
    "safe_spread_scaled": "Safe Spread",
    "junk_spread_scaled": "Junk Spread",
    "stock_strength_scaled": "Stock Strength",
}


@dataclass
class FearGreedData:
    summary: pd.DataFrame
    timeline: pd.DataFrame
    factors: pd.DataFrame
    latest_date: str
    source_age_days: int
    source_url: str = BASE_SITE_URL


def fetch_fear_greed_data(*, timeout: int = 20) -> FearGreedData:
    value_payload = _get_json(VALUE_URL, timeout=timeout)
    timeline_payload = _get_json(INDEX_URL, timeout=timeout)
    factor_payload = _get_json(FACTOR_STATUS_URL, timeout=timeout)

    summary = pd.DataFrame(
        [
            _summary_row("현재", value_payload.get("current"), value_payload.get("current_s")),
            _summary_row("1주 전", value_payload.get("week"), value_payload.get("week_s")),
            _summary_row("1개월 전", value_payload.get("month"), value_payload.get("month_s")),
            _summary_row("1년 전", value_payload.get("year"), value_payload.get("year_s")),
        ]
    )

    timeline = pd.DataFrame(timeline_payload.get("data", []))
    if timeline.empty:
        raise ValueError("Fear & Greed timeline data is empty.")
    timeline = timeline.rename(columns={"x": "date", "y": "kospi_close", "z": "fear_greed"})
    timeline["date"] = pd.to_datetime(timeline["date"])
    timeline = timeline.sort_values("date").reset_index(drop=True)

    latest_timestamp = timeline["date"].iloc[-1]
    latest_date = latest_timestamp.date().isoformat()
    source_age_days = (date.today() - latest_timestamp.date()).days

    factors = pd.DataFrame(
        [
            {
                "factor_key": key,
                "factor": FACTOR_LABELS.get(key, key),
                "score_0_1": float(value),
                "score_0_100": round(float(value) * 100, 2),
            }
            for key, value in factor_payload.items()
        ]
    )
    factors = factors.sort_values("score_0_100", ascending=False).reset_index(drop=True)

    return FearGreedData(
        summary=summary,
        timeline=timeline,
        factors=factors,
        latest_date=latest_date,
        source_age_days=source_age_days,
    )


def _get_json(url: str, *, timeout: int) -> dict:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _summary_row(label: str, value: float | None, status_code: str | None) -> dict:
    numeric = float(value) if value is not None else None
    return {
        "period": label,
        "score": round(numeric, 2) if numeric is not None else None,
        "status_code": status_code or "",
        "status": STATUS_MAP.get(status_code or "", "알 수 없음"),
    }
