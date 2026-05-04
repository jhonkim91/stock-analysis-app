from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests


BASE_SITE_URL = "https://feargreed.co.kr/"
API_URL = "https://feargree-api.vercel.app/api"


@dataclass
class FearGreedData:
    summary: pd.DataFrame
    timeline: pd.DataFrame
    factors: pd.DataFrame
    latest_date: str
    source_age_days: int
    source_url: str = BASE_SITE_URL


def fetch_fear_greed_data(*, timeout: int = 20) -> FearGreedData:
    payload = _get_json(API_URL, timeout=timeout)
    if not payload.get("success"):
        raise ValueError("Fear & Greed API did not return success.")

    kr_payload = dict(payload.get("kr") or {})
    historical_payload = dict(payload.get("historical") or {})
    history = pd.DataFrame(payload.get("history", []))
    if history.empty:
        raise ValueError("Fear & Greed history is empty.")

    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)
    history["fear_greed"] = pd.to_numeric(history["kr"], errors="coerce")
    history["us_fear_greed"] = pd.to_numeric(history.get("us"), errors="coerce")
    history["kospi_close"] = pd.NA

    latest_history_date = history["date"].max().date()
    latest_date = latest_history_date.isoformat()
    source_age_days = (date.today() - latest_history_date).days

    current_score = float(kr_payload.get("score") or history["fear_greed"].dropna().iloc[-1])
    week_ago_score = _optional_float(historical_payload.get("week_ago"))
    month_ago_score = _optional_float(historical_payload.get("month_ago"))
    year_ago_score = _optional_float(historical_payload.get("year_ago"))

    if week_ago_score is None:
        week_ago_score = _history_score_ago(history, days=7)
    if month_ago_score is None:
        month_ago_score = _history_score_ago(history, days=30)
    if year_ago_score is None:
        year_ago_score = _history_score_ago(history, days=365)

    summary = pd.DataFrame(
        [
            _summary_row("현재", current_score, status_override=str(kr_payload.get("label") or "").strip() or None),
            _summary_row("1주 전", week_ago_score, current_score=current_score),
            _summary_row("1개월 전", month_ago_score, current_score=current_score),
            _summary_row("1년 전", year_ago_score, current_score=current_score),
        ]
    )

    factors = pd.DataFrame(
        [
            {
                "factor_key": indicator.get("name", ""),
                "factor": indicator.get("name", ""),
                "score_0_1": round(float(indicator.get("value", 0)) / 100, 4),
                "score_0_100": round(float(indicator.get("value", 0)), 2),
                "raw": indicator.get("raw"),
                "unit": indicator.get("unit", ""),
            }
            for indicator in (kr_payload.get("indicators") or [])
        ]
    )
    if not factors.empty:
        factors = factors.sort_values("score_0_100", ascending=False).reset_index(drop=True)

    timeline = history.loc[:, ["date", "fear_greed", "us_fear_greed", "kospi_close"]].copy()

    return FearGreedData(
        summary=summary,
        timeline=timeline,
        factors=factors,
        latest_date=latest_date,
        source_age_days=source_age_days,
        source_url=BASE_SITE_URL,
    )


def _get_json(url: str, *, timeout: int) -> dict:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return response.json()


def _history_score_ago(history: pd.DataFrame, *, days: int) -> float | None:
    cutoff = history["date"].max() - pd.Timedelta(days=days)
    candidates = history.loc[history["date"] <= cutoff, "fear_greed"].dropna()
    if candidates.empty:
        return None
    return float(candidates.iloc[-1])


def _summary_row(
    label: str,
    score: float | None,
    *,
    current_score: float | None = None,
    status_override: str | None = None,
) -> dict:
    numeric = None if score is None else float(score)
    change = None
    if numeric is not None and current_score is not None:
        change = round(current_score - numeric, 2)
    return {
        "period": label,
        "score": None if numeric is None else round(numeric, 2),
        "status": status_override or score_to_status(numeric),
        "change": change,
    }


def score_to_status(score: float | None) -> str:
    if score is None:
        return "데이터 없음"
    if score <= 24:
        return "극단적 공포"
    if score <= 44:
        return "공포"
    if score <= 54:
        return "중립"
    if score <= 74:
        return "탐욕"
    return "극단적 탐욕"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
