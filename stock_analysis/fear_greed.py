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
    history = pd.DataFrame(payload.get("history", []))
    if history.empty:
        raise ValueError("Fear & Greed history is empty.")

    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)
    history["fear_greed"] = pd.to_numeric(history["kr"], errors="coerce")
    history["us_fear_greed"] = pd.to_numeric(history.get("us"), errors="coerce")
    history["kospi_close"] = pd.NA

    latest_timestamp = pd.to_datetime(payload.get("timestamp"))
    latest_date = latest_timestamp.date().isoformat()
    source_age_days = (date.today() - latest_timestamp.date()).days

    current_score = float(kr_payload.get("score") or history["fear_greed"].dropna().iloc[-1])
    summary = pd.DataFrame(
        [
            _summary_row("현재", current_score),
            _history_summary_row("1주 전", history, current_score, days=7),
            _history_summary_row("1개월 전", history, current_score, days=30),
            _history_summary_row("1년 전", history, current_score, days=365),
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


def _history_summary_row(label: str, history: pd.DataFrame, current_score: float, *, days: int) -> dict:
    cutoff = history["date"].max() - pd.Timedelta(days=days)
    candidates = history.loc[history["date"] <= cutoff, "fear_greed"].dropna()
    score = float(candidates.iloc[-1]) if not candidates.empty else None
    return _summary_row(label, score, current_score=current_score)


def _summary_row(label: str, score: float | None, *, current_score: float | None = None) -> dict:
    numeric = None if score is None else float(score)
    change = None
    if numeric is not None and current_score is not None:
        change = round(current_score - numeric, 2)
    return {
        "period": label,
        "score": None if numeric is None else round(numeric, 2),
        "status": score_to_status(numeric),
        "change": change,
    }


def score_to_status(score: float | None) -> str:
    if score is None:
        return "데이터 없음"
    if score <= 20:
        return "극단적 공포"
    if score <= 40:
        return "공포"
    if score <= 60:
        return "중립"
    if score <= 80:
        return "탐욕"
    return "극단적 탐욕"
