from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SettingProfile:
    ticker: str
    run_count: int
    recency_weight_sum: float
    latest_run_at: str | None
    preferred_threshold: float | None
    preferred_model_name: str | None
    preferred_model_label: str | None
    should_compare_tree: bool
    avg_accuracy_edge: float | None
    avg_walk_forward_edge: float | None
    avg_hit_rate: float | None
    avg_strategy_return: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_prediction_result_files(base_dir: str | Path) -> list[Path]:
    root = Path(base_dir)
    if not root.exists():
        return []
    files = [path for path in root.rglob("predictions.csv") if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def build_setting_profiles(base_dir: str | Path, *, min_records: int = 2) -> list[SettingProfile]:
    frames: list[pd.DataFrame] = []
    for path in list_prediction_result_files(base_dir):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "status" not in frame.columns or "ticker" not in frame.columns:
            continue
        frame = prepare_prediction_frame(frame[frame["status"] == "success"].copy())
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return []

    all_rows = pd.concat(frames, ignore_index=True)
    profiles: list[SettingProfile] = []
    for ticker, group in all_rows.groupby("ticker", dropna=True):
        run_count = int(len(group))
        if run_count < min_records:
            continue

        preferred_threshold = _weighted_median(group, "recommended_threshold", fallback_column="threshold")
        preferred_model_name = _weighted_mode_str(group, "model_name")
        preferred_model_label = _weighted_mode_str(group, "model_label")
        should_compare_tree = preferred_model_name == "tree"
        avg_accuracy_edge = weighted_mean_or_none(group, "accuracy_edge")
        avg_walk_forward_edge = weighted_mean_or_none(group, "walk_forward_edge")
        avg_hit_rate = weighted_mean_or_none(group, "backtest_hit_rate")
        avg_strategy_return = weighted_mean_or_none(group, "backtest_cumulative_strategy_return")
        latest_run_at = _latest_run_at(group)
        recency_weight_sum = float(group["recency_weight"].sum()) if "recency_weight" in group.columns else float(run_count)

        profiles.append(
            SettingProfile(
                ticker=str(ticker),
                run_count=run_count,
                recency_weight_sum=recency_weight_sum,
                latest_run_at=latest_run_at,
                preferred_threshold=preferred_threshold,
                preferred_model_name=preferred_model_name,
                preferred_model_label=preferred_model_label,
                should_compare_tree=should_compare_tree,
                avg_accuracy_edge=avg_accuracy_edge,
                avg_walk_forward_edge=avg_walk_forward_edge,
                avg_hit_rate=avg_hit_rate,
                avg_strategy_return=avg_strategy_return,
            )
        )

    profiles.sort(
        key=lambda profile: (
            profile.ticker,
        )
    )
    return profiles


def get_setting_profile(base_dir: str | Path, ticker: str, *, min_records: int = 2) -> SettingProfile | None:
    normalized = str(ticker).strip().upper()
    for profile in build_setting_profiles(base_dir, min_records=min_records):
        if profile.ticker.strip().upper() == normalized:
            return profile
    return None


def prepare_prediction_frame(frame: pd.DataFrame, *, half_life_days: float = 30.0) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["threshold"] = pd.to_numeric(prepared.get("threshold"), errors="coerce")
    prepared["recommended_threshold"] = pd.to_numeric(prepared.get("recommended_threshold"), errors="coerce")
    prepared["accuracy_edge"] = pd.to_numeric(prepared.get("accuracy_edge"), errors="coerce")
    if "accuracy" in prepared.columns and "baseline_accuracy" in prepared.columns:
        accuracy = pd.to_numeric(prepared["accuracy"], errors="coerce")
        baseline = pd.to_numeric(prepared["baseline_accuracy"], errors="coerce")
        prepared["accuracy_edge"] = prepared["accuracy_edge"].fillna(accuracy - baseline)
    prepared["walk_forward_edge"] = pd.to_numeric(prepared.get("walk_forward_edge"), errors="coerce")
    prepared["backtest_hit_rate"] = pd.to_numeric(prepared.get("backtest_hit_rate"), errors="coerce")
    prepared["backtest_cumulative_strategy_return"] = pd.to_numeric(
        prepared.get("backtest_cumulative_strategy_return"),
        errors="coerce",
    )
    prepared["probability_up"] = pd.to_numeric(prepared.get("probability_up"), errors="coerce")
    prepared["run_at_dt"] = pd.to_datetime(prepared.get("run_at"), errors="coerce")
    latest = prepared["run_at_dt"].dropna().max()
    if pd.isna(latest):
        prepared["recency_weight"] = 1.0
        return prepared
    age_days = (latest - prepared["run_at_dt"]).dt.total_seconds().div(86400.0)
    age_days = age_days.fillna(age_days.max() if age_days.notna().any() else 0.0).clip(lower=0.0)
    decay = np.exp(-np.log(2.0) * age_days / max(half_life_days, 1.0))
    prepared["recency_weight"] = decay.astype(float)
    return prepared


def weighted_mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame.get("recency_weight"), errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return None
    return float(np.average(values[mask], weights=weights[mask]))


def _weighted_mode_str(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = frame[column].astype(str).str.strip()
    weights = pd.to_numeric(frame.get("recency_weight"), errors="coerce").fillna(1.0)
    score_by_value: dict[str, float] = {}
    for value, weight in zip(values.tolist(), weights.tolist()):
        if not value or value.lower() == "nan":
            continue
        score_by_value[value] = score_by_value.get(value, 0.0) + float(weight)
    if not score_by_value:
        return None
    return max(score_by_value.items(), key=lambda item: item[1])[0]


def _weighted_median(frame: pd.DataFrame, column: str, *, fallback_column: str | None = None) -> float | None:
    series = pd.to_numeric(frame.get(column), errors="coerce")
    if series is None or series.dropna().empty:
        if fallback_column is None:
            return None
        series = pd.to_numeric(frame.get(fallback_column), errors="coerce")
    if series is None:
        return None

    weights = pd.to_numeric(frame.get("recency_weight"), errors="coerce").fillna(1.0)
    pairs = pd.DataFrame({"value": series, "weight": weights}).dropna()
    if pairs.empty:
        return None
    pairs = pairs.sort_values("value").reset_index(drop=True)
    cumulative = pairs["weight"].cumsum()
    cutoff = pairs["weight"].sum() / 2
    idx = int((cumulative >= cutoff).idxmax())
    return float(pairs.loc[idx, "value"])


def _latest_run_at(frame: pd.DataFrame) -> str | None:
    if "run_at_dt" not in frame.columns:
        return None
    latest = frame["run_at_dt"].dropna().max()
    if pd.isna(latest):
        return None
    if isinstance(latest, pd.Timestamp):
        return latest.isoformat()
    if isinstance(latest, datetime):
        return latest.isoformat()
    return str(latest)
