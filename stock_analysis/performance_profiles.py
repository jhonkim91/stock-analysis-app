from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SettingProfile:
    ticker: str
    run_count: int
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
        frame = frame[frame["status"] == "success"].copy()
        if frame.empty:
            continue
        frame["threshold"] = pd.to_numeric(frame.get("threshold"), errors="coerce")
        frame["recommended_threshold"] = pd.to_numeric(frame.get("recommended_threshold"), errors="coerce")
        frame["accuracy_edge"] = pd.to_numeric(frame.get("accuracy_edge"), errors="coerce")
        frame["walk_forward_edge"] = pd.to_numeric(frame.get("walk_forward_edge"), errors="coerce")
        frame["backtest_hit_rate"] = pd.to_numeric(frame.get("backtest_hit_rate"), errors="coerce")
        frame["backtest_cumulative_strategy_return"] = pd.to_numeric(
            frame.get("backtest_cumulative_strategy_return"),
            errors="coerce",
        )
        frames.append(frame)

    if not frames:
        return []

    all_rows = pd.concat(frames, ignore_index=True)
    profiles: list[SettingProfile] = []
    for ticker, group in all_rows.groupby("ticker", dropna=True):
        run_count = int(len(group))
        if run_count < min_records:
            continue

        preferred_threshold = _preferred_threshold(group)
        preferred_model_name = _mode_str(group.get("model_name"))
        preferred_model_label = _mode_str(group.get("model_label"))
        should_compare_tree = preferred_model_name == "tree"
        avg_accuracy_edge = _mean_or_none(group.get("accuracy_edge"))
        avg_walk_forward_edge = _mean_or_none(group.get("walk_forward_edge"))
        avg_hit_rate = _mean_or_none(group.get("backtest_hit_rate"))
        avg_strategy_return = _mean_or_none(group.get("backtest_cumulative_strategy_return"))

        profiles.append(
            SettingProfile(
                ticker=str(ticker),
                run_count=run_count,
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


def _preferred_threshold(frame: pd.DataFrame) -> float | None:
    candidates = frame["recommended_threshold"].dropna()
    if candidates.empty:
        candidates = frame["threshold"].dropna()
    if candidates.empty:
        return None
    return float(candidates.median())


def _mode_str(series: pd.Series | None) -> str | None:
    if series is None:
        return None
    values = [str(value).strip() for value in series.dropna().tolist() if str(value).strip()]
    if not values:
        return None
    counts = pd.Series(values).value_counts()
    return str(counts.index[0])


def _mean_or_none(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())
