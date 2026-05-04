from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from stock_analysis.user_store import SavedWatchlistItem, user_data_dir


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str
    access_token: str
    refresh_token: str


@dataclass(frozen=True)
class AuthResult:
    user: AuthenticatedUser | None
    requires_email_confirmation: bool
    message: str


@dataclass(frozen=True)
class AnalysisSnapshotSummary:
    snapshot_id: str
    run_type: str
    title: str
    row_count: int
    created_at: str
    summary: dict[str, Any]


@dataclass(frozen=True)
class AnalysisSnapshot(AnalysisSnapshotSummary):
    rows: list[dict[str, Any]]


def load_supabase_config(secret_source: Mapping[str, Any] | None = None) -> SupabaseConfig | None:
    source = secret_source or {}
    supabase_section = source.get("supabase", {}) if hasattr(source, "get") else {}

    url = _clean_value(_mapping_get(supabase_section, "url")) or _clean_value(os.getenv("SUPABASE_URL"))
    key = _clean_value(_mapping_get(supabase_section, "key")) or _clean_value(os.getenv("SUPABASE_KEY"))
    if not url or not key:
        return None
    return SupabaseConfig(url=url, key=key)


def sign_up_with_password(config: SupabaseConfig, email: str, password: str) -> AuthResult:
    client = _create_client(config)
    response = client.auth.sign_up({"email": email.strip(), "password": password})
    session = getattr(response, "session", None)
    if session is None:
        return AuthResult(
            user=None,
            requires_email_confirmation=True,
            message="가입 요청을 보냈습니다. 이메일 인증 후 로그인하세요.",
        )
    return AuthResult(
        user=_session_to_user(session),
        requires_email_confirmation=False,
        message="회원가입과 로그인에 성공했습니다.",
    )


def sign_in_with_password(config: SupabaseConfig, email: str, password: str) -> AuthenticatedUser:
    client = _create_client(config)
    response = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
    session = getattr(response, "session", None)
    if session is None:
        raise RuntimeError("로그인 세션을 만들지 못했습니다.")
    return _session_to_user(session)


def restore_session(config: SupabaseConfig, access_token: str, refresh_token: str) -> AuthenticatedUser:
    client = _create_client(config)
    response = client.auth.set_session(access_token, refresh_token)
    session = getattr(response, "session", None) or client.auth.get_session()
    if session is None:
        raise RuntimeError("세션을 복구하지 못했습니다.")
    return _session_to_user(session)


def sign_out(config: SupabaseConfig, access_token: str, refresh_token: str) -> None:
    client = _authorized_client(config, access_token, refresh_token)
    client.auth.sign_out()


def list_cloud_watchlist_items(
    config: SupabaseConfig,
    user: AuthenticatedUser,
) -> list[SavedWatchlistItem]:
    client = _authorized_client(config, user.access_token, user.refresh_token)
    response = (
        client.table("watchlist_items")
        .select("ticker, exchange, name, symbol, source, enabled, created_at, updated_at")
        .eq("user_id", user.user_id)
        .order("created_at")
        .execute()
    )
    rows = getattr(response, "data", []) or []
    return [
        SavedWatchlistItem(
            user_id=user.user_id,
            ticker=str(row.get("ticker") or ""),
            exchange=str(row.get("exchange") or ""),
            name=str(row.get("name") or ""),
            symbol=str(row.get("symbol") or ""),
            source=str(row.get("source") or ""),
            enabled=bool(row.get("enabled", True)),
            created_at=str(row.get("created_at") or ""),
            updated_at=str(row.get("updated_at") or ""),
        )
        for row in rows
    ]


def save_cloud_watchlist_item(
    config: SupabaseConfig,
    user: AuthenticatedUser,
    *,
    ticker: str,
    exchange: str = "",
    name: str = "",
    symbol: str,
    source: str = "",
    enabled: bool = True,
) -> None:
    client = _authorized_client(config, user.access_token, user.refresh_token)
    payload = {
        "user_id": user.user_id,
        "ticker": ticker.strip(),
        "exchange": exchange.strip().upper(),
        "name": name.strip(),
        "symbol": symbol.strip().upper(),
        "source": source.strip(),
        "enabled": bool(enabled),
    }
    (
        client.table("watchlist_items")
        .upsert(payload, on_conflict="user_id,symbol")
        .execute()
    )


def remove_cloud_watchlist_item(config: SupabaseConfig, user: AuthenticatedUser, symbol: str) -> None:
    client = _authorized_client(config, user.access_token, user.refresh_token)
    (
        client.table("watchlist_items")
        .delete()
        .eq("user_id", user.user_id)
        .eq("symbol", symbol.strip().upper())
        .execute()
    )


def export_cloud_watchlist_csv(config: SupabaseConfig, user: AuthenticatedUser) -> Path:
    items = [item for item in list_cloud_watchlist_items(config, user) if item.enabled]
    if not items:
        raise ValueError("등록된 관심종목이 없습니다.")

    path = user_data_dir(user.user_id) / "watchlist.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["ticker", "exchange", "name", "enabled"])
        writer.writeheader()
        writer.writerows(
            {
                "ticker": item.ticker,
                "exchange": item.exchange,
                "name": item.name,
                "enabled": 1 if item.enabled else 0,
            }
            for item in items
        )
    return path


def save_analysis_snapshot(
    config: SupabaseConfig,
    user: AuthenticatedUser,
    *,
    run_type: str,
    title: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> str:
    client = _authorized_client(config, user.access_token, user.refresh_token)
    payload = {
        "user_id": user.user_id,
        "run_type": run_type,
        "title": title,
        "row_count": len(rows),
        "summary": summary or {},
        "rows": rows,
    }
    response = client.table("analysis_snapshots").insert(payload).execute()
    data = getattr(response, "data", []) or []
    if data:
        return str(data[0].get("id") or "")
    return ""


def list_analysis_snapshots(
    config: SupabaseConfig,
    user: AuthenticatedUser,
    *,
    limit: int = 30,
) -> list[AnalysisSnapshotSummary]:
    client = _authorized_client(config, user.access_token, user.refresh_token)
    response = (
        client.table("analysis_snapshots")
        .select("id, run_type, title, row_count, created_at, summary")
        .eq("user_id", user.user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = getattr(response, "data", []) or []
    return [
        AnalysisSnapshotSummary(
            snapshot_id=str(row.get("id") or ""),
            run_type=str(row.get("run_type") or ""),
            title=str(row.get("title") or ""),
            row_count=int(row.get("row_count") or 0),
            created_at=str(row.get("created_at") or ""),
            summary=dict(row.get("summary") or {}),
        )
        for row in rows
    ]


def get_analysis_snapshot(
    config: SupabaseConfig,
    user: AuthenticatedUser,
    snapshot_id: str,
) -> AnalysisSnapshot | None:
    client = _authorized_client(config, user.access_token, user.refresh_token)
    response = (
        client.table("analysis_snapshots")
        .select("id, run_type, title, row_count, created_at, summary, rows")
        .eq("user_id", user.user_id)
        .eq("id", snapshot_id)
        .limit(1)
        .execute()
    )
    rows = getattr(response, "data", []) or []
    if not rows:
        return None
    row = rows[0]
    return AnalysisSnapshot(
        snapshot_id=str(row.get("id") or ""),
        run_type=str(row.get("run_type") or ""),
        title=str(row.get("title") or ""),
        row_count=int(row.get("row_count") or 0),
        created_at=str(row.get("created_at") or ""),
        summary=dict(row.get("summary") or {}),
        rows=[dict(item) for item in (row.get("rows") or [])],
    )


def _authorized_client(config: SupabaseConfig, access_token: str, refresh_token: str):
    client = _create_client(config)
    client.auth.set_session(access_token, refresh_token)
    return client


def _create_client(config: SupabaseConfig):
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "supabase package가 설치되어 있지 않습니다. "
            "python -m pip install -r requirements.txt 를 먼저 실행하세요."
        ) from exc

    return create_client(config.url, config.key)


def _session_to_user(session: Any) -> AuthenticatedUser:
    user = getattr(session, "user", None)
    user_id = _clean_value(getattr(user, "id", "")) or ""
    email = _clean_value(getattr(user, "email", "")) or ""
    access_token = _clean_value(getattr(session, "access_token", "")) or ""
    refresh_token = _clean_value(getattr(session, "refresh_token", "")) or ""
    if not user_id or not access_token or not refresh_token:
        raise RuntimeError("Supabase 세션 응답이 불완전합니다.")
    return AuthenticatedUser(
        user_id=user_id,
        email=email,
        access_token=access_token,
        refresh_token=refresh_token,
    )


def _mapping_get(mapping: Any, key: str) -> Any:
    if mapping is None:
        return None
    if hasattr(mapping, "get"):
        return mapping.get(key)
    try:
        return mapping[key]
    except Exception:
        return None


def _clean_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
