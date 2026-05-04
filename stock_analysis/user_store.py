from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "app_state.sqlite3"


@dataclass(frozen=True)
class SavedWatchlistItem:
    user_id: str
    ticker: str
    exchange: str
    name: str
    symbol: str
    source: str
    enabled: bool
    created_at: str
    updated_at: str


def normalize_user_id(value: str) -> str:
    normalized = re.sub(r"\s+", "-", value.strip())
    normalized = re.sub(r"[^\w.\-]+", "", normalized, flags=re.UNICODE)
    normalized = normalized.strip("._-").lower()
    return normalized[:40] or "guest"


def ensure_user(user_id: str) -> str:
    normalized = normalize_user_id(user_id)
    now = _timestamp()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (normalized, now, now),
        )
        conn.commit()
    return normalized


def list_user_ids(limit: int = 20) -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT user_id
            FROM users
            ORDER BY updated_at DESC, user_id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [str(row["user_id"]) for row in rows]


def save_watchlist_item(
    user_id: str,
    *,
    ticker: str,
    exchange: str = "",
    name: str = "",
    symbol: str,
    source: str = "",
    enabled: bool = True,
) -> None:
    normalized = ensure_user(user_id)
    now = _timestamp()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO watchlist_items (
                user_id,
                ticker,
                exchange,
                name,
                symbol,
                source,
                enabled,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, symbol) DO UPDATE SET
                ticker = excluded.ticker,
                exchange = excluded.exchange,
                name = excluded.name,
                source = excluded.source,
                enabled = excluded.enabled,
                updated_at = excluded.updated_at
            """,
            (
                normalized,
                ticker.strip(),
                exchange.strip().upper(),
                name.strip(),
                symbol.strip().upper(),
                source.strip(),
                int(enabled),
                now,
                now,
            ),
        )
        conn.execute("UPDATE users SET updated_at = ? WHERE user_id = ?", (now, normalized))
        conn.commit()


def list_watchlist_items(user_id: str) -> list[SavedWatchlistItem]:
    normalized = ensure_user(user_id)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                user_id,
                ticker,
                exchange,
                name,
                symbol,
                source,
                enabled,
                created_at,
                updated_at
            FROM watchlist_items
            WHERE user_id = ?
            ORDER BY created_at ASC, name ASC, symbol ASC
            """,
            (normalized,),
        ).fetchall()
    return [
        SavedWatchlistItem(
            user_id=str(row["user_id"]),
            ticker=str(row["ticker"]),
            exchange=str(row["exchange"]),
            name=str(row["name"]),
            symbol=str(row["symbol"]),
            source=str(row["source"]),
            enabled=bool(row["enabled"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    ]


def remove_watchlist_item(user_id: str, symbol: str) -> None:
    normalized = ensure_user(user_id)
    with _connect() as conn:
        conn.execute(
            "DELETE FROM watchlist_items WHERE user_id = ? AND symbol = ?",
            (normalized, symbol.strip().upper()),
        )
        conn.execute("UPDATE users SET updated_at = ? WHERE user_id = ?", (_timestamp(), normalized))
        conn.commit()


def export_watchlist_csv(user_id: str) -> Path:
    normalized = ensure_user(user_id)
    items = [item for item in list_watchlist_items(normalized) if item.enabled]
    if not items:
        raise ValueError("등록된 관심종목이 없습니다.")

    path = user_data_dir(normalized) / "watchlist.csv"
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


def user_data_dir(user_id: str) -> Path:
    return DATA_DIR / "users" / normalize_user_id(user_id)


def user_outputs_dir(user_id: str) -> Path:
    return ROOT / "outputs" / "users" / normalize_user_id(user_id)


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _initialize_db(conn)
    return conn


def _initialize_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS watchlist_items (
            user_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            exchange TEXT NOT NULL DEFAULT '',
            name TEXT NOT NULL DEFAULT '',
            symbol TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, symbol),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_watchlist_items_user_updated
        ON watchlist_items (user_id, updated_at DESC);
        """
    )


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
