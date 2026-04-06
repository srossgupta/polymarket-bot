"""SQLite-backed storage for trades, snapshots, watchlists, and metrics."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Generator

from .config import DB_PATH
from .models import Market, PerformanceSnapshot, PricePoint, TradeEvent

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    question TEXT,
    category TEXT,
    side TEXT,
    event_type TEXT,
    ts TEXT NOT NULL,
    price REAL,
    size_dollars REAL,
    shares REAL,
    pnl REAL DEFAULT 0,
    reason TEXT,
    entry_price REAL DEFAULT 0,
    hold_duration_seconds REAL DEFAULT 0,
    peak_price REAL DEFAULT 0,
    volume_at_entry REAL DEFAULT 0,
    velocity_at_entry REAL DEFAULT 0,
    volatility_at_entry REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    question TEXT,
    category TEXT,
    end_time TEXT,
    ts TEXT NOT NULL,
    yes_price REAL,
    no_price REAL,
    spread REAL DEFAULT 0,
    volume REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_ts TEXT NOT NULL,
    market_id TEXT NOT NULL,
    question TEXT,
    end_time TEXT,
    volume_usd REAL,
    category TEXT,
    yes_token_id TEXT,
    no_token_id TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    cash REAL,
    positions_value REAL,
    total_value REAL,
    open_positions INTEGER,
    total_trades INTEGER,
    wins INTEGER,
    losses INTEGER,
    total_pnl REAL,
    win_rate REAL,
    avg_win REAL,
    avg_loss REAL,
    sharpe_estimate REAL,
    expectancy REAL,
    best_category TEXT,
    worst_category TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
CREATE INDEX IF NOT EXISTS idx_trades_category ON trades(category);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);
CREATE INDEX IF NOT EXISTS idx_snapshots_market ON snapshots(market_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(ts);
"""


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript(SCHEMA)
    return _local.conn


@contextmanager
def _cursor() -> Generator[sqlite3.Cursor, None, None]:
    conn = _get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    _get_conn()


# --- Trade storage ---

def append_trade(event: TradeEvent) -> None:
    with _cursor() as cur:
        cur.execute(
            """INSERT INTO trades (market_id, question, category, side, event_type, ts,
               price, size_dollars, shares, pnl, reason, entry_price,
               hold_duration_seconds, peak_price, volume_at_entry,
               velocity_at_entry, volatility_at_entry)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (event.market_id, event.question, event.category, event.side,
             event.event_type, event.ts.isoformat(), event.price, event.size_dollars,
             event.shares, event.pnl, event.reason, event.entry_price,
             event.hold_duration_seconds, event.peak_price, event.volume_at_entry,
             event.velocity_at_entry, event.volatility_at_entry),
        )


def load_trade_events(limit: int = 5000) -> list[dict[str, Any]]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM trades ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]


def load_closed_trades(limit: int = 5000) -> list[dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM trades WHERE event_type IN ('SELL_MARKET','STOP_LOSS','FORCED_CLOSE') "
            "ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]


def load_trades_by_category() -> dict[str, list[dict]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM trades WHERE event_type IN ('SELL_MARKET','STOP_LOSS','FORCED_CLOSE') "
            "ORDER BY category, ts")
        rows = [dict(r) for r in cur.fetchall()]
    result: dict[str, list[dict]] = {}
    for r in rows:
        result.setdefault(r["category"], []).append(r)
    return result


# --- Snapshot storage ---

def append_snapshot(market: Market, point: PricePoint) -> None:
    with _cursor() as cur:
        cur.execute(
            """INSERT INTO snapshots (market_id, question, category, end_time, ts,
               yes_price, no_price, spread, volume)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (market.market_id, market.question, market.category,
             market.end_time.isoformat(), point.ts.isoformat(),
             point.yes, point.no, point.spread, point.volume_at_snapshot),
        )


def load_snapshots(market_id: str | None = None, limit: int = 50000) -> list[dict]:
    with _cursor() as cur:
        if market_id:
            cur.execute("SELECT * FROM snapshots WHERE market_id=? ORDER BY ts LIMIT ?",
                        (market_id, limit))
        else:
            cur.execute("SELECT * FROM snapshots ORDER BY ts LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


# --- Watchlist ---

def save_watchlist(markets: list[Market], scan_ts: datetime | None = None) -> None:
    ts = (scan_ts or datetime.now(timezone.utc)).isoformat()
    with _cursor() as cur:
        for m in markets:
            cur.execute(
                """INSERT INTO watchlist (scan_ts, market_id, question, end_time,
                   volume_usd, category, yes_token_id, no_token_id)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (ts, m.market_id, m.question, m.end_time.isoformat(),
                 m.volume_usd, m.category, m.yes_token_id, m.no_token_id),
            )


# --- Metrics ---

def append_metrics(metrics: dict) -> None:
    with _cursor() as cur:
        cur.execute("INSERT INTO metrics (ts, payload) VALUES (?, ?)",
                    (datetime.now(timezone.utc).isoformat(), json.dumps(metrics)))


def load_metrics(limit: int = 500) -> list[dict]:
    with _cursor() as cur:
        cur.execute("SELECT ts, payload FROM metrics ORDER BY ts DESC LIMIT ?", (limit,))
        return [{"ts": r["ts"], **json.loads(r["payload"])} for r in cur.fetchall()]


# --- Performance snapshots ---

def save_performance(snap: PerformanceSnapshot) -> None:
    with _cursor() as cur:
        cur.execute(
            """INSERT INTO performance (ts, cash, positions_value, total_value,
               open_positions, total_trades, wins, losses, total_pnl,
               win_rate, avg_win, avg_loss, sharpe_estimate, expectancy,
               best_category, worst_category)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (snap.ts.isoformat(), snap.cash, snap.positions_value, snap.total_value,
             snap.open_positions, snap.total_trades, snap.wins, snap.losses,
             snap.total_pnl, snap.win_rate, snap.avg_win, snap.avg_loss,
             snap.sharpe_estimate, snap.expectancy, snap.best_category, snap.worst_category),
        )


def load_performance_history(limit: int = 200) -> list[dict]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM performance ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


# --- Analytics queries ---

def get_pnl_by_parameter_set() -> list[dict]:
    """Group trades by the entry_price bucket to analyze parameter sensitivity."""
    with _cursor() as cur:
        cur.execute("""
            SELECT
                ROUND(entry_price * 100) as entry_cents,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(pnl), 2) as total_pnl,
                ROUND(AVG(pnl), 2) as avg_pnl,
                ROUND(AVG(hold_duration_seconds), 1) as avg_hold_secs
            FROM trades
            WHERE event_type IN ('SELL_MARKET','STOP_LOSS','FORCED_CLOSE')
            GROUP BY ROUND(entry_price * 100)
            ORDER BY entry_cents
        """)
        return [dict(r) for r in cur.fetchall()]


def get_category_performance() -> list[dict]:
    with _cursor() as cur:
        cur.execute("""
            SELECT
                category,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(pnl), 2) as total_pnl,
                ROUND(AVG(pnl), 2) as avg_pnl,
                ROUND(AVG(CASE WHEN pnl > 0 THEN pnl END), 2) as avg_win,
                ROUND(AVG(CASE WHEN pnl <= 0 THEN pnl END), 2) as avg_loss
            FROM trades
            WHERE event_type IN ('SELL_MARKET','STOP_LOSS','FORCED_CLOSE')
            GROUP BY category
            ORDER BY total_pnl DESC
        """)
        return [dict(r) for r in cur.fetchall()]


def get_hourly_pnl() -> list[dict]:
    """P&L bucketed by hour-of-day to find best trading windows."""
    with _cursor() as cur:
        cur.execute("""
            SELECT
                CAST(SUBSTR(ts, 12, 2) AS INTEGER) as hour_utc,
                COUNT(*) as trades,
                ROUND(SUM(pnl), 2) as total_pnl,
                ROUND(AVG(pnl), 2) as avg_pnl
            FROM trades
            WHERE event_type IN ('SELL_MARKET','STOP_LOSS','FORCED_CLOSE')
            GROUP BY hour_utc
            ORDER BY hour_utc
        """)
        return [dict(r) for r in cur.fetchall()]
