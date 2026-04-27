"""Analytics: P&L reports, category ranking, parameter sensitivity."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from polymarket_bot.data.storage import (
    get_category_performance, get_hourly_pnl, get_pnl_by_parameter_set,
    load_closed_trades, load_performance_history, load_snapshots,
)


def full_pnl_report() -> dict[str, Any]:
    trades = load_closed_trades()
    if not trades:
        return {"status": "no_trades"}

    total_pnl = sum(t.get("pnl", 0) for t in trades)
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    breakevens = [t for t in trades if t.get("pnl", 0) == 0]
    decisive = len(wins) + len(losses)

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakevens": len(breakevens),
        "win_rate": round(len(wins) / decisive, 4) if decisive else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
        "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0,
        "largest_win": round(max(t["pnl"] for t in wins), 2) if wins else 0,
        "largest_loss": round(min(t["pnl"] for t in losses), 2) if losses else 0,
    }


def category_ranking() -> list[dict]:
    perf = get_category_performance()
    for row in perf:
        n = row.get("trades", 0)
        row["win_rate"] = round(row.get("wins", 0) / n, 4) if n else 0
        row["expectancy"] = round(row.get("total_pnl", 0) / n, 4) if n else 0
    perf.sort(key=lambda x: x.get("expectancy", 0), reverse=True)
    return perf


def parameter_sensitivity_report() -> dict[str, Any]:
    return {"by_entry_price": get_pnl_by_parameter_set(), "by_hour_utc": get_hourly_pnl()}


def what_if_analysis(
    new_entry_cents: float | None = None,
    new_stop_cents: float | None = None,
    new_wake_minutes: int | None = None,
) -> dict[str, Any]:
    from polymarket_bot.backtest import run_snapshot_backtest
    from polymarket_bot.core import BotConfig
    snapshots = load_snapshots()
    if not snapshots:
        return {"error": "no snapshots"}
    cfg = BotConfig()
    if new_entry_cents is not None:
        cfg.strategy.entry_threshold_cents = new_entry_cents
    if new_wake_minutes is not None:
        cfg.strategy.wake_minutes_before_close = new_wake_minutes
    result = run_snapshot_backtest(cfg, snapshots)
    return {
        "params": {"entry_cents": cfg.strategy.entry_threshold_cents,
                    "wake_minutes": cfg.strategy.wake_minutes_before_close},
        "result": result.to_dict(),
    }


def equity_curve() -> list[dict]:
    history = load_performance_history()
    return [{"ts": h["ts"], "total_value": h["total_value"], "pnl": h["total_pnl"]}
            for h in reversed(history)]
