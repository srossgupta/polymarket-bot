"""Analytics module: P&L analysis, parameter sensitivity, category ranking."""

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
        return {"status": "no_trades", "message": "No closed trades yet"}

    total_pnl = sum(t.get("pnl", 0) for t in trades)
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    win_pnls = [t["pnl"] for t in wins]
    loss_pnls = [t["pnl"] for t in losses]

    max_consecutive_wins = max_consecutive_losses = current_streak = 0
    last_was_win = None
    for t in sorted(trades, key=lambda x: x.get("ts", "")):
        is_win = t.get("pnl", 0) > 0
        current_streak = current_streak + 1 if is_win == last_was_win else 1
        last_was_win = is_win
        if is_win:
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        else:
            max_consecutive_losses = max(max_consecutive_losses, current_streak)

    durations = [t.get("hold_duration_seconds", 0) for t in trades if t.get("hold_duration_seconds")]
    avg_hold = sum(durations) / len(durations) if durations else 0

    by_reason: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    for t in trades:
        by_reason[t.get("event_type", "unknown")]["trades"] += 1
        by_reason[t.get("event_type", "unknown")]["pnl"] += t.get("pnl", 0)

    return {
        "total_trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / len(trades), 2) if trades else 0,
        "avg_win": round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0,
        "avg_loss": round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0,
        "largest_win": round(max(win_pnls), 2) if win_pnls else 0,
        "largest_loss": round(min(loss_pnls), 2) if loss_pnls else 0,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "avg_hold_seconds": round(avg_hold, 1),
        "profit_factor": round(abs(sum(win_pnls) / sum(loss_pnls)), 4
                               ) if loss_pnls and sum(loss_pnls) != 0 else float("inf"),
        "by_exit_reason": dict(by_reason),
    }


def category_ranking() -> list[dict]:
    perf = get_category_performance()
    for row in perf:
        n, w, pnl = row.get("trades", 0), row.get("wins", 0), row.get("total_pnl", 0)
        row["win_rate"] = round(w / n, 4) if n else 0
        row["expectancy"] = round(pnl / n, 4) if n else 0
        row["score"] = round(row["expectancy"] * min(1.0, n / 10) + row["win_rate"] * 0.1, 4)
    perf.sort(key=lambda x: x.get("score", 0), reverse=True)
    return perf


def parameter_sensitivity_report() -> dict[str, Any]:
    by_entry = get_pnl_by_parameter_set()
    hourly = get_hourly_pnl()
    rec = ""
    if by_entry:
        best = max(by_entry, key=lambda x: x.get("avg_pnl", 0))
        worst = min(by_entry, key=lambda x: x.get("avg_pnl", 0))
        rec = (f"Best: {best.get('entry_cents')}c (avg ${best.get('avg_pnl', 0):.2f}) | "
               f"Worst: {worst.get('entry_cents')}c (avg ${worst.get('avg_pnl', 0):.2f})")
    return {"by_entry_price": by_entry, "by_hour_utc": hourly, "recommendation": rec}


def what_if_analysis(
    new_entry_cents: float | None = None,
    new_stop_cents: float | None = None,
    new_wake_minutes: int | None = None,
) -> dict[str, Any]:
    from polymarket_bot.backtest import run_snapshot_backtest
    from polymarket_bot.core import BotConfig
    snapshots = load_snapshots()
    if not snapshots:
        return {"error": "No snapshots available for what-if analysis"}
    cfg = BotConfig()
    if new_entry_cents is not None:
        cfg.strategy.entry_threshold_cents = new_entry_cents
    if new_stop_cents is not None:
        cfg.strategy.stop_loss_cents = new_stop_cents
    if new_wake_minutes is not None:
        cfg.strategy.wake_minutes_before_close = new_wake_minutes
    result = run_snapshot_backtest(cfg, snapshots)
    return {
        "parameters": {"entry_cents": cfg.strategy.entry_threshold_cents,
                        "stop_cents": cfg.strategy.stop_loss_cents,
                        "wake_minutes": cfg.strategy.wake_minutes_before_close},
        "result": result.to_dict(),
    }


def equity_curve() -> list[dict]:
    history = load_performance_history()
    return [{"ts": h["ts"], "total_value": h["total_value"], "pnl": h["total_pnl"]}
            for h in reversed(history)]
