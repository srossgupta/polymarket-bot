"""Backtesting: replay recorded snapshots, parameter sweep, Monte Carlo.

Uses the same logic as live trading:
  - Enter if price is in the entry band (95¢–99.5¢)
  - Hold to expiry (last snapshot = exit price)
  - No stop loss
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from polymarket_bot.core import BotConfig, Market, PricePoint, Side
from polymarket_bot.trading.strategy import pick_entry_side


@dataclass
class BacktestTrade:
    market_id: str
    category: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    reason: str


@dataclass
class BacktestResult:
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float
    sharpe: float
    trade_log: list[BacktestTrade]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades, "wins": self.wins, "losses": self.losses,
            "win_rate": round(self.win_rate, 4), "total_pnl": round(self.total_pnl, 2),
            "avg_pnl": round(self.avg_pnl, 2), "max_drawdown": round(self.max_drawdown, 2),
            "sharpe": round(self.sharpe, 4),
        }


def run_snapshot_backtest(cfg: BotConfig, snapshots: list[dict]) -> BacktestResult:
    """Replay snapshots: enter on band signal, hold to last snapshot."""
    by_market: dict[str, list[dict]] = defaultdict(list)
    for row in snapshots:
        by_market[row["market_id"]].append(row)

    trade_log: list[BacktestTrade] = []
    equity = [cfg.starting_cash]

    for market_id, rows in by_market.items():
        rows.sort(key=lambda r: r.get("ts", ""))
        category = rows[0].get("category", "unknown")
        entered = None

        for row in rows:
            yes = float(row.get("yes_price", row.get("yes", 0)))
            no = float(row.get("no_price", row.get("no", 0)))
            ts_str = row.get("ts", "")
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
            point = PricePoint(ts=ts, yes=yes, no=no)

            if entered is None:
                entry = pick_entry_side(point, cfg)
                if entry:
                    entered = {"side": entry[0], "price": entry[1]}
                continue

        # Close at last snapshot
        if entered:
            last = rows[-1]
            side = entered["side"]
            exit_price = float(last.get("yes_price", last.get("yes", 0))) if side == Side.YES \
                else float(last.get("no_price", last.get("no", 0)))
            size = cfg.strategy.max_dollars_per_market
            shares = size / entered["price"]
            pnl = shares * exit_price - size

            trade_log.append(BacktestTrade(
                market_id=market_id, category=category, side=side.value,
                entry_price=entered["price"], exit_price=exit_price,
                size=size, pnl=pnl, reason="market_expired"))
            equity.append(equity[-1] + pnl)

    # Stats
    wins = sum(1 for t in trade_log if t.pnl > 0)
    total_pnl = sum(t.pnl for t in trade_log)
    avg_pnl = total_pnl / len(trade_log) if trade_log else 0

    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        peak = max(peak, val)
        max_dd = max(max_dd, peak - val)

    pnls = [t.pnl for t in trade_log]
    sharpe = 0.0
    if len(pnls) >= 2:
        mean = sum(pnls) / len(pnls)
        std = (sum((p - mean) ** 2 for p in pnls) / len(pnls)) ** 0.5
        sharpe = mean / std if std > 1e-8 else 0.0

    return BacktestResult(
        trades=len(trade_log), wins=wins, losses=len(trade_log) - wins,
        win_rate=wins / len(trade_log) if trade_log else 0,
        total_pnl=total_pnl, avg_pnl=avg_pnl, max_drawdown=max_dd,
        sharpe=sharpe, trade_log=trade_log)


def parameter_sweep(
    snapshots: list[dict],
    entry_range: tuple[float, float, float] = (90, 97, 1),
    wake_range: tuple[int, int, int] = (3, 10, 2),
) -> list[dict]:
    """Try different entry thresholds and wake times, rank by PnL."""
    results: list[dict] = []
    entry = entry_range[0]
    while entry <= entry_range[1]:
        wake = wake_range[0]
        while wake <= wake_range[1]:
            cfg = BotConfig()
            cfg.strategy.entry_threshold_cents = entry
            cfg.strategy.wake_minutes_before_close = wake
            result = run_snapshot_backtest(cfg, snapshots)
            results.append({"entry_cents": entry, "wake_minutes": wake, **result.to_dict()})
            wake += wake_range[2]
        entry += entry_range[2]
    results.sort(key=lambda r: r["total_pnl"], reverse=True)
    return results


def monte_carlo_simulation(
    trade_pnls: list[float], num_sims: int = 1000,
    trades_per_sim: int = 100, starting_capital: float = 2000.0,
) -> dict[str, Any]:
    """Randomly resample trades to estimate future performance distribution."""
    if not trade_pnls:
        return {"error": "no trades"}
    finals: list[float] = []
    drawdowns: list[float] = []
    ruin = 0
    for _ in range(num_sims):
        equity = starting_capital
        peak = starting_capital
        max_dd = 0.0
        busted = False
        for _ in range(trades_per_sim):
            equity += random.choice(trade_pnls)
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
            if equity <= 0:
                busted = True
                break
        if busted:
            ruin += 1
            finals.append(0.0)
        else:
            finals.append(equity)
        drawdowns.append(max_dd)

    finals.sort()
    drawdowns.sort()
    p = lambda arr, pct: arr[min(int(len(arr) * pct / 100), len(arr) - 1)]

    return {
        "simulations": num_sims,
        "median_final": round(p(finals, 50), 2),
        "p5": round(p(finals, 5), 2),
        "p95": round(p(finals, 95), 2),
        "mean_final": round(sum(finals) / len(finals), 2),
        "median_drawdown": round(p(drawdowns, 50), 2),
        "prob_ruin": round(ruin / num_sims, 4),
        "prob_profit": round(sum(1 for v in finals if v > starting_capital) / num_sims, 4),
    }


def build_synthetic_snapshots(num_markets: int = 20) -> list[dict]:
    """Generate fake market data for testing when no real snapshots exist."""
    base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = []
    random.seed(42)
    categories = ["politics", "sports", "science", "entertainment", "business", "technology"]
    for i in range(num_markets):
        market_id = f"mkt_synthetic_{i:03d}"
        category = random.choice(categories)
        end_time = base + timedelta(minutes=random.randint(10, 300))
        price = random.uniform(0.88, 0.97)
        wins_market = random.random() < 0.75
        num_ticks = random.randint(15, 60)
        for t in range(num_ticks):
            drift = 0.001 if wins_market else -0.003
            price = max(0.01, min(0.99, price + drift + random.gauss(0, 0.008)))
            ts = base + timedelta(seconds=t * random.randint(8, 15))
            rows.append({
                "market_id": market_id, "question": f"Synthetic {i}",
                "category": category, "end_time": end_time.isoformat(),
                "ts": ts.isoformat(), "yes": round(price, 4), "no": round(1 - price, 4),
            })
    return rows
