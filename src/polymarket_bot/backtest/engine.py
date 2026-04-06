"""Backtesting engine: snapshot replay, parameter sweep, and Monte Carlo simulation."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from polymarket_bot.core import BotConfig, Market, PricePoint, PriceSeries, Side
from polymarket_bot.trading.strategy import entry_signal_from_price, stop_loss_hit


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
    entry_ts: str = ""
    exit_ts: str = ""
    hold_ticks: int = 0


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
    expectancy: float
    category_stats: dict[str, dict[str, float]]
    trade_log: list[BacktestTrade]
    params_used: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades, "wins": self.wins, "losses": self.losses,
            "win_rate": round(self.win_rate, 4), "total_pnl": round(self.total_pnl, 2),
            "avg_pnl": round(self.avg_pnl, 2), "max_drawdown": round(self.max_drawdown, 2),
            "sharpe": round(self.sharpe, 4), "expectancy": round(self.expectancy, 4),
            "category_stats": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in self.category_stats.items()},
            "params_used": self.params_used, "trade_count_detail": len(self.trade_log),
        }


def _row_to_market(row: dict) -> Market:
    end_time = row.get("end_time") or row.get("endTime") or ""
    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time)
    return Market(market_id=row["market_id"], question=row.get("question", ""),
                  end_time=end_time, volume_usd=float(row.get("volume_usd", 150000)),
                  category=row.get("category", "unknown"), yes_token_id="yes", no_token_id="no")


def run_snapshot_backtest(cfg: BotConfig, snapshots: list[dict]) -> BacktestResult:
    by_market: dict[str, list[dict]] = defaultdict(list)
    for row in snapshots:
        by_market[row["market_id"]].append(row)

    trade_log: list[BacktestTrade] = []
    equity_curve: list[float] = [cfg.starting_cash]
    by_category: dict[str, dict[str, float]] = defaultdict(
        lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "avg_pnl": 0.0})

    for market_id, rows in by_market.items():
        rows.sort(key=lambda r: r.get("ts", ""))
        market = _row_to_market(rows[0])
        entered: dict | None = None
        series = PriceSeries(market_id=market_id)
        hold_ticks = 0

        for row in rows:
            yes = float(row.get("yes_price", row.get("yes", 0)))
            no = float(row.get("no_price", row.get("no", 0)))
            ts_str = row.get("ts", "")
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
            point = PricePoint(ts=ts, yes=yes, no=no)
            series.add(point)

            if entered is None:
                signal = entry_signal_from_price(point, cfg, series)
                if signal:
                    entered = {"side": signal.side, "entry": signal.price,
                               "size": cfg.strategy.max_dollars_per_market, "entry_ts": ts_str}
                    hold_ticks = 0
                continue

            hold_ticks += 1
            side = entered["side"]
            current = point.yes if side == Side.YES else point.no
            hit, _ = stop_loss_hit(side, point, cfg, series)
            if hit:
                pnl = (entered["size"] / entered["entry"]) * current - entered["size"]
                trade_log.append(BacktestTrade(
                    market_id=market_id, category=market.category, side=side.value,
                    entry_price=entered["entry"], exit_price=current, size=entered["size"],
                    pnl=pnl, reason="stop_loss", entry_ts=entered["entry_ts"],
                    exit_ts=ts_str, hold_ticks=hold_ticks))
                equity_curve.append(equity_curve[-1] + pnl)
                by_category[market.category]["trades"] += 1
                by_category[market.category]["pnl"] += pnl
                entered = None
                continue

        if entered:
            last = rows[-1]
            side = entered["side"]
            exit_price = float(last.get("yes_price", last.get("yes", 0))) if side == Side.YES \
                else float(last.get("no_price", last.get("no", 0)))
            pnl = (entered["size"] / entered["entry"]) * exit_price - entered["size"]
            trade_log.append(BacktestTrade(
                market_id=market_id, category=market.category, side=side.value,
                entry_price=entered["entry"], exit_price=exit_price, size=entered["size"],
                pnl=pnl, reason="market_expired", entry_ts=entered["entry_ts"],
                exit_ts=last.get("ts", ""), hold_ticks=hold_ticks))
            equity_curve.append(equity_curve[-1] + pnl)
            by_category[market.category]["trades"] += 1
            by_category[market.category]["pnl"] += pnl
            if pnl > 0:
                by_category[market.category]["wins"] += 1

    wins = sum(1 for t in trade_log if t.pnl > 0)
    losses = len(trade_log) - wins
    total_pnl = sum(t.pnl for t in trade_log)
    avg_pnl = total_pnl / len(trade_log) if trade_log else 0

    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        max_dd = max(max_dd, peak - val)

    pnls = [t.pnl for t in trade_log]
    sharpe = 0.0
    if len(pnls) >= 2:
        mean = sum(pnls) / len(pnls)
        std = (sum((p - mean) ** 2 for p in pnls) / len(pnls)) ** 0.5
        sharpe = mean / std if std > 1e-8 else 0.0

    for cat in by_category:
        n = by_category[cat]["trades"]
        if n > 0:
            by_category[cat]["avg_pnl"] = by_category[cat]["pnl"] / n
            by_category[cat]["win_rate"] = by_category[cat]["wins"] / n

    return BacktestResult(
        trades=len(trade_log), wins=wins, losses=losses,
        win_rate=wins / len(trade_log) if trade_log else 0,
        total_pnl=total_pnl, avg_pnl=avg_pnl, max_drawdown=max_dd,
        sharpe=sharpe, expectancy=avg_pnl, category_stats=dict(by_category),
        trade_log=trade_log, params_used={
            "entry_cents": cfg.strategy.entry_threshold_cents,
            "stop_cents": cfg.strategy.stop_loss_cents,
            "wake_minutes": cfg.strategy.wake_minutes_before_close})


def parameter_sweep(
    snapshots: list[dict],
    entry_range: tuple[float, float, float] = (90, 98, 1),
    stop_range: tuple[float, float, float] = (55, 85, 5),
    wake_range: tuple[int, int, int] = (3, 15, 3),
) -> list[dict]:
    results: list[dict] = []
    entry_val = entry_range[0]
    while entry_val <= entry_range[1]:
        stop_val = stop_range[0]
        while stop_val <= stop_range[1]:
            wake_val = wake_range[0]
            while wake_val <= wake_range[1]:
                cfg = BotConfig()
                cfg.strategy.entry_threshold_cents = entry_val
                cfg.strategy.stop_loss_cents = stop_val
                cfg.strategy.wake_minutes_before_close = wake_val
                result = run_snapshot_backtest(cfg, snapshots)
                results.append({"entry_cents": entry_val, "stop_cents": stop_val,
                                "wake_minutes": wake_val, **result.to_dict()})
                wake_val += wake_range[2]
            stop_val += stop_range[2]
        entry_val += entry_range[2]
    results.sort(key=lambda r: (r["total_pnl"], r["sharpe"]), reverse=True)
    return results


def monte_carlo_simulation(
    trade_pnls: list[float], num_simulations: int = 1000,
    trades_per_sim: int = 100, starting_capital: float = 2000.0,
) -> dict[str, Any]:
    if not trade_pnls:
        return {"error": "no trades to simulate"}
    final_values: list[float] = []
    max_drawdowns: list[float] = []
    ruin_count = 0
    for _ in range(num_simulations):
        equity = starting_capital
        peak = starting_capital
        max_dd = 0.0
        ruined = False
        for _ in range(trades_per_sim):
            equity += random.choice(trade_pnls)
            if equity > peak:
                peak = equity
            max_dd = max(max_dd, peak - equity)
            if equity <= 0:
                ruined = True
                break
        if ruined:
            ruin_count += 1
            final_values.append(0.0)
        else:
            final_values.append(equity)
        max_drawdowns.append(max_dd)
    final_values.sort()
    max_drawdowns.sort()

    def pct(arr, p):
        return arr[min(int(len(arr) * p / 100), len(arr) - 1)]

    return {
        "simulations": num_simulations, "trades_per_sim": trades_per_sim,
        "starting_capital": starting_capital,
        "median_final_value": round(pct(final_values, 50), 2),
        "p5_final_value": round(pct(final_values, 5), 2),
        "p25_final_value": round(pct(final_values, 25), 2),
        "p75_final_value": round(pct(final_values, 75), 2),
        "p95_final_value": round(pct(final_values, 95), 2),
        "mean_final_value": round(sum(final_values) / len(final_values), 2),
        "median_max_drawdown": round(pct(max_drawdowns, 50), 2),
        "p95_max_drawdown": round(pct(max_drawdowns, 95), 2),
        "probability_of_ruin": round(ruin_count / num_simulations, 4),
        "probability_of_profit": round(
            sum(1 for v in final_values if v > starting_capital) / num_simulations, 4),
    }


def build_synthetic_snapshots(num_markets: int = 20) -> list[dict]:
    base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = []
    random.seed(42)
    categories = ["politics", "sports", "science", "entertainment",
                   "business", "weather", "legal", "technology"]
    for i in range(num_markets):
        market_id = f"mkt_synthetic_{i:03d}"
        category = random.choice(categories)
        end_time = base + timedelta(minutes=random.randint(10, 300))
        start_price = random.uniform(0.88, 0.97)
        price = start_price
        num_ticks = random.randint(15, 60)
        wins_market = random.random() < 0.75
        for t in range(num_ticks):
            drift = 0.001 if wins_market else -0.003
            price = max(0.01, min(0.99, price + drift + random.gauss(0, 0.008)))
            if t > num_ticks * 0.8:
                if wins_market:
                    price = min(0.99, price + random.uniform(0, 0.02))
                else:
                    price = max(0.01, price - random.uniform(0, 0.04))
            ts = base + timedelta(seconds=t * random.randint(8, 15))
            rows.append({"market_id": market_id, "question": f"Synthetic {i} ({category})",
                         "category": category, "end_time": end_time.isoformat(),
                         "ts": ts.isoformat(), "yes": round(price, 4),
                         "no": round(1 - price, 4),
                         "volume_usd": random.uniform(100000, 5000000)})
    return rows
