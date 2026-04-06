"""Paper portfolio: tracks positions, executes paper trades, computes P&L."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from polymarket_bot.core import (BotConfig, Market, PerformanceSnapshot, Position,
                                  PricePoint, Side, TradeEvent, TradeType)
from polymarket_bot.data.storage import append_trade, get_category_performance, save_performance
from polymarket_bot.trading.strategy import compute_position_size, volatility_adjusted_stop

logger = logging.getLogger(__name__)


class PaperPortfolio:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.cash: float = cfg.starting_cash
        self.initial_cash: float = cfg.starting_cash
        self.open_positions: dict[str, Position] = {}
        self.total_trades: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.total_pnl: float = 0.0
        self._pnl_history: list[float] = []

    def can_open_new(self) -> bool:
        return (len(self.open_positions) < self.cfg.max_open_positions
                and self.cash > 1.0)

    def open_position(
        self, market: Market, side: Side, price: float, reason: str,
        kelly_fraction: float = 0.0, velocity: float = 0.0, volatility: float = 0.0,
    ) -> Position | None:
        if market.market_id in self.open_positions:
            return None
        if not self.can_open_new():
            return None
        if kelly_fraction > 0:
            allocation = compute_position_size(
                price, kelly_fraction, self.cfg.strategy.max_dollars_per_market, self.cash)
        else:
            allocation = min(self.cfg.strategy.max_dollars_per_market, self.cash)
        if allocation < 1:
            return None

        shares = allocation / max(price, 1e-6)
        stop_price = volatility_adjusted_stop(price, volatility, self.cfg.strategy.stop_loss_cents)
        pos = Position(
            market_id=market.market_id, question=market.question, side=side,
            category=market.category, entry_ts=datetime.now(timezone.utc),
            entry_price=price, size_dollars=allocation, shares=shares,
            stop_loss_price=stop_price,
        )
        self.open_positions[market.market_id] = pos
        self.cash -= allocation

        event = TradeEvent(
            market_id=market.market_id, question=market.question,
            category=market.category, side=side.value,
            event_type=TradeType.BUY_LIMIT.value, ts=pos.entry_ts,
            price=price, size_dollars=allocation, shares=shares,
            reason=reason, volume_at_entry=market.volume_usd,
            velocity_at_entry=velocity, volatility_at_entry=volatility,
        )
        append_trade(event)
        logger.info("OPEN %s %s @ %.4f ($%.2f) | %s",
                     side.value, market.question[:50], price, allocation, reason)
        return pos

    def close_position(
        self, market: Market, point: PricePoint, reason: str,
        trade_type: TradeType = TradeType.SELL_MARKET,
    ) -> TradeEvent | None:
        pos = self.open_positions.get(market.market_id)
        if not pos:
            return None
        exit_price = point.yes if pos.side == Side.YES else point.no
        proceeds = pos.shares * exit_price
        pnl = proceeds - pos.size_dollars
        self.cash += proceeds
        del self.open_positions[market.market_id]

        self.total_trades += 1
        self.total_pnl += pnl
        self._pnl_history.append(pnl)
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        hold_seconds = (point.ts - pos.entry_ts).total_seconds()
        event = TradeEvent(
            market_id=market.market_id, question=market.question,
            category=market.category, side=pos.side.value,
            event_type=trade_type.value, ts=point.ts,
            price=exit_price, size_dollars=proceeds, shares=pos.shares,
            pnl=pnl, reason=reason, entry_price=pos.entry_price,
            hold_duration_seconds=hold_seconds, peak_price=pos.peak_price,
        )
        append_trade(event)
        logger.info("CLOSE %s %s @ %.4f PnL=$%.2f | %s",
                     pos.side.value, market.question[:50], exit_price, pnl, reason)
        return event

    def mark_to_market(self, prices: dict[str, PricePoint]) -> float:
        value = self.cash
        for mid, pos in self.open_positions.items():
            point = prices.get(mid)
            if point:
                px = point.yes if pos.side == Side.YES else point.no
                value += pos.shares * px
            else:
                value += pos.size_dollars
        return value

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades else 0.0

    @property
    def avg_win(self) -> float:
        wins = [p for p in self._pnl_history if p > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [p for p in self._pnl_history if p <= 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def sharpe_estimate(self) -> float:
        if len(self._pnl_history) < 3:
            return 0.0
        mean = sum(self._pnl_history) / len(self._pnl_history)
        var = sum((p - mean) ** 2 for p in self._pnl_history) / len(self._pnl_history)
        std = var ** 0.5
        return mean / std if std > 1e-8 else 0.0

    @property
    def expectancy(self) -> float:
        return self.total_pnl / self.total_trades if self.total_trades else 0.0

    def take_performance_snapshot(self, prices: dict[str, PricePoint] | None = None) -> PerformanceSnapshot:
        total_value = self.mark_to_market(prices or {})
        positions_value = total_value - self.cash
        cat_perf = get_category_performance()
        best_cat = cat_perf[0]["category"] if cat_perf else "none"
        worst_cat = cat_perf[-1]["category"] if cat_perf else "none"

        snap = PerformanceSnapshot(
            ts=datetime.now(timezone.utc), cash=round(self.cash, 2),
            positions_value=round(positions_value, 2),
            total_value=round(total_value, 2),
            open_positions=len(self.open_positions),
            total_trades=self.total_trades, wins=self.wins, losses=self.losses,
            total_pnl=round(self.total_pnl, 2), win_rate=round(self.win_rate, 4),
            avg_win=round(self.avg_win, 2), avg_loss=round(self.avg_loss, 2),
            sharpe_estimate=round(self.sharpe_estimate, 4),
            expectancy=round(self.expectancy, 4),
            best_category=best_cat, worst_category=worst_cat,
        )
        save_performance(snap)
        return snap

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": round(self.cash, 2),
            "total_value": round(self.cash + sum(p.size_dollars for p in self.open_positions.values()), 2),
            "open_positions": len(self.open_positions),
            "total_trades": self.total_trades, "wins": self.wins, "losses": self.losses,
            "total_pnl": round(self.total_pnl, 2), "win_rate": round(self.win_rate, 4),
            "sharpe": round(self.sharpe_estimate, 4), "expectancy": round(self.expectancy, 4),
        }
