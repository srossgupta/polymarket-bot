"""Core dataclasses for markets, prices, positions, and trade events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Side(str, Enum):
    YES = "YES"
    NO = "NO"


class TradeType(str, Enum):
    BUY_LIMIT = "BUY_LIMIT"
    SELL_MARKET = "SELL_MARKET"
    STOP_LOSS = "STOP_LOSS"
    FORCED_CLOSE = "FORCED_CLOSE"


@dataclass
class Market:
    market_id: str
    question: str
    end_time: datetime
    volume_usd: float
    category: str
    yes_token_id: str
    no_token_id: str
    active: bool = True
    slug: str = ""
    volume_24h: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    neg_risk: bool = False

    @property
    def seconds_to_close(self) -> float:
        return max(0, (self.end_time - datetime.now(timezone.utc)).total_seconds())

    @property
    def minutes_to_close(self) -> float:
        return self.seconds_to_close / 60.0


@dataclass
class PricePoint:
    ts: datetime
    yes: float
    no: float
    spread: float = 0.0
    volume_at_snapshot: float = 0.0

    @property
    def dominant_side(self) -> Side:
        return Side.YES if self.yes >= self.no else Side.NO

    @property
    def dominant_price(self) -> float:
        return max(self.yes, self.no)


@dataclass
class PriceSeries:
    """Rolling window of price observations for a market."""
    market_id: str
    points: list[PricePoint] = field(default_factory=list)
    max_points: int = 500

    def add(self, point: PricePoint) -> None:
        self.points.append(point)
        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points:]

    @property
    def latest(self) -> PricePoint | None:
        return self.points[-1] if self.points else None

    def prices(self, side: Side) -> list[float]:
        return [p.yes if side == Side.YES else p.no for p in self.points]

    def velocity(self, side: Side, window: int = 5) -> float:
        """Price change per second over the last `window` observations."""
        px = self.prices(side)
        if len(px) < 2 or len(self.points) < 2:
            return 0.0
        n = min(window, len(self.points))
        recent = px[-n:]
        dt = (self.points[-1].ts - self.points[-n].ts).total_seconds()
        if dt <= 0:
            return 0.0
        return (recent[-1] - recent[0]) / dt

    def volatility(self, side: Side, window: int = 20) -> float:
        """Standard deviation of returns over last `window` observations."""
        px = self.prices(side)
        if len(px) < 3:
            return 0.0
        recent = px[-window:]
        returns = [(recent[i] - recent[i - 1]) / max(recent[i - 1], 1e-6)
                   for i in range(1, len(recent))]
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        return var ** 0.5

    def mean_price(self, side: Side, window: int = 10) -> float:
        px = self.prices(side)
        if not px:
            return 0.0
        recent = px[-window:]
        return sum(recent) / len(recent)


@dataclass
class Position:
    market_id: str
    question: str
    side: Side
    category: str
    entry_ts: datetime
    entry_price: float
    size_dollars: float
    shares: float
    stop_loss_price: float
    peak_price: float = 0.0
    ticks_held: int = 0

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price

    @property
    def unrealized_pnl_at(self) -> float:
        """P&L if closed at peak price."""
        return self.shares * self.peak_price - self.size_dollars

    def update_peak(self, current_price: float) -> None:
        self.ticks_held += 1
        if current_price > self.peak_price:
            self.peak_price = current_price


@dataclass
class TradeEvent:
    market_id: str
    question: str
    category: str
    side: str
    event_type: str
    ts: datetime
    price: float
    size_dollars: float
    shares: float
    pnl: float = 0.0
    reason: str = ""
    entry_price: float = 0.0
    hold_duration_seconds: float = 0.0
    peak_price: float = 0.0
    volume_at_entry: float = 0.0
    velocity_at_entry: float = 0.0
    volatility_at_entry: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ts"] = self.ts.isoformat()
        return payload

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TradeEvent:
        d = dict(d)
        if isinstance(d.get("ts"), str):
            d["ts"] = datetime.fromisoformat(d["ts"])
        return TradeEvent(**{k: v for k, v in d.items() if k in TradeEvent.__dataclass_fields__})


@dataclass
class PerformanceSnapshot:
    """Point-in-time portfolio performance."""
    ts: datetime
    cash: float
    positions_value: float
    total_value: float
    open_positions: int
    total_trades: int
    wins: int
    losses: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe_estimate: float
    expectancy: float
    best_category: str
    worst_category: str
