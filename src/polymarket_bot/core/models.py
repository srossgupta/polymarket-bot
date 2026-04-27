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
    volume_at_entry: float = 0.0

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
