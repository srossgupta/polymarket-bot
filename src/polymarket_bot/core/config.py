"""Configuration with validation, parameter bounds, and adaptive state persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "polymarket.db")
ADAPTIVE_STATE_FILE = os.path.join(DATA_DIR, "adaptive_state.json")


@dataclass
class StrategyParams:
    # Scan schedule
    scan_hour_local: int = 8
    scan_minute_local: int = 0
    max_scan_horizon_hours: int = 24

    # Wake / monitoring
    wake_minutes_before_close: int = 6
    poll_seconds: float = 1.0  # fast polling for real-time
    min_time_to_close_seconds: int = 15

    # Volume filter
    min_volume_usd: float = 100_000.0

    # Entry
    entry_threshold_cents: float = 95.0

    # Stop loss
    stop_loss_cents: float = 70.0

    # Position sizing
    max_dollars_per_market: float = 100.0

    # Crypto exclusion keywords
    disallowed_category_keywords: list[str] = field(
        default_factory=lambda: [
            "crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
            "dogecoin", "doge", "altcoin", "defi", "nft", "blockchain",
            "binance", "coinbase", "token", "memecoin",
        ]
    )

    def validate(self, bounds: ParamBounds) -> None:
        self.entry_threshold_cents = max(bounds.min_entry_cents,
                                         min(bounds.max_entry_cents, self.entry_threshold_cents))
        self.stop_loss_cents = max(bounds.min_stop_cents,
                                  min(bounds.max_stop_cents, self.stop_loss_cents))
        self.wake_minutes_before_close = max(bounds.min_wake_minutes,
                                             min(bounds.max_wake_minutes, self.wake_minutes_before_close))


@dataclass
class ParamBounds:
    """Hard limits for self-correction parameter changes."""
    min_entry_cents: float = 90.0
    max_entry_cents: float = 98.0
    min_stop_cents: float = 55.0
    max_stop_cents: float = 85.0
    min_wake_minutes: int = 3
    max_wake_minutes: int = 15
    min_poll_seconds: float = 0.5
    max_poll_seconds: float = 5.0


@dataclass
class AdaptationConfig:
    """Controls for the self-correcting loop."""
    trade_window: int = 80
    min_trades_for_adaptation: int = 10
    target_win_rate: float = 0.62
    step_cents: float = 1.0
    min_category_samples: int = 5
    max_preferred_categories: int = 6
    # Exponential decay for weighting recent trades higher
    decay_factor: float = 0.95
    # Confidence: only adapt if we have enough data for statistical significance
    min_confidence_level: float = 0.6


@dataclass
class BotConfig:
    timezone: str = "America/Los_Angeles"
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    requests_timeout_seconds: int = 20

    starting_cash: float = 2_000.0
    max_open_positions: int = 15

    strategy: StrategyParams = field(default_factory=StrategyParams)
    bounds: ParamBounds = field(default_factory=ParamBounds)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    preferred_categories: list[str] = field(default_factory=list)

    @property
    def local_tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def scan_time(self) -> time:
        return time(self.strategy.scan_hour_local, self.strategy.scan_minute_local)


def _overlay_dict(obj: Any, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def load_config() -> BotConfig:
    """Load base config, then overlay adaptive state if available."""
    cfg = BotConfig()

    if os.path.exists(ADAPTIVE_STATE_FILE):
        try:
            with open(ADAPTIVE_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            _overlay_dict(cfg.strategy, state.get("strategy", {}))
            cfg.preferred_categories = state.get("preferred_categories", [])
            cfg.strategy.validate(cfg.bounds)
        except (OSError, json.JSONDecodeError):
            pass

    return cfg


def save_adaptive_strategy(params: StrategyParams, preferred_categories: list[str]) -> None:
    payload = {
        "strategy": asdict(params),
        "preferred_categories": preferred_categories,
    }
    with open(ADAPTIVE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
