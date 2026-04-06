"""Trading logic: strategy, portfolio, and self-correction."""

from .adaptive import adapt_strategy
from .paper import PaperPortfolio
from .strategy import (
    EntrySignal,
    compute_kelly_for_price,
    compute_position_size,
    eligible_for_tracking,
    entry_signal_from_price,
    is_crypto_market,
    select_markets_for_next_24h,
    should_wake_for_market,
    stop_loss_hit,
    volatility_adjusted_stop,
)

__all__ = [
    "EntrySignal",
    "PaperPortfolio",
    "adapt_strategy",
    "compute_kelly_for_price",
    "compute_position_size",
    "eligible_for_tracking",
    "entry_signal_from_price",
    "is_crypto_market",
    "select_markets_for_next_24h",
    "should_wake_for_market",
    "stop_loss_hit",
    "volatility_adjusted_stop",
]
