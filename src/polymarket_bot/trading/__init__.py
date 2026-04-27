"""Trading logic: strategy, portfolio, and self-correction."""

from .adaptive import adapt_strategy
from .paper import PaperPortfolio
from .strategy import (
    eligible_for_tracking,
    is_crypto_market,
    is_weather_market,
    pick_entry_side,
    select_markets_for_next_24h,
    should_wake_for_market,
)

__all__ = [
    "PaperPortfolio", "adapt_strategy", "eligible_for_tracking",
    "is_crypto_market", "is_weather_market", "pick_entry_side",
    "select_markets_for_next_24h", "should_wake_for_market",
]
