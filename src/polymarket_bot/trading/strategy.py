"""Trading strategy: market selection and entry signals.

Simple strategy:
  1. Filter markets (no crypto, no weather, closing within horizon)
  2. Wake 5 min before close
  3. Enter if price is in 95¢–99.5¢ band
  4. Hold to expiry — no stop loss
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polymarket_bot.core import BotConfig, Market, PricePoint, Side


# --- Keyword filters (skip noise markets) ---

CRYPTO_KEYWORDS = frozenset([
    "crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
    "dogecoin", "doge", "altcoin", "defi", "nft", "blockchain",
    "binance", "coinbase", "token", "memecoin", "cardano", "ripple",
    "xrp", "polygon", "matic", "avalanche", "avax", "polkadot",
    "chainlink", "uniswap", "aave", "stablecoin", "usdc", "usdt",
    "tether", "litecoin", "ltc",
])

WEATHER_KEYWORDS = frozenset([
    "temperature", "highest temperature", "lowest temperature",
    "rainfall", "precipitation", "humidity", "weather",
    "degrees", "celsius", "fahrenheit", "snowfall", "wind speed",
])


def _text_matches(market: Market, keywords: frozenset) -> bool:
    text = f"{market.category} {market.question} {market.slug}".lower()
    return any(kw in text for kw in keywords)


def is_crypto_market(market: Market) -> bool:
    return _text_matches(market, CRYPTO_KEYWORDS)


def is_weather_market(market: Market) -> bool:
    return _text_matches(market, WEATHER_KEYWORDS)


# --- Market selection ---

def select_markets_for_next_24h(
    markets: list[Market], cfg: BotConfig, now: datetime | None = None,
) -> list[Market]:
    """Pick active, non-crypto, non-weather markets closing within the scan horizon."""
    import logging
    log = logging.getLogger(__name__)
    now_utc = now or datetime.now(timezone.utc)
    horizon = now_utc + timedelta(hours=cfg.strategy.max_scan_horizon_hours)

    pref = {c.lower() for c in cfg.preferred_categories} if cfg.preferred_categories else set()
    selected = []
    drop_inactive = drop_noise = drop_pref = drop_time = 0

    for market in markets:
        if not market.active:
            drop_inactive += 1
            continue
        if is_crypto_market(market) or is_weather_market(market):
            drop_noise += 1
            continue
        if pref and market.category.lower() not in pref:
            drop_pref += 1
            continue
        if not (now_utc < market.end_time <= horizon):
            drop_time += 1
            continue
        selected.append(market)

    log.info(
        "select filter: %d in → %d out (dropped: %d inactive, %d noise, %d pref, %d time)",
        len(markets), len(selected), drop_inactive, drop_noise, drop_pref, drop_time,
    )
    return sorted(selected, key=lambda m: m.end_time)


# --- Simple helpers ---

def should_wake_for_market(market: Market, cfg: BotConfig,
                           now: datetime | None = None) -> bool:
    """True if it's time to start monitoring this market (within wake window)."""
    now_utc = now or datetime.now(timezone.utc)
    wake_time = market.end_time - timedelta(minutes=cfg.strategy.wake_minutes_before_close)
    return now_utc >= wake_time


def eligible_for_tracking(market: Market, cfg: BotConfig) -> bool:
    """True if market has enough volume to bother with."""
    return market.volume_usd >= cfg.strategy.min_volume_usd


# --- Entry signal ---

def in_entry_band(price: float, cfg: BotConfig) -> bool:
    """True if price is in the 95¢–99.5¢ sweet spot."""
    low = cfg.strategy.entry_threshold_cents / 100.0  # e.g. 0.95
    high = 0.995
    return low < price < high


def pick_entry_side(point: PricePoint, cfg: BotConfig) -> tuple[Side, float] | None:
    """If YES or NO price is in the entry band, return (side, price). Else None."""
    if in_entry_band(point.yes, cfg):
        return Side.YES, point.yes
    if in_entry_band(point.no, cfg):
        return Side.NO, point.no
    return None
