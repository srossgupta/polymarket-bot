"""Trading strategy: market selection, entry signals, stop-loss, and quantitative filters.

Uses Kelly criterion for sizing, expected value calculations for entry decisions,
and volatility-adjusted stop losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .config import BotConfig
from .models import Market, PricePoint, PriceSeries, Side


@dataclass
class EntrySignal:
    side: Side
    price: float
    reason: str
    expected_value: float = 0.0
    kelly_fraction: float = 0.0
    velocity: float = 0.0
    volatility: float = 0.0
    confidence: float = 0.0


# --- Crypto filter ---

CRYPTO_KEYWORDS = frozenset([
    "crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
    "dogecoin", "doge", "altcoin", "defi", "nft", "blockchain",
    "binance", "coinbase", "token", "memecoin", "cardano", "ripple",
    "xrp", "polygon", "matic", "avalanche", "avax", "polkadot",
    "chainlink", "uniswap", "aave", "stablecoin", "usdc", "usdt",
    "tether", "litecoin", "ltc",
])


def is_crypto_market(market: Market) -> bool:
    text = f"{market.category} {market.question} {market.slug}".lower()
    return any(kw in text for kw in CRYPTO_KEYWORDS)


# --- Market selection ---

def select_markets_for_next_24h(
    markets: list[Market],
    cfg: BotConfig,
    now: datetime | None = None,
) -> list[Market]:
    now_utc = now or datetime.now(timezone.utc)
    horizon = now_utc + timedelta(hours=cfg.strategy.max_scan_horizon_hours)
    pref = {c.lower() for c in cfg.preferred_categories} if cfg.preferred_categories else set()

    selected = []
    for market in markets:
        if not market.active:
            continue
        if is_crypto_market(market):
            continue
        if pref and market.category.lower() not in pref:
            continue
        if not (now_utc < market.end_time <= horizon):
            continue
        selected.append(market)

    return sorted(selected, key=lambda m: m.end_time)


def should_wake_for_market(market: Market, cfg: BotConfig,
                           now: datetime | None = None) -> bool:
    now_utc = now or datetime.now(timezone.utc)
    wake_time = market.end_time - timedelta(
        minutes=cfg.strategy.wake_minutes_before_close)
    return now_utc >= wake_time


def eligible_for_tracking(market: Market, cfg: BotConfig) -> bool:
    return market.volume_usd >= cfg.strategy.min_volume_usd


# --- Quantitative helpers ---

def implied_probability(price: float) -> float:
    """Convert market price to implied probability (assuming no vig)."""
    return max(0.0, min(1.0, price))


def expected_value(price: float, implied_prob: float, payout: float = 1.0) -> float:
    """EV of buying at `price` with `implied_prob` of winning.
    Payout is $1 for binary markets. EV = prob * payout - price.
    """
    return implied_prob * payout - price


def kelly_criterion(prob: float, odds: float) -> float:
    """Kelly fraction: f* = (bp - q) / b where b = net odds, p = prob, q = 1 - p.
    For binary markets: buying at price `c`, payout is $1.
    odds = (1 - c) / c  (net profit per dollar risked if you win).
    """
    if prob <= 0 or prob >= 1 or odds <= 0:
        return 0.0
    q = 1.0 - prob
    f = (odds * prob - q) / odds
    return max(0.0, min(1.0, f))


def compute_kelly_for_price(price: float) -> float:
    """For a high-prob market near expiry, estimate Kelly fraction.
    At 95c: you risk $0.95 to win $0.05.
    But the key insight: at near-expiry, the true probability may be
    SLIGHTLY higher than price due to market friction.
    We model true_prob = price + small_edge (1% edge assumption).
    """
    if price <= 0 or price >= 1:
        return 0.0
    # Assume a small edge: true prob slightly above market price
    edge = 0.01
    prob = min(0.995, price + edge)
    odds = (1.0 - price) / price  # net odds based on cost
    return kelly_criterion(prob, odds)


def volatility_adjusted_stop(
    entry_price: float,
    vol: float,
    base_stop_cents: float,
    multiplier: float = 2.0,
) -> float:
    """Adjust stop-loss based on observed volatility.
    Higher vol → wider stop to avoid premature exit.
    """
    base_stop = base_stop_cents / 100.0
    vol_adjustment = min(vol * multiplier, 0.10)  # cap at 10 cents
    adjusted = base_stop - vol_adjustment
    return max(0.50, min(entry_price - 0.05, adjusted))  # never tighter than 5c from entry


# --- Entry signal ---

def entry_signal_from_price(
    point: PricePoint,
    cfg: BotConfig,
    series: PriceSeries | None = None,
) -> EntrySignal | None:
    """Generate entry signal if YES or NO price exceeds threshold.
    Enriched with EV, Kelly, velocity, and volatility when series is available.
    """
    threshold = cfg.strategy.entry_threshold_cents / 100.0

    # Check both sides, prefer the higher-priced one
    candidates: list[tuple[Side, float]] = []
    if point.yes >= threshold:
        candidates.append((Side.YES, point.yes))
    if point.no >= threshold:
        candidates.append((Side.NO, point.no))

    if not candidates:
        return None

    # Pick the side with higher price (higher confidence)
    side, price = max(candidates, key=lambda x: x[1])

    # Compute quantitative metrics
    prob = implied_probability(price)
    ev = expected_value(price, prob)
    kf = compute_kelly_for_price(price)

    vel = 0.0
    vol = 0.0
    confidence = prob  # base confidence = implied probability

    if series and len(series.points) >= 3:
        vel = series.velocity(side, window=5)
        vol = series.volatility(side, window=10)

        # Boost confidence if price is stable/rising and vol is low
        if vel >= 0 and vol < 0.02:
            confidence = min(0.99, confidence + 0.02)
        # Reduce confidence if price is falling or vol is high
        elif vel < -0.001 or vol > 0.05:
            confidence = max(0.5, confidence - 0.05)

    # Reject if EV is significantly negative (price too high relative to prob)
    # For near-expiry markets, this is rare but protects edge cases
    if ev < -0.05:
        return None

    reason = (f"{side.value}>={cfg.strategy.entry_threshold_cents:.0f}c "
              f"EV={ev:.4f} Kelly={kf:.3f} vel={vel:.5f}")

    return EntrySignal(
        side=side,
        price=price,
        reason=reason,
        expected_value=ev,
        kelly_fraction=kf,
        velocity=vel,
        volatility=vol,
        confidence=confidence,
    )


# --- Stop loss ---

def stop_loss_hit(
    side: Side,
    point: PricePoint,
    cfg: BotConfig,
    series: PriceSeries | None = None,
) -> tuple[bool, float]:
    """Check if position should be stopped out.
    Uses base stop + optional volatility adjustment.
    """
    base_stop = cfg.strategy.stop_loss_cents / 100.0
    current = point.yes if side == Side.YES else point.no

    stop_level = base_stop
    if series and len(series.points) >= 5:
        vol = series.volatility(side, window=10)
        # If volatility is very high, slightly widen stop to avoid whipsaw
        if vol > 0.03:
            stop_level = max(0.50, base_stop - 0.03)

    return current < stop_level, current


# --- Position sizing ---

def compute_position_size(
    price: float,
    kelly_fraction: float,
    max_dollars: float,
    bankroll: float,
) -> float:
    """Position size using fractional Kelly (half-Kelly for safety)."""
    half_kelly = kelly_fraction * 0.5
    kelly_size = bankroll * half_kelly
    return min(kelly_size, max_dollars, bankroll)
