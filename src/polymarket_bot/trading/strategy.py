"""Trading strategy: market selection, entry signals, stop-loss, and quantitative filters.

Uses Kelly criterion for sizing, expected value calculations for entry decisions,
and volatility-adjusted stop losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from polymarket_bot.core import BotConfig, Market, PricePoint, PriceSeries, Side


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
    markets: list[Market], cfg: BotConfig, now: datetime | None = None,
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
    wake_time = market.end_time - timedelta(minutes=cfg.strategy.wake_minutes_before_close)
    return now_utc >= wake_time


def eligible_for_tracking(market: Market, cfg: BotConfig) -> bool:
    return market.volume_usd >= cfg.strategy.min_volume_usd


# --- Quantitative helpers ---

def implied_probability(price: float) -> float:
    return max(0.0, min(1.0, price))


def expected_value(price: float, implied_prob: float, payout: float = 1.0) -> float:
    return implied_prob * payout - price


def kelly_criterion(prob: float, odds: float) -> float:
    if prob <= 0 or prob >= 1 or odds <= 0:
        return 0.0
    q = 1.0 - prob
    f = (odds * prob - q) / odds
    return max(0.0, min(1.0, f))


def compute_kelly_for_price(price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    edge = 0.01
    prob = min(0.995, price + edge)
    odds = (1.0 - price) / price
    return kelly_criterion(prob, odds)


def volatility_adjusted_stop(
    entry_price: float, vol: float, base_stop_cents: float, multiplier: float = 2.0,
) -> float:
    base_stop = base_stop_cents / 100.0
    vol_adjustment = min(vol * multiplier, 0.10)
    adjusted = base_stop - vol_adjustment
    return max(0.50, min(entry_price - 0.05, adjusted))


# --- Entry signal ---

def entry_signal_from_price(
    point: PricePoint, cfg: BotConfig, series: PriceSeries | None = None,
) -> EntrySignal | None:
    threshold = cfg.strategy.entry_threshold_cents / 100.0
    candidates: list[tuple[Side, float]] = []
    if point.yes >= threshold:
        candidates.append((Side.YES, point.yes))
    if point.no >= threshold:
        candidates.append((Side.NO, point.no))
    if not candidates:
        return None

    side, price = max(candidates, key=lambda x: x[1])
    prob = implied_probability(price)
    ev = expected_value(price, prob)
    kf = compute_kelly_for_price(price)

    vel = 0.0
    vol = 0.0
    confidence = prob

    if series and len(series.points) >= 3:
        vel = series.velocity(side, window=5)
        vol = series.volatility(side, window=10)
        if vel >= 0 and vol < 0.02:
            confidence = min(0.99, confidence + 0.02)
        elif vel < -0.001 or vol > 0.05:
            confidence = max(0.5, confidence - 0.05)

    if ev < -0.05:
        return None

    reason = (f"{side.value}>={cfg.strategy.entry_threshold_cents:.0f}c "
              f"EV={ev:.4f} Kelly={kf:.3f} vel={vel:.5f}")
    return EntrySignal(side=side, price=price, reason=reason,
                       expected_value=ev, kelly_fraction=kf,
                       velocity=vel, volatility=vol, confidence=confidence)


# --- Stop loss ---

def stop_loss_hit(
    side: Side, point: PricePoint, cfg: BotConfig, series: PriceSeries | None = None,
) -> tuple[bool, float]:
    base_stop = cfg.strategy.stop_loss_cents / 100.0
    current = point.yes if side == Side.YES else point.no
    stop_level = base_stop
    if series and len(series.points) >= 5:
        vol = series.volatility(side, window=10)
        if vol > 0.03:
            stop_level = max(0.50, base_stop - 0.03)
    return current < stop_level, current


# --- Position sizing ---

def compute_position_size(
    price: float, kelly_fraction: float, max_dollars: float, bankroll: float,
) -> float:
    half_kelly = kelly_fraction * 0.5
    kelly_size = bankroll * half_kelly
    return min(kelly_size, max_dollars, bankroll)
