"""Tests for strategy module: entry signals, stop loss, Kelly, EV, crypto filter."""

import pytest
from datetime import datetime, timedelta, timezone

from polymarket_bot.core import BotConfig, Market, PricePoint, PriceSeries, Side
from polymarket_bot.trading.strategy import (
    compute_kelly_for_price, compute_position_size, entry_signal_from_price,
    expected_value, implied_probability, is_crypto_market, kelly_criterion,
    select_markets_for_next_24h, should_wake_for_market, stop_loss_hit,
    volatility_adjusted_stop,
)


def _make_market(market_id="mkt1", question="Will X happen?", end_time=None,
                 volume=200_000, category="politics", active=True) -> Market:
    if end_time is None:
        end_time = datetime.now(timezone.utc) + timedelta(hours=12)
    return Market(market_id=market_id, question=question, end_time=end_time,
                  volume_usd=volume, category=category, yes_token_id="yes_tok",
                  no_token_id="no_tok", active=active)


class TestCryptoFilter:
    def test_crypto_keywords_detected(self):
        assert is_crypto_market(_make_market(question="Will Bitcoin reach 100k?", category="crypto"))

    def test_ethereum_detected(self):
        assert is_crypto_market(_make_market(question="Will ETH price increase?", category="defi"))

    def test_non_crypto_passes(self):
        assert not is_crypto_market(_make_market(question="Will Team A win?", category="sports"))

    def test_slug_checked(self):
        m = _make_market(question="Price prediction", category="finance")
        m.slug = "bitcoin-100k-prediction"
        assert is_crypto_market(m)


class TestMarketSelection:
    def test_selects_within_horizon(self):
        now = datetime.now(timezone.utc)
        cfg = BotConfig()
        markets = [_make_market("m1", end_time=now + timedelta(hours=10)),
                   _make_market("m2", end_time=now + timedelta(hours=30)),
                   _make_market("m3", end_time=now + timedelta(hours=5))]
        selected = select_markets_for_next_24h(markets, cfg, now=now)
        ids = [m.market_id for m in selected]
        assert "m1" in ids and "m3" in ids and "m2" not in ids

    def test_excludes_crypto(self):
        now = datetime.now(timezone.utc)
        cfg = BotConfig()
        markets = [_make_market("m1", category="sports", end_time=now + timedelta(hours=5)),
                   _make_market("m2", question="Bitcoin price", category="crypto", end_time=now + timedelta(hours=5))]
        assert len(select_markets_for_next_24h(markets, cfg, now=now)) == 1

    def test_excludes_inactive(self):
        now = datetime.now(timezone.utc)
        assert select_markets_for_next_24h(
            [_make_market("m1", active=False, end_time=now + timedelta(hours=5))], BotConfig(), now=now) == []

    def test_sorted_by_end_time(self):
        now = datetime.now(timezone.utc)
        markets = [_make_market("late", end_time=now + timedelta(hours=20)),
                   _make_market("early", end_time=now + timedelta(hours=2)),
                   _make_market("mid", end_time=now + timedelta(hours=10))]
        assert [m.market_id for m in select_markets_for_next_24h(markets, BotConfig(), now=now)] == ["early", "mid", "late"]


class TestWakeLogic:
    def test_wake_at_t_minus_6(self):
        m = _make_market(end_time=datetime.now(timezone.utc) + timedelta(minutes=5))
        assert should_wake_for_market(m, BotConfig())

    def test_no_wake_too_early(self):
        m = _make_market(end_time=datetime.now(timezone.utc) + timedelta(minutes=30))
        assert not should_wake_for_market(m, BotConfig())


class TestEntrySignal:
    def test_yes_signal_above_threshold(self):
        point = PricePoint(ts=datetime.now(timezone.utc), yes=0.96, no=0.04)
        signal = entry_signal_from_price(point, BotConfig())
        assert signal is not None and signal.side == Side.YES and signal.price == 0.96

    def test_no_signal_above_threshold(self):
        point = PricePoint(ts=datetime.now(timezone.utc), yes=0.03, no=0.97)
        signal = entry_signal_from_price(point, BotConfig())
        assert signal is not None and signal.side == Side.NO

    def test_no_signal_below_threshold(self):
        assert entry_signal_from_price(PricePoint(ts=datetime.now(timezone.utc), yes=0.60, no=0.40), BotConfig()) is None

    def test_prefers_higher_price(self):
        cfg = BotConfig()
        cfg.strategy.entry_threshold_cents = 50
        signal = entry_signal_from_price(PricePoint(ts=datetime.now(timezone.utc), yes=0.96, no=0.55), cfg)
        assert signal is not None and signal.side == Side.YES

    def test_signal_with_series(self):
        series = PriceSeries(market_id="mkt1")
        ts = datetime.now(timezone.utc)
        for i in range(10):
            series.add(PricePoint(ts=ts + timedelta(seconds=i), yes=0.93 + i * 0.005, no=0.07 - i * 0.005))
        signal = entry_signal_from_price(PricePoint(ts=ts + timedelta(seconds=10), yes=0.96, no=0.04), BotConfig(), series)
        assert signal is not None and signal.velocity > 0


class TestStopLoss:
    def test_stop_triggered(self):
        hit, px = stop_loss_hit(Side.YES, PricePoint(ts=datetime.now(timezone.utc), yes=0.65, no=0.35), BotConfig())
        assert hit and px == 0.65

    def test_stop_not_triggered(self):
        hit, _ = stop_loss_hit(Side.YES, PricePoint(ts=datetime.now(timezone.utc), yes=0.85, no=0.15), BotConfig())
        assert not hit

    def test_stop_for_no_side(self):
        hit, _ = stop_loss_hit(Side.NO, PricePoint(ts=datetime.now(timezone.utc), yes=0.40, no=0.60), BotConfig())
        assert hit


class TestQuantitativeFunctions:
    def test_implied_probability_clamp(self):
        assert implied_probability(0.95) == 0.95 and implied_probability(1.5) == 1.0 and implied_probability(-0.1) == 0.0

    def test_expected_value(self):
        assert abs(expected_value(0.95, 0.95)) < 0.001

    def test_kelly_positive(self):
        assert kelly_criterion(0.95, 0.0526) >= 0

    def test_kelly_zero_for_bad_bet(self):
        assert kelly_criterion(0.3, 1.0) == 0.0

    def test_compute_kelly_for_price(self):
        assert 0 < compute_kelly_for_price(0.95) <= 1

    def test_position_sizing(self):
        size = compute_position_size(0.95, 0.5, 100, 2000)
        assert 0 < size <= 100

    def test_volatility_adjusted_stop(self):
        assert 0.50 <= volatility_adjusted_stop(0.96, 0.01, 70) <= 0.91


class TestPriceSeries:
    def test_velocity_positive(self):
        series = PriceSeries(market_id="test")
        ts = datetime.now(timezone.utc)
        for i in range(10):
            series.add(PricePoint(ts=ts + timedelta(seconds=i * 2), yes=0.90 + i * 0.01, no=0.10 - i * 0.01))
        assert series.velocity(Side.YES) > 0

    def test_velocity_negative(self):
        series = PriceSeries(market_id="test")
        ts = datetime.now(timezone.utc)
        for i in range(10):
            series.add(PricePoint(ts=ts + timedelta(seconds=i * 2), yes=0.99 - i * 0.01, no=0.01 + i * 0.01))
        assert series.velocity(Side.YES) < 0

    def test_volatility_nonzero(self):
        series = PriceSeries(market_id="test")
        ts = datetime.now(timezone.utc)
        for i in range(20):
            noise = 0.01 if i % 2 == 0 else -0.01
            series.add(PricePoint(ts=ts + timedelta(seconds=i), yes=0.95 + noise, no=0.05 - noise))
        assert series.volatility(Side.YES) > 0

    def test_max_points_trimmed(self):
        series = PriceSeries(market_id="test", max_points=5)
        ts = datetime.now(timezone.utc)
        for i in range(10):
            series.add(PricePoint(ts=ts + timedelta(seconds=i), yes=0.5, no=0.5))
        assert len(series.points) == 5
