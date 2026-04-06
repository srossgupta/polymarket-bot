"""Tests for strategy module: entry signals, stop loss, Kelly, EV, crypto filter."""

import pytest
from datetime import datetime, timedelta, timezone

from polymarket_bot.config import BotConfig
from polymarket_bot.models import Market, PricePoint, PriceSeries, Side
from polymarket_bot.strategy import (
    compute_kelly_for_price,
    compute_position_size,
    entry_signal_from_price,
    expected_value,
    implied_probability,
    is_crypto_market,
    kelly_criterion,
    select_markets_for_next_24h,
    should_wake_for_market,
    stop_loss_hit,
    volatility_adjusted_stop,
)


def _make_market(
    market_id="mkt1",
    question="Will X happen?",
    end_time=None,
    volume=200_000,
    category="politics",
    active=True,
) -> Market:
    if end_time is None:
        end_time = datetime.now(timezone.utc) + timedelta(hours=12)
    return Market(
        market_id=market_id,
        question=question,
        end_time=end_time,
        volume_usd=volume,
        category=category,
        yes_token_id="yes_tok",
        no_token_id="no_tok",
        active=active,
    )


class TestCryptoFilter:
    def test_crypto_keywords_detected(self):
        m = _make_market(question="Will Bitcoin reach 100k?", category="crypto")
        assert is_crypto_market(m) is True

    def test_ethereum_detected(self):
        m = _make_market(question="Will ETH price increase?", category="defi")
        assert is_crypto_market(m) is True

    def test_non_crypto_passes(self):
        m = _make_market(question="Will Team A win the championship?", category="sports")
        assert is_crypto_market(m) is False

    def test_slug_checked(self):
        m = _make_market(question="Price prediction", category="finance")
        m.slug = "bitcoin-100k-prediction"
        assert is_crypto_market(m) is True


class TestMarketSelection:
    def test_selects_within_horizon(self):
        now = datetime.now(timezone.utc)
        cfg = BotConfig()
        markets = [
            _make_market("m1", end_time=now + timedelta(hours=10)),
            _make_market("m2", end_time=now + timedelta(hours=30)),  # beyond 24h
            _make_market("m3", end_time=now + timedelta(hours=5)),
        ]
        selected = select_markets_for_next_24h(markets, cfg, now=now)
        ids = [m.market_id for m in selected]
        assert "m1" in ids
        assert "m3" in ids
        assert "m2" not in ids

    def test_excludes_crypto(self):
        now = datetime.now(timezone.utc)
        cfg = BotConfig()
        markets = [
            _make_market("m1", category="sports", end_time=now + timedelta(hours=5)),
            _make_market("m2", question="Bitcoin price", category="crypto",
                         end_time=now + timedelta(hours=5)),
        ]
        selected = select_markets_for_next_24h(markets, cfg, now=now)
        assert len(selected) == 1
        assert selected[0].market_id == "m1"

    def test_excludes_inactive(self):
        now = datetime.now(timezone.utc)
        cfg = BotConfig()
        markets = [_make_market("m1", active=False, end_time=now + timedelta(hours=5))]
        assert select_markets_for_next_24h(markets, cfg, now=now) == []

    def test_sorted_by_end_time(self):
        now = datetime.now(timezone.utc)
        cfg = BotConfig()
        markets = [
            _make_market("late", end_time=now + timedelta(hours=20)),
            _make_market("early", end_time=now + timedelta(hours=2)),
            _make_market("mid", end_time=now + timedelta(hours=10)),
        ]
        selected = select_markets_for_next_24h(markets, cfg, now=now)
        assert [m.market_id for m in selected] == ["early", "mid", "late"]


class TestWakeLogic:
    def test_wake_at_t_minus_6(self):
        cfg = BotConfig()
        end = datetime.now(timezone.utc) + timedelta(minutes=5)
        m = _make_market(end_time=end)
        assert should_wake_for_market(m, cfg) is True

    def test_no_wake_too_early(self):
        cfg = BotConfig()
        end = datetime.now(timezone.utc) + timedelta(minutes=30)
        m = _make_market(end_time=end)
        assert should_wake_for_market(m, cfg) is False


class TestEntrySignal:
    def test_yes_signal_above_threshold(self):
        cfg = BotConfig()
        point = PricePoint(ts=datetime.now(timezone.utc), yes=0.96, no=0.04)
        signal = entry_signal_from_price(point, cfg)
        assert signal is not None
        assert signal.side == Side.YES
        assert signal.price == 0.96

    def test_no_signal_above_threshold(self):
        cfg = BotConfig()
        point = PricePoint(ts=datetime.now(timezone.utc), yes=0.03, no=0.97)
        signal = entry_signal_from_price(point, cfg)
        assert signal is not None
        assert signal.side == Side.NO

    def test_no_signal_below_threshold(self):
        cfg = BotConfig()
        point = PricePoint(ts=datetime.now(timezone.utc), yes=0.60, no=0.40)
        signal = entry_signal_from_price(point, cfg)
        assert signal is None

    def test_prefers_higher_price(self):
        """If both sides somehow above threshold, pick the higher one."""
        cfg = BotConfig()
        cfg.strategy.entry_threshold_cents = 50  # low threshold for test
        point = PricePoint(ts=datetime.now(timezone.utc), yes=0.96, no=0.55)
        signal = entry_signal_from_price(point, cfg)
        assert signal is not None
        assert signal.side == Side.YES

    def test_signal_with_series(self):
        cfg = BotConfig()
        series = PriceSeries(market_id="mkt1")
        ts = datetime.now(timezone.utc)
        for i in range(10):
            series.add(PricePoint(
                ts=ts + timedelta(seconds=i),
                yes=0.93 + i * 0.005,
                no=0.07 - i * 0.005,
            ))
        point = PricePoint(ts=ts + timedelta(seconds=10), yes=0.96, no=0.04)
        signal = entry_signal_from_price(point, cfg, series)
        assert signal is not None
        assert signal.velocity > 0  # price was rising


class TestStopLoss:
    def test_stop_triggered(self):
        cfg = BotConfig()
        point = PricePoint(ts=datetime.now(timezone.utc), yes=0.65, no=0.35)
        hit, px = stop_loss_hit(Side.YES, point, cfg)
        assert hit is True
        assert px == 0.65

    def test_stop_not_triggered(self):
        cfg = BotConfig()
        point = PricePoint(ts=datetime.now(timezone.utc), yes=0.85, no=0.15)
        hit, px = stop_loss_hit(Side.YES, point, cfg)
        assert hit is False

    def test_stop_for_no_side(self):
        cfg = BotConfig()
        point = PricePoint(ts=datetime.now(timezone.utc), yes=0.40, no=0.60)
        hit, _ = stop_loss_hit(Side.NO, point, cfg)
        assert hit is True  # 0.60 < 0.70


class TestQuantitativeFunctions:
    def test_implied_probability_clamp(self):
        assert implied_probability(0.95) == 0.95
        assert implied_probability(1.5) == 1.0
        assert implied_probability(-0.1) == 0.0

    def test_expected_value(self):
        # Buying at 0.95 with true prob 0.95: EV = 0.95*1 - 0.95 = 0
        ev = expected_value(0.95, 0.95)
        assert abs(ev) < 0.001

    def test_kelly_positive(self):
        # prob=0.95, odds = 0.05/0.95 ≈ 0.0526
        kf = kelly_criterion(0.95, 0.0526)
        assert kf >= 0

    def test_kelly_zero_for_bad_bet(self):
        kf = kelly_criterion(0.3, 1.0)  # prob too low
        assert kf == 0.0

    def test_compute_kelly_for_price(self):
        kf = compute_kelly_for_price(0.95)
        assert 0 < kf <= 1

    def test_position_sizing(self):
        size = compute_position_size(0.95, 0.5, 100, 2000)
        assert size <= 100
        assert size > 0

    def test_volatility_adjusted_stop(self):
        stop = volatility_adjusted_stop(0.96, 0.01, 70)
        assert 0.50 <= stop <= 0.91


class TestPriceSeries:
    def test_velocity_positive(self):
        series = PriceSeries(market_id="test")
        ts = datetime.now(timezone.utc)
        for i in range(10):
            series.add(PricePoint(
                ts=ts + timedelta(seconds=i * 2),
                yes=0.90 + i * 0.01, no=0.10 - i * 0.01))
        vel = series.velocity(Side.YES)
        assert vel > 0

    def test_velocity_negative(self):
        series = PriceSeries(market_id="test")
        ts = datetime.now(timezone.utc)
        for i in range(10):
            series.add(PricePoint(
                ts=ts + timedelta(seconds=i * 2),
                yes=0.99 - i * 0.01, no=0.01 + i * 0.01))
        vel = series.velocity(Side.YES)
        assert vel < 0

    def test_volatility_nonzero(self):
        series = PriceSeries(market_id="test")
        ts = datetime.now(timezone.utc)
        for i in range(20):
            noise = 0.01 if i % 2 == 0 else -0.01
            series.add(PricePoint(
                ts=ts + timedelta(seconds=i),
                yes=0.95 + noise, no=0.05 - noise))
        vol = series.volatility(Side.YES)
        assert vol > 0

    def test_max_points_trimmed(self):
        series = PriceSeries(market_id="test", max_points=5)
        ts = datetime.now(timezone.utc)
        for i in range(10):
            series.add(PricePoint(ts=ts + timedelta(seconds=i), yes=0.5, no=0.5))
        assert len(series.points) == 5
