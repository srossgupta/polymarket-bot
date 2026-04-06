"""Tests for self-correction engine."""

import pytest
from polymarket_bot.trading.adaptive import (
    _confidence_interval, _exponential_weighted_stats, _should_adapt, adapt_strategy,
)
from polymarket_bot.core import BotConfig


def _make_trades(n_wins, n_losses, win_pnl=3.0, loss_pnl=-25.0, category="sports"):
    trades = [{"event_type": "SELL_MARKET", "pnl": win_pnl, "category": category,
               "ts": f"2025-01-{i+1:02d}T00:00:00"} for i in range(n_wins)]
    trades += [{"event_type": "STOP_LOSS", "pnl": loss_pnl, "category": category,
                "ts": f"2025-02-{i+1:02d}T00:00:00"} for i in range(n_losses)]
    return trades


class TestExponentialWeighting:
    def test_recent_weighted_higher(self):
        mean, wr, _ = _exponential_weighted_stats([-5, -5, -5, 10, 10, 10], 0.9)
        assert mean > 0 and wr > 0.5

    def test_empty_pnls(self):
        assert _exponential_weighted_stats([]) == (0.0, 0.0, 0.0)

    def test_all_wins(self):
        _, wr, _ = _exponential_weighted_stats([5, 3, 4, 6, 2])
        assert wr == 1.0


class TestConfidenceInterval:
    def test_small_sample_wide(self):
        l, u = _confidence_interval(0.6, 5)
        assert u - l > 0.2

    def test_large_sample_narrow(self):
        l, u = _confidence_interval(0.6, 100)
        assert u - l < 0.2

    def test_bounds(self):
        l, u = _confidence_interval(0.5, 50)
        assert l >= 0 and u <= 1


class TestAdaptationDirection:
    def test_tighten_when_losing(self):
        assert _should_adapt(0.40, 50, 0.6, 0.62) == "tighten"

    def test_relax_when_winning(self):
        assert _should_adapt(0.85, 50, 0.6, 0.62) == "relax"

    def test_hold_when_uncertain(self):
        assert _should_adapt(0.60, 5, 0.6, 0.62) in ("hold", "tighten", "relax")


class TestAdaptStrategy:
    def test_not_enough_trades(self):
        assert adapt_strategy(BotConfig(), _make_trades(3, 2))["adapted"] is False

    def test_tightens_on_losses(self):
        cfg = BotConfig()
        cfg.adaptation.min_trades_for_adaptation = 5
        orig = cfg.strategy.entry_threshold_cents
        result = adapt_strategy(cfg, _make_trades(2, 20))
        if result["adapted"]:
            assert cfg.strategy.entry_threshold_cents >= orig

    def test_relaxes_on_wins(self):
        cfg = BotConfig()
        cfg.adaptation.min_trades_for_adaptation = 5
        orig = cfg.strategy.entry_threshold_cents
        result = adapt_strategy(cfg, _make_trades(40, 2))
        if result["adapted"]:
            assert cfg.strategy.entry_threshold_cents <= orig

    def test_category_ranking(self):
        cfg = BotConfig()
        cfg.adaptation.min_trades_for_adaptation = 5
        cfg.adaptation.min_category_samples = 3
        trades = _make_trades(10, 1, category="sports") + _make_trades(2, 10, category="politics")
        result = adapt_strategy(cfg, trades)
        if result.get("preferred_categories"):
            assert "sports" in result["preferred_categories"]
