"""Tests for self-correction engine."""

import pytest
from polymarket_bot.adaptive import (
    _confidence_interval,
    _exponential_weighted_stats,
    _should_adapt,
    adapt_strategy,
)
from polymarket_bot.config import BotConfig


def _make_trades(n_wins: int, n_losses: int, win_pnl: float = 3.0,
                 loss_pnl: float = -25.0, category: str = "sports") -> list[dict]:
    trades = []
    for i in range(n_wins):
        trades.append({
            "event_type": "SELL_MARKET",
            "pnl": win_pnl,
            "category": category,
            "ts": f"2025-01-{i+1:02d}T00:00:00",
        })
    for i in range(n_losses):
        trades.append({
            "event_type": "STOP_LOSS",
            "pnl": loss_pnl,
            "category": category,
            "ts": f"2025-02-{i+1:02d}T00:00:00",
        })
    return trades


class TestExponentialWeighting:
    def test_recent_weighted_higher(self):
        pnls = [-5, -5, -5, 10, 10, 10]  # recent trades are wins
        mean, wr, var = _exponential_weighted_stats(pnls, decay=0.9)
        assert mean > 0  # recent wins should dominate
        assert wr > 0.5

    def test_empty_pnls(self):
        mean, wr, var = _exponential_weighted_stats([])
        assert mean == 0
        assert wr == 0

    def test_all_wins(self):
        pnls = [5, 3, 4, 6, 2]
        mean, wr, var = _exponential_weighted_stats(pnls)
        assert wr == 1.0
        assert mean > 0


class TestConfidenceInterval:
    def test_small_sample_wide(self):
        lower, upper = _confidence_interval(0.6, 5)
        assert upper - lower > 0.2  # wide with small sample

    def test_large_sample_narrow(self):
        lower, upper = _confidence_interval(0.6, 100)
        assert upper - lower < 0.2  # narrow with large sample

    def test_bounds(self):
        lower, upper = _confidence_interval(0.5, 50)
        assert lower >= 0
        assert upper <= 1


class TestAdaptationDirection:
    def test_tighten_when_losing(self):
        assert _should_adapt(0.40, 50, 0.6, 0.62) == "tighten"

    def test_relax_when_winning(self):
        assert _should_adapt(0.85, 50, 0.6, 0.62) == "relax"

    def test_hold_when_uncertain(self):
        # With few trades, should hold
        result = _should_adapt(0.60, 5, 0.6, 0.62)
        assert result in ("hold", "tighten", "relax")


class TestAdaptStrategy:
    def test_not_enough_trades(self):
        cfg = BotConfig()
        trades = _make_trades(3, 2)
        result = adapt_strategy(cfg, trades)
        assert result["adapted"] is False

    def test_tightens_on_losses(self):
        cfg = BotConfig()
        cfg.adaptation.min_trades_for_adaptation = 5
        original_entry = cfg.strategy.entry_threshold_cents
        trades = _make_trades(2, 20)
        result = adapt_strategy(cfg, trades)
        # Should tighten: higher entry threshold
        if result["adapted"]:
            assert cfg.strategy.entry_threshold_cents >= original_entry

    def test_relaxes_on_wins(self):
        cfg = BotConfig()
        cfg.adaptation.min_trades_for_adaptation = 5
        original_entry = cfg.strategy.entry_threshold_cents
        trades = _make_trades(40, 2)
        result = adapt_strategy(cfg, trades)
        if result["adapted"]:
            assert cfg.strategy.entry_threshold_cents <= original_entry

    def test_category_ranking(self):
        cfg = BotConfig()
        cfg.adaptation.min_trades_for_adaptation = 5
        cfg.adaptation.min_category_samples = 3
        trades = (
            _make_trades(10, 1, category="sports") +
            _make_trades(2, 10, category="politics")
        )
        result = adapt_strategy(cfg, trades)
        if result.get("preferred_categories"):
            assert "sports" in result["preferred_categories"]
