"""Tests for backtesting engine."""

import pytest
from datetime import datetime, timedelta, timezone

from polymarket_bot.backtest import (BacktestResult, build_synthetic_snapshots,
                                      monte_carlo_simulation, parameter_sweep,
                                      run_snapshot_backtest)
from polymarket_bot.core import BotConfig


def _make_snapshots_winning():
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = base + timedelta(minutes=10)
    return [{"market_id": "win_mkt", "question": "Win market", "category": "sports",
             "end_time": end.isoformat(), "ts": (base + timedelta(seconds=i * 60)).isoformat(),
             "yes": yes, "no": round(1 - yes, 4)}
            for i, yes in enumerate([0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99])]


def _make_snapshots_losing():
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = base + timedelta(minutes=10)
    return [{"market_id": "loss_mkt", "question": "Loss market", "category": "politics",
             "end_time": end.isoformat(), "ts": (base + timedelta(seconds=i * 60)).isoformat(),
             "yes": yes, "no": round(1 - yes, 4)}
            for i, yes in enumerate([0.93, 0.96, 0.85, 0.72, 0.65])]


class TestSnapshotBacktest:
    def test_winning_trade(self):
        result = run_snapshot_backtest(BotConfig(), _make_snapshots_winning())
        assert result.trades >= 1 and result.total_pnl > 0 and result.wins >= 1

    def test_losing_trade_stop_loss(self):
        result = run_snapshot_backtest(BotConfig(), _make_snapshots_losing())
        assert result.trades >= 1 and result.losses >= 1

    def test_mixed_markets(self):
        assert run_snapshot_backtest(BotConfig(), _make_snapshots_winning() + _make_snapshots_losing()).trades >= 2

    def test_category_stats_populated(self):
        result = run_snapshot_backtest(BotConfig(), _make_snapshots_winning() + _make_snapshots_losing())
        assert "sports" in result.category_stats or "politics" in result.category_stats

    def test_max_drawdown_nonnegative(self):
        assert run_snapshot_backtest(BotConfig(), _make_snapshots_losing()).max_drawdown >= 0

    def test_empty_snapshots(self):
        result = run_snapshot_backtest(BotConfig(), [])
        assert result.trades == 0 and result.total_pnl == 0


class TestSyntheticData:
    def test_generates_data(self):
        snapshots = build_synthetic_snapshots(num_markets=5)
        assert len(snapshots) > 0 and len(set(s["market_id"] for s in snapshots)) == 5

    def test_deterministic(self):
        s1, s2 = build_synthetic_snapshots(3), build_synthetic_snapshots(3)
        assert len(s1) == len(s2) and s1[0]["yes"] == s2[0]["yes"]

    def test_backtest_on_synthetic(self):
        assert isinstance(run_snapshot_backtest(BotConfig(), build_synthetic_snapshots(10)), BacktestResult)


class TestParameterSweep:
    def test_sweep_returns_results(self):
        results = parameter_sweep(_make_snapshots_winning() + _make_snapshots_losing(),
                                   entry_range=(93, 97, 2), stop_range=(65, 75, 5), wake_range=(6, 6, 1))
        assert len(results) > 0
        if len(results) >= 2:
            assert results[0]["total_pnl"] >= results[-1]["total_pnl"]

    def test_sweep_has_params(self):
        results = parameter_sweep(_make_snapshots_winning(), entry_range=(95, 95, 1),
                                   stop_range=(70, 70, 1), wake_range=(6, 6, 1))
        assert len(results) == 1 and results[0]["entry_cents"] == 95


class TestMonteCarlo:
    def test_with_positive_trades(self):
        result = monte_carlo_simulation([5.0, 3.0, -2.0, 4.0, 1.0, -1.0, 6.0, 2.0], 100, 50)
        assert result["probability_of_profit"] > 0

    def test_with_all_losses(self):
        result = monte_carlo_simulation([-10.0, -5.0, -8.0, -3.0], 500, 500)
        assert result["median_final_value"] < 2000

    def test_empty_trades(self):
        assert "error" in monte_carlo_simulation([])
