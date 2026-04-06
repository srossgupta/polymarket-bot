"""Tests for backtesting engine: replay, parameter sweep, Monte Carlo."""

import pytest
from datetime import datetime, timedelta, timezone

from polymarket_bot.backtest import (
    BacktestResult,
    build_synthetic_snapshots,
    monte_carlo_simulation,
    parameter_sweep,
    run_snapshot_backtest,
)
from polymarket_bot.config import BotConfig


def _make_snapshots_winning() -> list[dict]:
    """Market where YES climbs above 95c and stays there → win."""
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = base + timedelta(minutes=10)
    rows = []
    prices = [0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99]
    for i, yes in enumerate(prices):
        rows.append({
            "market_id": "win_mkt",
            "question": "Win market",
            "category": "sports",
            "end_time": end.isoformat(),
            "ts": (base + timedelta(seconds=i * 60)).isoformat(),
            "yes": yes,
            "no": round(1 - yes, 4),
        })
    return rows


def _make_snapshots_losing() -> list[dict]:
    """Market where YES crosses 95c then crashes below 70c → stop loss."""
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = base + timedelta(minutes=10)
    rows = []
    prices = [0.93, 0.96, 0.85, 0.72, 0.65]
    for i, yes in enumerate(prices):
        rows.append({
            "market_id": "loss_mkt",
            "question": "Loss market",
            "category": "politics",
            "end_time": end.isoformat(),
            "ts": (base + timedelta(seconds=i * 60)).isoformat(),
            "yes": yes,
            "no": round(1 - yes, 4),
        })
    return rows


class TestSnapshotBacktest:
    def test_winning_trade(self):
        cfg = BotConfig()
        result = run_snapshot_backtest(cfg, _make_snapshots_winning())
        assert result.trades >= 1
        assert result.total_pnl > 0
        assert result.wins >= 1

    def test_losing_trade_stop_loss(self):
        cfg = BotConfig()
        result = run_snapshot_backtest(cfg, _make_snapshots_losing())
        assert result.trades >= 1
        assert result.losses >= 1

    def test_mixed_markets(self):
        cfg = BotConfig()
        snapshots = _make_snapshots_winning() + _make_snapshots_losing()
        result = run_snapshot_backtest(cfg, snapshots)
        assert result.trades >= 2

    def test_category_stats_populated(self):
        cfg = BotConfig()
        snapshots = _make_snapshots_winning() + _make_snapshots_losing()
        result = run_snapshot_backtest(cfg, snapshots)
        assert "sports" in result.category_stats or "politics" in result.category_stats

    def test_max_drawdown_nonnegative(self):
        cfg = BotConfig()
        result = run_snapshot_backtest(cfg, _make_snapshots_losing())
        assert result.max_drawdown >= 0

    def test_empty_snapshots(self):
        cfg = BotConfig()
        result = run_snapshot_backtest(cfg, [])
        assert result.trades == 0
        assert result.total_pnl == 0


class TestSyntheticData:
    def test_generates_data(self):
        snapshots = build_synthetic_snapshots(num_markets=5)
        assert len(snapshots) > 0
        market_ids = set(s["market_id"] for s in snapshots)
        assert len(market_ids) == 5

    def test_deterministic(self):
        s1 = build_synthetic_snapshots(num_markets=3)
        s2 = build_synthetic_snapshots(num_markets=3)
        assert len(s1) == len(s2)
        assert s1[0]["yes"] == s2[0]["yes"]  # same seed

    def test_backtest_on_synthetic(self):
        cfg = BotConfig()
        snapshots = build_synthetic_snapshots(num_markets=10)
        result = run_snapshot_backtest(cfg, snapshots)
        # Should produce at least some trades on 10 markets
        assert isinstance(result, BacktestResult)


class TestParameterSweep:
    def test_sweep_returns_results(self):
        snapshots = _make_snapshots_winning() + _make_snapshots_losing()
        results = parameter_sweep(
            snapshots,
            entry_range=(93, 97, 2),
            stop_range=(65, 75, 5),
            wake_range=(6, 6, 1),
        )
        assert len(results) > 0
        # Should be sorted by PnL descending
        if len(results) >= 2:
            assert results[0]["total_pnl"] >= results[-1]["total_pnl"]

    def test_sweep_has_params(self):
        snapshots = _make_snapshots_winning()
        results = parameter_sweep(
            snapshots,
            entry_range=(95, 95, 1),
            stop_range=(70, 70, 1),
            wake_range=(6, 6, 1),
        )
        assert len(results) == 1
        assert results[0]["entry_cents"] == 95


class TestMonteCarlo:
    def test_with_positive_trades(self):
        pnls = [5.0, 3.0, -2.0, 4.0, 1.0, -1.0, 6.0, 2.0]
        result = monte_carlo_simulation(pnls, num_simulations=100, trades_per_sim=50)
        assert result["simulations"] == 100
        assert result["probability_of_profit"] > 0
        assert result["median_final_value"] > 0

    def test_with_all_losses(self):
        pnls = [-10.0, -5.0, -8.0, -3.0]
        result = monte_carlo_simulation(pnls, num_simulations=500, trades_per_sim=500)
        # With all negative trades and enough simulation, should see some ruin or low values
        assert result["median_final_value"] < 2000  # should lose money

    def test_empty_trades(self):
        result = monte_carlo_simulation([])
        assert "error" in result
