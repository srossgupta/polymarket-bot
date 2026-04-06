"""Backtesting: snapshot replay, parameter sweep, Monte Carlo."""

from .engine import (
    BacktestResult,
    BacktestTrade,
    build_synthetic_snapshots,
    monte_carlo_simulation,
    parameter_sweep,
    run_snapshot_backtest,
)

__all__ = [
    "BacktestResult", "BacktestTrade", "build_synthetic_snapshots",
    "monte_carlo_simulation", "parameter_sweep", "run_snapshot_backtest",
]
