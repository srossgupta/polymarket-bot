"""Polymarket high-probability paper trading bot.

Scans markets closing within 24h, monitors at T-6 minutes,
enters at 95c+ YES/NO, stops at 70c. Self-correcting strategy.
"""

from polymarket_bot.core import BotConfig, load_config
from polymarket_bot.engine import run_backtest, run_daily_once, run_parameter_sweep

__all__ = ["BotConfig", "load_config", "run_backtest", "run_daily_once", "run_parameter_sweep"]
