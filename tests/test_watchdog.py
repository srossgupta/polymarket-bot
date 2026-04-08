"""Tests for LiveWatchdog."""
from __future__ import annotations

import time
from threading import Event
from unittest.mock import patch

from polymarket_bot.watchdog import (
    COLD_STREAK_LIMIT,
    DANGER_WIN_RATE,
    LiveWatchdog,
    _interruptible_sleep,
)


# ── LiveWatchdog unit tests ───────────────────────────────────────────────────

def test_initial_state():
    wd = LiveWatchdog()
    assert not wd.should_restart()
    assert wd.checks_done == 0
    assert wd.restarts_done == 0


def test_trigger_restart_sets_flag():
    wd = LiveWatchdog()
    wd._trigger_restart(reason="test", severity="TEST")
    assert wd.should_restart()


def test_acknowledge_restart_clears_flag_and_increments():
    wd = LiveWatchdog()
    wd._trigger_restart(reason="test", severity="TEST")
    assert wd.should_restart()
    wd.acknowledge_restart()
    assert not wd.should_restart()
    assert wd.restarts_done == 1


def test_no_restart_when_too_few_trades():
    wd = LiveWatchdog()
    # Only 2 trades — below MIN_TRADES_TO_JUDGE (5)
    trades = [{"pnl": -10}, {"pnl": -10}]
    with patch("polymarket_bot.watchdog.load_closed_trades", return_value=trades):
        wd._check_health()
    assert not wd.should_restart()
    assert wd.checks_done == 1


def test_restart_triggered_on_low_win_rate():
    wd = LiveWatchdog()
    # 1 win, 9 losses → 10% win rate < 40% danger threshold
    trades = [{"pnl": 5, "ts": "2024-01-01T00:00:00"}] + \
             [{"pnl": -5, "ts": "2024-01-01T00:00:01"}] * 9
    with patch("polymarket_bot.watchdog.load_closed_trades", return_value=trades):
        wd._check_health()
    assert wd.should_restart()


def test_restart_triggered_on_cold_streak():
    wd = LiveWatchdog()
    # Win rate is fine (7/10 = 70%) but last 3 are all losses
    trades = [{"pnl": 5, "ts": f"2024-01-01T00:00:0{i}"} for i in range(7)] + \
             [{"pnl": -5, "ts": f"2024-01-01T00:00:0{i}"} for i in range(7, 10)]
    with patch("polymarket_bot.watchdog.load_closed_trades", return_value=trades):
        wd._check_health()
    assert wd.should_restart()


def test_no_restart_when_healthy():
    wd = LiveWatchdog()
    # 8 wins, 2 losses = 80% win rate, last 3 are wins
    trades = [{"pnl": -5, "ts": "2024-01-01T00:00:00"},
              {"pnl": -5, "ts": "2024-01-01T00:00:01"}] + \
             [{"pnl": 5, "ts": f"2024-01-01T00:00:{i+2:02d}"} for i in range(8)]
    with patch("polymarket_bot.watchdog.load_closed_trades", return_value=trades):
        wd._check_health()
    assert not wd.should_restart()


def test_start_stop():
    wd = LiveWatchdog()
    wd.start()
    assert wd._thread is not None
    assert wd._thread.is_alive()
    wd.stop()
    assert not wd._thread.is_alive()


# ── _interruptible_sleep tests ────────────────────────────────────────────────

def test_interruptible_sleep_completes():
    shutdown = Event()
    start = time.monotonic()
    _interruptible_sleep(2, shutdown)
    elapsed = time.monotonic() - start
    assert elapsed >= 1.9


def test_interruptible_sleep_interrupted_by_shutdown():
    shutdown = Event()
    # Set the flag immediately — sleep should return almost instantly
    shutdown.set()
    start = time.monotonic()
    _interruptible_sleep(60, shutdown)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0


def test_interruptible_sleep_interrupted_by_watchdog():
    shutdown = Event()
    wd = LiveWatchdog()
    wd._trigger_restart(reason="test", severity="TEST")
    start = time.monotonic()
    _interruptible_sleep(60, shutdown, wd)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0
