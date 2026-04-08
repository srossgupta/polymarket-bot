"""
Live self-correcting watchdog.

Plain English of what this does:

  WHILE bot is running:
    every 30 seconds → peek at recent trades
    IF win rate is tanking  → STOP bot, tighten params, RESTART
    IF we're on a cold streak → STOP bot, raise entry bar, RESTART
    IF everything is fine    → keep running, do nothing

Think of it like a pit crew watching a race car:
  - Car is fine? Keep racing.
  - Tires going bad? Pull in, change tires, send it back out.
  - Car on fire? Stop completely and figure it out.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Any

from polymarket_bot.core import load_config
from polymarket_bot.data.storage import init_db, load_closed_trades
from polymarket_bot.trading.adaptive import adapt_strategy

logger = logging.getLogger("watchdog")


# ── What counts as "going bad" ────────────────────────────────────────────────

WATCH_INTERVAL_SECONDS = 30   # check every 30 seconds
MIN_TRADES_TO_JUDGE    = 5    # need at least 5 trades before we form an opinion
DANGER_WIN_RATE        = 0.40  # below 40% win rate = pull the car in
COLD_STREAK_LIMIT      = 3    # 3 losses in a row = pause and re-evaluate


# ── The actual watchdog ────────────────────────────────────────────────────────

class LiveWatchdog:
    """
    Runs in a background thread while the main bot loop runs.
    Checks performance every WATCH_INTERVAL_SECONDS.
    If things look bad:  flags the bot to restart with new params.
    If things look OK:   does nothing, lets it run.
    """

    def __init__(self):
        self._stop_event   = Event()   # signal: "stop the watchdog thread"
        self._restart_flag = Event()   # signal: "bot should restart"
        self._thread: Thread | None = None
        self.checks_done    = 0
        self.restarts_done  = 0

    # ── Start / stop ──────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = Thread(target=self._loop, daemon=True, name="watchdog")
        self._thread.start()
        logger.info("🐕 Watchdog started (checking every %ds)", WATCH_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("🐕 Watchdog stopped (%d checks, %d restarts)", self.checks_done, self.restarts_done)

    def should_restart(self) -> bool:
        """Bot's main loop calls this each iteration to ask: 'should I restart?'"""
        return self._restart_flag.is_set()

    def acknowledge_restart(self) -> None:
        """Bot calls this after it has restarted to clear the flag."""
        self._restart_flag.clear()
        self.restarts_done += 1

    # ── Main watchdog loop ────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            # wait() returns False on timeout (do a check), True if stop was signalled (exit)
            stopped = self._stop_event.wait(timeout=WATCH_INTERVAL_SECONDS)
            if stopped:
                break
            self._check_health()

    def _check_health(self) -> None:
        self.checks_done += 1
        trades = load_closed_trades(limit=20)  # only look at the 20 most recent

        if len(trades) < MIN_TRADES_TO_JUDGE:
            logger.info("🐕 [check %d] Only %d trades so far — too early to judge, watching...",
                        self.checks_done, len(trades))
            return

        # ── Check 1: overall win rate ─────────────────────────────────────────
        win_rate = sum(1 for t in trades if t.get("pnl", 0) > 0) / len(trades)
        total_pnl = sum(t.get("pnl", 0) for t in trades)

        # ── Check 2: recent losing streak ────────────────────────────────────
        recent = sorted(trades, key=lambda t: t.get("ts", ""), reverse=True)[:COLD_STREAK_LIMIT]
        consecutive_losses = all(t.get("pnl", 0) <= 0 for t in recent)

        logger.info(
            "🐕 [check %d] win_rate=%.0f%% | pnl=$%.2f | last_%d_all_losses=%s",
            self.checks_done, win_rate * 100, total_pnl,
            COLD_STREAK_LIMIT, consecutive_losses,
        )

        # ── Decision ──────────────────────────────────────────────────────────
        if win_rate < DANGER_WIN_RATE:
            self._trigger_restart(
                reason=f"Win rate {win_rate:.0%} is below danger threshold {DANGER_WIN_RATE:.0%}",
                severity="DANGER",
            )
        elif consecutive_losses:
            self._trigger_restart(
                reason=f"Last {COLD_STREAK_LIMIT} trades were all losses",
                severity="COLD_STREAK",
            )
        else:
            logger.info("🐕 All good — bot keeps running")

    def _trigger_restart(self, reason: str, severity: str) -> None:
        logger.warning("🚨 WATCHDOG [%s]: %s", severity, reason)
        logger.warning("🛑 Signalling bot to STOP → adapt params → RESTART")
        self._restart_flag.set()


# ── The main self-correcting run loop ─────────────────────────────────────────

def run_with_watchdog() -> None:
    """
    This is the top-level function you run.

    What happens:
      1. Bot starts running its daily cycle
      2. Watchdog starts watching in background
      3. If watchdog raises a flag → bot finishes current market → stops
      4. Strategy auto-adapts (entry/stop/wake params)
      5. Bot restarts with the new params
      6. Repeat forever

    To stop: Ctrl+C
    """

    # Setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    init_db()

    # Handle Ctrl+C gracefully
    shutdown = Event()
    def _handle_ctrl_c(sig, frame):
        print("\n\n⏹  Ctrl+C caught — shutting down cleanly...")
        shutdown.set()
    signal.signal(signal.SIGINT, _handle_ctrl_c)

    watchdog = LiveWatchdog()
    watchdog.start()

    cycle = 0

    print("""
╔══════════════════════════════════════════════════════╗
║        POLYMARKET BOT — LIVE WATCHDOG MODE           ║
║                                                      ║
║  Ctrl+C to stop                                      ║
║  Watchdog checks every 30s                           ║
║  Auto-restarts if win rate < 40% or 3 losses in a row║
╚══════════════════════════════════════════════════════╝
""")

    while not shutdown.is_set():
        cycle += 1
        cfg = load_config()  # reload config each cycle (picks up any saved adaptations)

        logger.info("═" * 55)
        logger.info("▶  CYCLE %d STARTING  (entry=%.0fc  stop=%.0fc  wake=%dmin)",
                    cycle,
                    cfg.strategy.entry_threshold_cents,
                    cfg.strategy.stop_loss_cents,
                    cfg.strategy.wake_minutes_before_close)
        logger.info("═" * 55)

        # Run one daily cycle
        try:
            _run_one_cycle_with_watchdog_check(cfg, watchdog, shutdown)
        except Exception as exc:
            logger.error("💥 Cycle %d crashed: %s", cycle, exc, exc_info=True)
            logger.info("Waiting 10s before retry...")
            time.sleep(10)
            continue

        if shutdown.is_set():
            break

        # ── Did the watchdog raise a flag? ────────────────────────────────────
        if watchdog.should_restart():
            logger.warning("🔄 RESTART triggered — adapting strategy now...")
            _do_adaptation(cfg)
            watchdog.acknowledge_restart()
            logger.info("✅ Params updated. Restarting bot in 5s...")
            time.sleep(5)
            continue

        # Normal end of cycle — sleep until next scheduled scan
        logger.info("✅ Cycle %d complete. Sleeping 60s before next cycle.", cycle)
        _interruptible_sleep(60, shutdown)

    watchdog.stop()
    logger.info("👋 Bot shut down cleanly after %d cycles.", cycle)


def _run_one_cycle_with_watchdog_check(cfg, watchdog: LiveWatchdog, shutdown: Event) -> None:
    """
    Runs the scan → monitor cycle.
    Checks for watchdog/shutdown signals between markets
    so we can stop cleanly mid-cycle if needed.
    """
    from polymarket_bot.api import PolymarketClient
    from polymarket_bot.engine import _monitor_market_cluster, scan_watchlist
    from polymarket_bot.trading import PaperPortfolio, should_wake_for_market
    from polymarket_bot.data.storage import append_metrics

    client    = PolymarketClient(cfg)
    portfolio = PaperPortfolio(cfg)
    now       = datetime.now(timezone.utc)

    watchlist = scan_watchlist(client, cfg, now=now)
    pending   = [m for m in watchlist if m.end_time > now]
    run_stats: list[dict] = []

    while pending and not shutdown.is_set():
        pending.sort(key=lambda m: m.end_time)
        current = datetime.now(timezone.utc)

        due = [m for m in pending if should_wake_for_market(m, cfg, now=current)]

        if not due:
            next_m   = pending[0]
            wake_at  = next_m.end_time - timedelta(minutes=cfg.strategy.wake_minutes_before_close)
            sleep_s  = max(1.0, min(30.0, (wake_at - current).total_seconds()))

            # Sleep in small chunks so we can react to watchdog/shutdown
            _interruptible_sleep(sleep_s, shutdown, watchdog)
            continue

        # ── Check flags before processing next market ─────────────────────────
        if watchdog.should_restart() or shutdown.is_set():
            logger.warning("⚡ Watchdog/shutdown flag — wrapping up current cycle early")
            break

        # Group into clusters (markets closing at the same time)
        clusters = _cluster_by_close_time(due)
        for cluster in clusters:
            stats = _monitor_market_cluster(client, portfolio, cluster, cfg)
            run_stats.extend(stats)

        processed_ids = {m.market_id for m in due}
        pending = [m for m in pending if m.market_id not in processed_ids]

    # Save cycle summary
    summary = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "watchlist_count": len(watchlist),
        "processed_count": len(run_stats),
        "portfolio":       portfolio.to_dict(),
    }
    append_metrics(summary)


def _do_adaptation(cfg) -> None:
    """Run the self-correction algorithm and save new params to disk."""
    trades     = load_closed_trades()
    adaptation = adapt_strategy(cfg, trades)

    if adaptation.get("adapted"):
        logger.info("📐 New params saved:")
        logger.info("   entry_threshold = %.0fc  (was tightened/relaxed)",
                    adaptation["new_entry_cents"])
        logger.info("   stop_loss       = %.0fc",
                    adaptation["new_stop_cents"])
        logger.info("   wake_minutes    = %d min",
                    adaptation["new_wake_minutes"])
        if adaptation.get("preferred_categories"):
            logger.info("   focus_categories = %s",
                        adaptation["preferred_categories"])
    else:
        logger.info("ℹ️  Adaptation held: %s", adaptation.get("reason", ""))


def _cluster_by_close_time(markets) -> list[list]:
    """Group markets closing within 3 minutes of each other."""
    markets = sorted(markets, key=lambda m: m.end_time)
    clusters: list[list] = []
    current = [markets[0]]
    for m in markets[1:]:
        if (m.end_time - current[0].end_time).total_seconds() <= 180:
            current.append(m)
        else:
            clusters.append(current)
            current = [m]
    clusters.append(current)
    return clusters


def _interruptible_sleep(seconds: float, shutdown: Event,
                          watchdog: LiveWatchdog | None = None) -> None:
    """Sleep in 1-second chunks so we wake up fast if shutdown/restart is signalled."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if shutdown.is_set():
            return
        if watchdog and watchdog.should_restart():
            return
        time.sleep(1)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_with_watchdog()
