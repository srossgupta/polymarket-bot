"""Live self-correcting watchdog.

Runs in a background thread while the bot trades.
Every 30 seconds it checks recent trades:
  - Win rate below 40%? → restart with tighter params.
  - 3 losses in a row?  → restart with tighter params.
  - Everything fine?    → keep running.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from threading import Event, Thread

from polymarket_bot.core import load_config
from polymarket_bot.data.storage import init_db, load_closed_trades
from polymarket_bot.trading.adaptive import adapt_strategy

logger = logging.getLogger("watchdog")

WATCH_INTERVAL = 15       # seconds between checks (faster feedback)
MIN_TRADES     = 3        # need this many before judging (react sooner)
DANGER_WIN_RATE = 0.35    # below this = pull in (slightly more tolerant)
COLD_STREAK     = 2       # this many losses in a row = pause (react faster)


class LiveWatchdog:
    def __init__(self):
        self._stop = Event()
        self._restart = Event()
        self._thread: Thread | None = None
        self.checks_done = 0
        self.restarts_done = 0

    def start(self) -> None:
        self._thread = Thread(target=self._loop, daemon=True, name="watchdog")
        self._thread.start()
        logger.info("Watchdog started (checking every %ds)", WATCH_INTERVAL)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Watchdog stopped (%d checks, %d restarts)", self.checks_done, self.restarts_done)

    def should_restart(self) -> bool:
        return self._restart.is_set()

    def acknowledge_restart(self) -> None:
        self._restart.clear()
        self.restarts_done += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(timeout=WATCH_INTERVAL):
                break
            self._check()

    def _check(self) -> None:
        self.checks_done += 1
        trades = load_closed_trades(limit=20)

        if len(trades) < MIN_TRADES:
            logger.info("[check %d] %d trades — too few to judge", self.checks_done, len(trades))
            return

        # Win rate over decisive trades only (pnl != 0)
        decisive = [t for t in trades if t.get("pnl", 0) != 0]
        wins = sum(1 for t in decisive if t.get("pnl", 0) > 0)
        win_rate = wins / len(decisive) if decisive else 1.0
        total_pnl = sum(t.get("pnl", 0) for t in trades)

        # Recent losing streak
        recent = sorted(trades, key=lambda t: t.get("ts", ""), reverse=True)[:COLD_STREAK]
        all_losses = all(t.get("pnl", 0) < 0 for t in recent)

        logger.info("[check %d] win_rate=%.0f%% pnl=$%.2f streak_bad=%s",
                    self.checks_done, win_rate * 100, total_pnl, all_losses)

        if len(decisive) < 2:
            return

        if win_rate < DANGER_WIN_RATE:
            logger.warning("WATCHDOG: win rate %.0f%% < %.0f%% — triggering restart",
                           win_rate * 100, DANGER_WIN_RATE * 100)
            self._restart.set()
        elif all_losses:
            logger.warning("WATCHDOG: %d consecutive losses — triggering restart", COLD_STREAK)
            self._restart.set()


# --- Main loop with watchdog ---

def run_with_watchdog() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    init_db()

    shutdown = Event()
    def _handle_sigint(sig, frame):
        print("\nShutting down...")
        shutdown.set()
    signal.signal(signal.SIGINT, _handle_sigint)

    watchdog = LiveWatchdog()
    watchdog.start()
    cycle = 0

    print("""
=== POLYMARKET BOT — WATCHDOG MODE ===
  Ctrl+C to stop
  Checks every 30s
  Auto-restarts on bad performance
======================================
""")

    while not shutdown.is_set():
        cycle += 1
        cfg = load_config()
        logger.info("=== CYCLE %d (entry=%.0fc, wake=%dmin) ===",
                    cycle, cfg.strategy.entry_threshold_cents,
                    cfg.strategy.wake_minutes_before_close)

        try:
            _run_one_cycle(cfg, watchdog, shutdown)
        except Exception as exc:
            logger.error("Cycle %d crashed: %s", cycle, exc, exc_info=True)
            time.sleep(10)
            continue

        if shutdown.is_set():
            break

        if watchdog.should_restart():
            logger.warning("Adapting strategy after watchdog flag...")
            trades = load_closed_trades()
            adapt_strategy(cfg, trades)
            watchdog.acknowledge_restart()
            logger.info("Params updated. Restarting in 5s...")
            time.sleep(5)
            continue

        logger.info("Cycle %d done. Starting next cycle immediately.", cycle)
        _interruptible_sleep(5, shutdown)  # tiny pause then rescan

    watchdog.stop()
    logger.info("Bot shut down after %d cycles.", cycle)


def _run_one_cycle(cfg, watchdog: LiveWatchdog, shutdown: Event) -> None:
    """Scan → monitor → close. Checks watchdog/shutdown between markets."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from polymarket_bot.api import PolymarketClient
    from polymarket_bot.engine import _monitor_market_cluster, scan_watchlist
    from polymarket_bot.trading import PaperPortfolio, eligible_for_tracking, should_wake_for_market
    from polymarket_bot.data.storage import append_metrics

    client = PolymarketClient(cfg)
    portfolio = PaperPortfolio(cfg)
    now = datetime.now(timezone.utc)

    watchlist = scan_watchlist(client, cfg, now=now)

    # Pre-filter: skip already-resolved and zero-price markets
    # Run in parallel — fetching 2500+ markets serially takes 15+ minutes
    candidates = [m for m in watchlist
                  if m.end_time > now and eligible_for_tracking(m, cfg)]

    def _check_price(m):
        try:
            pt = client.fetch_market_prices(m)
            best = max(pt.yes, pt.no)
            if best >= 0.99:
                return None, f"near-resolved ({best:.2f})"
            if best <= 0.01:
                return None, f"near-zero ({best:.2f})"
            return m, None
        except Exception:
            return m, None  # can't fetch — let monitoring handle it

    pending: list = []
    skipped_pre = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_check_price, m): m for m in candidates}
        for future in as_completed(futures):
            market, _ = future.result()
            if market is None:
                skipped_pre += 1
            else:
                pending.append(market)

    logger.info("%d markets pass filters (%d pre-screened out)", len(pending), skipped_pre)
    run_stats: list[dict] = []

    while pending and not shutdown.is_set():
        pending.sort(key=lambda m: m.end_time)
        current = datetime.now(timezone.utc)
        due = [m for m in pending if should_wake_for_market(m, cfg, now=current)]

        if not due:
            next_m = pending[0]
            wake_at = next_m.end_time - timedelta(minutes=cfg.strategy.wake_minutes_before_close)
            sleep_s = max(1.0, min(30.0, (wake_at - current).total_seconds()))
            _interruptible_sleep(sleep_s, shutdown, watchdog)
            continue

        if watchdog.should_restart() or shutdown.is_set():
            break

        # Cluster markets closing within 3 min of each other
        due.sort(key=lambda m: m.end_time)
        clusters: list[list] = [[due[0]]]
        for m in due[1:]:
            if (m.end_time - clusters[-1][0].end_time).total_seconds() <= 180:
                clusters[-1].append(m)
            else:
                clusters.append([m])

        for cluster in clusters:
            run_stats.extend(_monitor_market_cluster(client, portfolio, cluster, cfg))

        processed = {m.market_id for m in due}
        pending = [m for m in pending if m.market_id not in processed]

    entries = sum(1 for s in run_stats if s.get("entered"))
    skips = sum(1 for s in run_stats if s.get("skipped"))
    if run_stats:
        logger.info("Cycle: %d processed, %d entered, %d skipped", len(run_stats), entries, skips)

    append_metrics({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "watchlist_count": len(watchlist),
        "processed_count": len(run_stats),
        "entries": entries,
        "portfolio": portfolio.to_dict(),
    })


def _interruptible_sleep(seconds: float, shutdown: Event,
                          watchdog: LiveWatchdog | None = None) -> None:
    """Sleep in 1s chunks so we react quickly to shutdown/restart."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if shutdown.is_set():
            return
        if watchdog and watchdog.should_restart():
            return
        time.sleep(1)


if __name__ == "__main__":
    run_with_watchdog()
