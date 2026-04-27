"""Runtime: scan markets, monitor prices, execute paper trades.

Flow:
  1. Scan all open markets → filter to ones closing within 24h
  2. Wait until 5 min before each market closes
  3. Poll price every 0.3s
  4. If price is in 95¢–99.5¢ band → enter
  5. Hold to expiry → close at final price
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from polymarket_bot.api import PolymarketClient
from polymarket_bot.core import BotConfig, Market, PriceSeries, load_config
from polymarket_bot.data.storage import (append_metrics, append_snapshot, load_closed_trades,
                                          load_snapshots, save_watchlist)
from polymarket_bot.trading import (PaperPortfolio, adapt_strategy, eligible_for_tracking,
                                     select_markets_for_next_24h, should_wake_for_market)
from polymarket_bot.trading.strategy import pick_entry_side

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --- Scan ---

def scan_watchlist(client: PolymarketClient, cfg: BotConfig, now: datetime | None = None) -> list[Market]:
    """Fetch all open markets, filter down to ones closing soon."""
    now = now or _utcnow()
    markets = client.fetch_open_markets()
    logger.info("Fetched %d total open markets", len(markets))
    selected = select_markets_for_next_24h(markets, cfg, now=now)
    save_watchlist(selected, scan_ts=now)
    logger.info("Selected %d markets closing within %dh", len(selected), cfg.strategy.max_scan_horizon_hours)
    for m in selected[:10]:
        logger.info("  [%s] %s (vol=$%.0f, closes=%s)",
                     m.category, m.question[:60], m.volume_usd, m.end_time.strftime("%H:%M UTC"))
    if len(selected) > 10:
        logger.info("  ... and %d more", len(selected) - 10)
    return selected


# --- Monitor a single market ---

def monitor_market_until_close(
    client: PolymarketClient, portfolio: PaperPortfolio, market: Market, cfg: BotConfig,
    now_fn: Callable[[], datetime] = _utcnow, sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """
    Poll price every 0.3s in the 5-min window before close.
    Enter if 95¢ < price < 99.5¢.
    Hold to expiry — no stop loss.
    """
    stats: dict[str, Any] = {
        "market_id": market.market_id, "question": market.question,
        "category": market.category, "entered": False,
    }

    if not eligible_for_tracking(market, cfg):
        stats["skipped"] = f"volume ${market.volume_usd:.0f} < ${cfg.strategy.min_volume_usd:.0f}"
        return stats

    logger.info("MONITORING %s (closes in %.1f min)", market.question[:60], market.minutes_to_close)
    series = PriceSeries(market_id=market.market_id)
    last_point = None
    failures = 0

    while now_fn() < market.end_time:
        # Fetch price
        try:
            point = client.fetch_market_prices(market)
        except Exception as exc:
            logger.warning("Price fetch failed for %s: %s", market.market_id, exc)
            failures += 1
            if failures >= 5:
                stats["skipped"] = "clob_unavailable"
                return stats
            sleeper(cfg.strategy.poll_seconds)
            continue

        if point.yes == 0.0 and point.no == 0.0:
            failures += 1
            if failures >= 5:
                stats["skipped"] = "clob_price_zero"
                return stats
            sleeper(cfg.strategy.poll_seconds)
            continue

        failures = 0
        last_point = point
        series.add(point)
        append_snapshot(market, point)

        # Entry: pick YES or NO side if price is in 95¢–99.5¢ band
        if market.market_id not in portfolio.open_positions:
            entry = pick_entry_side(point, cfg)
            if entry:
                side, price = entry
                reason = f"{side.value}@{price:.4f}"
                portfolio.open_position(market, side, price, reason)
                stats.update(entered=True, side=side.value, entry_price=price)

        # Stop polling when market is about to close
        if (market.end_time - now_fn()).total_seconds() <= cfg.strategy.min_time_to_close_seconds:
            break

        sleeper(cfg.strategy.poll_seconds)

    # Close at expiry
    if market.market_id in portfolio.open_positions and last_point is not None:
        portfolio.close_position(market, last_point, reason="market_expired")
    return stats


# --- Run multiple markets concurrently ---

def _monitor_market_cluster(client, portfolio, markets, cfg) -> list[dict]:
    if len(markets) == 1:
        return [monitor_market_until_close(client, portfolio, markets[0], cfg)]
    results = []
    with ThreadPoolExecutor(max_workers=min(len(markets), 8)) as executor:
        futures = {executor.submit(monitor_market_until_close, client, portfolio, m, cfg): m for m in markets}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"market_id": futures[future].market_id, "error": str(exc)})
    return results


# --- Daily cycle ---

def run_daily_once(cfg: BotConfig | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    client = PolymarketClient(cfg)
    portfolio = PaperPortfolio(cfg)
    now = _utcnow()
    logger.info("=== Daily cycle started at %s ===", now.isoformat())

    watchlist = scan_watchlist(client, cfg, now=now)
    pending = [m for m in watchlist if m.end_time > now]
    run_stats: list[dict] = []

    while pending:
        pending.sort(key=lambda m: m.end_time)
        current = _utcnow()

        due = [m for m in pending if should_wake_for_market(m, cfg, now=current)]
        if not due:
            next_m = pending[0]
            wake_at = next_m.end_time - timedelta(minutes=cfg.strategy.wake_minutes_before_close)
            time.sleep(max(1.0, min(60.0, (wake_at - current).total_seconds())))
            continue

        # Group markets closing within 3 min of each other
        due.sort(key=lambda m: m.end_time)
        clusters: list[list[Market]] = [[due[0]]]
        for m in due[1:]:
            if (m.end_time - clusters[-1][0].end_time).total_seconds() <= 180:
                clusters[-1].append(m)
            else:
                clusters.append([m])

        for cluster in clusters:
            run_stats.extend(_monitor_market_cluster(client, portfolio, cluster, cfg))

        processed_ids = {m.market_id for m in due}
        pending = [m for m in pending if m.market_id not in processed_ids]

    # Adapt strategy based on results
    closed_trades = load_closed_trades()
    adaptation = adapt_strategy(cfg, closed_trades)
    perf = portfolio.take_performance_snapshot()

    summary = {
        "timestamp": _utcnow().isoformat(),
        "watchlist_count": len(watchlist),
        "processed_count": len(run_stats),
        "portfolio": portfolio.to_dict(),
        "adaptation": adaptation,
        "performance": {
            "total_value": perf.total_value, "pnl": perf.total_pnl,
            "win_rate": perf.win_rate, "sharpe": perf.sharpe_estimate,
        },
    }
    append_metrics(summary)
    logger.info("=== Daily cycle complete: %d processed, PnL=$%.2f ===", len(run_stats), portfolio.total_pnl)
    return summary


# --- Backtest ---

def run_backtest(cfg: BotConfig | None = None) -> dict[str, Any]:
    from polymarket_bot.backtest import build_synthetic_snapshots, run_snapshot_backtest, monte_carlo_simulation
    cfg = cfg or load_config()
    snapshots = load_snapshots()
    source = "recorded_snapshots"
    if not snapshots:
        snapshots = build_synthetic_snapshots(num_markets=30)
        source = "synthetic"
    result = run_snapshot_backtest(cfg, snapshots)
    mc = monte_carlo_simulation([t.pnl for t in result.trade_log]) if result.trade_log else None
    payload = {"source": source, "snapshot_count": len(snapshots),
               "result": result.to_dict(), "monte_carlo": mc}
    append_metrics({"timestamp": _utcnow().isoformat(), "backtest": payload})
    return payload


def run_parameter_sweep(cfg: BotConfig | None = None) -> dict[str, Any]:
    from polymarket_bot.backtest import parameter_sweep
    cfg = cfg or load_config()
    snapshots = load_snapshots()
    if not snapshots:
        from polymarket_bot.backtest import build_synthetic_snapshots
        snapshots = build_synthetic_snapshots(num_markets=30)
    results = parameter_sweep(snapshots)
    top_5 = results[:5] if results else []
    return {"total_combinations": len(results), "top_5_params": top_5}
