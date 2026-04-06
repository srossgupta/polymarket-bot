"""Runtime orchestration: scanning, concurrent monitoring, and paper execution.

Handles the full lifecycle:
1. Daily scan at fixed time → build watchlist of markets closing in 24h
2. Sleep until T-6 (configurable) before each market closes
3. Wake and monitor concurrently (multiple markets can close simultaneously)
4. Execute paper trades based on strategy signals
5. Self-correct after each cycle
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .adaptive import adapt_strategy
from .analytics import full_pnl_report
from .backtest import (BacktestResult, build_synthetic_snapshots,
                       monte_carlo_simulation, parameter_sweep,
                       run_snapshot_backtest)
from .client import PolymarketClient
from .config import BotConfig, load_config
from .models import Market, PriceSeries, Side, TradeType
from .paper import PaperPortfolio
from .storage import (append_metrics, append_snapshot, load_closed_trades,
                      load_snapshots, save_watchlist)
from .strategy import (eligible_for_tracking, entry_signal_from_price,
                       select_markets_for_next_24h, should_wake_for_market,
                       stop_loss_hit)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --- Market scanning ---

def scan_watchlist(
    client: PolymarketClient,
    cfg: BotConfig,
    now: datetime | None = None,
) -> list[Market]:
    """Scan all open markets and select those closing in next 24 hours."""
    now = now or _utcnow()
    logger.info("Scanning markets...")
    markets = client.fetch_open_markets()
    logger.info("Found %d total open markets", len(markets))

    selected = select_markets_for_next_24h(markets, cfg, now=now)
    save_watchlist(selected, scan_ts=now)
    logger.info("Selected %d markets closing within %dh",
                len(selected), cfg.strategy.max_scan_horizon_hours)

    for m in selected[:10]:
        logger.info("  [%s] %s (vol=$%.0f, closes=%s)",
                     m.category, m.question[:60], m.volume_usd,
                     m.end_time.strftime("%H:%M UTC"))
    if len(selected) > 10:
        logger.info("  ... and %d more", len(selected) - 10)

    return selected


# --- Single market monitor ---

def monitor_market_until_close(
    client: PolymarketClient,
    portfolio: PaperPortfolio,
    market: Market,
    cfg: BotConfig,
    now_fn: Callable[[], datetime] = _utcnow,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Monitor a single market from T-wake to close.
    Tracks prices, generates entry signals, manages stop-loss.
    """
    stats: dict[str, Any] = {
        "market_id": market.market_id,
        "question": market.question,
        "category": market.category,
        "entered": False,
        "stopped": False,
        "volume": market.volume_usd,
    }

    # Volume check
    if not eligible_for_tracking(market, cfg):
        stats["skipped"] = f"volume ${market.volume_usd:.0f} < ${cfg.strategy.min_volume_usd:.0f}"
        logger.info("SKIP %s: %s", market.question[:50], stats["skipped"])
        return stats

    logger.info("MONITORING %s (closes in %.1f min)",
                market.question[:60], market.minutes_to_close)

    series = PriceSeries(market_id=market.market_id)
    last_point = None

    while now_fn() < market.end_time:
        try:
            point = client.fetch_market_prices(market)
        except Exception as exc:
            logger.warning("Price fetch failed for %s: %s", market.market_id, exc)
            sleeper(cfg.strategy.poll_seconds)
            continue

        last_point = point
        series.add(point)
        append_snapshot(market, point)

        # --- Entry logic ---
        if market.market_id not in portfolio.open_positions:
            signal = entry_signal_from_price(point, cfg, series)
            if signal:
                pos = portfolio.open_position(
                    market, signal.side, signal.price, signal.reason,
                    kelly_fraction=signal.kelly_fraction,
                    velocity=signal.velocity,
                    volatility=signal.volatility,
                )
                if pos:
                    stats["entered"] = True
                    stats["side"] = signal.side.value
                    stats["entry_price"] = signal.price
                    stats["kelly"] = signal.kelly_fraction
                    stats["ev"] = signal.expected_value

        # --- Position management ---
        if market.market_id in portfolio.open_positions:
            pos = portfolio.open_positions[market.market_id]
            current_price = point.yes if pos.side == Side.YES else point.no
            pos.update_peak(current_price)

            # Stop loss check
            hit, px = stop_loss_hit(pos.side, point, cfg, series)
            if hit:
                portfolio.close_position(
                    market, point,
                    reason=f"stop_loss<{cfg.strategy.stop_loss_cents:.0f}c (px={px:.4f})",
                    trade_type=TradeType.STOP_LOSS,
                )
                stats["stopped"] = True
                stats["stop_price"] = px
                return stats

        # Check if too close to expiry
        seconds_left = (market.end_time - now_fn()).total_seconds()
        if seconds_left <= cfg.strategy.min_time_to_close_seconds:
            break

        sleeper(cfg.strategy.poll_seconds)

    # Force-close any open position at expiry
    if market.market_id in portfolio.open_positions and last_point is not None:
        portfolio.close_position(
            market, last_point,
            reason="market_expired",
            trade_type=TradeType.FORCED_CLOSE,
        )
        stats["forced_close"] = True

    return stats


# --- Concurrent market monitoring ---

def _monitor_market_cluster(
    client: PolymarketClient,
    portfolio: PaperPortfolio,
    markets: list[Market],
    cfg: BotConfig,
) -> list[dict]:
    """Monitor a cluster of markets closing at roughly the same time.
    Uses thread pool for concurrent execution.
    """
    if len(markets) == 1:
        return [monitor_market_until_close(client, portfolio, markets[0], cfg)]

    results = []
    max_workers = min(len(markets), 8)
    logger.info("Processing cluster of %d markets concurrently", len(markets))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                monitor_market_until_close, client, portfolio, m, cfg
            ): m
            for m in markets
        }
        for future in as_completed(futures):
            market = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logger.error("Error monitoring %s: %s", market.market_id, exc)
                results.append({
                    "market_id": market.market_id,
                    "error": str(exc),
                })

    return results


# --- Main daily cycle ---

def run_daily_once(cfg: BotConfig | None = None) -> dict[str, Any]:
    """Execute one full daily cycle: scan → monitor → trade → adapt."""
    cfg = cfg or load_config()
    client = PolymarketClient(cfg)
    portfolio = PaperPortfolio(cfg)

    now = _utcnow()
    logger.info("=== Daily cycle started at %s ===", now.isoformat())

    # Step 1: Scan
    watchlist = scan_watchlist(client, cfg, now=now)

    # Step 2: Process markets in time-ordered clusters
    pending = [m for m in watchlist if m.end_time > now]
    run_stats: list[dict] = []

    while pending:
        pending.sort(key=lambda m: m.end_time)
        current = _utcnow()

        # Find markets that are due (within wake window)
        due = [m for m in pending if should_wake_for_market(m, cfg, now=current)]

        if not due:
            # Sleep until next market's wake time
            next_market = pending[0]
            wake_at = next_market.end_time - timedelta(
                minutes=cfg.strategy.wake_minutes_before_close)
            sleep_secs = max(1.0, min(60.0, (wake_at - current).total_seconds()))
            logger.info("Sleeping %.1fs until next wake (%s)",
                        sleep_secs, next_market.question[:40])
            time.sleep(sleep_secs)
            continue

        # Group due markets by close time (within 3-minute clusters)
        clusters: list[list[Market]] = []
        due.sort(key=lambda m: m.end_time)
        current_cluster: list[Market] = [due[0]]

        for m in due[1:]:
            if (m.end_time - current_cluster[0].end_time).total_seconds() <= 180:
                current_cluster.append(m)
            else:
                clusters.append(current_cluster)
                current_cluster = [m]
        clusters.append(current_cluster)

        # Process each cluster
        for cluster in clusters:
            stats = _monitor_market_cluster(client, portfolio, cluster, cfg)
            run_stats.extend(stats)

        # Remove processed markets
        processed_ids = {m.market_id for m in due}
        pending = [m for m in pending if m.market_id not in processed_ids]

    # Step 3: Self-correct
    closed_trades = load_closed_trades()
    adaptation = adapt_strategy(cfg, closed_trades)

    # Step 4: Performance snapshot
    perf = portfolio.take_performance_snapshot()

    summary = {
        "timestamp": _utcnow().isoformat(),
        "watchlist_count": len(watchlist),
        "processed_count": len(run_stats),
        "portfolio": portfolio.to_dict(),
        "run_stats": run_stats,
        "adaptation": adaptation,
        "performance": {
            "total_value": perf.total_value,
            "pnl": perf.total_pnl,
            "win_rate": perf.win_rate,
            "sharpe": perf.sharpe_estimate,
        },
    }
    append_metrics(summary)
    logger.info("=== Daily cycle complete: %d processed, PnL=$%.2f ===",
                len(run_stats), portfolio.total_pnl)
    return summary


# --- Backtest entry points ---

def run_backtest(cfg: BotConfig | None = None) -> dict[str, Any]:
    """Run backtest from recorded snapshots or synthetic data."""
    cfg = cfg or load_config()
    snapshots = load_snapshots()
    source = "recorded_snapshots"

    if not snapshots:
        snapshots = build_synthetic_snapshots(num_markets=30)
        source = "synthetic_bootstrap"
        logger.info("No recorded snapshots, using synthetic data (%d markets)",
                     len(set(s["market_id"] for s in snapshots)))

    result = run_snapshot_backtest(cfg, snapshots)

    # Also run Monte Carlo if we have trade data
    mc_result = None
    if result.trade_log:
        pnls = [t.pnl for t in result.trade_log]
        mc_result = monte_carlo_simulation(pnls)

    payload = {
        "source": source,
        "snapshot_count": len(snapshots),
        "result": result.to_dict(),
        "monte_carlo": mc_result,
    }

    append_metrics({"timestamp": _utcnow().isoformat(), "backtest": payload})
    return payload


def run_parameter_sweep(cfg: BotConfig | None = None) -> dict[str, Any]:
    """Sweep parameters and find optimal settings."""
    cfg = cfg or load_config()
    snapshots = load_snapshots()

    if not snapshots:
        snapshots = build_synthetic_snapshots(num_markets=30)

    results = parameter_sweep(snapshots)
    top_5 = results[:5] if results else []

    payload = {
        "total_combinations": len(results),
        "top_5_params": top_5,
    }

    if top_5:
        best = top_5[0]
        logger.info("Best params: entry=%sc, stop=%sc, wake=%smin (PnL=$%.2f, Sharpe=%.3f)",
                     best["entry_cents"], best["stop_cents"],
                     best["wake_minutes"], best["total_pnl"], best["sharpe"])

    return payload
