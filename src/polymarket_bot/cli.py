"""CLI entrypoint for Polymarket paper trading bot."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta

from polymarket_bot.core import load_config
from polymarket_bot.data import (category_ranking, equity_curve, full_pnl_report,
                                  parameter_sensitivity_report, what_if_analysis)
from polymarket_bot.engine import run_backtest, run_daily_once, run_parameter_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket paper trading bot")
    parser.add_argument("-v", "--verbose", action="store_true")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--backtest", action="store_true", help="Backtest with recorded/synthetic data")
    group.add_argument("--sweep", action="store_true", help="Parameter sweep optimization")
    group.add_argument("--paper-once", action="store_true", help="Run one daily cycle")
    group.add_argument("--run-loop", action="store_true", help="Run forever on schedule")
    group.add_argument("--watchdog", action="store_true", help="Run with live self-correction")
    group.add_argument("--pnl", action="store_true", help="Full P&L report")
    group.add_argument("--categories", action="store_true", help="Category ranking")
    group.add_argument("--sensitivity", action="store_true", help="Parameter sensitivity")
    group.add_argument("--whatif", action="store_true", help="What-if analysis")
    group.add_argument("--equity", action="store_true", help="Equity curve")

    parser.add_argument("--entry", type=float, help="Entry threshold in cents (for --whatif)")
    parser.add_argument("--wake", type=int, help="Wake minutes before close (for --whatif)")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    out = lambda data: print(json.dumps(data, indent=2, default=str))

    if args.backtest:
        out(run_backtest())
    elif args.sweep:
        out(run_parameter_sweep())
    elif args.paper_once:
        out(run_daily_once())
    elif args.run_loop:
        cfg = load_config()
        while True:
            now = datetime.now(cfg.local_tz)
            target = now.replace(hour=cfg.strategy.scan_hour_local,
                                 minute=cfg.strategy.scan_minute_local, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            time.sleep(max((target - now).total_seconds(), 1.0))
            try:
                out(run_daily_once(cfg))
            except Exception as exc:
                logging.error("Daily cycle failed: %s", exc, exc_info=True)
    elif args.pnl:
        out(full_pnl_report())
    elif args.categories:
        out(category_ranking())
    elif args.sensitivity:
        out(parameter_sensitivity_report())
    elif args.whatif:
        out(what_if_analysis(args.entry, None, args.wake))
    elif args.equity:
        out(equity_curve())
    elif args.watchdog:
        from polymarket_bot.watchdog import run_with_watchdog
        run_with_watchdog()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
