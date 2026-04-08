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


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket high-probability paper trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m polymarket_bot --backtest          # Backtest with synthetic/recorded data
  python -m polymarket_bot --sweep             # Parameter sweep optimization
  python -m polymarket_bot --paper-once        # Run one daily cycle
  python -m polymarket_bot --run-loop          # Run forever on schedule
  python -m polymarket_bot --watchdog          # Run forever with live self-correction
  python -m polymarket_bot --pnl              # Full P&L report
  python -m polymarket_bot --categories        # Category ranking
  python -m polymarket_bot --whatif --entry 93 --stop 65  # What-if analysis
        """)
    parser.add_argument("-v", "--verbose", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--backtest", action="store_true")
    group.add_argument("--sweep", action="store_true")
    group.add_argument("--paper-once", action="store_true")
    group.add_argument("--run-loop", action="store_true")
    group.add_argument("--pnl", action="store_true")
    group.add_argument("--categories", action="store_true")
    group.add_argument("--sensitivity", action="store_true")
    group.add_argument("--whatif", action="store_true")
    group.add_argument("--equity", action="store_true")
    group.add_argument("--watchdog", action="store_true")
    parser.add_argument("--entry", type=float)
    parser.add_argument("--stop", type=float)
    parser.add_argument("--wake", type=int)
    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.backtest:
        _print_json(run_backtest())
    elif args.sweep:
        _print_json(run_parameter_sweep())
    elif args.paper_once:
        _print_json(run_daily_once())
    elif args.run_loop:
        cfg = load_config()
        logging.info("Starting daily loop (scan at %02d:%02d local)", cfg.strategy.scan_hour_local, cfg.strategy.scan_minute_local)
        while True:
            now_local = datetime.now(cfg.local_tz)
            target = now_local.replace(hour=cfg.strategy.scan_hour_local, minute=cfg.strategy.scan_minute_local, second=0, microsecond=0)
            if target <= now_local:
                target += timedelta(days=1)
            time.sleep(max((target - now_local).total_seconds(), 1.0))
            try:
                _print_json(run_daily_once(cfg))
            except Exception as exc:
                logging.error("Daily cycle failed: %s", exc, exc_info=True)
    elif args.pnl:
        _print_json(full_pnl_report())
    elif args.categories:
        _print_json(category_ranking())
    elif args.sensitivity:
        _print_json(parameter_sensitivity_report())
    elif args.whatif:
        _print_json(what_if_analysis(args.entry, args.stop, args.wake))
    elif args.equity:
        _print_json(equity_curve())
    elif args.watchdog:
        from polymarket_bot.watchdog import run_with_watchdog
        run_with_watchdog()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
