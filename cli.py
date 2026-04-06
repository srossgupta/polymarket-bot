"""CLI entrypoint for Polymarket paper trading bot."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta

from .analytics import (category_ranking, equity_curve, full_pnl_report,
                        parameter_sensitivity_report, what_if_analysis)
from .config import load_config
from .engine import run_backtest, run_daily_once, run_parameter_sweep


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, default=str))


def cmd_backtest(args: argparse.Namespace) -> None:
    result = run_backtest()
    _print_json(result)


def cmd_sweep(args: argparse.Namespace) -> None:
    result = run_parameter_sweep()
    _print_json(result)


def cmd_paper_once(args: argparse.Namespace) -> None:
    summary = run_daily_once()
    _print_json(summary)


def cmd_run_loop(args: argparse.Namespace) -> None:
    cfg = load_config()
    logging.info("Starting daily loop (scan at %02d:%02d local, tz=%s)",
                 cfg.strategy.scan_hour_local, cfg.strategy.scan_minute_local,
                 cfg.timezone)

    while True:
        now_local = datetime.now(cfg.local_tz)
        target = now_local.replace(
            hour=cfg.strategy.scan_hour_local,
            minute=cfg.strategy.scan_minute_local,
            second=0, microsecond=0,
        )
        if target <= now_local:
            target += timedelta(days=1)

        wait = (target - now_local).total_seconds()
        logging.info("Sleeping until next scan at %s (%.1f min)",
                     target.isoformat(), wait / 60)
        time.sleep(max(wait, 1.0))

        try:
            summary = run_daily_once(cfg)
            _print_json(summary)
        except Exception as exc:
            logging.error("Daily cycle failed: %s", exc, exc_info=True)


def cmd_pnl(args: argparse.Namespace) -> None:
    report = full_pnl_report()
    _print_json(report)


def cmd_categories(args: argparse.Namespace) -> None:
    ranking = category_ranking()
    _print_json(ranking)


def cmd_sensitivity(args: argparse.Namespace) -> None:
    report = parameter_sensitivity_report()
    _print_json(report)


def cmd_whatif(args: argparse.Namespace) -> None:
    result = what_if_analysis(
        new_entry_cents=args.entry,
        new_stop_cents=args.stop,
        new_wake_minutes=args.wake,
    )
    _print_json(result)


def cmd_equity(args: argparse.Namespace) -> None:
    curve = equity_curve()
    _print_json(curve)


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
  python -m polymarket_bot --pnl              # Full P&L report
  python -m polymarket_bot --categories        # Category ranking
  python -m polymarket_bot --whatif --entry 93 --stop 65  # What-if analysis
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--backtest", action="store_true",
                       help="Run backtest from snapshots or synthetic data")
    group.add_argument("--sweep", action="store_true",
                       help="Parameter sweep optimization")
    group.add_argument("--paper-once", action="store_true",
                       help="Run one full scan-to-close paper cycle")
    group.add_argument("--run-loop", action="store_true",
                       help="Run forever on fixed daily schedule")
    group.add_argument("--pnl", action="store_true",
                       help="Show full P&L report")
    group.add_argument("--categories", action="store_true",
                       help="Show category performance ranking")
    group.add_argument("--sensitivity", action="store_true",
                       help="Parameter sensitivity analysis")
    group.add_argument("--whatif", action="store_true",
                       help="What-if analysis with different parameters")
    group.add_argument("--equity", action="store_true",
                       help="Show equity curve")

    # What-if parameters
    parser.add_argument("--entry", type=float, help="Entry threshold (cents)")
    parser.add_argument("--stop", type=float, help="Stop loss (cents)")
    parser.add_argument("--wake", type=int, help="Wake minutes before close")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    commands = {
        "backtest": cmd_backtest,
        "sweep": cmd_sweep,
        "paper_once": cmd_paper_once,
        "run_loop": cmd_run_loop,
        "pnl": cmd_pnl,
        "categories": cmd_categories,
        "sensitivity": cmd_sensitivity,
        "whatif": cmd_whatif,
        "equity": cmd_equity,
    }

    for name, handler in commands.items():
        if getattr(args, name, False):
            handler(args)
            return

    parser.print_help()


if __name__ == "__main__":
    main()
