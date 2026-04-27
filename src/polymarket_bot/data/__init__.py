"""Data layer: SQLite storage and analytics."""

from .analytics import (
    category_ranking,
    equity_curve,
    full_pnl_report,
    parameter_sensitivity_report,
    what_if_analysis,
)
from .storage import (
    append_metrics, append_snapshot, append_trade,
    get_category_performance, get_hourly_pnl, get_pnl_by_parameter_set,
    init_db, load_closed_trades, load_metrics, load_performance_history,
    load_snapshots, save_performance, save_watchlist,
)

__all__ = [
    "append_metrics", "append_snapshot", "append_trade",
    "category_ranking", "equity_curve", "full_pnl_report",
    "get_category_performance", "get_hourly_pnl", "get_pnl_by_parameter_set",
    "init_db", "load_closed_trades", "load_metrics", "load_performance_history",
    "load_snapshots", "parameter_sensitivity_report", "save_performance",
    "save_watchlist", "what_if_analysis",
]
