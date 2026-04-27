"""Self-correction: adjust entry threshold and wake time based on recent performance.

If win rate is below target → tighten (raise entry bar, shorter wake).
If win rate is above target → relax (lower entry bar, longer wake).
If not enough data → hold.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

from polymarket_bot.core import BotConfig, save_adaptive_strategy

logger = logging.getLogger(__name__)


def _weighted_win_rate(pnls: list[float], decay: float = 0.95) -> float:
    """Win rate weighted so recent trades matter more."""
    if not pnls:
        return 0.0
    n = len(pnls)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total = sum(weights)
    if total < 1e-10:
        return 0.0
    return sum(w for w, p in zip(weights, pnls) if p > 0) / total


def _confidence_interval(win_rate: float, n: int, z: float = 1.645) -> tuple[float, float]:
    """Wilson score interval — should we tighten, relax, or hold?"""
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    center = (win_rate + z * z / (2 * n)) / denom
    spread = z * math.sqrt((win_rate * (1 - win_rate) + z * z / (4 * n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


def adapt_strategy(cfg: BotConfig, closed_trade_events: list[dict]) -> dict[str, Any]:
    """Look at recent trades, decide if we should tighten or relax the entry bar."""
    trades = [t for t in closed_trade_events
              if t.get("event_type") in {"SELL_MARKET", "STOP_LOSS", "FORCED_CLOSE"}]

    if len(trades) < cfg.adaptation.min_trades_for_adaptation:
        return {"adapted": False, "reason": f"need >= {cfg.adaptation.min_trades_for_adaptation} trades, have {len(trades)}"}

    window = trades[-cfg.adaptation.trade_window:]
    pnls = [t.get("pnl", 0) for t in window]

    # Only judge on decisive trades (pnl != 0)
    decisive = [p for p in pnls if p != 0]
    if not decisive:
        return {"adapted": False, "reason": "all breakevens, nothing to adapt on"}

    win_rate = sum(1 for p in decisive if p > 0) / len(decisive)
    lower, upper = _confidence_interval(win_rate, len(decisive))
    target = cfg.adaptation.target_win_rate

    # Decide direction
    if upper < target:
        direction = "tighten"
    elif lower > target:
        direction = "relax"
    else:
        direction = "hold"

    params = cfg.strategy
    step = cfg.adaptation.step_cents
    changes: dict[str, str] = {}

    if direction == "tighten":
        old = params.entry_threshold_cents
        params.entry_threshold_cents = min(cfg.bounds.max_entry_cents, old + step)
        if params.entry_threshold_cents != old:
            changes["entry"] = f"{old}c → {params.entry_threshold_cents}c"

    elif direction == "relax":
        old = params.entry_threshold_cents
        params.entry_threshold_cents = max(cfg.bounds.min_entry_cents, old - step)
        if params.entry_threshold_cents != old:
            changes["entry"] = f"{old}c → {params.entry_threshold_cents}c"

    # Clamp
    params.entry_threshold_cents = max(
        cfg.bounds.min_entry_cents,
        min(cfg.bounds.max_entry_cents, params.entry_threshold_cents))
    params.wake_minutes_before_close = max(
        cfg.bounds.min_wake_minutes,
        min(cfg.bounds.max_wake_minutes, params.wake_minutes_before_close))

    # Category performance — focus on winners
    by_cat: dict[str, dict[str, Any]] = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in window:
        cat = t.get("category", "unknown")
        by_cat[cat]["trades"] += 1
        if t.get("pnl", 0) > 0:
            by_cat[cat]["wins"] += 1
        by_cat[cat]["pnl"] += t.get("pnl", 0)

    preferred = []
    for cat, stats in by_cat.items():
        if stats["trades"] < cfg.adaptation.min_category_samples:
            continue
        expectancy = stats["pnl"] / stats["trades"]
        score = expectancy * math.sqrt(stats["trades"])
        if score > 1.0:
            preferred.append(cat)
    preferred = preferred[:cfg.adaptation.max_preferred_categories]

    save_adaptive_strategy(params, preferred)

    result = {
        "adapted": bool(changes), "direction": direction, "changes": changes,
        "new_entry_cents": params.entry_threshold_cents,
        "new_wake_minutes": params.wake_minutes_before_close,
        "preferred_categories": preferred,
        "win_rate": round(win_rate, 4),
        "window_size": len(window),
    }
    if changes:
        logger.info("Strategy adapted: %s", changes)
    else:
        logger.info("Strategy held (direction=%s, win_rate=%.1f%%)", direction, win_rate * 100)
    return result
