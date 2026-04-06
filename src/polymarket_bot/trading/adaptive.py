"""Self-correction engine with exponential decay weighting and statistical rigor."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

from polymarket_bot.core import BotConfig, save_adaptive_strategy

logger = logging.getLogger(__name__)


def _exponential_weighted_stats(
    pnls: list[float], decay: float = 0.95,
) -> tuple[float, float, float]:
    if not pnls:
        return 0.0, 0.0, 0.0
    n = len(pnls)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_weight = sum(weights)
    if total_weight < 1e-10:
        return 0.0, 0.0, 0.0
    weighted_mean = sum(w * p for w, p in zip(weights, pnls)) / total_weight
    weighted_wins = sum(w for w, p in zip(weights, pnls) if p > 0) / total_weight
    weighted_var = sum(w * (p - weighted_mean) ** 2 for w, p in zip(weights, pnls)) / total_weight
    return weighted_mean, weighted_wins, weighted_var


def _confidence_interval(win_rate: float, n: int, z: float = 1.645) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    denominator = 1 + z * z / n
    center = (win_rate + z * z / (2 * n)) / denominator
    spread = z * math.sqrt((win_rate * (1 - win_rate) + z * z / (4 * n)) / n) / denominator
    return max(0, center - spread), min(1, center + spread)


def _should_adapt(win_rate: float, n_trades: int, min_confidence: float, target: float) -> str:
    lower, upper = _confidence_interval(win_rate, n_trades)
    if upper < target:
        return "tighten"
    elif lower > target:
        return "relax"
    return "hold"


def adapt_strategy(cfg: BotConfig, closed_trade_events: list[dict]) -> dict[str, Any]:
    trades = [t for t in closed_trade_events
              if t.get("event_type") in {"SELL_MARKET", "STOP_LOSS", "FORCED_CLOSE"}]
    min_trades = cfg.adaptation.min_trades_for_adaptation
    if len(trades) < min_trades:
        return {"adapted": False, "reason": f"need >= {min_trades} closed trades, have {len(trades)}"}

    window = trades[-cfg.adaptation.trade_window:]
    pnls = [t.get("pnl", 0) for t in window]
    weighted_mean, weighted_wr, weighted_var = _exponential_weighted_stats(pnls, cfg.adaptation.decay_factor)
    simple_wr = sum(1 for p in pnls if p > 0) / len(pnls)
    direction = _should_adapt(simple_wr, len(pnls), cfg.adaptation.min_confidence_level, cfg.adaptation.target_win_rate)

    params = cfg.strategy
    step = cfg.adaptation.step_cents
    changes: dict[str, str] = {}

    if direction == "tighten":
        old = params.entry_threshold_cents
        params.entry_threshold_cents = min(cfg.bounds.max_entry_cents, params.entry_threshold_cents + step)
        if params.entry_threshold_cents != old:
            changes["entry_threshold"] = f"{old} -> {params.entry_threshold_cents}"
        old = params.stop_loss_cents
        params.stop_loss_cents = max(cfg.bounds.min_stop_cents, params.stop_loss_cents - step)
        if params.stop_loss_cents != old:
            changes["stop_loss"] = f"{old} -> {params.stop_loss_cents}"
        if weighted_mean < 0:
            old = params.wake_minutes_before_close
            params.wake_minutes_before_close = max(cfg.bounds.min_wake_minutes, params.wake_minutes_before_close - 1)
            if params.wake_minutes_before_close != old:
                changes["wake_minutes"] = f"{old} -> {params.wake_minutes_before_close}"
    elif direction == "relax":
        old = params.entry_threshold_cents
        params.entry_threshold_cents = max(cfg.bounds.min_entry_cents, params.entry_threshold_cents - step)
        if params.entry_threshold_cents != old:
            changes["entry_threshold"] = f"{old} -> {params.entry_threshold_cents}"
        old = params.stop_loss_cents
        params.stop_loss_cents = min(cfg.bounds.max_stop_cents, params.stop_loss_cents + step)
        if params.stop_loss_cents != old:
            changes["stop_loss"] = f"{old} -> {params.stop_loss_cents}"
        old = params.wake_minutes_before_close
        params.wake_minutes_before_close = min(cfg.bounds.max_wake_minutes, params.wake_minutes_before_close + 1)
        if params.wake_minutes_before_close != old:
            changes["wake_minutes"] = f"{old} -> {params.wake_minutes_before_close}"

    params.validate(cfg.bounds)

    by_cat: dict[str, dict[str, Any]] = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "pnls": []})
    for t in window:
        cat = t.get("category", "unknown")
        by_cat[cat]["trades"] += 1
        by_cat[cat]["wins"] += 1 if t.get("pnl", 0) > 0 else 0
        by_cat[cat]["pnl"] += t.get("pnl", 0)

    ranked = []
    for cat, stats in by_cat.items():
        if stats["trades"] < cfg.adaptation.min_category_samples:
            continue
        wr = stats["wins"] / stats["trades"]
        expectancy = stats["pnl"] / stats["trades"]
        score = expectancy * math.sqrt(stats["trades"])
        ranked.append((cat, wr, expectancy, score))
    ranked.sort(key=lambda x: x[3], reverse=True)
    preferred = [cat for cat, _, _, score in ranked if score > 0][:cfg.adaptation.max_preferred_categories]

    save_adaptive_strategy(params, preferred)

    result = {
        "adapted": bool(changes), "direction": direction, "changes": changes,
        "new_entry_cents": params.entry_threshold_cents,
        "new_stop_cents": params.stop_loss_cents,
        "new_wake_minutes": params.wake_minutes_before_close,
        "preferred_categories": preferred, "window_size": len(window),
        "weighted_win_rate": round(weighted_wr, 4),
        "simple_win_rate": round(simple_wr, 4),
        "weighted_mean_pnl": round(weighted_mean, 4),
        "confidence_interval": _confidence_interval(simple_wr, len(pnls)),
    }
    if changes:
        logger.info("Strategy adapted: %s", changes)
    else:
        logger.info("Strategy held (direction=%s, wr=%.3f)", direction, simple_wr)
    return result
