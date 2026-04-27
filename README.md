# Polymarket Paper Trading Bot

Algorithmic paper trading bot for [Polymarket](https://polymarket.com/) prediction markets.

Targets markets closing within the next 24 hours, enters when price hits the 95¢–99.5¢ band 5 minutes before expiry, holds to resolution. No stop loss — let the market decide.

---

## How It Works

```
Every cycle:

  1. Fetch all open markets (~47,000)
         │
  2. Filter: closing within 24h, non-crypto, non-weather, volume > $5k
         │
  3. For each market — sleep until T-5 min before close
         │
  4. Wake up → poll price every 300ms
         │
  5. YES or NO price in 95¢–99.5¢? → Enter (flat $50 per market)
         │
  6. Market closes → settle at final price (win $X or lose $Y)
         │
  7. Watchdog checks win rate every 30s → auto-adapts entry threshold
```

---

## Strategy

| Parameter | Value | Description |
|-----------|-------|-------------|
| Entry band | 95¢ – 99.5¢ | Price must be high-prob but not yet fully resolved |
| Wake window | 5 min before close | Only monitor in the final window |
| Poll interval | 300ms | Fast price polling via CLOB API |
| Max per trade | $50 | Flat position sizing |
| Min volume | $5,000 | Liquidity filter |
| Stop loss | None | Hold to expiry — market resolves it |

**Why no stop loss?** Near-expiry binary markets at 95¢+ rarely reverse. Stop losses cause more losses than they prevent (verified from live data: 2 stop-loss trades = -$79.80 of -$76.79 total losses).

---

## Self-Correcting Watchdog

A background thread watches live performance every 30 seconds:

- **Win rate < 40%** → tighten entry threshold (require higher prices), restart
- **3 losses in a row** → tighten entry threshold, restart
- **Everything fine** → keep running

Entry threshold auto-adjusts between 90¢ and 97¢ based on what's working.

---

## Quick Start

```bash
# Install
pip install -e src/

# Run with live watchdog (recommended)
python -m polymarket_bot --watchdog

# Run one cycle and exit
python -m polymarket_bot --paper-once

# Run forever on daily schedule (no watchdog)
python -m polymarket_bot --run-loop
```

---

## Analytics

```bash
# Full P&L report
python -m polymarket_bot --pnl

# Category performance ranking
python -m polymarket_bot --categories

# Parameter sensitivity by entry price and hour
python -m polymarket_bot --sensitivity

# What-if with different params (uses recorded snapshots)
python -m polymarket_bot --whatif --entry 93 --wake 7

# Equity curve over time
python -m polymarket_bot --equity

# Backtest with recorded or synthetic data
python -m polymarket_bot --backtest

# Parameter sweep (tries all entry/wake combinations)
python -m polymarket_bot --sweep
```

---

## Project Structure

```
polymarket_bot/
├── src/polymarket_bot/
│   ├── api/
│   │   └── client.py        # Gamma + CLOB API client (paginated, rate-limited)
│   ├── core/
│   │   ├── models.py        # Market, Position, PricePoint, TradeEvent, etc.
│   │   └── config.py        # BotConfig, StrategyParams — loads from adaptive_state.json
│   ├── trading/
│   │   ├── strategy.py      # Market filters + entry band logic
│   │   ├── paper.py         # Paper portfolio — tracks P&L, wins, losses
│   │   └── adaptive.py      # Self-correction — adjusts entry threshold
│   ├── data/
│   │   ├── storage.py       # SQLite — trades, snapshots, watchlist, metrics
│   │   └── analytics.py     # P&L reports, category ranking, what-if
│   ├── backtest/
│   │   └── engine.py        # Snapshot replay, parameter sweep, Monte Carlo
│   ├── engine.py            # Main loop: scan → monitor → trade → adapt
│   ├── watchdog.py          # Live self-correcting run loop
│   └── cli.py               # CLI entry point
├── data/
│   ├── polymarket.db        # SQLite database (gitignored)
│   └── adaptive_state.json  # Saved strategy params (entry threshold, wake time)
└── requirements.txt
```

---

## Data Storage

All state in `data/polymarket.db` (SQLite):

| Table | Contents |
|-------|----------|
| `trades` | Every BUY and FORCED_CLOSE with price, P&L, hold duration |
| `snapshots` | Price observations for every monitored market |
| `watchlist` | Daily scan results |
| `metrics` | Cycle summaries and backtest results |
| `performance` | Portfolio snapshots over time |

---

## Market Filters

Markets are excluded if they match any of these:

- **Crypto**: bitcoin, eth, solana, dogecoin, defi, nft, blockchain, binance, coinbase, and 20+ more
- **Weather**: temperature, rainfall, precipitation, humidity, degrees, snowfall, etc.
- **Already resolved**: price ≥ 99¢ at pre-screen (skipped before monitoring starts)
- **No liquidity**: volume < $5,000

---

## Notes

- **Paper trading only** — no real orders are placed
- No API key required — uses Polymarket's public Gamma and CLOB APIs
- Concurrent monitoring handles multiple markets closing at the same time (up to 8 threads)
- Crypto markets are filtered at two levels: keyword match on question/category/slug, and volume pre-screen
