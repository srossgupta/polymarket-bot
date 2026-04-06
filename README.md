# Polymarket High-Probability Paper Trading Bot

Algorithmic paper trading bot for [Polymarket](https://polymarket.com/) prediction markets. Targets high-probability markets (>95c) near expiry for consistent, low-risk returns.

## How It Works

```
Scan all markets ──► Filter: closing <24h, non-crypto, vol >$100k
                           │
                     Sleep until T-6 min
                           │
                     Wake & poll prices (1s intervals)
                           │
              YES or NO ≥ 95c? ──► Buy (half-Kelly sized, max $100)
                           │
              Price < 70c? ──► Stop-loss sell
                           │
              Market closes ──► Settle at $1 or $0
                           │
              Adapt strategy ──► Adjust entry/stop/wake params
```

## Strategy

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Entry threshold | 95c | 90-98c | Minimum price to enter |
| Stop loss | 70c | 55-85c | Exit if price falls below |
| Wake window | T-6 min | 3-15 min | How early to start monitoring |
| Max per market | $100 | - | Position size cap |
| Min volume | $100,000 | - | Liquidity filter |

## Quantitative Features

- **Kelly Criterion** — optimal position sizing based on edge & odds
- **Expected Value** — EV calculation filters negative-EV entries
- **Volatility-adjusted stops** — wider stops in choppy markets to avoid whipsaw
- **Price velocity** — momentum tracking boosts/reduces entry confidence
- **Sharpe ratio** — risk-adjusted return measurement
- **Monte Carlo** — 1000-path simulation for confidence intervals on P&L

## Self-Correcting Loop

After trades accumulate, the bot auto-adjusts using **exponential decay weighting** (recent trades count more) and **Wilson confidence intervals** (only adapts when statistically significant):

- **Losing?** → Tighten entry (require higher prices), widen stop
- **Winning?** → Relax entry (capture more trades), tighten stop
- **Category tracking** → Focus on profitable categories, avoid losers

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run backtest (uses synthetic data if no recorded snapshots)
python -m polymarket_bot --backtest

# Parameter sweep (finds optimal entry/stop/wake)
python -m polymarket_bot --sweep

# Run one daily cycle (scan → monitor → trade → adapt)
python -m polymarket_bot --paper-once

# Run forever on daily schedule
python -m polymarket_bot --run-loop
```

## Analytics & Tuning

```bash
# Full P&L report
python -m polymarket_bot --pnl

# Category performance ranking
python -m polymarket_bot --categories

# Parameter sensitivity analysis
python -m polymarket_bot --sensitivity

# What-if: instantly see P&L with different params
python -m polymarket_bot --whatif --entry 93 --stop 65
python -m polymarket_bot --whatif --wake 15

# Equity curve
python -m polymarket_bot --equity
```

## Backtest Results (Synthetic Data)

```
Trades:     20          Win Rate:   85%
Total P&L:  $20.87      Sharpe:     0.24
Max DD:     $22.83      Expectancy: $1.04/trade

Monte Carlo (1000 sims, 100 trades each):
  Median final value:  $2,103
  P(profit):           98.3%
  P(ruin):             0.0%
  95th pctl drawdown:  $55.78
```

## Project Structure

```
polymarket_bot/
├── README.md
├── requirements.txt
├── src/
│   └── polymarket_bot/
│       ├── __init__.py          # Package entry
│       ├── __main__.py          # python -m polymarket_bot
│       ├── cli.py               # Command-line interface
│       ├── engine.py            # Orchestration (scan → monitor → trade → adapt)
│       ├── core/                # Data models & configuration
│       │   ├── models.py        # Market, Position, PricePoint, PriceSeries, TradeEvent
│       │   └── config.py        # BotConfig, StrategyParams, ParamBounds
│       ├── api/                 # External API
│       │   └── client.py        # Polymarket API (paginated, rate-limited)
│       ├── trading/             # Trading logic
│       │   ├── strategy.py      # Entry signals, stop-loss, Kelly, EV, volatility
│       │   ├── paper.py         # Paper portfolio with full P&L tracking
│       │   └── adaptive.py      # Self-correction (decay weighting, Wilson CIs)
│       ├── data/                # Persistence & analytics
│       │   ├── storage.py       # SQLite storage with indexed queries
│       │   └── analytics.py     # P&L reports, category ranking, what-if
│       └── backtest/            # Backtesting
│           └── engine.py        # Snapshot replay, parameter sweep, Monte Carlo
├── tests/
│   ├── test_strategy.py
│   ├── test_backtest.py
│   ├── test_paper.py
│   ├── test_adaptive.py
│   └── test_engine.py
└── data/                        # Runtime data (gitignored)
    └── polymarket.db
```

## Data Storage

All data in SQLite (`data/polymarket.db`):

| Table | Contents |
|-------|----------|
| `trades` | Every trade with entry/exit price, P&L, hold time, velocity, volatility |
| `snapshots` | Price observations for all monitored markets |
| `watchlist` | Daily scan results |
| `metrics` | Run summaries, backtest results |
| `performance` | Point-in-time portfolio snapshots |

## Testing

```bash
pytest tests/ -v
```

68 tests covering strategy logic, backtesting, portfolio management, self-correction, and engine orchestration.

## Notes

- **Paper trading only** — no real orders are placed
- Requires network access for live market scanning via Polymarket's public API
- Crypto markets are automatically excluded (30+ keyword filter)
- Concurrent monitoring handles multiple markets closing simultaneously
