# Polymarket High-Probability Paper Trading Bot

Algorithmic paper trading bot for Polymarket prediction markets. Targets high-probability markets (>95c) near expiry for consistent returns.

## Strategy

1. **Daily Scan**: At a fixed time, scan all active Polymarket markets (excluding crypto)
2. **Filter**: Select markets closing within 24 hours
3. **Sleep/Wake**: Sleep until T-6 minutes before each market closes
4. **Volume Check**: Only trade markets with volume > $100,000
5. **Entry**: When YES or NO price >= 95c, place a paper limit buy
6. **Position Sizing**: Half-Kelly criterion, capped at $100/market
7. **Stop Loss**: If price falls below 70c, market sell
8. **Expiry**: Force-close any open position at market close

## Quantitative Features

- **Kelly Criterion**: Optimal position sizing based on implied probability
- **Expected Value**: EV calculation for each entry signal
- **Volatility Analysis**: Rolling volatility with adjusted stop-losses
- **Price Velocity**: Momentum tracking for confidence scoring
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Monte Carlo Simulation**: Confidence intervals on expected P&L

## Self-Correcting Loop

After trades accumulate, the bot auto-adjusts using exponential decay weighting and Wilson confidence intervals:

- `entry_threshold_cents` (90-98): tightens when losing, relaxes when winning
- `stop_loss_cents` (55-85): widens to avoid whipsaw, tightens when profitable
- `wake_minutes_before_close` (3-15): adjusts monitoring window
- `preferred_categories`: focuses on profitable categories

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest (synthetic data if no recorded snapshots)
python -m polymarket_bot --backtest

# Parameter sweep optimization
python -m polymarket_bot --sweep

# Run one daily cycle (scan + monitor + trade + adapt)
python -m polymarket_bot --paper-once

# Run forever on daily schedule
python -m polymarket_bot --run-loop

# Analytics
python -m polymarket_bot --pnl              # Full P&L report
python -m polymarket_bot --categories        # Category ranking
python -m polymarket_bot --sensitivity       # Parameter sensitivity
python -m polymarket_bot --whatif --entry 93 --stop 65  # What-if analysis
python -m polymarket_bot --equity            # Equity curve

# Tests
pytest polymarket_bot/tests/ -v
```

## Data Storage (SQLite)

All data stored in `polymarket_bot/data/polymarket.db`:

- **trades**: Full trade log with entry/exit prices, PnL, hold duration, velocity, volatility
- **snapshots**: Price observations for every monitored market
- **watchlist**: Daily scan results
- **metrics**: Run summaries and backtest results
- **performance**: Point-in-time portfolio snapshots

## Parameter Tuning

The bot logs every trade with rich metadata so you can instantly re-evaluate P&L when changing parameters:

```bash
# "What if we used T-15 instead of T-6?"
python -m polymarket_bot --whatif --wake 15

# "What if entry was 93c instead of 95c?"
python -m polymarket_bot --whatif --entry 93

# "What if stop was 65c instead of 70c?"
python -m polymarket_bot --whatif --stop 65
```

## Architecture

```
polymarket_bot/
├── models.py      # Data classes (Market, Position, PriceSeries, TradeEvent)
├── config.py      # Configuration with validation and adaptive state
├── client.py      # Polymarket API client (paginated, rate-limited)
├── strategy.py    # Trading rules (Kelly, EV, volatility, stop-loss)
├── paper.py       # Paper portfolio with full lifecycle tracking
├── backtest.py    # Snapshot replay, parameter sweep, Monte Carlo
├── adaptive.py    # Self-correction with exponential decay + confidence intervals
├── analytics.py   # P&L reports, category ranking, what-if analysis
├── engine.py      # Orchestration with concurrent market monitoring
├── storage.py     # SQLite storage with analytics queries
├── cli.py         # Command-line interface
└── tests/         # Comprehensive test suite
```
