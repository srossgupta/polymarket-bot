"""Tests for paper portfolio: position management, P&L tracking."""

import pytest
from datetime import datetime, timedelta, timezone

from polymarket_bot.config import BotConfig
from polymarket_bot.models import Market, PricePoint, Side, TradeType
from polymarket_bot.paper import PaperPortfolio
from polymarket_bot.storage import init_db


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    """Use temp DB for all tests."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("polymarket_bot.storage.DB_PATH", db_path)
    monkeypatch.setattr("polymarket_bot.config.DB_PATH", db_path)
    # Reset thread-local connection
    import polymarket_bot.storage as storage_mod
    if hasattr(storage_mod._local, "conn"):
        storage_mod._local.conn = None
    init_db()


def _make_market(market_id="mkt1", category="sports") -> Market:
    return Market(
        market_id=market_id,
        question="Test market",
        end_time=datetime.now(timezone.utc) + timedelta(hours=1),
        volume_usd=200_000,
        category=category,
        yes_token_id="yes_tok",
        no_token_id="no_tok",
    )


class TestPaperPortfolio:
    def test_open_position(self):
        cfg = BotConfig()
        portfolio = PaperPortfolio(cfg)
        market = _make_market()
        pos = portfolio.open_position(market, Side.YES, 0.96, "test")
        assert pos is not None
        assert pos.side == Side.YES
        assert pos.entry_price == 0.96
        assert market.market_id in portfolio.open_positions
        assert portfolio.cash < cfg.starting_cash

    def test_duplicate_position_rejected(self):
        cfg = BotConfig()
        portfolio = PaperPortfolio(cfg)
        market = _make_market()
        pos1 = portfolio.open_position(market, Side.YES, 0.96, "test")
        pos2 = portfolio.open_position(market, Side.YES, 0.97, "test2")
        assert pos1 is not None
        assert pos2 is None

    def test_close_winning_position(self):
        cfg = BotConfig()
        portfolio = PaperPortfolio(cfg)
        market = _make_market()
        portfolio.open_position(market, Side.YES, 0.96, "test")

        exit_point = PricePoint(ts=datetime.now(timezone.utc), yes=0.99, no=0.01)
        event = portfolio.close_position(market, exit_point, "win",
                                         trade_type=TradeType.SELL_MARKET)
        assert event is not None
        assert event.pnl > 0
        assert portfolio.wins == 1
        assert portfolio.total_pnl > 0

    def test_close_losing_position(self):
        cfg = BotConfig()
        portfolio = PaperPortfolio(cfg)
        market = _make_market()
        portfolio.open_position(market, Side.YES, 0.96, "test")

        exit_point = PricePoint(ts=datetime.now(timezone.utc), yes=0.65, no=0.35)
        event = portfolio.close_position(market, exit_point, "stop_loss",
                                         trade_type=TradeType.STOP_LOSS)
        assert event is not None
        assert event.pnl < 0
        assert portfolio.losses == 1

    def test_max_positions_enforced(self):
        cfg = BotConfig()
        cfg.max_open_positions = 2
        portfolio = PaperPortfolio(cfg)

        m1 = _make_market("m1")
        m2 = _make_market("m2")
        m3 = _make_market("m3")

        assert portfolio.open_position(m1, Side.YES, 0.96, "t") is not None
        assert portfolio.open_position(m2, Side.YES, 0.96, "t") is not None
        assert portfolio.open_position(m3, Side.YES, 0.96, "t") is None

    def test_win_rate_calculation(self):
        cfg = BotConfig()
        portfolio = PaperPortfolio(cfg)

        for i in range(3):
            m = _make_market(f"win_{i}")
            portfolio.open_position(m, Side.YES, 0.96, "test")
            exit_point = PricePoint(ts=datetime.now(timezone.utc), yes=0.99, no=0.01)
            portfolio.close_position(m, exit_point, "win")

        m = _make_market("loss_0")
        portfolio.open_position(m, Side.YES, 0.96, "test")
        exit_point = PricePoint(ts=datetime.now(timezone.utc), yes=0.65, no=0.35)
        portfolio.close_position(m, exit_point, "loss")

        assert portfolio.win_rate == 0.75

    def test_sharpe_calculation(self):
        cfg = BotConfig()
        portfolio = PaperPortfolio(cfg)
        portfolio._pnl_history = [5.0, 3.0, -2.0, 4.0, 1.0]
        portfolio.total_trades = 5
        assert portfolio.sharpe_estimate != 0

    def test_position_peak_tracking(self):
        cfg = BotConfig()
        portfolio = PaperPortfolio(cfg)
        m = _make_market()
        pos = portfolio.open_position(m, Side.YES, 0.96, "test")
        assert pos is not None
        pos.update_peak(0.98)
        assert pos.peak_price == 0.98
        pos.update_peak(0.95)
        assert pos.peak_price == 0.98  # doesn't decrease
        assert pos.ticks_held == 2
