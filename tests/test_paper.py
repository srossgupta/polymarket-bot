"""Tests for paper portfolio."""

import pytest
from datetime import datetime, timedelta, timezone

from polymarket_bot.core import BotConfig, Market, PricePoint, Side, TradeType
from polymarket_bot.trading import PaperPortfolio
from polymarket_bot.data.storage import init_db


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("polymarket_bot.data.storage.DB_PATH", db_path)
    monkeypatch.setattr("polymarket_bot.core.config.DB_PATH", db_path)
    import polymarket_bot.data.storage as storage_mod
    if hasattr(storage_mod._local, "conn"):
        storage_mod._local.conn = None
    init_db()


def _make_market(market_id="mkt1", category="sports") -> Market:
    return Market(market_id=market_id, question="Test market",
                  end_time=datetime.now(timezone.utc) + timedelta(hours=1),
                  volume_usd=200_000, category=category,
                  yes_token_id="yes_tok", no_token_id="no_tok")


class TestPaperPortfolio:
    def test_open_position(self):
        cfg = BotConfig()
        p = PaperPortfolio(cfg)
        pos = p.open_position(_make_market(), Side.YES, 0.96, "test")
        assert pos is not None and pos.side == Side.YES and p.cash < cfg.starting_cash

    def test_duplicate_position_rejected(self):
        p = PaperPortfolio(BotConfig())
        m = _make_market()
        assert p.open_position(m, Side.YES, 0.96, "t") is not None
        assert p.open_position(m, Side.YES, 0.97, "t2") is None

    def test_close_winning_position(self):
        p = PaperPortfolio(BotConfig())
        p.open_position(_make_market(), Side.YES, 0.96, "test")
        event = p.close_position(_make_market(), PricePoint(ts=datetime.now(timezone.utc), yes=0.99, no=0.01),
                                  "win", TradeType.SELL_MARKET)
        assert event is not None and event.pnl > 0 and p.wins == 1

    def test_close_losing_position(self):
        p = PaperPortfolio(BotConfig())
        p.open_position(_make_market(), Side.YES, 0.96, "test")
        event = p.close_position(_make_market(), PricePoint(ts=datetime.now(timezone.utc), yes=0.65, no=0.35),
                                  "stop", TradeType.STOP_LOSS)
        assert event is not None and event.pnl < 0 and p.losses == 1

    def test_max_positions_enforced(self):
        cfg = BotConfig()
        cfg.max_open_positions = 2
        p = PaperPortfolio(cfg)
        assert p.open_position(_make_market("m1"), Side.YES, 0.96, "t") is not None
        assert p.open_position(_make_market("m2"), Side.YES, 0.96, "t") is not None
        assert p.open_position(_make_market("m3"), Side.YES, 0.96, "t") is None

    def test_win_rate_calculation(self):
        p = PaperPortfolio(BotConfig())
        for i in range(3):
            m = _make_market(f"win_{i}")
            p.open_position(m, Side.YES, 0.96, "test")
            p.close_position(m, PricePoint(ts=datetime.now(timezone.utc), yes=0.99, no=0.01), "win")
        m = _make_market("loss_0")
        p.open_position(m, Side.YES, 0.96, "test")
        p.close_position(m, PricePoint(ts=datetime.now(timezone.utc), yes=0.65, no=0.35), "loss")
        assert p.win_rate == 0.75

    def test_sharpe_calculation(self):
        p = PaperPortfolio(BotConfig())
        p._pnl_history = [5.0, 3.0, -2.0, 4.0, 1.0]
        p.total_trades = 5
        assert p.sharpe_estimate != 0

    def test_position_peak_tracking(self):
        p = PaperPortfolio(BotConfig())
        pos = p.open_position(_make_market(), Side.YES, 0.96, "test")
        pos.update_peak(0.98)
        assert pos.peak_price == 0.98
        pos.update_peak(0.95)
        assert pos.peak_price == 0.98 and pos.ticks_held == 2
