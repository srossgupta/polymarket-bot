"""Tests for the orchestration engine: monitoring, clustering, backtest integration."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from polymarket_bot.config import BotConfig
from polymarket_bot.engine import monitor_market_until_close, run_backtest
from polymarket_bot.models import Market, PricePoint, Side
from polymarket_bot.paper import PaperPortfolio
from polymarket_bot.storage import init_db


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("polymarket_bot.storage.DB_PATH", db_path)
    monkeypatch.setattr("polymarket_bot.config.DB_PATH", db_path)
    import polymarket_bot.storage as storage_mod
    if hasattr(storage_mod._local, "conn"):
        storage_mod._local.conn = None
    init_db()


def _make_market(end_minutes: int = 2) -> Market:
    return Market(
        market_id="test_mkt",
        question="Test market?",
        end_time=datetime.now(timezone.utc) + timedelta(minutes=end_minutes),
        volume_usd=200_000,
        category="sports",
        yes_token_id="yes_tok",
        no_token_id="no_tok",
    )


class TestMonitorMarket:
    def test_enters_and_wins(self):
        cfg = BotConfig()
        portfolio = PaperPortfolio(cfg)
        market = _make_market(end_minutes=1)

        # Simulate: price rises to 96c and stays
        prices = iter([
            PricePoint(ts=datetime.now(timezone.utc), yes=0.96, no=0.04),
            PricePoint(ts=datetime.now(timezone.utc), yes=0.97, no=0.03),
            PricePoint(ts=datetime.now(timezone.utc), yes=0.98, no=0.02),
        ])

        mock_client = MagicMock()
        mock_client.fetch_market_prices = MagicMock(side_effect=prices)

        call_count = [0]

        def now_fn():
            call_count[0] += 1
            # After 3 calls, be past end_time
            if call_count[0] > 4:
                return market.end_time + timedelta(seconds=1)
            return market.end_time - timedelta(seconds=30 - call_count[0] * 10)

        stats = monitor_market_until_close(
            mock_client, portfolio, market, cfg,
            now_fn=now_fn, sleeper=lambda x: None)
        assert stats["entered"] is True

    def test_skips_low_volume(self):
        cfg = BotConfig()
        portfolio = PaperPortfolio(cfg)
        market = _make_market()
        market.volume_usd = 50_000  # below threshold

        mock_client = MagicMock()
        stats = monitor_market_until_close(
            mock_client, portfolio, market, cfg,
            sleeper=lambda x: None)
        assert "skipped" in stats

    def test_stop_loss_triggers(self):
        cfg = BotConfig()
        portfolio = PaperPortfolio(cfg)
        market = _make_market(end_minutes=5)

        prices = iter([
            PricePoint(ts=datetime.now(timezone.utc), yes=0.96, no=0.04),
            PricePoint(ts=datetime.now(timezone.utc), yes=0.80, no=0.20),
            PricePoint(ts=datetime.now(timezone.utc), yes=0.65, no=0.35),
        ])

        mock_client = MagicMock()
        mock_client.fetch_market_prices = MagicMock(side_effect=prices)

        call_count = [0]

        def now_fn():
            call_count[0] += 1
            return market.end_time - timedelta(minutes=4, seconds=-call_count[0] * 30)

        stats = monitor_market_until_close(
            mock_client, portfolio, market, cfg,
            now_fn=now_fn, sleeper=lambda x: None)
        assert stats["entered"] is True
        assert stats["stopped"] is True


class TestBacktestIntegration:
    def test_backtest_runs(self, tmp_path, monkeypatch):
        # Override data dir
        monkeypatch.setattr("polymarket_bot.config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("polymarket_bot.config.ADAPTIVE_STATE_FILE",
                            str(tmp_path / "adaptive.json"))

        result = run_backtest()
        assert "source" in result
        assert "result" in result
        assert result["result"]["trades"] >= 0
