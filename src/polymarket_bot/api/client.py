"""Polymarket API client: fetch markets and prices."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

from polymarket_bot.core import BotConfig, Market, PricePoint

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, calls_per_second: float = 5.0):
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


class PolymarketClient:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._rate = RateLimiter(calls_per_second=4.0)

    def _get(self, url: str, params: dict[str, Any] | None = None,
             retries: int = 3) -> Any:
        last_error: Exception | None = None
        for attempt in range(retries):
            self._rate.wait()
            try:
                resp = self.session.get(url, params=params,
                                        timeout=self.cfg.requests_timeout_seconds)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    last_error = exc
                    continue
                last_error = exc
            except Exception as exc:
                last_error = exc
                time.sleep(0.75 * (attempt + 1))
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _parse_end_time(raw: dict[str, Any]) -> datetime:
        for key in ("endDate", "endTime", "end_date_iso", "end_time", "endDateIso"):
            val = raw.get(key)
            if not val:
                continue
            if isinstance(val, (int, float)):
                return datetime.fromtimestamp(float(val), tz=timezone.utc)
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                except ValueError:
                    continue
        raise ValueError("No parseable end time")

    @staticmethod
    def _get_token_ids(raw: dict[str, Any]) -> tuple[str, str]:
        """Extract YES/NO token IDs. clobTokenIds comes as a JSON string, not a list."""
        clob_ids = raw.get("clobTokenIds") or []
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except (ValueError, TypeError):
                clob_ids = []
        if isinstance(clob_ids, list) and len(clob_ids) >= 2:
            return str(clob_ids[0]), str(clob_ids[1])
        # Fallback: look in tokens array
        tokens = raw.get("tokens") or []
        yes_id, no_id = "", ""
        for token in tokens:
            outcome = str(token.get("outcome", "")).strip().lower()
            token_id = str(token.get("token_id") or token.get("tokenId") or "")
            if outcome == "yes" and token_id:
                yes_id = token_id
            elif outcome == "no" and token_id:
                no_id = token_id
        return yes_id, no_id

    def _parse_market(self, raw: dict[str, Any]) -> Market | None:
        try:
            end_time = self._parse_end_time(raw)
        except (ValueError, KeyError):
            return None

        yes_id, no_id = self._get_token_ids(raw)
        if not yes_id or not no_id:
            return None

        market_id = str(raw.get("id") or raw.get("conditionId") or "")
        if not market_id:
            return None

        volume = float(raw.get("volumeNum") or raw.get("volume") or 0)

        # Category from events array or slug
        category = "uncategorized"
        events = raw.get("events") or []
        if events and isinstance(events, list):
            category = str(events[0].get("slug") or events[0].get("title") or "uncategorized")
        if category == "uncategorized":
            category = str(raw.get("groupItemTitle") or raw.get("slug") or "uncategorized")

        return Market(
            market_id=market_id,
            question=str(raw.get("question") or raw.get("title") or ""),
            end_time=end_time,
            volume_usd=volume,
            category=category.lower(),
            yes_token_id=yes_id,
            no_token_id=no_id,
            active=bool(raw.get("active", True)),
            slug=str(raw.get("slug") or ""),
        )

    def fetch_open_markets(self) -> list[Market]:
        """Paginate through Gamma API, parse all open markets."""
        all_markets: list[Market] = []
        total_rows = 0
        offset = 0
        page_size = 500

        while True:
            params = {"closed": "false", "active": "true",
                      "limit": str(page_size), "offset": str(offset)}
            try:
                payload = self._get(f"{self.cfg.gamma_base_url}/markets", params=params)
            except Exception as exc:
                logger.error("Failed to fetch markets at offset %d: %s", offset, exc)
                break

            rows = payload if isinstance(payload, list) else payload.get("data", [])
            if not rows:
                break

            for raw in rows:
                total_rows += 1
                market = self._parse_market(raw)
                if market:
                    all_markets.append(market)

            if len(rows) < page_size:
                break
            offset += page_size

        logger.info("Fetched %d rows → %d parsed markets", total_rows, len(all_markets))
        return all_markets

    def fetch_price(self, token_id: str) -> float:
        try:
            payload = self._get(f"{self.cfg.clob_base_url}/midpoint",
                                params={"token_id": token_id})
            mid = payload.get("mid") if isinstance(payload, dict) else payload
            return float(mid)
        except Exception:
            return 0.0

    def fetch_market_prices(self, market: Market) -> PricePoint:
        ts = datetime.now(timezone.utc)
        yes_price = self.fetch_price(market.yes_token_id)
        no_price = self.fetch_price(market.no_token_id)
        spread = abs(yes_price + no_price - 1.0) if (yes_price and no_price) else 0.0
        return PricePoint(ts=ts, yes=yes_price, no=no_price, spread=spread,
                          volume_at_snapshot=market.volume_usd)
