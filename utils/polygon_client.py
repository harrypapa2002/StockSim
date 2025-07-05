"""
Enhanced Polygon.io API Client for StockSim with Crypto Support

This module provides a comprehensive client for accessing financial data from Polygon.io,
supporting stocks, crypto, options, and forex data with caching and rate limiting.

Key Features:
- Stock market data (OHLCV aggregates, ticks, fundamentals)
- Cryptocurrency data (OHLCV aggregates, snapshots)
- News data with sentiment analysis and keyword extraction
- Corporate fundamentals (earnings, balance sheets, cash flow)
- Comprehensive caching system for efficient data retrieval
- Rate limiting and error handling for production use
- Support for multiple time resolutions and date ranges
- Unified data format for StockSim integration
"""

from __future__ import annotations

import os
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
import pytz

from dotenv import load_dotenv
import requests

from polygon.stocks import StocksClient

from utils.time_utils import parse_datetime_utc, parse_interval_to_timedelta


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation for deduplication."""
    return re.sub(r"\W+", "", text.lower())


def parse_resolution_polygon(interval: str) -> Tuple[int, str]:
    """
    Convert our flexible syntax to Polygon API format.
    Returns (multiplier, timespan) tuple.

    Examples:
    '1s' -> (1, 'second')
    '1m' -> (1, 'minute')
    '5m' -> (5, 'minute')
    '1h' -> (1, 'hour')
    '1d' -> (1, 'day')
    '1w' -> (1, 'week')
    '1mo' -> (1, 'month')
    """
    interval = interval.lower().strip()

    # Extract number and unit
    match = re.match(r'^(\d+)(s|sec|m|min|h|hour|d|day|w|week|mo|month)$', interval)
    if not match:
        raise ValueError(f"Unsupported interval format: {interval}")

    multiplier = int(match.group(1))
    unit = match.group(2)

    # Map to Polygon timespan
    timespan_map = {
        's': 'second', 'sec': 'second',
        'm': 'minute', 'min': 'minute',
        'h': 'hour', 'hour': 'hour',
        'd': 'day', 'day': 'day',
        'w': 'week', 'week': 'week',
        'mo': 'month', 'month': 'month'
    }

    if unit not in timespan_map:
        raise ValueError(f"Unsupported time unit: {unit}")

    return multiplier, timespan_map[unit]


class PolygonClient:
    """
    Comprehensive Polygon.io client supporting both stocks and cryptocurrency data.

    Leverages your subscription features:
    - Unlimited API calls
    - 5 years historical data
    - 15-minute delayed real-time data
    - Technical indicators
    - Minute-level aggregates
    - Snapshots
    - Reference data
    - Fundamentals (stocks only)
    - Corporate actions (stocks only)
    - Cryptocurrency data (separate endpoints)
    """

    def __init__(self, api_key: Optional[str] = None, base_cache_dir: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Please set POLYGON_API_KEY in your environment.")

        # Initialize StocksClient for stock market data
        self.stocks_client = StocksClient(api_key=self.api_key)

        # Set up caching
        self.base_cache_dir = base_cache_dir or os.path.join(os.path.dirname(__file__), "..", "data", "polygon")
        self.base_cache_dir = os.path.abspath(self.base_cache_dir)

        self.cache_dirs = {
            "aggregates": os.path.join(self.base_cache_dir, "aggregates"),
            "crypto_aggregates": os.path.join(self.base_cache_dir, "crypto_aggregates"),
            "ticker_details": os.path.join(self.base_cache_dir, "ticker_details"),
            "corporate_actions": os.path.join(self.base_cache_dir, "corporate_actions"),
            "fundamentals": os.path.join(self.base_cache_dir, "fundamentals"),
            "news": os.path.join(self.base_cache_dir, "news")
        }

        for folder in self.cache_dirs.values():
            os.makedirs(folder, exist_ok=True)

    def _get_cache_path(self, filename: str, data_type: str) -> str:
        """Construct the full cache file path for a given data type."""
        return os.path.join(self.cache_dirs[data_type], filename)

    def _load_from_cache(self, cache_path: str) -> Optional[Any]:
        """Load data from cache if it exists."""
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    def _save_to_cache(self, data: Any, cache_path: str) -> None:
        """Save data to cache."""
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _direct_api_call(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make direct API call for endpoints not covered by StocksClient."""
        if params is None:
            params = {}
        params["apikey"] = self.api_key

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if data.get("status") == "ERROR":
            raise ValueError(f"Polygon API error: {data.get('error', 'Unknown error')}")

        return data

    def get_related_tickers(self, symbol: str, use_cache: bool = True) -> List[str]:
        """
        Get related tickers for a given symbol (stocks only).
        """
        cache_key = f"{symbol}_related_tickers"
        cache_path = self._get_cache_path(f"{cache_key}.json", "ticker_details")

        if use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                return cached_data

        url = f"https://api.polygon.io/v1/related-companies/{symbol}"
        data = self._direct_api_call(url)

        related_tickers = [item['ticker'] for item in data.get('results', [])]

        if use_cache:
            self._save_to_cache(related_tickers, cache_path)

        return related_tickers

    def load_aggregates(
            self,
            symbol: str,
            interval: str = "1d",
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            adjusted: bool = True,
            sort: str = "asc",
            limit: int = 50000,
            use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV aggregate bars for stocks from Polygon's aggregates endpoint.

        - `interval`: e.g. "1m", "5m", "1h", "1d", etc.
        - `start_date` / `end_date`: UTC ISO strings (e.g. "2024-01-01T00:00:00").
        - They are converted → America/New_York → epoch milliseconds.
        - Response timestamps remain in UTC.

        IMPORTANT: Always returns the candle with the latest close time that occurs
        at or before start_date, followed by all subsequent candles.
        For daily intervals, ensures you get the previous trading day regardless
        of what time within the day you specify as start_date.
        """
        # 1) Parse resolution (multiplier & timespan)
        multiplier, timespan = parse_resolution_polygon(interval)

        # Parse the interval to get the duration
        interval_delta = parse_interval_to_timedelta(interval)

        # 2) Build cache filename
        sd_key = start_date.replace(":", "").replace("-", "") if start_date else "start"
        ed_key = end_date.replace(":", "").replace("-", "") if end_date else "end"
        filename = f"{symbol}_{multiplier}{timespan}_{sd_key}_{ed_key}.json"
        cache_path = self._get_cache_path(filename, "aggregates")

        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                # Filter cached results to ensure proper alignment
                if start_date:
                    return self._filter_candles_by_start_date(cached, start_date, interval_delta, timespan)
                return cached

        # 3) Convert UTC ISO strings → aware UTC → convert to America/New_York → epoch ms
        eastern = pytz.timezone("America/New_York")

        def to_et_epoch_ms(iso_str: str) -> int:
            # parse incoming ISO as UTC‐aware
            dt_utc = datetime.fromisoformat(iso_str)
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            else:
                dt_utc = dt_utc.astimezone(timezone.utc)

            # convert to ET
            dt_et = dt_utc.astimezone(eastern)

            # epoch seconds of that ET‐aware moment
            # (same absolute instant as UTC, but dt_et.timestamp() is correct)
            return int(dt_et.timestamp() * 1000)

        # For the API call, determine how far back to go based on interval type
        adjusted_start_date = start_date
        if start_date:
            start_dt_utc = parse_datetime_utc(start_date)

            if timespan == "day":
                # For daily data, go back to ensure we get the previous trading day
                # Go back by several days to account for weekends/holidays
                adjusted_start_dt = start_dt_utc - timedelta(days=10)
            elif timespan in ["week", "month"]:
                # For weekly/monthly, go back proportionally more
                if timespan == "week":
                    adjusted_start_dt = start_dt_utc - timedelta(weeks=4)
                else:  # month
                    adjusted_start_dt = start_dt_utc - timedelta(days=90)
            else:
                # For intraday intervals (minute, hour, second), use the original logic
                adjusted_start_dt = start_dt_utc - (3 * interval_delta)

            adjusted_start_date = adjusted_start_dt.isoformat()

        from_ms = to_et_epoch_ms(adjusted_start_date) if adjusted_start_date else ""
        to_ms = to_et_epoch_ms(end_date) if end_date else ""

        # 4) Build the stocks aggregates URL:
        #    /v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_ms}/{to_ms}
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/" \
              f"{multiplier}/{timespan}/{from_ms}/{to_ms}"

        params: Dict[str, Any] = {
            "adjusted": str(adjusted).lower(),  # "true" or "false"
            "sort": sort,
            "limit": limit
        }

        data = self._direct_api_call(url, params)
        results = data.get("results", [])

        candles: List[Dict[str, Any]] = []
        for item in results:
            # 't' is a Unix‐ms timestamp in UTC
            dt_utc = datetime.fromtimestamp(item["t"] / 1000, tz=timezone.utc)
            candle = {
                "timestamp": dt_utc.isoformat(),
                "open": float(item.get("o", 0)),
                "high": float(item.get("h", 0)),
                "low": float(item.get("l", 0)),
                "close": float(item.get("c", 0)),
                "volume": int(item.get("v", 0)),
                **({"transactions": int(item["n"])} if "n" in item else {}),
                **({"vwap": float(item["vw"])} if "vw" in item else {}),
            }
            candles.append(candle)

        # Filter results to ensure proper alignment
        if start_date and candles:
            candles = self._filter_candles_by_start_date(candles, start_date, interval_delta, timespan)

        # 5) Cache & return
        if use_cache:
            self._save_to_cache(candles, cache_path)

        return candles

    def load_crypto_aggregates(
            self,
            symbol: str,
            interval: str = "1d",
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            market: str = "USD",
            sort: str = "asc",
            limit: int = 50000,
            use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV aggregate bars for cryptocurrency from Polygon's crypto aggregates endpoint.

        - `symbol`: crypto symbol (e.g. "BTC", "ETH")
        - `interval`: e.g. "1m", "5m", "1h", "1d", etc.
        - `start_date` / `end_date`: UTC ISO strings (e.g. "2024-01-01T00:00:00")
        - `market`: target market currency (default: "USD")
        - Response timestamps are in UTC

        Uses Polygon's crypto aggregates endpoint:
        /v2/aggs/ticker/X:{symbol}{market}/range/{multiplier}/{timespan}/{from_ms}/{to_ms}
        """
        # 1) Parse resolution (multiplier & timespan)
        multiplier, timespan = parse_resolution_polygon(interval)

        # Parse the interval to get the duration
        interval_delta = parse_interval_to_timedelta(interval)

        # 2) Build cache filename for crypto
        sd_key = start_date.replace(":", "").replace("-", "") if start_date else "start"
        ed_key = end_date.replace(":", "").replace("-", "") if end_date else "end"
        filename = f"{symbol}_{market}_{multiplier}{timespan}_{sd_key}_{ed_key}.json"
        cache_path = self._get_cache_path(filename, "crypto_aggregates")

        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                # Filter cached results to ensure proper alignment
                if start_date:
                    return self._filter_crypto_candles_by_start_date(cached, start_date, interval_delta)
                return cached

        # 3) Convert UTC ISO strings to epoch milliseconds (crypto trades 24/7, use UTC)
        def to_utc_epoch_ms(iso_str: str) -> int:
            dt_utc = datetime.fromisoformat(iso_str)
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            else:
                dt_utc = dt_utc.astimezone(timezone.utc)
            return int(dt_utc.timestamp() * 1000)

        # For crypto, we go back a bit to ensure we get a warmup candle
        adjusted_start_date = start_date
        if start_date:
            start_dt_utc = parse_datetime_utc(start_date)
            # Go back by a few intervals to get warmup data
            adjusted_start_dt = start_dt_utc - (3 * interval_delta)
            adjusted_start_date = adjusted_start_dt.isoformat()

        from_ms = to_utc_epoch_ms(adjusted_start_date) if adjusted_start_date else ""
        to_ms = to_utc_epoch_ms(end_date) if end_date else ""

        # 4) Build the crypto aggregates URL using X: prefix
        #    /v2/aggs/ticker/X:{symbol}{market}/range/{multiplier}/{timespan}/{from_ms}/{to_ms}
        crypto_ticker = f"X:{symbol}{market}"
        url = f"https://api.polygon.io/v2/aggs/ticker/{crypto_ticker}/range/" \
              f"{multiplier}/{timespan}/{from_ms}/{to_ms}"

        params: Dict[str, Any] = {
            "sort": sort,
            "limit": limit
        }

        data = self._direct_api_call(url, params)
        results = data.get("results", [])

        candles: List[Dict[str, Any]] = []
        for item in results:
            # 't' is a Unix‐ms timestamp in UTC
            dt_utc = datetime.fromtimestamp(item["t"] / 1000, tz=timezone.utc)
            candle = {
                "timestamp": dt_utc.isoformat(),
                "open": float(item.get("o", 0)),
                "high": float(item.get("h", 0)),
                "low": float(item.get("l", 0)),
                "close": float(item.get("c", 0)),
                "volume": float(item.get("v", 0)),  # Crypto volume can be decimal
                **({"transactions": int(item["n"])} if "n" in item else {}),
                **({"vwap": float(item["vw"])} if "vw" in item else {}),
            }
            candles.append(candle)

        # Filter results to ensure proper alignment for crypto
        if start_date and candles:
            candles = self._filter_crypto_candles_by_start_date(candles, start_date, interval_delta)

        # 5) Cache & return
        if use_cache:
            self._save_to_cache(candles, cache_path)

        return candles

    def _filter_candles_by_start_date(
            self,
            candles: List[Dict[str, Any]],
            start_date: str,
            interval_delta,
            timespan: str
    ) -> List[Dict[str, Any]]:
        """
        Filter candles to return the appropriate previous candle followed by all subsequent candles.

        For daily intervals: Returns the previous trading day's candle regardless of the time
        specified in start_date, followed by all candles on or after the start_date's date.

        For intraday intervals: Returns the candle from exactly one interval before start_date,
        followed by all subsequent candles.
        """
        if not candles:
            return candles

        start_dt_utc = parse_datetime_utc(start_date)

        if timespan == "day":
            # For daily data, we want the previous trading day regardless of time
            start_date_only = start_dt_utc.date()

            # Find the latest candle before the start date
            previous_candle = None
            subsequent_candles = []

            for candle in candles:
                candle_ts = parse_datetime_utc(candle["timestamp"])
                candle_date = candle_ts.date()

                if candle_date < start_date_only:
                    # This is a potential previous candle
                    if previous_candle is None:
                        previous_candle = candle
                    else:
                        # Keep the latest one before start_date
                        prev_ts = parse_datetime_utc(previous_candle["timestamp"])
                        if candle_ts > prev_ts:
                            previous_candle = candle
                elif candle_date >= start_date_only:
                    # This is on or after our start date
                    subsequent_candles.append(candle)

            # Build result: previous candle + subsequent candles
            result = []
            if previous_candle:
                result.append(previous_candle)
            result.extend(subsequent_candles)
            return result

        else:
            # For intraday intervals, use the original logic
            target_warmup_start = start_dt_utc - interval_delta

            # Find the candle closest to this target time
            best_candle = None
            min_time_diff = float('inf')

            for candle in candles:
                candle_ts = parse_datetime_utc(candle["timestamp"])
                time_diff = abs((candle_ts - target_warmup_start).total_seconds())

                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_candle = candle

            if best_candle is None:
                return candles

            # Build result: warmup candle + all candles at or after start_date
            filtered_candles = [best_candle]

            # Add all candles whose timestamp is >= start_date
            for candle in candles:
                candle_ts = parse_datetime_utc(candle["timestamp"])
                if candle_ts >= start_dt_utc and candle != best_candle:
                    filtered_candles.append(candle)

            return filtered_candles

    def _filter_crypto_candles_by_start_date(
            self,
            candles: List[Dict[str, Any]],
            start_date: str,
            interval_delta
    ) -> List[Dict[str, Any]]:
        """
        Filter crypto candles for 24/7 markets (no market hours to consider).

        Returns the candle from one interval before start_date, followed by all subsequent candles.
        """
        if not candles:
            return candles

        start_dt_utc = parse_datetime_utc(start_date)
        target_warmup_start = start_dt_utc - interval_delta

        # Find the candle closest to this target time
        best_candle = None
        min_time_diff = float('inf')

        for candle in candles:
            candle_ts = parse_datetime_utc(candle["timestamp"])
            time_diff = abs((candle_ts - target_warmup_start).total_seconds())

            if time_diff < min_time_diff:
                min_time_diff = time_diff
                best_candle = candle

        if best_candle is None:
            return candles

        # Build result: warmup candle + all candles at or after start_date
        filtered_candles = [best_candle]

        # Add all candles whose timestamp is >= start_date
        for candle in candles:
            candle_ts = parse_datetime_utc(candle["timestamp"])
            if candle_ts >= start_dt_utc and candle != best_candle:
                filtered_candles.append(candle)

        return filtered_candles

    def load_daily_market_summary(
        self,
        date: str,
        adjusted: bool = True,
        include_otc: bool = False,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve daily OHLC, volume, and VWAP for all U.S. stocks on a given date,
        using Polygon's "Daily Market Summary" endpoint:
        GET /v2/aggs/grouped/locale/us/market/stocks/{date}

        - `date` should be "YYYY-MM-DD" (UTC).
        - `adjusted`: if True (default), returns split‐adjusted values.
        - `include_otc`: if True, includes OTC securities (default is False).
        - Caches results under "aggregates/" so repeated calls for the same date/flags are fast.
        """
        # Build cache path
        cache_key = f"daily_summary_{date}_{adjusted}_{include_otc}"
        cache_path = self._get_cache_path(f"{cache_key}.json", "aggregates")

        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                return cached

        # Construct the URL
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"
        params: Dict[str, Any] = {
            "adjusted": str(adjusted).lower(),   # "true" or "false"
            "include_otc": str(include_otc).lower()  # "true" or "false"
        }

        # Direct API call
        data = self._direct_api_call(url, params)
        results = data.get("results", [])

        summary: List[Dict[str, Any]] = []
        for item in results:
            # 't' in grouped‐daily response is Unix ms for that date
            ts_utc = datetime.fromtimestamp(item["t"] / 1000, tz=timezone.utc)
            summary.append({
                "ticker": item.get("T"),               # ticker symbol
                "timestamp": ts_utc.isoformat(),       # ISO‐formatted UTC
                "open": float(item.get("o", 0)),
                "high": float(item.get("h", 0)),
                "low": float(item.get("l", 0)),
                "close": float(item.get("c", 0)),
                "volume": int(item.get("v", 0)),
                "vwap": float(item.get("vw")) if item.get("vw") is not None else None,
                "transactions": int(item.get("n")) if item.get("n") is not None else None
            })

        # Cache and return
        if use_cache:
            self._save_to_cache(summary, cache_path)

        return summary

    def load_daily_summary_for_related(
        self,
        symbol: str,
        date: str,
        adjusted: bool = True,
        include_otc: bool = False,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch the daily market summary for `date`, then filter it to only include tickers
        related to `symbol` (stocks only).

        - `symbol`: e.g. "AAPL"
        - `date`: "YYYY-MM-DD" (UTC)
        - `adjusted`: if True, returns split‐adjusted values.
        - `include_otc`: if True, includes OTC securities.
        - `use_cache`: whether to read/write cached JSON for both related_tickers and the daily summary.

        Returns a list of dicts (same format as `load_daily_market_summary`) but only for tickers
        in Polygon's related‐tickers list for `symbol`.
        """
        # 1) Fetch related tickers (cached if available)
        related = self.get_related_tickers(symbol, use_cache=use_cache)

        # 2) Load the full daily market summary for that date
        summary = self.load_daily_market_summary(
            date=date,
            adjusted=adjusted,
            include_otc=include_otc,
            use_cache=use_cache
        )

        # 3) Filter only those entries whose ticker is in the related‐tickers list
        filtered: List[Dict[str, Any]] = [
            entry for entry in summary
            if entry["ticker"] in related
        ]

        return filtered

    def load_all_corporate_fundamentals(
            self,
            symbol: str,
            as_of_date: str,
            ipo_params: Optional[Dict[str, Any]] = None,
            splits_params: Optional[Dict[str, Any]] = None,
            dividends_params: Optional[Dict[str, Any]] = None,
            ticker_events_params: Optional[Dict[str, Any]] = None,
            financials_params: Optional[Dict[str, Any]] = None,
            use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch corporate‐actions and fundamental data for `symbol` as of (and not after) `as_of_date`:
          - IPOs              (/v3/reference/ipos)
          - Splits            (/v3/reference/splits)
          - Dividends         (/v3/reference/dividends)
          - Ticker Events     (/vX/reference/tickers/{symbol}/events)
          - Financials        (/vX/reference/financials)

        Note: This is only available for stocks, not cryptocurrency.
        """

        # Helper: call API, cache results, return raw list/dict
        def _fetch_and_cache(
                url: str,
                params: Dict[str, Any],
                cache_subdir: str,
                cache_key: str
        ) -> Any:
            filename = f"{cache_key}.json"
            cache_path = self._get_cache_path(filename, cache_subdir)
            if use_cache:
                cached = self._load_from_cache(cache_path)
                if cached is not None:
                    return cached

            data = self._direct_api_call(url, params)
            results = data.get("results", data)
            self._save_to_cache(results, cache_path)
            return results

        # 1) IPOs: fetch all IPOs for symbol, then filter by announced_date ≤ as_of_date
        ipo_params = ipo_params.copy() if ipo_params else {}
        ipo_params["ticker"] = symbol
        raw_ipos = _fetch_and_cache(
            url="https://api.polygon.io/vX/reference/ipos",
            params=ipo_params,
            cache_subdir="corporate_actions",
            cache_key=f"{symbol}_all_ipos"
        )
        # Filter out any IPOs announced after as_of_date
        filtered_ipos = []
        for ipo in raw_ipos:
            announced = ipo.get("announced_date")
            if announced and announced <= as_of_date:
                filtered_ipos.append(ipo)

        # 2) Splits: fetch all splits for symbol, then filter by execution_date ≤ as_of_date
        splits_params = splits_params.copy() if splits_params else {}
        splits_params["ticker"] = symbol
        raw_splits = _fetch_and_cache(
            url="https://api.polygon.io/v3/reference/splits",
            params=splits_params,
            cache_subdir="corporate_actions",
            cache_key=f"{symbol}_all_splits"
        )
        filtered_splits = []
        for split in raw_splits:
            exec_date = split.get("execution_date")
            if exec_date and exec_date <= as_of_date:
                filtered_splits.append(split)

        # 3) Dividends: keep API‐level filter declaration_date ≤ as_of_date
        dividends_params = dividends_params.copy() if dividends_params else {}
        dividends_params["ticker"] = symbol
        dividends_params["declaration_date.lte"] = as_of_date
        dividends = _fetch_and_cache(
            url="https://api.polygon.io/v3/reference/dividends",
            params=dividends_params,
            cache_subdir="corporate_actions",
            cache_key=f"{symbol}_dividends_up_to_{as_of_date.replace('-', '')}"
        )

        # 4) Ticker Events: fetch all events, no additional client‐side filtering
        ticker_events_params = ticker_events_params.copy() if ticker_events_params else {}
        ticker_events_url = f"https://api.polygon.io/vX/reference/tickers/{symbol}/events"
        ticker_events = _fetch_and_cache(
            url=ticker_events_url,
            params=ticker_events_params,
            cache_subdir="corporate_actions",
            cache_key=f"{symbol}_ticker_events"
        )

        # 5) Financials: keep API‐level filter filing_date ≤ as_of_date
        financials_params = financials_params.copy() if financials_params else {}
        financials_params["ticker"] = symbol
        financials_params["filing_date.lte"] = as_of_date
        financials = _fetch_and_cache(
            url="https://api.polygon.io/vX/reference/financials",
            params=financials_params,
            cache_subdir="fundamentals",
            cache_key=f"{symbol}_financials_up_to_{as_of_date.replace('-', '')}"
        )

        return {
            "ipos": filtered_ipos,
            "splits": filtered_splits,
            "dividends": dividends,
            "ticker_events": ticker_events,
            "financials": financials
        }

    def load_news(
            self,
            ticker: Optional[str] = None,
            published_utc_gte: Optional[str] = None,
            published_utc_lte: Optional[str] = None,
            sort: str = "published_utc",
            order: str = "desc",
            limit: int = 100,
            use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles from Polygon's "Stocks News" endpoint (v2/reference/news),
        allowing time‐range filtering. Polygon's API only accepts YYYY-MM-DD for
        published_utc.gte / published_utc.lte, so we pass just the date part to the API,
        then post‐filter by full timestamp in UTC.

        Returns a list of standardized dicts containing:
          - timestamp    (ISO string in UTC)
          - headline     (article title)
          - description  (article description)
          - url          (link to the full article)
          - source       (publisher name)
          - keywords     (list of keywords)
          - tickers      (list of tickers mentioned)

        Query Parameters:
          - ticker: string (case‐sensitive ticker symbol)
          - published_utc.gte: ISO datetime (e.g. "2024-06-01T08:00:00Z")
          - published_utc.lte: ISO datetime (e.g. "2024-06-01T12:00:00Z")
          - sort: field to sort by (default: "published_utc")
          - order: "asc" or "desc" (default: "desc")
          - limit: max number of articles (default: 100)

        We cache results under "news/" by ticker + date filters + limit + sort + order.
        """
        # 1) Build cache filename (use date‐only for naming)
        parts = []
        if ticker:
            parts.append(f"ticker_{ticker}")
        if published_utc_gte:
            # strip time characters for the key
            parts.append(f"from_{published_utc_gte[:10].replace('-', '')}")
        if published_utc_lte:
            parts.append(f"to_{published_utc_lte[:10].replace('-', '')}")
        parts.append(f"sort_{sort}")
        parts.append(f"order_{order}")
        parts.append(f"lim_{limit}")
        key = "_".join(parts) if parts else "all"
        filename = f"news_{key}.json"
        cache_path = self._get_cache_path(filename, "news")

        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                # cached items already have ISO‐formatted timestamp strings
                return cached

        # 2) Build request to /v2/reference/news, passing date‐only for Polygon API
        url = "https://api.polygon.io/v2/reference/news"
        params: Dict[str, Any] = {
            "sort": sort,
            "order": order,
            "limit": limit
        }
        if ticker:
            params["ticker"] = ticker

        # Polygon expects YYYY-MM-DD for these filters
        if published_utc_gte:
            params["published_utc.gte"] = published_utc_gte[:10]
        if published_utc_lte:
            params["published_utc.lte"] = published_utc_lte[:10]

        data = self._direct_api_call(url, params)
        raw_results = data.get("results", [])

        # Determine the full datetime bounds for post‐filtering
        dt_lower: Optional[datetime] = None
        dt_upper: Optional[datetime] = None
        if published_utc_gte:
            # convert "2024-06-01T08:00:00Z" → "2024-06-01T08:00:00+00:00"
            dt_lower = datetime.fromisoformat(published_utc_gte.replace("Z", "+00:00"))
        if published_utc_lte:
            dt_upper = datetime.fromisoformat(published_utc_lte.replace("Z", "+00:00"))

        # 3) Standardize each article's fields and post‐filter by full timestamp
        output: List[Dict[str, Any]] = []
        for article in raw_results:
            ts_raw = article.get("published_utc")  # e.g. "2024-06-15T12:34:56Z"
            if not ts_raw:
                continue
            try:
                dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                # skip if timestamp is malformed
                continue

            # Post‐filter: include only if within [dt_lower, dt_upper]
            if dt_lower and dt < dt_lower:
                continue
            if dt_upper and dt > dt_upper:
                continue

            iso_ts = dt.isoformat()

            publisher = None
            if isinstance(article.get("publisher"), dict):
                publisher = article["publisher"].get("name")

            output.append({
                "timestamp": iso_ts,
                "headline": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("article_url", ""),
                "source": publisher,
                "keywords": article.get("keywords", []),
                "tickers": article.get("tickers", [])
            })

        # 4) Cache and return
        if use_cache:
            self._save_to_cache(output, cache_path)

        return output

    def get_crypto_snapshot(self, symbol: str, market: str = "USD", use_cache: bool = True) -> Dict[str, Any]:
        """
        Get real-time snapshot data for a cryptocurrency pair.

        Uses Polygon's crypto snapshot endpoint:
        GET /v2/snapshot/locale/global/markets/crypto/tickers/X:{symbol}{market}

        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            market: Market currency (default: "USD")
            use_cache: Whether to use cached data

        Returns:
            Dictionary containing current market data for the crypto pair
        """
        cache_key = f"{symbol}_{market}_snapshot"
        cache_path = self._get_cache_path(f"{cache_key}.json", "crypto_aggregates")

        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                return cached

        # Build crypto ticker format
        crypto_ticker = f"X:{symbol}{market}"
        url = f"https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/{crypto_ticker}"

        data = self._direct_api_call(url)
        ticker_data = data.get("ticker", {})

        # Standardize the response format
        snapshot = {
            "symbol": symbol,
            "market": market,
            "last_quote": ticker_data.get("lastQuote", {}),
            "last_trade": ticker_data.get("lastTrade", {}),
            "min": ticker_data.get("min", {}),
            "prevDay": ticker_data.get("prevDay", {}),
            "updated": ticker_data.get("updated"),
            "fmv": ticker_data.get("fmv"),  # Fair Market Value
        }

        if use_cache:
            self._save_to_cache(snapshot, cache_path)

        return snapshot

    def get_all_crypto_snapshots(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get snapshots for all available cryptocurrency pairs.

        Uses Polygon's crypto snapshots endpoint:
        GET /v2/snapshot/locale/global/markets/crypto/tickers

        Args:
            use_cache: Whether to use cached data

        Returns:
            List of dictionaries containing market data for all crypto pairs
        """
        cache_key = "all_crypto_snapshots"
        cache_path = self._get_cache_path(f"{cache_key}.json", "crypto_aggregates")

        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                return cached

        url = "https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers"

        data = self._direct_api_call(url)
        tickers = data.get("tickers", [])

        # Standardize the response format
        snapshots = []
        for ticker in tickers:
            # Extract symbol and market from ticker (e.g., "X:BTCUSD" -> "BTC", "USD")
            ticker_symbol = ticker.get("ticker", "")
            if ticker_symbol.startswith("X:"):
                pair = ticker_symbol[2:]  # Remove "X:" prefix
                # Try to identify common market currencies
                for market in ["USD", "USDT", "EUR", "BTC", "ETH"]:
                    if pair.endswith(market):
                        symbol = pair[:-len(market)]
                        break
                else:
                    # Fallback: assume last 3 chars are market
                    symbol = pair[:-3]
                    market = pair[-3:]
            else:
                symbol = ticker_symbol
                market = "USD"

            snapshot = {
                "symbol": symbol,
                "market": market,
                "ticker": ticker_symbol,
                "last_quote": ticker.get("lastQuote", {}),
                "last_trade": ticker.get("lastTrade", {}),
                "min": ticker.get("min", {}),
                "prevDay": ticker.get("prevDay", {}),
                "updated": ticker.get("updated"),
                "fmv": ticker.get("fmv"),
            }
            snapshots.append(snapshot)

        if use_cache:
            self._save_to_cache(snapshots, cache_path)

        return snapshots


if __name__ == "__main__":
    # Test the enhanced client with both stocks and crypto
    client = PolygonClient()

    print("=== Testing Stock Data ===")
    symbol = "AAPL"

    # Get related tickers
    related_tickers = client.get_related_tickers(symbol)
    print(f"Related tickers for {symbol}: {related_tickers}")

    # Get stock candles
    stock_candles = client.load_aggregates(
        symbol=symbol,
        interval="1d",
        start_date="2024-01-01T00:00:00",
        end_date="2024-01-10T00:00:00",
        adjusted=True,
        sort="asc",
        limit=1000,
        use_cache=False
    )
    print(f"Stock candles for {symbol}: {len(stock_candles)} candles")
    if stock_candles:
        print(f"First candle: {stock_candles[0]}")

    print("\n=== Testing Crypto Data ===")
    crypto_symbol = "BTC"

    # Get crypto candles
    crypto_candles = client.load_crypto_aggregates(
        symbol=crypto_symbol,
        interval="1d",
        start_date="2024-01-01T00:00:00",
        end_date="2024-01-10T00:00:00",
        market="USD",
        sort="asc",
        limit=1000,
        use_cache=False
    )
    print(f"Crypto candles for {crypto_symbol}USD: {len(crypto_candles)} candles")
    if crypto_candles:
        print(f"First candle: {crypto_candles[0]}")

    # Get crypto snapshot
    try:
        crypto_snapshot = client.get_crypto_snapshot(crypto_symbol, "USD", use_cache=False)
        print(f"Crypto snapshot for {crypto_symbol}USD: {crypto_snapshot}")
    except Exception as e:
        print(f"Error getting crypto snapshot: {e}")

    print("\n=== Testing News Data ===")
    # Get news for stocks
    try:
        news_articles = client.load_news(
            ticker=symbol,
            published_utc_gte="2024-01-01T00:00:00Z",
            published_utc_lte="2024-01-02T00:00:00Z",
            sort="published_utc",
            order="desc",
            limit=5,
            use_cache=False
        )
        print(f"Retrieved {len(news_articles)} articles for {symbol}")
        if news_articles:
            print(f"First article: {news_articles[0]['headline']}")
    except Exception as e:
        print(f"Error getting news: {e}")

    print("\n=== Testing Fundamentals ===")
    # Load corporate fundamentals (stocks only)
    try:
        fundamentals = client.load_all_corporate_fundamentals(
            symbol=symbol,
            as_of_date="2024-01-01",
            use_cache=False
        )
        print(f"Corporate fundamentals for {symbol}:")
        print(f"  IPOs: {len(fundamentals.get('ipos', []))}")
        print(f"  Splits: {len(fundamentals.get('splits', []))}")
        print(f"  Dividends: {len(fundamentals.get('dividends', []))}")
        print(f"  Financials: {len(fundamentals.get('financials', []))}")
    except Exception as e:
        print(f"Error getting fundamentals: {e}")