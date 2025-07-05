"""
Enhanced Alpha Vantage Client

This module provides a comprehensive client for Alpha Vantage API supporting:
- Stock data (fundamental and time series)
- Cryptocurrency data (time series)
- News data (for both stocks and crypto)
- Unified interface for StockSim integration
"""

import os
import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, List

import requests
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries

from utils.time_utils import parse_datetime_utc


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation for deduplication."""
    return re.sub(r"\W+", "", text.lower())


def _extract_keywords_from_article(article: Dict[str, Any]) -> List[str]:
    """
    Extract keywords from Alpha Vantage news article.

    Args:
        article: Alpha Vantage news article dictionary

    Returns:
        List of extracted keywords
    """
    keywords = []

    # Add ticker symbols as keywords
    ticker_sentiment = article.get("ticker_sentiment", [])
    for item in ticker_sentiment:
        ticker = item.get("ticker", "")
        if ticker:
            keywords.append(ticker)

    # Add topic classifications
    topics = article.get("topics", [])
    for topic in topics:
        topic_name = topic.get("topic", "")
        if topic_name:
            keywords.append(topic_name.lower())

    # Extract from title and summary
    title = article.get("title", "").lower()
    summary = article.get("summary", "").lower()

    # Common financial keywords
    financial_keywords = [
        "earnings", "revenue", "profit", "loss", "growth", "decline",
        "bullish", "bearish", "rally", "crash", "volatility", "trading",
        "investment", "market", "stock", "crypto", "bitcoin", "ethereum"
    ]

    text_content = f"{title} {summary}"
    for keyword in financial_keywords:
        if keyword in text_content:
            keywords.append(keyword)

    # Remove duplicates and limit to 10
    return list(set(keywords))[:10]


class AlphaVantageClient:
    """
    A unified client for retrieving various data from the Alpha Vantage API.
    Supports stocks, cryptocurrency, and news data with caching capabilities.
    """

    def __init__(self, api_key: Optional[str] = None, base_cache_dir: Optional[str] = None):
        """
        Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY env var)
            base_cache_dir: Base directory for caching data
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Please set ALPHA_VANTAGE_API_KEY in your environment.")
        
        # Set up cache directories
        self.base_cache_dir = base_cache_dir or os.path.join(os.path.dirname(__file__), "..", "data", "alpha")
        self.base_cache_dir = os.path.abspath(self.base_cache_dir)

        self.cache_dirs = {
            "overview": os.path.join(self.base_cache_dir, "overview"),
            "income_statement": os.path.join(self.base_cache_dir, "income_statement"),
            "balance_sheet": os.path.join(self.base_cache_dir, "balance_sheet"),
            "cash_flow": os.path.join(self.base_cache_dir, "cash_flow"),
            "earnings": os.path.join(self.base_cache_dir, "earnings"),
            "earnings_call_transcripts": os.path.join(self.base_cache_dir, "earnings_call_transcripts"),
            "insider": os.path.join(self.base_cache_dir, "insider"),
            "news": os.path.join(self.base_cache_dir, "news"),
            "candles": os.path.join(self.base_cache_dir, "candles"),
            "crypto_candles": os.path.join(self.base_cache_dir, "crypto_candles"),
            "sliding_analytics": os.path.join(self.base_cache_dir, "sliding_analytics"),
            "fixed_analytics": os.path.join(self.base_cache_dir, "fixed_analytics")
        }
        
        for folder in self.cache_dirs.values():
            os.makedirs(folder, exist_ok=True)

    def _get_cache_path(self, filename: str, data_type: str) -> str:
        """Construct the full cache file path for a given data type."""
        return os.path.join(self.cache_dirs[data_type], filename)

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to Alpha Vantage API."""
        base_url = "https://www.alphavantage.co/query"
        params["apikey"] = self.api_key
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()

    # ========================
    # Crypto Currency Methods
    # ========================

    def load_crypto_aggregates(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        market: str = "USD",
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load cryptocurrency OHLCV data from Alpha Vantage.
        
        This method provides a unified interface that matches the Polygon client
        for seamless integration with existing StockSim code.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            interval: Time interval ("1d", "1w", "1mo", "1min", "5min", "15min", "30min", "60min")
            start_date: Start date in ISO format
            end_date: End date in ISO format  
            market: Market currency (default: "USD")
            adjusted: Compatibility parameter (ignored)
            sort: Sort order ("asc" or "desc")
            limit: Maximum number of records (compatibility)
            use_cache: Whether to use cached data
            
        Returns:
            List of OHLCV dictionaries in StockSim format
        """
        # Create cache filename
        cache_key = f"{symbol}_{market}_{interval}_{start_date[:10]}_{end_date[:10]}"
        cache_filename = f"{cache_key}.json"
        cache_path = self._get_cache_path(cache_filename, "crypto_candles")
        
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                if sort == "desc":
                    cached_data.reverse()
                return cached_data
        
        # Map interval to Alpha Vantage function
        interval_map = {
            "1min": ("CRYPTO_INTRADAY", "1min"),
            "5min": ("CRYPTO_INTRADAY", "5min"), 
            "15min": ("CRYPTO_INTRADAY", "15min"),
            "30min": ("CRYPTO_INTRADAY", "30min"),
            "60min": ("CRYPTO_INTRADAY", "60min"),
            "1d": ("DIGITAL_CURRENCY_DAILY", None),
            "1w": ("DIGITAL_CURRENCY_WEEKLY", None),
            "1mo": ("DIGITAL_CURRENCY_MONTHLY", None)
        }
        
        if interval not in interval_map:
            raise ValueError(f"Unsupported interval: {interval}")
        
        function, av_interval = interval_map[interval]
        
        # Build request parameters
        params = {
            "function": function,
            "symbol": symbol,
            "market": market
        }
        
        if av_interval:  # Intraday data
            params["interval"] = av_interval
            params["outputsize"] = "full"
        
        # Make API request
        try:
            data = self._request(params)
            
            # Parse response based on function type
            if function == "CRYPTO_INTRADAY":
                time_series_key = f"Time Series Crypto ({av_interval})"
                time_series = data.get(time_series_key, {})
            else:
                time_series_key = f"Time Series (Digital Currency {function.split('_')[-1].title()})"
                time_series = data.get(time_series_key, {})
            
            candles = []
            start_dt = parse_datetime_utc(start_date)
            end_dt = parse_datetime_utc(end_date)
            
            for timestamp_str, values in time_series.items():
                try:
                    # Parse timestamp
                    if function == "CRYPTO_INTRADAY":
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    else:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    # Filter by date range
                    if timestamp < start_dt or timestamp > end_dt:
                        continue
                    
                    # Extract OHLCV values
                    if function == "CRYPTO_INTRADAY":
                        # Intraday format
                        open_price = float(values["1. open"])
                        high_price = float(values["2. high"])
                        low_price = float(values["3. low"])
                        close_price = float(values["4. close"])
                        volume = float(values["5. volume"])
                    else:
                        # Daily/Weekly/Monthly format - check which format is available
                        # Try market-specific format first (1a. open (USD)), fallback to simple format (1. open)
                        market_key = f"1a. open ({market})"
                        simple_key = "1. open"
                        
                        if market_key in values:
                            # Market-specific format (older API format)
                            open_price = float(values[f"1a. open ({market})"])
                            high_price = float(values[f"2a. high ({market})"])
                            low_price = float(values[f"3a. low ({market})"])
                            close_price = float(values[f"4a. close ({market})"])
                            volume = float(values["5. volume"])
                        elif simple_key in values:
                            # Simple format (current API format)
                            open_price = float(values["1. open"])
                            high_price = float(values["2. high"])
                            low_price = float(values["3. low"])
                            close_price = float(values["4. close"])
                            volume = float(values["5. volume"])
                        else:
                            # Skip this data point if format is unrecognized
                            continue
                    
                    candle = {
                        "timestamp": timestamp.isoformat(),
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                        "vwap": None,  # Not provided by Alpha Vantage
                        "transactions": None  # Not provided by Alpha Vantage
                    }
                    
                    candles.append(candle)
                    
                except (KeyError, ValueError) as e:
                    print(f"Error parsing candle data for {timestamp_str}: {e}")
                    continue
            
            # Sort by timestamp
            candles.sort(key=lambda x: x["timestamp"])
            
            # Cache the results
            with open(cache_path, "w") as f:
                json.dump(candles, f, indent=2)
            
            # Apply sort order
            if sort == "desc":
                candles.reverse()
            
            # Apply limit
            if limit and len(candles) > limit:
                candles = candles[:limit]
            
            return candles
            
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            return []

    # ========================
    # Stock Time Series Methods  
    # ========================

    def load_aggregates(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load stock OHLCV data from Alpha Vantage.
        
        This method provides a unified interface that matches the Polygon client.
        """
        # Create cache filename
        cache_key = f"{symbol}_{interval}_{start_date[:10]}_{end_date[:10]}"
        cache_filename = f"{cache_key}.json"
        cache_path = self._get_cache_path(cache_filename, "candles")
        
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                if sort == "desc":
                    cached_data.reverse()
                return cached_data
        
        # Use existing TimeSeries client
        ts = TimeSeries(self.api_key, output_format="json")
        
        # Map interval to Alpha Vantage function
        is_intraday = interval.lower() in {"1min", "5min", "15min", "30min", "60min"}
        
        try:
            if is_intraday:
                data, _ = ts.get_intraday(
                    symbol=symbol,
                    interval=interval.lower(),
                    outputsize="full",
                    extended_hours="false"
                )
            elif interval.lower() == "1d":
                data, _ = ts.get_daily(symbol=symbol, outputsize="full")
            elif interval.lower() == "1w":
                data, _ = ts.get_weekly(symbol=symbol)
            elif interval.lower() == "1mo":
                data, _ = ts.get_monthly(symbol=symbol)
            else:
                raise ValueError(f"Unsupported interval: {interval}")
            
            candles = []
            start_dt = parse_datetime_utc(start_date)
            end_dt = parse_datetime_utc(end_date)
            eastern = ZoneInfo("US/Eastern")
            
            for timestamp_str, values in data.items():
                try:
                    # Parse timestamp
                    if is_intraday:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        timestamp = timestamp.replace(tzinfo=eastern).astimezone(timezone.utc)
                    else:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    # Filter by date range  
                    if timestamp < start_dt or timestamp > end_dt:
                        continue
                    
                    # Extract OHLCV values
                    open_price = float(values["1. open"])
                    high_price = float(values["2. high"])
                    low_price = float(values["3. low"])
                    close_price = float(values["4. close"])
                    volume = int(values["5. volume"])
                    
                    candle = {
                        "timestamp": timestamp.isoformat(),
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                        "vwap": None,  # Not provided by Alpha Vantage
                        "transactions": None  # Not provided by Alpha Vantage
                    }
                    
                    candles.append(candle)
                    
                except (KeyError, ValueError) as e:
                    print(f"Error parsing candle data for {timestamp_str}: {e}")
                    continue
            
            # Sort by timestamp
            candles.sort(key=lambda x: x["timestamp"])
            
            # Cache the results
            with open(cache_path, "w") as f:
                json.dump(candles, f, indent=2)
            
            # Apply sort order
            if sort == "desc":
                candles.reverse()
            
            # Apply limit
            if limit and len(candles) > limit:
                candles = candles[:limit]
            
            return candles
            
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return []

    # ========================
    # News Methods
    # ========================

    def load_news(
        self,
        ticker: Optional[str] = None,
        published_utc_gte: Optional[str] = None,
        published_utc_lte: Optional[str] = None,
        sort: str = "published_utc",
        order: str = "desc",
        limit: int = 50,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load news articles from Alpha Vantage.
        
        This method provides a unified interface that matches the Polygon client
        for seamless integration with existing StockSim code.
        
        Args:
            ticker: Ticker symbol (for stocks) or crypto symbol
            published_utc_gte: Start date in ISO format
            published_utc_lte: End date in ISO format
            sort: Sort field (for compatibility)
            order: Sort order ("asc" or "desc")
            limit: Maximum number of articles
            use_cache: Whether to use cached data
            
        Returns:
            List of news articles in StockSim format
        """
        # Create cache key based on parameters
        cache_params = [
            ticker or "general",
            published_utc_gte[:10] if published_utc_gte else "no_start",
            published_utc_lte[:10] if published_utc_lte else "no_end",
            str(limit)
        ]
        cache_key = "_".join(cache_params)
        cache_filename = f"news_{cache_key}.json"
        cache_path = self._get_cache_path(cache_filename, "news")
        
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                if order == "asc":
                    cached_data.reverse()
                return cached_data
        
        # Build request parameters
        params = {
            "function": "NEWS_SENTIMENT"
        }
        
        if ticker:
            # For crypto symbols, add CRYPTO: prefix if not already present
            if ticker in ["BTC", "ETH", "LTC", "XRP", "ADA", "DOT", "SOL", "DOGE", "MATIC", "AVAX"] and not ticker.startswith("CRYPTO:"):
                params["tickers"] = f"CRYPTO:{ticker}"
            else:
                params["tickers"] = ticker
        
        if published_utc_gte:
            # Convert to Alpha Vantage format (YYYYMMDDTHHMM)
            start_dt = datetime.fromisoformat(published_utc_gte.replace('Z', '+00:00'))
            params["time_from"] = start_dt.strftime("%Y%m%dT%H%M")
        
        if published_utc_lte:
            # Convert to Alpha Vantage format (YYYYMMDDTHHMM)
            end_dt = datetime.fromisoformat(published_utc_lte.replace('Z', '+00:00'))
            params["time_to"] = end_dt.strftime("%Y%m%dT%H%M")
        
        params["limit"] = str(min(limit, 1000))  # Alpha Vantage limit
        
        try:
            data = self._request(params)
            
            # Extract articles from response
            feed = data.get("feed", [])
            articles = []
            
            for article in feed:
                try:
                    # Parse publication time
                    time_published = article.get("time_published", "")
                    if time_published:
                        # Alpha Vantage format: YYYYMMDDTHHMMSS
                        pub_dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                        timestamp = pub_dt.isoformat()
                    else:
                        timestamp = datetime.now(timezone.utc).isoformat()
                    
                    # Extract ticker symbols
                    ticker_sentiment = article.get("ticker_sentiment", [])
                    tickers = [item.get("ticker", "") for item in ticker_sentiment]
                    
                    # Format for StockSim compatibility
                    formatted_article = {
                        "timestamp": timestamp,
                        "headline": article.get("title", ""),
                        "source": article.get("source", ""),
                        "description": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "keywords": _extract_keywords_from_article(article),
                        "tickers": tickers,
                    }
                    
                    articles.append(formatted_article)
                    
                except Exception as e:
                    print(f"Error processing news article: {e}")
                    continue
            
            # Sort by publication date (newest first for desc order)
            articles.sort(key=lambda x: x["timestamp"], reverse=(order == "desc"))
            
            # Apply limit
            if len(articles) > limit:
                articles = articles[:limit]
            
            # Cache the results
            with open(cache_path, "w") as f:
                json.dump(articles, f, indent=2)
            
            return articles
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    # ========================
    # Fundamental Data Methods (Stocks Only)
    # ========================

    def get_overview(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get company overview data."""
        filename = f"{symbol}_overview.json"
        cache_path = self._get_cache_path(filename, "overview")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        params = {"function": "OVERVIEW", "symbol": symbol}
        data = self._request(params)
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    def get_income_statement(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get income statement data."""
        filename = f"{symbol}_income_statement.json"
        cache_path = self._get_cache_path(filename, "income_statement")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        params = {"function": "INCOME_STATEMENT", "symbol": symbol}
        data = self._request(params)
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    def get_balance_sheet(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get balance sheet data."""
        filename = f"{symbol}_balance_sheet.json"
        cache_path = self._get_cache_path(filename, "balance_sheet")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        params = {"function": "BALANCE_SHEET", "symbol": symbol}
        data = self._request(params)
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    def get_cash_flow(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get cash flow data."""
        filename = f"{symbol}_cash_flow.json"
        cache_path = self._get_cache_path(filename, "cash_flow")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        params = {"function": "CASH_FLOW", "symbol": symbol}
        data = self._request(params)
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    def get_earnings(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get earnings data."""
        filename = f"{symbol}_earnings.json"
        cache_path = self._get_cache_path(filename, "earnings")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        params = {"function": "EARNINGS", "symbol": symbol}
        data = self._request(params)
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    # ========================
    # Compatibility Methods
    # ========================

    def load_all_corporate_fundamentals(
        self,
        symbol: str,
        as_of_date: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Load comprehensive fundamental data for compatibility with Polygon client.
        
        This aggregates multiple Alpha Vantage endpoints to provide similar
        data structure as Polygon's corporate fundamentals.
        """
        fundamentals = {
            "overview": self.get_overview(symbol, use_cache),
            "income_statement": self.get_income_statement(symbol, use_cache),
            "balance_sheet": self.get_balance_sheet(symbol, use_cache),
            "cash_flow": self.get_cash_flow(symbol, use_cache),
            "earnings": self.get_earnings(symbol, use_cache),
            "ipos": [],  # Not available in Alpha Vantage
            "splits": [],  # Not available in Alpha Vantage  
            "dividends": [],  # Not available in Alpha Vantage
            "ticker_events": {"events": []},  # Not available in Alpha Vantage
            "financials": []  # Converted from other data
        }
        
        return fundamentals


if __name__ == "__main__":
    # Test the Alpha Vantage client with both stocks and crypto
    client = AlphaVantageClient()

    print("=== Testing Stock Data ===")
    symbol = "AAPL"

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
        print(f"Last candle: {stock_candles[-1]}")

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
        print(f"Last candle: {crypto_candles[-1]}")

    print("\n=== Testing News Data ===")
    # Get news for crypto
    try:
        news_articles = client.load_news(
            ticker=crypto_symbol,
            published_utc_gte="2025-05-01T00:00:00Z",
            published_utc_lte="2025-07-01T00:00:00Z",
            sort="published_utc",
            order="desc",
            limit=5,
            use_cache=False
        )
        print(f"Retrieved {len(news_articles)} articles for {crypto_symbol}")
        if news_articles:
            print(f"First article: {news_articles[0]['headline']}")
    except Exception as e:
        print(f"Error getting news: {e}")

    print("\n=== Testing Fundamentals ===")
    # Load fundamental data (stocks only for Alpha Vantage)
    try:
        fundamentals = client.load_all_corporate_fundamentals(
            symbol=symbol,
            as_of_date="2024-01-01",
            use_cache=False
        )
        print(f"Corporate fundamentals for {symbol}:")
        print(f"  Overview: {'Available' if fundamentals.get('overview') else 'None'}")
        print(f"  Income Statement: {'Available' if fundamentals.get('income_statement') else 'None'}")
        print(f"  Balance Sheet: {'Available' if fundamentals.get('balance_sheet') else 'None'}")
        print(f"  Cash Flow: {'Available' if fundamentals.get('cash_flow') else 'None'}")
        print(f"  Earnings: {'Available' if fundamentals.get('earnings') else 'None'}")
    except Exception as e:
        print(f"Error getting fundamentals: {e}")

    print("\n=== Testing Intraday Data ===")
    # Test intraday intervals
    try:
        intraday_candles = client.load_aggregates(
            symbol=symbol,
            interval="60min",
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-02T00:00:00",
            adjusted=True,
            sort="asc",
            limit=50,
            use_cache=False
        )
        print(f"Intraday (60min) candles for {symbol}: {len(intraday_candles)} candles")
        if intraday_candles:
            print(f"First intraday candle: {intraday_candles[0]}")
    except Exception as e:
        print(f"Error getting intraday data: {e}")

    print("\n=== Testing Crypto Intraday Data ===")
    # Test crypto intraday intervals
    try:
        crypto_intraday = client.load_crypto_aggregates(
            symbol=crypto_symbol,
            interval="60min",
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-02T00:00:00",
            market="USD",
            sort="asc",
            limit=50,
            use_cache=False
        )
        print(f"Crypto intraday (60min) candles for {crypto_symbol}USD: {len(crypto_intraday)} candles")
        if crypto_intraday:
            print(f"First crypto intraday candle: {crypto_intraday[0]}")
    except Exception as e:
        print(f"Error getting crypto intraday data: {e}")

    print("\n=== Summary ===")
    print("Alpha Vantage client testing completed.")
    print("Key differences from Polygon:")
    print("- Limited to 5 API calls per minute (free tier) or 500+ calls per minute (premium)")
    print("- Some fundamental data features differ")
    print("- Crypto data uses different endpoint structure")
    print("- News sentiment analysis includes additional metadata")