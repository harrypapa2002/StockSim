"""
Candle-Based Exchange Agent

Historical backtesting exchange that processes OHLCV data with dual-market support
for stocks and cryptocurrency. Provides deterministic order matching for reproducible
research results.

Key Features:
- Dual-market data sources (Polygon.io, Alpha Vantage)
- Realistic intra-candle price simulation
- Technical indicator computation
- News and fundamental data integration
- Deterministic order matching for reproducible backtests
"""

import asyncio
import uuid
from bisect import bisect_right
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Set

from utils.data_validators import parse_quantity
from utils.fundamentals_processor import extract_polygon_fundamentals
from utils.indicators_tracker import IndicatorsTracker
from utils.messages import MessageType
from utils.orders import Side, OrderType, OrderStatus
from utils.polygon_client import PolygonClient
from utils.alpha_vantage_client import AlphaVantageClient
from utils.role import Role
from agents.agent import Agent
from utils.subscription_manager import create_unsubscription_response, create_subscription_response
from utils.time_utils import parse_datetime_utc, parse_interval_to_timedelta


class SubscriptionStatus(Enum):
    """Subscription status for market data feeds."""
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


def _get_trigger_price(order: Dict[str, Any], seg_start: float, seg_end: float) -> Optional[float]:
    """
    Determine if an order triggers during a price segment and return trigger price.
    
    This function simulates realistic order execution by determining when orders
    would trigger based on price movement within a candle segment.
    
    Args:
        order: Order dictionary with type, side, and price information
        seg_start: Starting price of the segment
        seg_end: Ending price of the segment
        
    Returns:
        Trigger price if order executes, None otherwise
    """
    otype = order["order_type"].upper()
    side = order["side"].upper()
    trigger_price = order["price"]

    # No movement = no triggers
    if seg_start == seg_end:
        return None

    if otype == OrderType.MARKET.value:
        # Market orders execute immediately at segment start
        return seg_start

    elif otype == OrderType.LIMIT.value:
        if side == Side.BUY.value:
            # Limit buy: execute if price at or below limit
            if seg_start <= trigger_price:
                return seg_start
            elif seg_start > trigger_price >= seg_end:
                return trigger_price
        else:  # SELL
            # Limit sell: execute if price at or above limit
            if seg_start >= trigger_price:
                return seg_start
            elif seg_start < trigger_price <= seg_end:
                return trigger_price

    elif otype == OrderType.STOP.value:
        if side == Side.BUY.value:
            # Stop buy: execute if price at or above stop
            if seg_start >= trigger_price:
                return seg_start
            elif seg_start < trigger_price <= seg_end:
                return trigger_price
        else:  # SELL
            # Stop sell: execute if price at or below stop
            if seg_start <= trigger_price:
                return seg_start
            elif seg_start > trigger_price >= seg_end:
                return trigger_price

    return None


class CandleBasedExchangeAgent(Agent):
    """
    Candle-based exchange agent supporting dual-market historical backtesting.
    
    This agent processes historical OHLCV data and simulates realistic order execution
    with support for both traditional stock markets and cryptocurrency markets via
    Polygon.io. Alpha Vantage is used only for news feeds. It provides deterministic
    backtesting with sophisticated intra-candle price simulation and configuration-driven
    symbol type classification.
    """

    def __init__(
        self,
        instrument: str,
        resolution: str,
        start_date: str,
        end_date: str,
        warmup_candles: int = 250,
        agent_id: str = "candle_exchange",
        rabbitmq_host: str = "localhost",
        tickers: Optional[List[str]] = None,
        spread_factor: float = 0.001,
        limit_news: int = 50,
        indicator_kwargs: Optional[Dict[str, Any]] = None,
        data_source: str = "polygon",
        symbol_type: str = "stock"
    ):
        """
        Initialize candle-based exchange with dual-market support.
        
        Args:
            instrument: Trading instrument symbol (e.g., "NVDA", "BTCUSDT")
            resolution: Candle interval (e.g., "1d", "1h", "15m")
            start_date: Simulation start date in ISO format
            end_date: Simulation end date in ISO format
            warmup_candles: Number of candles for indicator warmup
            agent_id: Unique agent identifier
            rabbitmq_host: RabbitMQ server hostname
            tickers: List of tickers for news feeds
            spread_factor: Bid-ask spread factor for market making
            limit_news: Maximum number of news articles to fetch
            indicator_kwargs: Technical indicator configuration
            data_source: Data source ("polygon" or "alpha_vantage")
            symbol_type: Symbol type ("stock" or "crypto")
        """
        super().__init__(agent_id=agent_id, rabbitmq_host=rabbitmq_host)
        
        # Core configuration
        self.instrument = instrument
        self.resolution = resolution
        self.start_date = start_date
        self.end_date = end_date
        self.warmup_candles = warmup_candles
        self.spread_factor = spread_factor
        self.data_source = data_source.lower()
        self.symbol_type = symbol_type.lower()
        
        # Market data
        self.candles = []
        self.latest_close_price: Optional[float] = None
        self.subscribed_agents: Set[str] = set()
        self.candle_interval = parse_interval_to_timedelta(resolution)
        
        # News and fundamental data configuration
        self.tickers = tickers
        self.limit_news = limit_news
        
        # Initialize data clients
        self.polygon_client = PolygonClient()
        self.alpha_vantage_client = AlphaVantageClient()

        # Set primary data client based on user preference
        if self.data_source == "alpha_vantage":
            self.client = self.alpha_vantage_client
            self.logger.info(f"Initialized Alpha Vantage client for market data: {instrument}")
        else:
            self.client = self.polygon_client
            self.logger.info(f"Initialized Polygon client for market data: {instrument}")

        self.logger.info(f"Both Polygon.io and Alpha Vantage clients available for flexible data sourcing")

        # Initialize technical indicators
        self.indicators_tracker = IndicatorsTracker(**(indicator_kwargs or {}))

        # Load main candle data based on source
        self._load_candle_data()
        
        # Load historical data for indicator warmup
        self._load_warmup_data()
        
        # Load fundamental data (stocks only)
        self._load_fundamental_data()
        
        # Order management
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.candle_timestamps = [parse_datetime_utc(c["timestamp"]) for c in self.candles]
        self.last_processed_candle_time: Optional[datetime] = None

        self.logger.info(f"CandleBasedExchangeAgent initialized for {instrument} using {data_source} data source")

    def _load_candle_data(self):
        """Load main simulation candle data based on configured data source."""
        if self.data_source == "alpha_vantage":
            # Use Alpha Vantage for candlestick data
            if self.symbol_type == "crypto":
                self.candles = self.alpha_vantage_client.load_crypto_aggregates(
                    symbol=self.instrument,
                    interval=self.resolution,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    market="USD",
                    sort="asc",
                    limit=50000,
                    use_cache=True
                )
            else:
                self.candles = self.alpha_vantage_client.load_aggregates(
                    symbol=self.instrument,
                    interval=self.resolution,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    adjusted=True,
                    sort="asc",
                    limit=50000,
                    use_cache=True
                )
            self.logger.info(f"Loaded {len(self.candles)} candles for simulation period using Alpha Vantage")
        else:
            # Use Polygon.io for candlestick data (default)
            self.candles = self.polygon_client.load_aggregates(
                symbol=self.instrument,
                interval=self.resolution,
                start_date=self.start_date,
                end_date=self.end_date,
                adjusted=True,
                sort="asc",
                limit=50000,
                use_cache=True
            )
            self.logger.info(f"Loaded {len(self.candles)} candles for simulation period using Polygon.io")

    def _load_warmup_data(self):
        """Load historical data for technical indicator warmup based on configured data source."""
        sim_start_dt = parse_datetime_utc(self.start_date)
        historical_start_dt = sim_start_dt - self.warmup_candles * self.candle_interval

        if self.data_source == "alpha_vantage":
            # Use Alpha Vantage for warmup data
            if self.symbol_type == "crypto":
                historical_candles = self.alpha_vantage_client.load_crypto_aggregates(
                    symbol=self.instrument,
                    interval=self.resolution,
                    start_date=historical_start_dt.isoformat(),
                    end_date=self.start_date,
                    market="USD",
                    sort="asc",
                    limit=50000,
                    use_cache=True
                )
            else:
                historical_candles = self.alpha_vantage_client.load_aggregates(
                    symbol=self.instrument,
                    interval=self.resolution,
                    start_date=historical_start_dt.isoformat(),
                    end_date=self.start_date,
                    adjusted=True,
                    sort="asc",
                    limit=50000,
                    use_cache=True
                )
            self.logger.info(f"Warmed up indicators using Alpha Vantage data")
        else:
            # Use Polygon.io for warmup data (default)
            historical_candles = self.polygon_client.load_aggregates(
                symbol=self.instrument,
                interval=self.resolution,
                start_date=historical_start_dt.isoformat(),
                end_date=self.start_date,
                adjusted=True,
                sort="asc",
                limit=50000,
                use_cache=True
            )
            self.logger.info(f"Warmed up indicators using Polygon.io data")

        # Warm up indicators with historical data
        warmup_count = 0
        for candle in historical_candles:
            candle_ts = parse_datetime_utc(candle["timestamp"])
            if candle_ts + self.candle_interval < sim_start_dt:
                self.indicators_tracker.update(candle)
                warmup_count += 1
                
        self.logger.info(f"Warmed up indicators with {warmup_count} historical candles")

    def _load_fundamental_data(self):
        """Load fundamental data (stocks only - not available for crypto)."""
        # Only load fundamentals for stocks, not crypto
        if self.symbol_type == "stock" and self.client:
            try:
                fundamentals_raw = self.client.load_all_corporate_fundamentals(
                    symbol=self.instrument,
                    as_of_date=self.end_date.split("T")[0],
                    use_cache=True
                )
                self.fundamentals = fundamentals_raw
                self.logger.info(f"Loaded fundamental data for {self.instrument}")
            except Exception as e:
                self.logger.warning(f"Failed to load fundamentals for {self.instrument}: {e}")
                self.fundamentals = {}
        else:
            # No fundamental data for crypto
            self.fundamentals = {}
            self.logger.info(f"No fundamental data available for {self.symbol_type} asset {self.instrument}")

    async def _handle_regular_message(self, msg: Dict[str, Any]):
        """Handle incoming messages from trading agents."""
        msg_type_str = msg.get("type")
        if not msg_type_str:
            self.logger.error("Missing message type in CandleBasedExchangeAgent.")
            return

        msg_type = MessageType(msg_type_str)
        sender = msg.get("sender", "")
        payload = msg.get("payload", {})

        if msg_type == MessageType.SUBSCRIBE:
            asyncio.create_task(self._handle_subscribe(sender))
        elif msg_type == MessageType.UNSUBSCRIBE:
            asyncio.create_task(self._handle_unsubscribe(sender))
        elif msg_type == MessageType.ORDER_SUBMISSION:
            asyncio.create_task(self._handle_trade_request(sender, payload))
        elif msg_type == MessageType.MARKET_DATA_SNAPSHOT_REQUEST:
            asyncio.create_task(self._handle_market_data_snapshot_request(sender, payload))
        elif msg_type == MessageType.NEWS_SNAPSHOT_REQUEST:
            asyncio.create_task(self._handle_news_snapshot_request(sender, payload))
        elif msg_type == MessageType.FUNDAMENTALS_REQUEST:
            asyncio.create_task(self._handle_fundamentals_request(sender, payload))
        else:
            self.logger.warning(f"Unsupported message type: {msg_type_str}")

    async def _handle_subscribe(self, sender: str):
        """Handle market data subscription requests."""
        if not sender:
            return

        self.subscribed_agents.add(sender)
        confirmation = create_subscription_response(self.instrument)
        await self.send_message(sender, MessageType.SUBSCRIPTION_CONFIRMATION, confirmation)
        self.logger.info(f"Agent {sender} subscribed to candle data for {self.instrument}.")

    async def _handle_unsubscribe(self, sender: str):
        """Handle market data unsubscription requests."""
        if not sender:
            return
            
        if sender in self.subscribed_agents:
            self.subscribed_agents.discard(sender)
        confirmation = create_unsubscription_response(self.instrument)
        await self.send_message(sender, MessageType.UNSUBSCRIPTION_CONFIRMATION, confirmation)
        self.logger.info(f"Agent {sender} unsubscribed from candle data for {self.instrument}.")

    async def _handle_trade_request(self, sender: str, payload: Dict[str, Any]):
        """
        Handle trade order submissions with robust validation.
        
        Supports MARKET and LIMIT orders with proper quantity parsing
        and error handling for various input formats.
        """
        side_str = payload.get("side", "").upper()
        quantity_raw = payload.get("quantity", 0)
        order_type_str = payload.get("order_type", "LIMIT").upper()

        # Validate order side
        try:
            side = Side[side_str]
        except KeyError:
            self.logger.error(f"Invalid side in trade request: {side_str}")
            return

        # Validate order type
        try:
            order_type = OrderType[order_type_str]
        except KeyError:
            self.logger.error(f"Invalid order_type in trade request: {order_type_str}")
            return

        # Parse and validate quantity
        try:
            quantity = parse_quantity(quantity_raw, default=0)

        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid quantity in trade request: {quantity_raw} - {e}")
            await self.send_message(sender, MessageType.ERROR, 
                                  {"error": f"Invalid quantity in trade request: {quantity_raw}"})
            return

        request_id = payload.get("order_id") or str(uuid.uuid4())
        order_price = float(payload.get("price", 0))

        # Create order for execution in next candle
        order = {
            "sender": sender,
            "side": side.value,
            "price": order_price,
            "quantity": quantity,
            "order_id": request_id,
            "order_type": order_type.value,
            "status": OrderStatus.ACTIVE.value,
            "oco_group": payload.get("oco_group"),
            "explanation": payload.get("explanation"),
            "is_short": payload.get("is_short", False),
            "is_short_cover": payload.get("is_short_cover", False)
        }

        self.open_orders[request_id] = order
        self.logger.info(f"Queued order {request_id} for {sender}: {order_type.value} {side.value} {quantity} @ {order_price}")

    async def _handle_market_data_snapshot_request(self, sender: str, payload: Dict[str, Any]):
        """Provide aggregated market data snapshot for specified time window."""
        if not sender:
            return
            
        try:
            window_start = parse_datetime_utc(payload["window_start"])
            window_end = parse_datetime_utc(payload["window_end"])
        except (KeyError, ValueError) as e:
            self.logger.error(f"Invalid window params for MARKET_DATA_SNAPSHOT_REQUEST: {e}")
            return

        # Filter candles within the requested window
        filtered = [c for c in self.candles 
                   if window_start < parse_datetime_utc(c["timestamp"]) + self.candle_interval <= window_end]
        
        if not filtered:
            response_payload = {
                "instrument": self.instrument,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "data": {},
                "indicators": {}
            }
        else:
            # Aggregate OHLCV data
            opens = [c["open"] for c in filtered]
            highs = [c["high"] for c in filtered]
            lows = [c["low"] for c in filtered]
            closes = [c["close"] for c in filtered]
            volumes = [c["volume"] for c in filtered]

            overall = {
                "open": opens[0], 
                "high": max(highs), 
                "low": min(lows), 
                "close": closes[-1], 
                "volume": sum(volumes)
            }

            # Calculate VWAP if available
            vwap_sum = 0.0
            vwap_volume_sum = 0
            have_vwap = True
            for c in filtered:
                if "vwap" in c and c["vwap"] is not None:
                    vwap_sum += c["vwap"] * c["volume"]
                    vwap_volume_sum += c["volume"]
                else:
                    have_vwap = False
                    break

            if have_vwap and vwap_volume_sum > 0:
                overall["vwap"] = round(vwap_sum / vwap_volume_sum, 6)
            else:
                overall["vwap"] = None

            # Calculate transaction count if available
            transactions_sum = 0
            have_tx = True
            for c in filtered:
                if "transactions" in c and c["transactions"] is not None:
                    transactions_sum += c["transactions"]
                else:
                    have_tx = False
                    break

            if have_tx:
                overall["transactions"] = transactions_sum
            else:
                overall["transactions"] = None

            # Get current technical indicators
            indicators = self.indicators_tracker.get_latest_values()

            # Add bid/ask spread simulation
            close_price = overall.get("close")
            if close_price:
                overall["best_bid"] = round(close_price * (1 - self.spread_factor), 4)
                overall["best_ask"] = round(close_price * (1 + self.spread_factor), 4)

            response_payload = {
                "instrument": self.instrument,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "data": overall,
                "indicators": indicators,
            }

        await self.send_message(sender, MessageType.MARKET_DATA_SNAPSHOT_RESPONSE, response_payload)

    async def _handle_news_snapshot_request(self, sender: str, payload: Dict[str, Any]):
        """Handle news data requests with configurable news source support."""
        try:
            window_start = parse_datetime_utc(payload["window_start"])
            window_end = parse_datetime_utc(payload["window_end"])
        except (KeyError, ValueError) as e:
            self.logger.error(f"Invalid window params for NEWS_SNAPSHOT_REQUEST: {e}")
            return

        # Use Alpha Vantage for news if data_source is set to alpha_vantage, otherwise use Polygon.io
        news_client = self.alpha_vantage_client if self.data_source == "alpha_vantage" else self.polygon_client

        try:
            news_list = news_client.load_news(
                ticker=self.tickers[0] if self.tickers else None,
                published_utc_gte=window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                published_utc_lte=window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                sort="published_utc",
                order="desc",
                limit=self.limit_news,
                use_cache=True
            )

            if not news_list:
                self.logger.info(f"No news articles available for {window_start} to {window_end}.")

            response = {
                "instrument": self.instrument,
                "news": [
                    {
                        "timestamp": n["timestamp"],
                        "headline": n["headline"],
                        "source": n["source"],
                        "description": n["description"],
                        "url": n["url"],
                        "keywords": n.get("keywords", []),
                        "tickers": n.get("tickers", []),
                    } for n in news_list
                ]
            }

            self.logger.debug(f"Fetched {len(news_list)} news articles using {self.data_source} client")

        except Exception as e:
            self.logger.error(f"Error fetching news for {self.instrument} using {self.data_source}: {e}")
            response = {"instrument": self.instrument, "news": []}

        await self.send_message(sender, MessageType.NEWS_SNAPSHOT_RESPONSE, response)

    async def _handle_fundamentals_request(self, sender: str, payload: Dict[str, Any]):
        """Handle fundamental data requests (stocks only)."""
        if not sender:
            return

        # Extract date range parameters
        prev_cutoff_raw = payload.get("window_start")
        as_of_raw = payload.get("window_end")

        try:
            if prev_cutoff_raw:
                prev_dt = parse_datetime_utc(prev_cutoff_raw)
                prev_cutoff_str = prev_dt.date().isoformat()
            else:
                prev_cutoff_str = None
        except Exception as e:
            self.logger.error(f"Invalid window_start for FUNDAMENTALS_REQUEST: {e}")
            return

        try:
            as_of_dt = parse_datetime_utc(as_of_raw)
            as_of_str = as_of_dt.date().isoformat()
        except Exception as e:
            self.logger.error(f"Invalid window_end for FUNDAMENTALS_REQUEST: {e}")
            return

        # Filter fundamental data within date range
        if self.data_source == "polygon":
            try:
                fundamentals = extract_polygon_fundamentals(
                    raw=self.fundamentals,
                    prev_cutoff=prev_cutoff_str,
                    as_of_date=as_of_str
                )

                # Check if any fundamental data exists
                has_data = False
                if fundamentals:
                    ipos = fundamentals.get("ipos", [])
                    splits = fundamentals.get("splits", [])
                    dividends = fundamentals.get("dividends", [])
                    ticker_events = fundamentals.get("ticker_events", {}).get("events", [])
                    financials = fundamentals.get("financials", [])
                    has_data = bool(ipos or splits or dividends or ticker_events or financials)

                if has_data:
                    self.logger.debug(f"Fundamentals data found for {self.instrument}")
                else:
                    self.logger.debug(f"No fundamentals data in window [{prev_cutoff_str}, {as_of_str}]")

                response = {
                    "instrument": self.instrument,
                    "fundamentals": fundamentals if has_data else {}
                }
            except Exception as e:
                self.logger.error(f"Error processing fundamentals for {self.instrument}: {e}")
                response = {"instrument": self.instrument, "fundamentals": {}}
        else:
            # No fundamental data for crypto
            response = {"instrument": self.instrument, "fundamentals": {}}

        await self.send_message(sender, MessageType.FUNDAMENTALS_RESPONSE, response)

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        """
        Process time tick and execute orders based on current candle data.
        
        This method handles the core simulation logic including order execution,
        indicator updates, and portfolio notifications.
        """
        await super().handle_time_tick(payload)
        candle_index = self.current_tick_id
        current_dt = self.current_time - self.candle_interval

        # Find the most recent candle before current time
        idx = bisect_right(self.candle_timestamps, current_dt) - 1

        if idx < 0:
            self.logger.info(f"No candle available before {current_dt.isoformat()}. Skipping tick.")
            await self.publish_time(
                msg_type=MessageType.BARRIER_RESPONSE,
                payload={"tick_id": candle_index},
                routing_key="simulation_clock"
            )
            return

        ref_candle = self.candles[idx]
        ref_time = self.candle_timestamps[idx]

        # Log the candle being processed for debugging
        self.logger.debug(f"Processing candle at {ref_time.isoformat()} for tick {current_dt.isoformat()}")

        # Process new candle data
        if self.last_processed_candle_time is None or ref_time > self.last_processed_candle_time:
            # Update technical indicators
            self.indicators_tracker.update(ref_candle)
            self.last_processed_candle_time = ref_time
            self.latest_close_price = ref_candle["close"]
            
            # Execute pending orders
            await self._fill_orders_for_candle(ref_candle)
            
            # Notify subscribed agents of price update
            for agent_id in self.subscribed_agents:
                await self.send_message(agent_id, MessageType.PORTFOLIO_UPDATE,
                                      {"instrument": self.instrument, "close_price": self.latest_close_price})
        else:
            # Clear pending orders if no new candle
            self.open_orders = {}

        # Send barrier response to simulation clock
        await self.publish_time(msg_type=MessageType.BARRIER_RESPONSE,
                              payload={"tick_id": candle_index},
                              routing_key="simulation_clock")

        self.logger.debug(f"Sent BARRIER_RESPONSE for tick_id {candle_index}.")

    async def _fill_orders_for_candle(self, candle: Dict[str, Any]):
        """
        Simulate realistic order execution within a candle using price path modeling.
        
        This method implements sophisticated intra-candle price simulation to
        determine realistic order execution timing and prices.
        """
        c_open = candle["open"]
        c_close = candle["close"]
        c_high = candle["high"]
        c_low = candle["low"]

        # Model realistic intra-candle price path
        # If close >= open: assume path goes open -> low -> high -> close
        # If close < open: assume path goes open -> high -> low -> close
        if c_close >= c_open:
            price_path = [c_open, c_low, c_high, c_close]
        else:
            price_path = [c_open, c_high, c_low, c_close]

        # Process each price segment for order execution
        for i in range(len(price_path) - 1):
            segment_start = price_path[i]
            segment_end = price_path[i + 1]

            # Find orders triggered in this segment
            triggered_orders = self._collect_triggered_orders(segment_start, segment_end)

            # Sort by trigger price (price priority)
            if segment_end > segment_start:
                triggered_orders.sort(key=lambda o: o["trigger_price"])
            else:
                triggered_orders.sort(key=lambda o: o["trigger_price"], reverse=True)

            # Execute triggered orders
            for order_info in triggered_orders:
                if order_info["order"]["status"] != OrderStatus.ACTIVE.value:
                    continue

                await self._fill_order(order_info["order"], order_info["trigger_price"])

                # Handle OCO (One-Cancels-Other) groups
                oco = order_info["order"]["oco_group"]
                if oco:
                    await self._cancel_oco_siblings(order_info["order"])

        # Clear all processed orders
        self.open_orders = {}

    def _collect_triggered_orders(self, segment_start: float, segment_end: float) -> List[Dict[str, Any]]:
        """
        Identify orders that trigger during a price movement segment.
        
        Args:
            segment_start: Starting price of the segment
            segment_end: Ending price of the segment
            
        Returns:
            List of triggered orders with their trigger prices
        """
        triggered = []
        for oid, order in self.open_orders.items():
            if order["status"] != OrderStatus.ACTIVE.value:
                continue

            maybe_price = _get_trigger_price(order, segment_start, segment_end)
            if maybe_price is not None:
                triggered.append({"order": order, "trigger_price": maybe_price})
                
        return triggered

    async def _fill_order(self, order: Dict[str, Any], fill_price: float):
        """
        Execute order fill and notify the originating agent.
        
        Args:
            order: Order dictionary to fill
            fill_price: Execution price for the order
        """
        order["status"] = OrderStatus.FILLED.value
        
        fill_payload = {
            "instrument": self.instrument,
            "price": fill_price,
            "quantity": order["quantity"],
            "order_type": order["order_type"],
            "order_id": order["order_id"],
            "role": Role.BUYER.value if order["side"] == Side.BUY.value else Role.SELLER.value,
            "order_status": OrderStatus.FILLED.value,
            "explanation": order.get("explanation"),
            "is_short": order.get("is_short", False),
            "is_short_cover": order.get("is_short_cover", False)
        }
        
        await self.send_message(order["sender"], MessageType.TRADE_EXECUTION, fill_payload)
        self.logger.info(f"Filled order {order['order_id']} for {order['sender']} at {fill_price}")

    async def _cancel_oco_siblings(self, filled_order: Dict[str, Any]):
        """
        Cancel other orders in the same OCO (One-Cancels-Other) group.
        
        Args:
            filled_order: The order that was just filled
        """
        oco_group = filled_order.get("oco_group")
        if not oco_group:
            return

        for oid, o in self.open_orders.items():
            if (o["status"] == OrderStatus.ACTIVE.value
                    and o["oco_group"] == oco_group
                    and o["sender"] == filled_order["sender"]
                    and oid != filled_order["order_id"]):
                o["status"] = OrderStatus.CANCELED.value
                self.logger.info(f"Canceled OCO sibling {oid} because {filled_order['order_id']} was filled.")

