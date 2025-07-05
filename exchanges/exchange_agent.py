"""
Real-Time Order Book Exchange Agent

Implements production-grade order matching engine with microsecond precision for
real-time market simulation. Supports sub-second order matching with realistic
latency simulation for studying LLM response time effects on execution quality.

Key Features:
- Real-time order book management with price-time priority
- Market impact modeling and price discovery
- Technical indicator computation
- Asynchronous RabbitMQ messaging
"""

import asyncio
import bisect
import json
import uuid
import websockets
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set, List, Tuple


from utils.orders import Side, OrderType, Order, OrderBook, OrderStatus
from utils.messages import MessageType
from utils.role import Role
from agents.agent import Agent
from utils.time_utils import parse_datetime_utc, interval_to_seconds, parse_interval_to_timedelta
from utils.indicators_tracker import IndicatorsTracker
from utils.polygon_client import PolygonClient
from utils.alpha_vantage_client import AlphaVantageClient
from utils.data_validators import parse_quantity
from utils.fundamentals_processor import extract_polygon_fundamentals
from utils.subscription_manager import create_subscription_response, create_unsubscription_response

class ExchangeAgent(Agent):
    """
    Real-Time Order Book Exchange Agent
    
    This agent manages order book operations for a specific financial instrument,
    implementing production-grade order matching with realistic market dynamics.

    Key Features:
    - Sub-second order matching with realistic latency simulation
    - Market impact modeling for LLM adaptation research
    - Historical trade data loading and persistence
    - Deterministic replay for reproducible research
    
    The exchange agent serves as the core market simulation engine in StockSim,
    enabling rigorous evaluation of LLM trading strategies under realistic
    market conditions.
    """

    def __init__(
        self,
        instrument: str,
        agent_id: str = "exchange",
        rabbitmq_host: str = 'localhost',
        trades_output_file: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        limit_news: int = 50,
        indicator_kwargs: Optional[Dict[str, Any]] = None,
        data_source: str = "polygon",
        symbol_type: str = "stock",
        data_start_date: Optional[str] = None,
        data_end_date: Optional[str] = None,
        warmup_resolution: str = "1d",
        warmup_candles: int = 250,
        resolutions: Optional[List[str]] = None,
        dashboard_port: int = 8765
    ):
        """
        Initialize the Real-Time Order Book Exchange Agent.

        Args:
            instrument: Financial instrument symbol (e.g., 'AAPL', 'BTC-USD')
            agent_id: Unique identifier for this exchange agent instance
            rabbitmq_host: RabbitMQ server hostname for message coordination
            trades_output_file: Path for persisting new trades at shutdown
            tickers: List of tickers for news feeds
            limit_news: Maximum number of news articles to fetch
            indicator_kwargs: Technical indicator configuration
            data_source: Data source ("polygon" or "alpha_vantage")
            symbol_type: Symbol type ("stock" or "crypto")
            data_start_date: Start date for indicator data (ISO format: YYYY-MM-DD)
            data_end_date: End date for indicator data (ISO format: YYYY-MM-DD)
            warmup_resolution: Candle resolution for warmup data (e.g., "1d", "1h", "5m")
            warmup_candles: Number of warmup candles (used for validation, default: 250)
            resolutions: List of resolutions to aggregate trade data into (e.g., ["1m", "15m", "1h", "1d"])
            dashboard_port: Port for the WebSocket dashboard server (default: 8765)
        """
        super().__init__(agent_id=agent_id, rabbitmq_host=rabbitmq_host)
        self.instrument: str = instrument
        self.order_book = OrderBook(instrument=instrument)

        # Trade dispatch tracking for incremental updates
        self.last_dispatched_trade_key: Optional[Tuple[datetime, int]] = None
        self.last_indicator_update_key: Optional[Tuple[datetime, int]] = None
        self.subscribed_agents: Set[str] = set()

        # Candle cache for performance optimization
        # Format: {(resolution_seconds, window_start_timestamp, window_end_timestamp): candles_list}
        self._candle_cache: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}

        # News and fundamental data configuration
        self.tickers = tickers or [instrument]
        self.limit_news = limit_news
        self.data_source = data_source.lower()
        self.symbol_type = symbol_type.lower()
        
        # Initialize data clients for external feeds
        self.polygon_client = PolygonClient()
        self.alpha_vantage_client = AlphaVantageClient()
        
        # Set primary data client based on user preference
        if self.data_source == "alpha_vantage":
            self.client = self.alpha_vantage_client
            self.logger.info(f"Initialized Alpha Vantage client for external data: {instrument}")
        else:
            self.client = self.polygon_client
            self.logger.info(f"Initialized Polygon client for external data: {instrument}")
        
        # Store warmup parameters for indicator initialization
        self.warmup_start_date = data_start_date
        self.warmup_end_date = data_end_date
        self.warmup_resolution = warmup_resolution
        self.warmup_candles = warmup_candles
        
        # Multi-resolution support
        self.resolutions = resolutions or ["1m", "15m", "1h", "1d"]
        self.resolution_seconds = {res: interval_to_seconds(res) for res in self.resolutions}
        self.resolution_timedeltas = {res: parse_interval_to_timedelta(res) for res in self.resolutions}
        
        # Multi-resolution data storage - simplified to only store completed candles
        self.resolution_candles: Dict[str, List[Dict[str, Any]]] = {res: [] for res in self.resolutions}
        
        # Dashboard WebSocket server
        self.dashboard_port = dashboard_port
        self.dashboard_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.dashboard_server = None
        self.dashboard_task = None

        # Initialize technical indicators tracker for each resolution
        self.indicators_trackers: Dict[str, IndicatorsTracker] = {
            res: IndicatorsTracker(**(indicator_kwargs or {})) for res in self.resolutions
        }
        
        # Load historical data for indicator warmup
        self._load_warmup_data()
        
        # Load fundamental data (stocks only)
        self.fundamentals = {}
        if self.symbol_type == "stock":
            self._load_fundamental_data()

        self.trades_output_file = trades_output_file
        
        # Start dashboard server
        self._start_dashboard_server()
        
        self.logger.info(f"ExchangeAgent {self.agent_id} initialized for instrument {self.instrument}.")

    def _start_dashboard_server(self):
        def runner():
            async def dashboard_main():
                # start and keep the server alive
                server = await websockets.serve(
                    self._handle_dashboard_connection,
                    "localhost",
                    self.dashboard_port
                )
                self.logger.info(f"Dashboard server started on ws://localhost:{self.dashboard_port}")
                # hang here forever
                await asyncio.Future()

            # this creates and runs the event loop properly
            asyncio.run(dashboard_main())

        self.dashboard_task = threading.Thread(target=runner, daemon=True)
        self.dashboard_task.start()

    async def _handle_dashboard_connection(self, websocket):
        """Handle new dashboard WebSocket connections."""
        self.dashboard_clients.add(websocket)
        self.logger.info(f"Dashboard client connected: {websocket.remote_address}")
        
        try:
            # Send initial data to new client
            await self._send_initial_dashboard_data(websocket)
            
            # Keep connection alive and handle client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON from dashboard client: {message}")
                except Exception as e:
                    self.logger.error(f"Error handling dashboard message: {e}")
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Dashboard client disconnected: {websocket.remote_address}")
        except Exception as e:
            self.logger.error(f"Dashboard connection error: {e}")
        finally:
            self.dashboard_clients.discard(websocket)

    async def _send_initial_dashboard_data(self, websocket):
        """Send initial candle data to a new dashboard client."""
        try:
            initial_data = {
                "type": "initial_data",
                "instrument": self.instrument,
                "resolutions": self.resolutions,
                "candles": {
                    resolution: candles[-100:] if len(candles) > 100 else candles  # Last 100 candles max
                    for resolution, candles in self.resolution_candles.items()
                },
                "current_time": self.current_time.isoformat() if self.current_time else None,
                "order_book": {
                    "best_bid": self.order_book.get_best_bid()[0],
                    "best_ask": self.order_book.get_best_ask()[0],
                    "bid_quantity": self.order_book.get_best_bid()[1],
                    "ask_quantity": self.order_book.get_best_ask()[1],
                    "spread": (self.order_book.get_best_ask()[0] - self.order_book.get_best_bid()[0]) if self.order_book.get_best_bid()[0] and self.order_book.get_best_ask()[0] else None
                }
            }
            await websocket.send(json.dumps(initial_data))
        except Exception as e:
            self.logger.error(f"Error sending initial dashboard data: {e}")

    def _broadcast_candle_update(self, resolution: str, candle: Dict[str, Any]):
        """Broadcast candle updates to all connected dashboard clients."""
        if not self.dashboard_clients:
            return

        message = {
            "type": "candle_update",
            "instrument": self.instrument,
            "resolution": resolution,
            "candle": candle,
            "timestamp": self.current_time.isoformat() if self.current_time else None
        }

        # Create a task to broadcast to all clients
        asyncio.create_task(self._broadcast_to_clients(message))

    async def _broadcast_to_clients(self, message: Dict[str, Any]):
        """Broadcast message to all connected dashboard clients."""
        if not self.dashboard_clients:
            return

        message_json = json.dumps(message)
        disconnected_clients = set()

        for client in self.dashboard_clients.copy():
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                self.logger.error(f"Error broadcasting to dashboard client: {e}")
                disconnected_clients.add(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.dashboard_clients.discard(client)

    def _load_fundamental_data(self):
        """Load fundamental data (stocks only - not available for crypto)."""
        # Only load fundamentals for stocks, not crypto
        if self.symbol_type == "stock" and self.client:
            try:
                # Use data_end_date if available, otherwise current time
                if self.warmup_end_date:
                    end_date_str = self.warmup_end_date.split("T")[0]  # Extract date part if ISO format
                else:
                    end_date_str = datetime.now().strftime("%Y-%m-%d")

                fundamentals_raw = self.client.load_all_corporate_fundamentals(
                    symbol=self.instrument,
                    as_of_date=end_date_str,
                    use_cache=True
                )
                self.fundamentals = fundamentals_raw
                self.logger.info(f"Loaded fundamental data for {self.instrument} as of {end_date_str}")
            except Exception as e:
                self.logger.warning(f"Failed to load fundamentals for {self.instrument}: {e}")
                self.fundamentals = {}
        else:
            # No fundamental data for crypto
            self.fundamentals = {}
            self.logger.info(f"No fundamental data available for {self.symbol_type} asset {self.instrument}")

    def _load_warmup_data(self):
        """
        Load historical data for technical indicator warmup across all resolutions.
        
        This method loads historical candle data to initialize technical indicators
        with sufficient historical context before real-time trading begins. Data is
        loaded for each configured resolution to properly warm up all indicator trackers.
        """
        if not self.warmup_start_date or not self.warmup_end_date:
            self.logger.info("No warmup dates specified, skipping indicator warmup")
            return
            
        self.logger.info(f"Starting multi-resolution indicator warmup for {self.instrument} from {self.warmup_start_date} to {self.warmup_end_date}")
        
        # Load warmup data for each resolution
        for resolution in self.resolutions:
            self._load_warmup_data_for_resolution(resolution)
    
    def _load_warmup_data_for_resolution(self, resolution: str):
        """
        Load historical warmup data for a specific resolution.
        
        Args:
            resolution: Time resolution (e.g., "1m", "15m", "1h", "1d")
        """
        try:
            self.logger.info(f"Loading warmup data for {resolution} resolution")
            
            # Load historical candles for warmup based on data source
            if self.data_source == "alpha_vantage":
                if self.symbol_type == "crypto":
                    historical_candles = self.alpha_vantage_client.load_crypto_aggregates(
                        symbol=self.instrument,
                        interval=resolution,
                        start_date=self.warmup_start_date,
                        end_date=self.warmup_end_date,
                        market="USD",
                        sort="asc",
                        limit=50000,
                        use_cache=True
                    )
                else:
                    historical_candles = self.alpha_vantage_client.load_aggregates(
                        symbol=self.instrument,
                        interval=resolution,
                        start_date=self.warmup_start_date,
                        end_date=self.warmup_end_date,
                        adjusted=True,
                        sort="asc",
                        limit=50000,
                        use_cache=True
                    )
            else:
                # Use Polygon.io for warmup data (default)
                historical_candles = self.polygon_client.load_aggregates(
                    symbol=self.instrument,
                    interval=resolution,
                    start_date=self.warmup_start_date,
                    end_date=self.warmup_end_date,
                    adjusted=True,
                    sort="asc",
                    limit=50000,
                    use_cache=True
                )
            
            # Store historical candles in resolution_candles for dashboard and responses
            if historical_candles:
                # Convert to the format expected by the dashboard and responses
                formatted_candles = []
                for candle in historical_candles:
                    # Ensure timestamp is in ISO format
                    if isinstance(candle.get("timestamp"), str):
                        timestamp = candle["timestamp"]
                    else:
                        # Handle datetime objects
                        timestamp = candle["timestamp"].isoformat() if hasattr(candle["timestamp"], 'isoformat') else str(candle["timestamp"])

                    formatted_candle = {
                        "timestamp": timestamp,
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"],
                        "volume": candle["volume"]
                    }
                    formatted_candles.append(formatted_candle)

                # Store historical candles for this resolution
                self.resolution_candles[resolution] = formatted_candles
                self.logger.info(f"Stored {len(formatted_candles)} historical candles for {resolution} resolution")

            # Warm up indicators for this resolution
            warmup_count = 0
            indicators_tracker = self.indicators_trackers[resolution]
            
            for candle in historical_candles:
                indicators_tracker.update(candle)
                warmup_count += 1
                
            self.logger.info(f"Successfully warmed up {resolution} indicators with {warmup_count} historical candles")
            
        except Exception as e:
            self.logger.warning(f"Failed to load warmup data for {self.instrument} at {resolution} resolution: {e}")
            self.logger.info(f"Continuing without indicator warmup for {resolution} resolution")


    async def _handle_regular_message(self, msg: Dict[str, Any]):
        """Handle incoming messages from trading agents."""
        msg_type_str = msg.get("type")
        if not msg_type_str:
            self.logger.error("Missing message type in ExchangeAgent.")
            return

        try:
            msg_type = MessageType(msg_type_str)
        except ValueError:
            sender = msg.get("sender")
            if sender:
                await self.send_message(sender, MessageType.ERROR, {"error": "Invalid message type"})
            self.logger.error(f"ExchangeAgent {self.agent_id} received invalid message type: {msg_type_str}")
            return

        sender = msg.get("sender", "")
        payload = msg.get("payload", {})

        if msg_type == MessageType.SUBSCRIBE:
            asyncio.create_task(self._handle_subscribe(sender))
        elif msg_type == MessageType.UNSUBSCRIBE:
            asyncio.create_task(self._handle_unsubscribe(sender))
        elif msg_type == MessageType.ORDER_SUBMISSION:
            await self._handle_order_submission(sender, payload)
        elif msg_type == MessageType.CANCEL_ORDER:
            await self._handle_cancel_order(sender, payload)
        elif msg_type == MessageType.MARKET_DATA_SNAPSHOT_REQUEST:
            asyncio.create_task(self._handle_market_data_snapshot_request(sender, payload))
        elif msg_type == MessageType.NEWS_SNAPSHOT_REQUEST:
            asyncio.create_task(self._handle_news_snapshot_request(sender, payload))
        elif msg_type == MessageType.FUNDAMENTALS_REQUEST:
            asyncio.create_task(self._handle_fundamentals_request(sender, payload))
        else:
            self.logger.warning(f"Unsupported message type: {msg_type_str}")

    async def _handle_subscribe(self, sender: Optional[str]):
        """
        Handle agent subscription requests for market data feeds.

        Args:
            sender: Agent ID requesting subscription
        """
        if not sender:
            self.logger.error(f"ExchangeAgent {self.agent_id} received SUBSCRIBE message without sender.")
            return

        self.subscribed_agents.add(sender)
        confirmation = create_subscription_response(self.instrument)
        await self.send_message(sender, MessageType.SUBSCRIPTION_CONFIRMATION, confirmation)
        self.logger.info(f"ExchangeAgent {self.agent_id} confirmed SUBSCRIBE to agent {sender} for "
                    f"instrument {self.instrument}.")

    async def _handle_unsubscribe(self, sender: Optional[str]):
        """
        Handle agent unsubscription requests from market data feeds.

        Args:
            sender: Agent ID requesting unsubscription
        """
        if not sender:
            self.logger.error(f"ExchangeAgent {self.agent_id} received UNSUBSCRIBE message without sender.")
            return

        self.subscribed_agents.discard(sender)
        confirmation = create_unsubscription_response(self.instrument)
        await self.send_message(sender, MessageType.UNSUBSCRIPTION_CONFIRMATION, confirmation)
        self.logger.info(f"ExchangeAgent {self.agent_id} confirmed UNSUBSCRIBE to agent {sender} "
                    f"for instrument {self.instrument}.")

    async def _handle_order_submission(self, sender: Optional[str], payload: Dict[str, Any]):
        """
        Process order submissions with comprehensive validation and matching.
        
        Implements production-grade order processing with realistic market impact
        modeling for LLM trading strategy evaluation.

        Args:
            sender: Agent ID submitting the order
            payload: Order details including side, type, quantity, price
        """
        if not sender:
            self.logger.error(f"ExchangeAgent {self.agent_id} received ORDER_SUBMISSION without sender.")
            return

        # Validate order side (BUY/SELL)
        try:
            side_str = payload["side"].upper()
            side = Side[side_str]
        except (KeyError, ValueError):
            await self.send_message(sender, MessageType.ERROR,
                              {"error": "Invalid or missing 'side' in order submission."})
            self.logger.error(f"ExchangeAgent {self.agent_id} received ORDER_SUBMISSION"
                         f" with invalid or missing 'side': {payload.get('side')}")
            return

        # Validate order type (LIMIT/MARKET/STOP)
        try:
            order_type_str = payload["order_type"].upper()
            order_type = OrderType[order_type_str]
        except (KeyError, ValueError):
            await self.send_message(sender, MessageType.ERROR,
                              {"error": "Invalid or missing 'order_type' in order submission."})
            self.logger.error(f"ExchangeAgent {self.agent_id} received ORDER_SUBMISSION "
                         f"with invalid or missing 'order_type': {payload.get('order_type')}")
            return

        # Validate and convert quantity
        quantity_raw = payload.get("quantity")
        try:
            quantity = parse_quantity(quantity_raw, default=0)
        except (ValueError, TypeError) as e:
            await self.send_message(sender, MessageType.ERROR,
                              {"error": f"Invalid or missing 'quantity' in order submission: {quantity_raw}"})
            self.logger.error(f"ExchangeAgent {self.agent_id} received ORDER_SUBMISSION "
                         f"with invalid or missing 'quantity': {quantity_raw} - {e}")
            return

        # Generate order ID if not provided
        order_id = payload.get("order_id")
        if not order_id:
            order_id = str(uuid.uuid4())
            self.logger.warning(f"ExchangeAgent {self.agent_id} received ORDER_SUBMISSION without 'order_id' "
                           f"from agent {sender}. Generated order_id: {order_id}")

        # Validate price for LIMIT and STOP orders
        price = payload.get("price")
        if order_type in [OrderType.LIMIT, OrderType.STOP] and (price is None or not isinstance(price, (int, float))):
            await self.send_message(sender, MessageType.ERROR,
                              {"error": f"Missing or invalid 'price' for {order_type.value} order."})
            self.logger.error(f"ExchangeAgent {self.agent_id} received {order_type.value} ORDER_SUBMISSION "
                         f"with missing or invalid 'price': {price}")
            return
        elif order_type == OrderType.MARKET and price is not None:
            self.logger.warning(f"ExchangeAgent {self.agent_id} received MARKET ORDER_SUBMISSION with "
                           f"unnecessary 'price': {price}. Ignoring price.")
            price = None  # Ignore price for MARKET orders

        timestamp = self.current_time

        # Create and add order to order book
        try:
            order = Order(
                order_id=uuid.UUID(order_id),
                instrument=self.instrument,
                side=side,
                original_quantity=quantity,
                order_type=order_type,
                agent_id=sender,
                price=price,
                timestamp=timestamp,
                oco_group=payload.get("oco_group"),
                explanation=payload.get("explanation"),
                is_short=payload.get("is_short", False),
                is_short_cover=payload.get("is_short_cover", False)
            )
        except Exception as e:
            await self.send_message(sender, MessageType.ERROR, {"error": f"Failed to create order: {str(e)}"})
            self.logger.error(f"ExchangeAgent {self.agent_id} failed to create Order object: {e}")
            return

        try:
            self.order_book.add_order(order)
            self.logger.info(f"ExchangeAgent {self.agent_id} added order: {order}")

            # Send order confirmation to the submitting agent
            confirmation_payload = {
                "order_id": order_id,
                "status": "ACTIVE",
                "instrument": self.instrument,
                "side": side.value,
                "quantity": quantity,
                "order_type": order_type.value,
                "price": price,
                "timestamp": timestamp.isoformat() if timestamp else None
            }
            await self.send_message(sender, MessageType.ORDER_CONFIRMATION, confirmation_payload)
            self.logger.debug(f"Sent ORDER_CONFIRMATION for order {order_id} to {sender}")

        except Exception as e:
            await self.send_message(sender, MessageType.ERROR,
                              {"error": f"Failed to add order to order book: {str(e)}"})
            self.logger.error(f"ExchangeAgent {self.agent_id} failed to add order to order book: {e}")

    async def _handle_cancel_order(self, sender: Optional[str], payload: Dict[str, Any]):
        """
        Process order cancellation requests with validation.

        Args:
            sender: Agent ID requesting cancellation
            payload: Cancellation details including order_id
        """
        if not sender:
            self.logger.error(f"ExchangeAgent {self.agent_id} received CANCEL_ORDER without sender.")
            return

        order_id = payload.get("order_id")
        if not order_id:
            await self.send_message(sender, MessageType.ERROR,
                              {"error": "Missing 'order_id' in cancel order request."})
            self.logger.error(f"ExchangeAgent {self.agent_id} received CANCEL_ORDER without 'order_id' "
                              f"from agent {sender}.")
            return

        try:
            order_uuid = uuid.UUID(order_id)
        except ValueError:
            await self.send_message(sender, MessageType.ERROR,
                              {"error": "Invalid 'order_id' format."})
            self.logger.error(f"ExchangeAgent {self.agent_id} received CANCEL_ORDER "
                              f"with invalid 'order_id': {order_id}")
            return

        try:
            success = self.order_book.cancel_order(order_uuid)
        except Exception as e:
            await self.send_message(sender, MessageType.ERROR, {"error": f"Error cancelling order: {str(e)}"})
            self.logger.error(f"ExchangeAgent {self.agent_id} encountered an error while "
                              f"cancelling order {order_id}: {e}")
            return

        if success:
            confirmation = {"order_id": order_id, "status": OrderStatus.CANCELED.value}
            await self.send_message(sender, MessageType.ORDER_CANCELLATION_CONFIRMATION, confirmation)
            self.logger.info(f"ExchangeAgent {self.agent_id} canceled order {order_id} for agent {sender}.")
        else:
            await self.send_message(sender, MessageType.ERROR,
                              {"error": "Unable to cancel order. It may already be filled or canceled."})
            self.logger.warning(f"ExchangeAgent {self.agent_id} failed to cancel order {order_id} for agent {sender}.")


    def _aggregate_trade_data_multi_resolution(
        self,
        window_start: datetime,
        window_end: datetime,
        resolution: str = "1d"
    ) -> Dict[str, Any]:
        """
        Aggregate trade data and compute technical indicators for specified time window and resolution.
        
        This method enables comprehensive market analysis by computing OHLCV data,
        technical indicators, and current order book state for LLM agent decision-making.
        Uses resolution-specific IndicatorsTracker for technical analysis computations.

        Args:
            window_start: Start of aggregation time window
            window_end: End of aggregation time window
            resolution: Time resolution for aggregation (e.g., "1m", "15m", "1h", "1d")

        Returns:
            Dictionary containing aggregated data and technical indicators for the specified resolution
        """
        if resolution not in self.resolutions:
            raise ValueError(f"Unsupported resolution: {resolution}. Available: {self.resolutions}")

        # Get historical candles for this resolution using binary search for efficient filtering
        historical_candles = self.resolution_candles.get(resolution, [])
        filtered_historical_candles = []

        if historical_candles:
            # Use binary search to find candles within the time window
            def candle_key(candle):
                return parse_datetime_utc(candle["timestamp"])

            # Find start index using binary search
            start_index = bisect.bisect_left(
                historical_candles,
                window_start,
                key=candle_key
            )

            # Find end index using binary search
            end_index = bisect.bisect_right(
                historical_candles,
                window_end,
                key=candle_key
            )

            # Extract candles in the time window
            filtered_historical_candles = historical_candles[start_index:end_index]

        # Get trades for this time window
        trades = self.order_book.trade_history
        start_index = trades.bisect_left({"timestamp": window_start, "seq": -1})
        end_index = trades.bisect_right({"timestamp": window_end, "seq": float("inf")})
        relevant_trades = list(trades[start_index:end_index])
        
        # Get current order book state
        best_bid, bid_qty = self.order_book.get_best_bid()
        best_ask, ask_qty = self.order_book.get_best_ask()
        
        # Create candles from trades for this resolution
        trade_based_candles = self._create_candles_from_trades(
            relevant_trades,
            window_start,
            window_end,
            self.resolution_seconds[resolution]
        )

        # Combine historical candles with trade-based candles
        # No overlap check needed since historical data is from warmup period
        # and trade-based candles are from simulation period
        all_candles = filtered_historical_candles + trade_based_candles

        # Sort candles by timestamp (they should already be mostly sorted)
        all_candles.sort(key=lambda x: parse_datetime_utc(x["timestamp"]))

        # Get current technical indicators for this resolution
        indicators = self.indicators_trackers[resolution].get_latest_values()

        return {
            "resolution": resolution,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "candles": all_candles,
            "best_bid": best_bid,
            "bid_quantity": bid_qty,
            "best_ask": best_ask,
            "ask_quantity": ask_qty,
            "indicators": indicators,
            "historical_candles_count": len(filtered_historical_candles),
            "trade_based_candles_count": len(trade_based_candles),
            "total_candles_count": len(all_candles)
        }
    
    def _create_candles_from_trades(
        self, 
        trades: List[Dict[str, Any]], 
        window_start: datetime, 
        window_end: datetime, 
        resolution_seconds: int
    ) -> List[Dict[str, Any]]:
        """
        Create OHLCV candles from trade data for a specific resolution with caching.

        Args:
            trades: List of trade records
            window_start: Start of time window
            window_end: End of time window
            resolution_seconds: Resolution in seconds
            
        Returns:
            List of OHLCV candle dictionaries
        """
        if not trades:
            return []

        # Create cache key
        start_ts = int(window_start.timestamp())
        end_ts = int(window_end.timestamp())
        cache_key = (resolution_seconds, start_ts, end_ts)

        # Check cache first
        if cache_key in self._candle_cache:
            self.logger.debug(f"Cache hit for candles: resolution={resolution_seconds}s, window=[{window_start}, {window_end}]")
            return self._candle_cache[cache_key].copy()

        candles = []
        
        # Align window_start to resolution boundary
        start_timestamp = int(window_start.timestamp())
        aligned_start = start_timestamp - (start_timestamp % resolution_seconds)
        current_candle_start = datetime.fromtimestamp(aligned_start, tz=window_start.tzinfo)
        
        while current_candle_start < window_end:
            current_candle_end = current_candle_start + timedelta(seconds=resolution_seconds)
            
            # Find trades in this candle period using binary search
            def trade_key(trade):
                return trade["timestamp"], trade.get("seq", 0)

            start_key = (current_candle_start, -1)
            end_key = (current_candle_end, float("inf"))

            start_index = bisect.bisect_left(trades, start_key, key=trade_key)
            end_index = bisect.bisect_right(trades, end_key, key=trade_key)
            candle_trades = trades[start_index:end_index]

            if candle_trades:
                prices = [t["price"] for t in candle_trades]
                volumes = [t["quantity"] for t in candle_trades]
                
                candle = {
                    "timestamp": current_candle_start.isoformat(),
                    "open": prices[0],
                    "high": max(prices),
                    "low": min(prices),
                    "close": prices[-1],
                    "volume": sum(volumes),
                    "vwap": round(
                        sum(t["price"] * t["quantity"] for t in candle_trades) / sum(volumes),
                        4
                    ) if sum(volumes) > 0 else None,
                    "trade_count": len(candle_trades)
                }
                candles.append(candle)
            
            current_candle_start = current_candle_end

        # Cache the result
        self._candle_cache[cache_key] = candles.copy()
        self.logger.debug(f"Cached candles: resolution={resolution_seconds}s, window=[{window_start}, {window_end}], count={len(candles)}")

        return candles

    async def _handle_market_data_snapshot_request(self, sender: Optional[str], payload: Dict[str, Any]):
        """
        Process market data snapshot requests with multi-resolution technical indicator computation.
        
        Provides comprehensive market analysis data for LLM agent decision-making,
        including OHLCV data, technical indicators, and current order book state
        for the requested resolution.

        Args:
            sender: Agent ID requesting market data
            payload: Request parameters including time window and resolution
        """
        if not sender:
            self.logger.error("MARKET_DATA_SNAPSHOT_REQUEST missing sender.")
            return
            
        try:
            window_start = parse_datetime_utc(payload["window_start"])
            window_end = parse_datetime_utc(payload["window_end"])
        except (KeyError, ValueError) as e:
            await self.send_message(sender, MessageType.ERROR, {"error": f"Invalid window parameters: {e}"})
            return

        # Extract resolution parameter (default to "1d" if not specified)
        resolution = payload.get("resolution", "1d")
        
        if resolution not in self.resolutions:
            await self.send_message(sender, MessageType.ERROR, {
                "error": f"Unsupported resolution: {resolution}. Available: {self.resolutions}"
            })
            return

        try:
            aggregated_data = self._aggregate_trade_data_multi_resolution(
                window_start,
                window_end,
                resolution
            )
        except Exception as e:
            await self.send_message(sender, MessageType.ERROR, {"error": f"Failed to aggregate data: {e}"})
            return

        response_payload = {
            "instrument": self.instrument,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "resolution": resolution,
            "aggregated_data": aggregated_data,
            "available_resolutions": self.resolutions
        }
        await self.send_message(sender, MessageType.MARKET_DATA_SNAPSHOT_RESPONSE, response_payload)
        self.logger.info(
            f"ExchangeAgent {self.agent_id} responded to {sender} with {resolution} market data snapshot for "
            f"window [{window_start}, {window_end}]."
        )

    async def _handle_news_snapshot_request(self, sender: str, payload: Dict[str, Any]):
        """Handle news data requests with configurable news source support."""
        if not sender:
            self.logger.error("NEWS_SNAPSHOT_REQUEST missing sender.")
            return
            
        try:
            window_start = parse_datetime_utc(payload["window_start"])
            window_end = parse_datetime_utc(payload["window_end"])
        except (KeyError, ValueError) as e:
            await self.send_message(sender, MessageType.ERROR, {"error": f"Invalid window parameters: {e}"})
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
            self.logger.error("FUNDAMENTALS_REQUEST missing sender.")
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
            await self.send_message(sender, MessageType.ERROR, {"error": f"Invalid window_start: {e}"})
            return

        try:
            as_of_dt = parse_datetime_utc(as_of_raw)
            as_of_str = as_of_dt.date().isoformat()
        except Exception as e:
            await self.send_message(sender, MessageType.ERROR, {"error": f"Invalid window_end: {e}"})
            return

        # Filter fundamental data within date range
        if self.data_source == "polygon" and self.symbol_type == "stock":
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
            # No fundamental data for crypto or non-polygon sources
            response = {"instrument": self.instrument, "fundamentals": {}}

        await self.send_message(sender, MessageType.FUNDAMENTALS_RESPONSE, response)

    async def _dispatch_new_trades(self):
        """
        Dispatch trade execution notifications to participating agents.
        
        Implements incremental trade dispatch to avoid duplicate notifications
        while ensuring all agents receive real-time trade confirmations.
        """
        trades = self.order_book.trade_history
        if self.last_dispatched_trade_key is None:
            start_index = 0
        else:
            start_index = trades.bisect_right({
                "timestamp": self.last_dispatched_trade_key[0],
                "seq": self.last_dispatched_trade_key[1]
            })

        new_trades = list(trades[start_index:])

        for trade in new_trades:
            dt = trade["timestamp"]
            dt_str = dt.isoformat() if isinstance(dt, datetime) else str(dt)
            trade_msg = {
                "instrument": trade["instrument"],
                "price": trade["price"],
                "quantity": trade["quantity"],
                "timestamp": dt_str,
                "seq": trade["seq"]
            }
            
            # Notify buyer with extended order information
            buy_agent = trade["buy_agent"]
            buy_order_id = trade.get("buy_order_id")
            buy_order = self.order_book.orders_by_id.get(buy_order_id) if buy_order_id else None
            
            buy_payload = {
                **trade_msg, 
                "role": Role.BUYER.value, 
                "order_id": str(buy_order_id) if buy_order_id else None, 
                "order_status": trade["buy_order_status"],
                "explanation": buy_order.explanation if buy_order else None,
                "is_short": buy_order.is_short if buy_order else False,
                "is_short_cover": buy_order.is_short_cover if buy_order else False
            }
            
            await self.send_message(buy_agent, MessageType.TRADE_EXECUTION, buy_payload)

            # Notify seller with extended order information
            sell_agent = trade["sell_agent"]
            sell_order_id = trade.get("sell_order_id")
            sell_order = self.order_book.orders_by_id.get(sell_order_id) if sell_order_id else None
            
            sell_payload = {
                **trade_msg, 
                "role": Role.SELLER.value, 
                "order_id": str(sell_order_id) if sell_order_id else None, 
                "order_status": trade["sell_order_status"],
                "explanation": sell_order.explanation if sell_order else None,
                "is_short": sell_order.is_short if sell_order else False,
                "is_short_cover": sell_order.is_short_cover if sell_order else False
            }
            
            await self.send_message(sell_agent, MessageType.TRADE_EXECUTION, sell_payload)
            
            # Handle OCO cancellations for filled orders
            for order_id, order in [(buy_order_id, buy_order), (sell_order_id, sell_order)]:
                if order_id and order and trade.get(f"{'buy' if order_id == buy_order_id else 'sell'}_order_status") == OrderStatus.FILLED.value:
                    await self._cancel_oco_siblings(order)

            self.logger.info(f"ExchangeAgent {self.agent_id} dispatched TRADE_EXECUTION "
                             f"to {buy_agent} and {sell_agent} for trade: {trade_msg}")

        if new_trades:
            last_trade = new_trades[-1]
            self.last_dispatched_trade_key = (last_trade["timestamp"], last_trade["seq"])

    async def _cancel_oco_siblings(self, filled_order: Order):
        """
        Cancel other orders in the same OCO (One-Cancels-Other) group.
        
        Args:
            filled_order: The order that was just filled
        """
        if not filled_order.oco_group:
            return
            
        orders_to_cancel = []
        
        # Find other orders in the same OCO group
        for order in self.order_book.orders_by_id.values():
            if (order.status == OrderStatus.ACTIVE and
                order.oco_group == filled_order.oco_group and
                order.agent_id == filled_order.agent_id and
                order.order_id != filled_order.order_id):
                orders_to_cancel.append(order)
        
        # Cancel the sibling orders
        for order in orders_to_cancel:
            try:
                success = self.order_book.cancel_order(order.order_id)
                if success:
                    self.logger.info(f"Canceled OCO sibling order {order.order_id} because {filled_order.order_id} was filled.")
                else:
                    self.logger.warning(f"Failed to cancel OCO sibling order {order.order_id}")
            except Exception as e:
                self.logger.error(f"Error canceling OCO sibling order {order.order_id}: {e}")


    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        """
        Handle simulation time tick with trade dispatch and barrier synchronization.
        
        Process time tick similar to candle-based exchange for consistent behavior
        across different exchange modes.

        Args:
            payload: Time tick information including tick_id
        """
        await super().handle_time_tick(payload)
        tick_id = payload.get("tick_id")

        try:
            await self._dispatch_new_trades()
        except Exception as e:
            self.logger.error(f"ExchangeAgent {self.agent_id} encountered an error during dispatch: {e}")

        # Update indicators with latest trade data
        await self._update_indicators_with_recent_trades()

        # Notify subscribed agents of price update using latest trade close price
        if self.order_book.trade_history:
            latest_trade = self.order_book.trade_history[-1]
            latest_close_price = latest_trade["price"]

            for agent_id in self.subscribed_agents:
                await self.send_message(agent_id, MessageType.PORTFOLIO_UPDATE,
                                      {"instrument": self.instrument, "close_price": latest_close_price})

        # Signal completion to simulation clock
        await self.publish_time(
            msg_type=MessageType.BARRIER_RESPONSE,
            payload={"tick_id": tick_id},
            routing_key="simulation_clock"
        )
        self.logger.info(f"Sent BARRIER_RESPONSE for tick {tick_id}.")

    async def _update_indicators_with_recent_trades(self):
        """
        Update technical indicators with recent trade data for all resolutions.

        Creates synthetic candles from recent trades to feed all IndicatorsTrackers,
        maintaining separate indicator state for each resolution.
        """
        if not self.order_book.trade_history:
            return

        # Get only new trades since last update for efficiency
        trades = self.order_book.trade_history
        if self.last_indicator_update_key is None:
            start_index = 0
        else:
            start_index = trades.bisect_right({
                "timestamp": self.last_indicator_update_key[0],
                "seq": self.last_indicator_update_key[1]
            })

        new_trades = list(trades[start_index:])

        if not new_trades:
            return

        # Update indicators for each resolution using only new trades
        for resolution in self.resolutions:
            await self._update_resolution_indicators(resolution, new_trades)

        # Update the last processed trade key
        if new_trades:
            last_trade = new_trades[-1]
            self.last_indicator_update_key = (last_trade["timestamp"], last_trade["seq"])

    async def _update_resolution_indicators(self, resolution: str, trades: List[Dict[str, Any]]):
        """
        Update indicators for a specific resolution using trade data.

        Args:
            resolution: Time resolution (e.g., "1m", "15m", "1h", "1d")
            trades: List of trade records
        """
        if not trades:
            return

        resolution_seconds = self.resolution_seconds[resolution]
        current_time = self.current_time

        # Find the current candle period for this resolution
        current_timestamp = int(current_time.timestamp())
        aligned_start = current_timestamp - (current_timestamp % resolution_seconds)
        candle_start = datetime.fromtimestamp(aligned_start, tz=current_time.tzinfo)
        candle_end = candle_start + timedelta(seconds=resolution_seconds)

        # Get trades in the current candle period using binary search for efficiency
        def trade_key(trade):
            return trade["timestamp"], trade.get("seq", 0)

        start_key = (candle_start, -1)
        end_key = (candle_end, float("inf"))

        start_index = bisect.bisect_left(trades, start_key, key=trade_key)
        end_index = bisect.bisect_right(trades, end_key, key=trade_key)
        candle_trades = trades[start_index:end_index]

        if candle_trades:
            prices = [trade["price"] for trade in candle_trades]
            volumes = [trade["quantity"] for trade in candle_trades]

            synthetic_candle = {
                "timestamp": candle_start.isoformat(),
                "open": prices[0],
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],
                "volume": sum(volumes)
            }

            # Update indicators for this resolution
            self.indicators_trackers[resolution].update(synthetic_candle)

            # Store completed candle if this is a new candle period
            if (not self.resolution_candles[resolution] or
                self.resolution_candles[resolution][-1]["timestamp"] != synthetic_candle["timestamp"]):

                # Add the new candle to the history
                self.resolution_candles[resolution].append(synthetic_candle.copy())

                # Broadcast candle update to dashboard
                self._broadcast_candle_update(resolution, synthetic_candle)
