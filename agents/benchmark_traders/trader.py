"""
Base Trader Agent for StockSim

This module implements the foundational trading agent architecture for the StockSim
simulation platform, providing comprehensive portfolio management and execution capabilities.

Key Features:
- Support for long/short positions with realistic P&L tracking
- Asynchronous message handling for real-time trading simulation
- Robust performance metrics calculation (Sharpe ratio, drawdown, etc.)
- Production-grade order management and execution tracking
- Extensible architecture for both LLM and algorithmic trading strategies

Technical Architecture:
- Inherits from base Agent class for RabbitMQ messaging
- Implements realistic trading mechanics with proper position tracking
- Supports multiple order types (MARKET, LIMIT) and advanced features
- Comprehensive logging and metrics export for analysis
- Thread-safe portfolio updates with mark-to-market calculations
- Multi-instrument support for portfolio-level coordination

This enables multi-agent trading capabilities, supporting both competitive and
cooperative agent interactions in realistic financial simulation environments.
"""


import asyncio
import json
import uuid
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Optional, List
from abc import ABC

from agents.agent import Agent
from utils.messages import MessageType
from utils.metrics import RationalityMetrics
from utils.role import Role


class TraderAgent(Agent, ABC):
    """
    Base trading agent implementing comprehensive portfolio management and execution logic.

    This agent provides the foundational trading capabilities required for the StockSim
    simulation platform, supporting both traditional algorithmic strategies and advanced
    LLM-based trading approaches.

    Core Capabilities:
    - Multi-instrument portfolio management with FIFO accounting
    - Long and short position support with realistic P&L calculation
    - Comprehensive performance metrics tracking
    - Asynchronous order execution with proper risk management
    - Real-time market data processing and response
    - Robust logging and audit trail generation

    The agent maintains separate tracking for:
    - Cash balance and available buying power
    - Long positions (shares owned) with cost basis tracking
    - Short positions (shares borrowed) with mark-to-market P&L
    - Realized and unrealized profit/loss calculations
    - Trade execution history for performance analysis
    """

    DEFAULT_INITIAL_CASH = 100000.0

    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        agent_id: Optional[str] = None,
        rabbitmq_host: str = 'localhost',
        initial_cash: float = DEFAULT_INITIAL_CASH,
        initial_positions: Optional[Dict[str, int]] = None,
        initial_cost_basis: Optional[Dict[str, float]] = None,
        action_interval_seconds: int = 10
    ):
        """
        Initialize the TraderAgent with portfolio and messaging configuration.

        Args:
            instrument_exchange_map: Mapping of instruments to their exchange IDs
            agent_id: Unique identifier for the agent (UUID generated if None)
            rabbitmq_host: RabbitMQ server hostname for messaging
            initial_cash: Starting cash balance for trading
            initial_positions: Pre-existing positions (symbol -> quantity)
            initial_cost_basis: Cost basis for initial positions (symbol -> average price)
            action_interval_seconds: Minimum interval between scheduled actions
        """
        super().__init__(
            agent_id=agent_id,
            rabbitmq_host=rabbitmq_host
        )

        # Core trading configuration
        self.instrument_exchange_map = instrument_exchange_map
        self.cash: float = initial_cash

        # Position tracking with FIFO accounting
        self.long_qty: Dict[str, int] = defaultdict(int, initial_positions or {})
        self.short_qty: Dict[str, int] = defaultdict(int)  # Stored as positive quantities

        # FIFO lot tracking for accurate P&L calculation
        self.long_lots: Dict[str, deque] = defaultdict(deque)  # (quantity, cost_basis)
        self.short_lots: Dict[str, deque] = defaultdict(deque)  # (quantity, short_price)

        # Initialize cost basis for existing positions
        if initial_positions and initial_cost_basis:
            for instrument, quantity in initial_positions.items():
                cost_basis = initial_cost_basis.get(instrument, 0.0)
                self.long_lots[instrument].append((quantity, cost_basis))

        # Market data and portfolio tracking
        self.prices: Dict[str, float] = defaultdict(float)
        self.portfolio_value: float = initial_cash
        self.realized_pnl: Dict[str, float] = defaultdict(float)

        # Performance tracking and metrics
        self.metrics = RationalityMetrics()
        self.session_executed_orders: List[Dict[str, Any]] = []
        
        # Order tracking and management
        self.pending_orders: Dict[str, Dict[str, Any]] = {}  # order_id -> order_info
        self.order_status: Dict[str, str] = {}  # order_id -> status

        # Action scheduling and exchange management
        self.action_schedule: Dict[datetime, list] = defaultdict(list)
        self.subscribed_exchanges: set = set()
        self.last_market_snapshot: Dict[str, Any] = {
            instrument: None for instrument in instrument_exchange_map.keys()
        }

        # Timing configuration
        self.action_interval = timedelta(seconds=action_interval_seconds)
        self.next_action_time: Optional[datetime] = None

        self.logger.info(
            f"TraderAgent {self.agent_id} initialized with cash balance ${self.cash:,.2f}"
        )

    async def initialize(self):
        """
        Initialize the agent's market data subscriptions after connection establishment.

        This method is called after the RabbitMQ connection is established and
        automatically subscribes to market data feeds for all configured instruments.
        """
        await super().initialize()
        await self.subscribe_to_exchanges()

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        """
        Process time tick events and execute scheduled trading actions.

        This method is called on each simulation time step and:
        1. Updates the agent's current time
        2. Executes any scheduled actions that are due
        3. Maintains deterministic execution ordering

        Args:
            payload: Time tick payload containing current_time and tick_id
        """
        await super().handle_time_tick(payload)

        # Execute all scheduled actions that are due
        due_times = sorted([
            action_time for action_time in self.action_schedule
            if action_time <= self.current_time
        ])

        for action_time in due_times:
            actions = self.action_schedule[action_time]
            for action in actions:
                try:
                    await action()
                    self.logger.debug(
                        f"Executed scheduled action for {action_time.isoformat()}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error executing scheduled action at {action_time.isoformat()}: {e}"
                    )
            del self.action_schedule[action_time]

    def schedule_action(self, action_time: datetime, action_func: Callable[[], Any]) -> None:
        """
        Schedule an asynchronous action to be executed at a specific simulation time.

        This enables precise timing control for trading strategies, allowing actions
        to be scheduled based on market events or time-based triggers.

        Args:
            action_time: Simulation time when the action should be executed
            action_func: Async function to execute (coroutine)
        """
        self.action_schedule[action_time].append(action_func)
        self.logger.debug(f"Scheduled action for {action_time.isoformat()}")

    def get_performance_metrics(self, risk_free_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for research analysis.

        Computes a full suite of financial performance indicators including:
        - Return on Investment (ROI) and Return on Invested Capital (ROIC)
        - Risk-adjusted returns (Sharpe ratio, Sortino ratio)
        - Risk metrics (maximum drawdown, volatility)
        - Trading activity metrics (win rate, profit factor, trade frequency)

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation (annualized).
                          If None, calculated based on action interval and standard assumptions.

        Returns:
            Dictionary containing all calculated performance metrics
        """
        # Calculate risk-free rate if not provided
        if risk_free_rate is None:
            risk_free_rate = self._calculate_risk_free_rate()

        closing_trades = [
            trade for trade in self.metrics.trade_history
            if trade["action"] in {"SELL", "SHORT_COVER"}
        ]

        profit_per_trade = 0.0
        if closing_trades:
            total_profit = sum(trade.get("realized_profit", 0) for trade in closing_trades)
            profit_per_trade = round(total_profit / len(closing_trades), 4)

        return {
            "ROI": self.metrics.compute_roi(),
            "Sharpe Ratio": self.metrics.compute_sharpe_ratio(risk_free_rate),
            "Annualized Sharpe Ratio": self.metrics.compute_sharpe_ratio(
                risk_free_rate, annualize=True
            ),
            "Sortino Ratio": self.metrics.compute_sortino_ratio(risk_free_rate),
            "Win Rate": self.metrics.compute_win_rate(),
            "Profit Factor": self.metrics.compute_profit_factor(),
            "Max Drawdown": self.metrics.compute_max_drawdown(),
            "Num Trades": len(self.metrics.trade_history),
            "Num Closed Trades": len(closing_trades),
            "Total Traded Volume": self.metrics.compute_total_traded_volume(),
            "Average Trade Size": self.metrics.compute_average_trade_size(),
            "ROIC": self.metrics.compute_roic(),
            "Profit per Trade": profit_per_trade,
            "Last Portfolio Value": self.metrics.get_last_portfolio_value(),
            "Risk Free Rate Used": risk_free_rate
        }

    def _calculate_risk_free_rate(self) -> float:
        """
        Calculate appropriate risk-free rate based on trading interval.

        Uses standard assumptions:
        - Annual risk-free rate of ~5% (0.05)
        - Adjusts based on action_interval for period-appropriate rate

        Returns:
            Period-appropriate risk-free rate
        """
        # Standard annual risk-free rate assumption (5%)
        annual_risk_free_rate = 0.05

        # Convert action interval to years for proper scaling
        action_interval_seconds = self.action_interval.total_seconds()

        # Common trading intervals and their annual scaling factors
        seconds_per_year = 365.25 * 24 * 3600  # ~31,557,600 seconds

        if action_interval_seconds >= 86400:  # Daily or longer
            # For daily+ intervals, use daily risk-free rate
            period_risk_free_rate = annual_risk_free_rate / 365.25
        elif action_interval_seconds >= 3600:  # Hourly intervals
            # For hourly intervals, use hourly risk-free rate (assuming 24/7 markets)
            period_risk_free_rate = annual_risk_free_rate / (365.25 * 24)
        elif action_interval_seconds >= 60:  # Minute intervals
            # For minute intervals, use minute risk-free rate
            period_risk_free_rate = annual_risk_free_rate / (365.25 * 24 * 60)
        else:
            # For very short intervals, use a minimal rate
            period_risk_free_rate = annual_risk_free_rate / seconds_per_year

        self.logger.debug(
            f"Calculated risk-free rate: {period_risk_free_rate:.6f} "
            f"(annual: {annual_risk_free_rate:.2%}, interval: {action_interval_seconds}s)"
        )

        return period_risk_free_rate

    async def _handle_regular_message(self, msg: Dict[str, Any]) -> None:
        """
        Route incoming messages to appropriate handlers.

        Processes all non-time-tick messages including market data updates,
        trade executions, order confirmations, and error messages.

        Args:
            msg: Decoded message dictionary with type and payload
        """
        msg_type_str = msg.get("type")
        msg_type = MessageType(msg_type_str)
        if msg_type == MessageType.MARKET_DATA_SNAPSHOT_RESPONSE:
            asyncio.create_task(self._handle_market_data_snapshot(msg.get("payload", {})))
        elif msg_type == MessageType.TRADE_EXECUTION:
            await self._handle_trade_execution(msg.get("payload", {}))
        elif msg_type == MessageType.ORDER_CONFIRMATION:
            await self._handle_order_confirmation(msg.get("payload", {}))
        elif msg_type == MessageType.ORDER_CANCELLATION_CONFIRMATION:
            await self._handle_order_cancellation_confirmation(msg.get("payload", {}))
        elif msg_type == MessageType.SUBSCRIPTION_CONFIRMATION:
            asyncio.create_task(self._handle_subscription_confirmation(msg.get("payload", {})))
        elif msg_type == MessageType.UNSUBSCRIPTION_CONFIRMATION:
            asyncio.create_task(self._handle_unsubscription_confirmation(msg.get("payload", {})))
        elif msg_type == MessageType.PORTFOLIO_UPDATE:
            asyncio.create_task(self._handle_portfolio_update(msg.get("payload", {})))
        elif msg_type == MessageType.NEWS_SNAPSHOT_RESPONSE:
            asyncio.create_task(self._handle_news_snapshot_response(msg.get("payload", {})))
        elif msg_type == MessageType.FUNDAMENTALS_RESPONSE:
            asyncio.create_task(self._handle_fundamentals_response(msg.get("payload", {})))
        elif msg_type == MessageType.ERROR:
            asyncio.create_task(self._handle_error(msg.get("payload", {}).get("error", "Unknown error")))
        else:
            self.logger.warning(
                f"TraderAgent {self.agent_id} received unknown message type: {msg_type}"
            )


    async def _handle_market_data_snapshot(self, payload: Dict[str, Any]) -> None:
        """
        Process market data snapshot updates.

        Updates the agent's view of current market prices and triggers
        the on_market_data_update callback for strategy-specific processing.

        Args:
            payload: Market data snapshot containing price and volume information
        """
        instrument = payload.get("instrument")
        if instrument is not None:
            self.last_market_snapshot[instrument] = payload
            asyncio.create_task(self.on_market_data_update(instrument, payload))
        else:
            self.logger.warning("Received market data snapshot with missing instrument")

    async def _handle_trade_execution(self, payload: Dict[str, Any]) -> None:
        """
        Process trade execution confirmations.

        Updates portfolio positions, cash balance, and P&L based on executed trades.
        This is a critical method that maintains portfolio consistency.

        Args:
            payload: Trade execution details including price, quantity, and flags
        """
        await self.on_trade_execution(payload)

    async def _handle_news_snapshot_response(self, payload: Dict[str, Any]):
        """Handle news data updates (can be overridden for news-aware strategies)."""
        pass

    async def _handle_fundamentals_response(self, payload: Dict[str, Any]):
        """Handle fundamental data updates (can be overridden for fundamental strategies)."""
        pass

    async def _handle_portfolio_update(self, payload: Dict[str, Any]) -> None:
        """
        Process portfolio value updates from exchanges.

        Updates current prices and recalculates mark-to-market portfolio value.

        Args:
            payload: Portfolio update containing instrument and current price
        """
        instrument = payload.get("instrument")
        close_price = payload.get("close_price")

        if instrument and close_price:
            self.prices[instrument] = close_price
        self._mark_to_market()

    def _mark_to_market(self):
        """
        Calculate current portfolio value based on market prices.

        Computes total portfolio value including:
        - Cash balance
        - Market value of long positions
        - Mark-to-market value of short positions

        Updates the metrics tracking for performance analysis.
        """
        value = self.cash

        for symbol, price in self.prices.items():
            value += self.long_qty[symbol] * price   # Add long position value
            value -= self.short_qty[symbol] * price  # Subtract short position liability

        self.portfolio_value = value
        self.metrics.record_portfolio_value(
            value,
            timestamp=self.current_time.isoformat() if self.current_time else None
        )

        self.logger.debug(
            f"Portfolio value updated at {self.current_time.isoformat() if self.current_time else None}: "
            f"${value:,.2f}"
        )


    async def _handle_order_cancellation_confirmation(self, payload: Dict[str, Any]) -> None:
        """Handle order cancellation confirmations."""
        order_id = payload.get("order_id")
        status = payload.get("status", "CANCELED")
        
        if order_id:
            # Update order status
            self.order_status[order_id] = status
            
            # Remove from pending orders if canceled
            if status == "CANCELED" and order_id in self.pending_orders:
                canceled_order = self.pending_orders.pop(order_id)
                self.logger.info(f"Order {order_id} canceled: {canceled_order['instrument']} {canceled_order['side']} {canceled_order['quantity']}")
            
            # Call strategy-specific handler
            await self.on_order_canceled(order_id, payload)
        else:
            self.logger.warning(f"Order cancellation confirmation missing order_id: {payload}")

    async def _handle_subscription_confirmation(self, payload: Dict[str, Any]) -> None:
        """Handle market data subscription confirmations."""
        instrument = payload.get("instrument")
        status = payload.get("status")

        if instrument and status:
            self.logger.info(f"Subscription confirmation for {instrument}: {status}")
            if status == "subscribed":
                self.subscribed_exchanges.add(instrument)
        else:
            self.logger.warning(f"Incomplete subscription confirmation: {payload}")

    async def _handle_unsubscription_confirmation(self, payload: Dict[str, Any]) -> None:
        """Handle market data unsubscription confirmations."""
        instrument = payload.get("instrument")
        status = payload.get("status")

        if instrument and status:
            self.logger.info(f"Unsubscription confirmation for {instrument}: {status}")
            if status == "unsubscribed":
                self.subscribed_exchanges.discard(instrument)
        else:
            self.logger.warning(f"Incomplete unsubscription confirmation: {payload}")

    async def _handle_error(self, error_msg: str) -> None:
        """Handle error messages from exchanges or other agents."""
        self.logger.error(f"Agent encountered error: {error_msg}")

    async def on_market_data_update(self, instrument: str, snapshot: Dict[str, Any]) -> None:
        """
        Handle market data updates - override in strategy implementations.

        This method is called whenever new market data is received and should
        be overridden by concrete trading strategies to implement decision logic.

        Args:
            instrument: The financial instrument that was updated
            snapshot: Complete market data snapshot with price/volume information
        """
        pass

    async def on_trade_execution(self, trade_data: Dict[str, Any]) -> None:
        """
        Update portfolio state based on executed trades.

        This method implements comprehensive trade processing including:
        - Position updates for long/short trades
        - FIFO cost basis tracking for accurate P&L calculation
        - Cash balance adjustments
        - Metrics recording for performance analysis

        Supports all trade types:
        - Regular BUY/SELL for long positions
        - SHORT (opening short positions)
        - SHORT_COVER (closing short positions)
        - Partial fills and position sizing constraints

        Args:
            trade_data: Trade execution details from exchange
        """
        try:
            symbol = trade_data["instrument"]
            qty_requested = int(trade_data["quantity"])
            price = float(trade_data["price"])
            role = trade_data["role"]
            is_short = trade_data.get("is_short", False)
            is_short_cover = trade_data.get("is_short_cover", False)
            order_type = trade_data.get("order_type")
            explanation = trade_data.get("explanation")
            timestamp = self.current_time.isoformat() if self.current_time else None

            executed_qty = qty_requested

            if role == Role.BUYER.value:
                if is_short_cover:
                    # Cover short positions (buy to close shorts)
                    cover_qty = min(qty_requested, self.short_qty[symbol])
                    if cover_qty == 0:
                        self.logger.warning(
                            f"SHORT_COVER {qty_requested} {symbol} ignored - no shorts to cover"
                        )
                        return

                    if executed_qty < qty_requested:
                        self.logger.warning(
                            f"{role} requested {qty_requested}, executed {executed_qty}.  Cancelling remainder."
                        )
                        # send a cancellation to wipe out the rest
                        order_id = trade_data.get("order_id")
                        if order_id:
                            await self.cancel_order(order_id)

                    executed_qty = cover_qty
                    pnl = 0.0
                    remaining = cover_qty

                    # Calculate P&L using FIFO short lot accounting
                    while remaining > 0 and self.short_lots[symbol]:
                        lot_qty, lot_price = self.short_lots[symbol][0]
                        qty_to_cover = min(lot_qty, remaining)
                        pnl += (lot_price - price) * qty_to_cover  # Profit if price dropped
                        remaining -= qty_to_cover

                        if qty_to_cover == lot_qty:
                            self.short_lots[symbol].popleft()
                        else:
                            self.short_lots[symbol][0] = (lot_qty - qty_to_cover, lot_price)

                    # Update positions and cash
                    self.short_qty[symbol] -= cover_qty
                    self.realized_pnl[symbol] += pnl
                    self.cash -= price * cover_qty

                    # Record trade for metrics
                    self.metrics.record_trade({
                        "timestamp": timestamp,
                        "instrument": symbol,
                        "action": "SHORT_COVER",
                        "quantity": cover_qty,
                        "price": price,
                        "realized_profit": pnl
                    })

                    self.session_executed_orders.append({
                        "timestamp": timestamp,
                        "instrument": symbol,
                        "action": "SHORT_COVER",
                        "orderType": order_type,
                        "price": price,
                        "quantity": cover_qty,
                        "explanation": explanation
                    })

                    if cover_qty < qty_requested:
                        self.logger.warning(
                            f"SHORT_COVER requested {qty_requested}, executed {cover_qty}"
                        )

                else:
                    # Regular buy (open/add to long position)
                    if qty_requested > 0:
                        self.long_qty[symbol] += qty_requested
                        self.long_lots[symbol].append((qty_requested, price))
                        self.cash -= price * qty_requested

                        self.metrics.record_trade({
                            "timestamp": timestamp,
                            "instrument": symbol,
                            "action": "BUY",
                            "quantity": qty_requested,
                            "price": price,
                            "realized_profit": 0.0
                        })

                        self.session_executed_orders.append({
                            "timestamp": timestamp,
                            "instrument": symbol,
                            "action": "BUY",
                            "orderType": order_type,
                            "price": price,
                            "quantity": qty_requested,
                            "explanation": explanation
                        })

            elif role == Role.SELLER.value:
                if is_short:
                    # Open short position (sell short)
                    self.short_qty[symbol] += qty_requested
                    self.short_lots[symbol].append((qty_requested, price))
                    self.cash += price * qty_requested

                    self.metrics.record_trade({
                        "timestamp": timestamp,
                        "instrument": symbol,
                        "action": "SHORT",
                        "quantity": qty_requested,
                        "price": price,
                        "realized_profit": 0.0
                    })

                    self.session_executed_orders.append({
                        "timestamp": timestamp,
                        "instrument": symbol,
                        "action": "SHORT",
                        "orderType": order_type,
                        "price": price,
                        "quantity": qty_requested,
                        "explanation": explanation
                    })

                else:
                    # Regular sell (close long position)
                    held_qty = self.long_qty[symbol]
                    executed_qty = min(qty_requested, held_qty)

                    if executed_qty == 0:
                        self.logger.warning(
                            f"SELL {qty_requested} {symbol} ignored - no long positions"
                        )
                        return

                    if executed_qty < qty_requested:
                        self.logger.warning(
                            f"{role} requested {qty_requested}, executed {executed_qty}.  Cancelling remainder."
                        )
                        # send a cancellation to wipe out the rest
                        order_id = trade_data.get("order_id")
                        if order_id:
                            await self.cancel_order(order_id)

                    pnl = 0.0
                    remaining = executed_qty

                    # Calculate P&L using FIFO long lot accounting
                    while remaining > 0 and self.long_lots[symbol]:
                        lot_qty, lot_price = self.long_lots[symbol][0]
                        qty_to_sell = min(lot_qty, remaining)
                        pnl += (price - lot_price) * qty_to_sell
                        remaining -= qty_to_sell

                        if qty_to_sell == lot_qty:
                            self.long_lots[symbol].popleft()
                        else:
                            self.long_lots[symbol][0] = (lot_qty - qty_to_sell, lot_price)

                    # Update positions and cash
                    self.long_qty[symbol] -= executed_qty
                    self.realized_pnl[symbol] += pnl
                    self.cash += price * executed_qty

                    self.metrics.record_trade({
                        "timestamp": timestamp,
                        "instrument": symbol,
                        "action": "SELL",
                        "quantity": executed_qty,
                        "price": price,
                        "realized_profit": pnl
                    })

                    self.session_executed_orders.append({
                        "timestamp": timestamp,
                        "instrument": symbol,
                        "action": "SELL",
                        "orderType": order_type,
                        "price": price,
                        "quantity": executed_qty,
                        "explanation": explanation
                    })

                    if executed_qty < qty_requested:
                        self.logger.warning(
                            f"SELL requested {qty_requested}, executed {executed_qty}"
                        )

            else:
                self.logger.error(f"Unknown trade role: {role}")
                return

            # Update order status if we have the order_id
            order_id = trade_data.get("order_id")
            order_status = trade_data.get("order_status")
            if order_id:
                self.order_status[order_id] = order_status
                
                # Remove from pending if fully filled
                if order_status == "FILLED" and order_id in self.pending_orders:
                    filled_order = self.pending_orders.pop(order_id)
                    self.logger.debug(f"Order {order_id} fully filled and removed from pending")
                elif order_status == "PARTIALLY_FILLED":
                    # Update remaining quantity for partially filled orders
                    if order_id in self.pending_orders:
                        self.pending_orders[order_id]["remaining_quantity"] = (
                            self.pending_orders[order_id].get("remaining_quantity", 
                            self.pending_orders[order_id]["quantity"]) - executed_qty
                        )

            self.logger.debug(
                f"Trade executed: {symbol} {role} {executed_qty}@${price:.2f} | "
                f"Cash: ${self.cash:,.2f}, Long: {self.long_qty[symbol]}, "
                f"Short: {self.short_qty[symbol]}"
            )

        except Exception as e:
            self.logger.error(f"Error processing trade execution: {e}")

    async def place_order(
        self,
        instrument: str,
        side: str,
        quantity: int,
        order_type: str,
        price: Optional[float] = None,
        oco_group: Optional[str] = None,
        explanation: Optional[str] = None,
        is_short: bool = False,
        is_short_cover: bool = False
    ) -> Optional[str]:
        """
        Submit a trading order to the exchange.

        This method provides a comprehensive order submission interface supporting
        all order types and trading scenarios in the StockSim platform.

        Args:
            instrument: Financial instrument symbol (e.g., "AAPL", "BTC")
            side: Order side ("BUY" or "SELL")
            quantity: Number of shares/units to trade
            order_type: Order type ("MARKET" or "LIMIT")
            price: Limit price for LIMIT orders (required for LIMIT orders)
            oco_group: One-Cancels-Other group ID for linked orders
            explanation: Human-readable explanation for the order (for LLM agents)
            is_short: True if this is a short sale (opening short position)
            is_short_cover: True if this is covering a short position

        Returns:
            Order ID if submission successful, None if failed
        """
        exchange_id = self._get_exchange_for_instrument(instrument)
        if exchange_id is None:
            self.logger.error(f"No exchange found for instrument {instrument}")
            return None

        if side == "SELL" and not is_short:
            held = self.long_qty[instrument]
            if held <= 0:
                self.logger.warning(f"Cannot sell {quantity} {instrument}: no long position")
                return None
            if quantity > held:
                self.logger.warning(f"Trimming SELL {instrument} from {quantity} to {held} (held qty)")
                quantity = held

        if is_short_cover:
            covered = self.short_qty[instrument]
            if covered <= 0:
                self.logger.warning(f"Cannot cover {quantity} {instrument}: no short position")
                return None
            if quantity > covered:
                self.logger.warning(f"Trimming SHORT_COVER {instrument} from {quantity} to {covered}")
                quantity = covered

        order_id = str(uuid.uuid4())

        payload: Dict[str, Any] = {
            "order_id": order_id,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "is_short": is_short,
            "is_short_cover": is_short_cover
        }

        if price is not None:
            payload["price"] = price
        if oco_group is not None:
            payload["oco_group"] = oco_group
        if explanation is not None:
            payload["explanation"] = explanation

        success = await self.send_message(exchange_id, MessageType.ORDER_SUBMISSION, payload)
        if success:
            # Track the pending order
            self.pending_orders[order_id] = {
                "order_id": order_id,
                "instrument": instrument,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "oco_group": oco_group,
                "explanation": explanation,
                "is_short": is_short,
                "is_short_cover": is_short_cover,
                "timestamp": self.current_time.isoformat() if self.current_time else None,
                "exchange_id": exchange_id
            }
            self.order_status[order_id] = "PENDING"
            
            self.logger.info(f"Order {order_id} submitted: {side} {quantity} {instrument} @ {price if price else 'MARKET'}")
            return order_id
        else:
            self.logger.error(f"Failed to submit order {order_id} for {quantity} {instrument}")
            return None

    def _get_exchange_for_instrument(self, instrument: str) -> Optional[str]:
        """
        Get the exchange ID for a given financial instrument.

        Args:
            instrument: Financial instrument symbol

        Returns:
            Exchange ID string, or None if not found
        """
        exchange_id = self.instrument_exchange_map.get(instrument)
        if exchange_id is None:
            self.logger.warning(f"No exchange mapped for instrument {instrument}")
        return exchange_id

    async def subscribe_to_exchanges(self) -> None:
        """
        Subscribe to market data feeds for all configured instruments.

        Sends subscription requests to all exchanges managing the instruments
        that this agent is configured to trade.
        """
        for instrument, exchange_id in self.instrument_exchange_map.items():
            success = await self.send_message(
                exchange_id,
                MessageType.SUBSCRIBE,
                {"instrument": instrument}
            )
            if not success:
                self.logger.error(
                    f"Failed to subscribe to {exchange_id} for {instrument}"
                )

    async def unsubscribe_from_exchange(self, instrument: str) -> None:
        """
        Unsubscribe from market data for a specific instrument.

        Args:
            instrument: Financial instrument to unsubscribe from
        """
        exchange_id = self._get_exchange_for_instrument(instrument)
        if exchange_id is not None:
            success = await self.send_message(
                exchange_id,
                MessageType.UNSUBSCRIBE,
                {"instrument": instrument}
            )
            if success:
                self.logger.info(f"Unsubscribed from {exchange_id} for {instrument}")
            else:
                self.logger.error(f"Failed to unsubscribe from {exchange_id} for {instrument}")

    def get_realized_pnl(self) -> Dict[str, float]:
        """
        Get realized profit/loss for each instrument.

        Returns:
            Dictionary mapping instrument symbols to realized P&L values
        """
        return {
            instrument: round(pnl, 2)
            for instrument, pnl in self.realized_pnl.items()
        }

    def stop(self):
        """
        Gracefully shutdown the agent and export performance data.

        This method:
        1. Calculates final portfolio value
        2. Computes comprehensive performance metrics
        3. Exports all data to JSON files for research analysis
        4. Logs final performance summary
        5. Calls parent shutdown procedures
        """
        # Final portfolio valuation
        self._mark_to_market()
        final_value = self.portfolio_value

        # Log final performance summary
        self.logger.info(f"Final portfolio value: ${final_value:,.2f}")
        self.logger.info(f"Realized P&L: {self.get_realized_pnl()}")

        # Calculate comprehensive metrics with automatic risk-free rate calculation
        metrics = self.get_performance_metrics()  # Now uses calculated risk-free rate
        metrics["Realized P&L"] = self.get_realized_pnl()

        self.logger.info("Final Performance Metrics:\n" + json.dumps(metrics, indent=2))

        # Export data for research analysis
        output_dir = os.getenv("METRICS_OUTPUT_DIR", "metrics")
        os.makedirs(output_dir, exist_ok=True)

        # Export performance metrics
        metrics_file = os.path.join(output_dir, f"metrics_{self.agent_id}.json")
        try:
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            self.logger.info(f"Performance metrics exported to {metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")

        # Export executed orders
        orders_file = os.path.join(output_dir, f"executed_orders_{self.agent_id}.json")
        try:
            with open(orders_file, "w") as f:
                json.dump(self.session_executed_orders, f, indent=2)
            self.logger.info(f"Executed orders exported to {orders_file}")
        except Exception as e:
            self.logger.error(f"Failed to export orders: {e}")

        # Export portfolio time series
        portfolio_file = os.path.join(output_dir, f"portfolio_timeseries_{self.agent_id}.json")
        try:
            with open(portfolio_file, "w") as f:
                json.dump(self.metrics.portfolio_time_series, f, indent=2)
            self.logger.info(f"Portfolio time series exported to {portfolio_file}")
        except Exception as e:
            self.logger.error(f"Failed to export portfolio data: {e}")
            
        # Export pending orders (if any remain)
        if self.pending_orders:
            pending_orders_file = os.path.join(output_dir, f"pending_orders_{self.agent_id}.json")
            try:
                with open(pending_orders_file, "w") as f:
                    json.dump(self.pending_orders, f, indent=2)
                self.logger.info(f"Pending orders exported to {pending_orders_file}")
            except Exception as e:
                self.logger.error(f"Failed to export pending orders: {e}")

        # Call parent shutdown
        super().stop()

    async def _handle_order_confirmation(self, payload: Dict[str, Any]) -> None:
        """Handle order confirmation messages from exchange."""
        order_id = payload.get("order_id")
        status = payload.get("status", "ACTIVE")
        
        if order_id:
            self.order_status[order_id] = status
            
            if order_id in self.pending_orders:
                self.logger.info(f"Order {order_id} confirmed with status: {status}")
                # Call strategy-specific handler
                await self.on_order_confirmed(order_id, payload)
            else:
                self.logger.warning(f"Received confirmation for unknown order {order_id}")
        else:
            self.logger.warning(f"Order confirmation missing order_id: {payload}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            True if cancellation request was sent successfully
        """
        if order_id not in self.pending_orders:
            self.logger.warning(f"Cannot cancel order {order_id} - not found in pending orders")
            return False
            
        order_info = self.pending_orders[order_id]
        exchange_id = order_info["exchange_id"]
        
        payload = {"order_id": order_id}
        
        success = await self.send_message(exchange_id, MessageType.CANCEL_ORDER, payload)
        if success:
            self.logger.info(f"Cancellation request sent for order {order_id}")
            return True
        else:
            self.logger.error(f"Failed to send cancellation request for order {order_id}")
            return False

    def get_pending_orders(self, instrument: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get all pending orders, optionally filtered by instrument.
        
        Args:
            instrument: If provided, only return orders for this instrument
            
        Returns:
            Dictionary of order_id -> order_info for pending orders
        """
        if instrument is None:
            return self.pending_orders.copy()
        else:
            return {
                order_id: order_info 
                for order_id, order_info in self.pending_orders.items()
                if order_info["instrument"] == instrument
            }

    def get_order_status(self, order_id: str) -> Optional[str]:
        """
        Get the current status of an order.
        
        Args:
            order_id: The order ID to check
            
        Returns:
            Order status string or None if order not found
        """
        return self.order_status.get(order_id)

    async def cancel_all_orders(self, instrument: Optional[str] = None) -> int:
        """
        Cancel all pending orders, optionally filtered by instrument.
        
        Args:
            instrument: If provided, only cancel orders for this instrument
            
        Returns:
            Number of cancellation requests sent
        """
        orders_to_cancel = self.get_pending_orders(instrument)
        cancel_count = 0
        
        for order_id in list(orders_to_cancel.keys()):
            success = await self.cancel_order(order_id)
            if success:
                cancel_count += 1
                
        return cancel_count

    async def on_order_confirmed(self, order_id: str, payload: Dict[str, Any]) -> None:
        """
        Handle order confirmation - override in strategy implementations.
        
        Args:
            order_id: The confirmed order ID
            payload: Full confirmation payload
        """
        pass

    async def on_order_canceled(self, order_id: str, payload: Dict[str, Any]) -> None:
        """
        Handle order cancellation - override in strategy implementations.
        
        Args:
            order_id: The canceled order ID
            payload: Full cancellation payload
        """
        pass
