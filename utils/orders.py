"""
Order Management System for StockSim Trading Simulation

This module provides a comprehensive order book implementation for financial trading
simulations. It includes order types, status tracking, and a full order matching engine
with trade execution capabilities.

Classes:
    Side: Enumeration for buy/sell sides
    OrderType: Enumeration for different order types
    OrderStatus: Enumeration for order lifecycle states
    Order: Individual order representation
    OrderBook: Complete order book with matching engine

Example Usage:
    # Create an order book for a stock
    order_book = OrderBook("AAPL")

    # Create a limit buy order
    buy_order = Order(
        order_id=uuid.uuid4(),
        instrument="AAPL",
        side=Side.BUY,
        original_quantity=100,
        order_type=OrderType.LIMIT,
        agent_id="trader_1",
        price=150.00,
        timestamp=datetime.now(),
        oco_group="trade_group_1",
        explanation="Long position entry",
        is_short=False,
        is_short_cover=False
    )

    # Add the order to the book
    order_book.add_order(buy_order)
"""

import heapq
import os
import uuid
from collections import deque, defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from sortedcontainers import SortedList

from utils.logging_setup import setup_logger


class Side(Enum):
    """
    Enumeration representing the side of a trade order.

    Attributes:
        BUY: Represents a buy order (bid)
        SELL: Represents a sell order (ask/offer)
    """
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """
    Enumeration representing different types of orders.

    Attributes:
        LIMIT: Order with a specific price that waits for matching
        MARKET: Order that executes immediately at best available price
        STOP: Stop order that triggers when market price reaches stop price
             - STOP BUY: triggers when market price >= stop price
             - STOP SELL: triggers when market price <= stop price
    """
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"


class OrderStatus(Enum):
    """
    Enumeration representing the lifecycle status of an order.

    Attributes:
        ACTIVE: Order is in the book and waiting for execution
        PARTIALLY_FILLED: Order has been partially executed
        FILLED: Order has been completely executed
        CANCELED: Order has been canceled before completion
    """
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"


@dataclass
class Order:
    """
    Represents a single trading order in the system.

    An order contains all necessary information for trade execution including
    identification, trading parameters, execution tracking, and extended order features.

    Attributes:
        order_id (uuid.UUID): Unique identifier for the order
        instrument (str): Trading symbol (e.g., "AAPL", "MSFT")
        side (Side): Whether this is a buy or sell order
        original_quantity (int): Initial quantity when order was created
        order_type (OrderType): Type of order (LIMIT, MARKET, etc.)
        agent_id (str): Identifier of the trading agent placing the order
        price (Optional[float]): Price for limit orders, None for market orders
        timestamp (Optional[datetime]): When the order was created
        oco_group (Optional[str]): One-Cancels-Other group identifier
        explanation (Optional[str]): Human-readable explanation for the order
        is_short (bool): Whether this is a short sale order
        is_short_cover (bool): Whether this order covers a short position
        quantity (int): Current remaining quantity (auto-initialized)
        status (OrderStatus): Current status of the order (auto-initialized)

    Methods:
        is_filled(): Check if order is completely executed
        fill(traded_qty): Execute a portion of the order
    """
    order_id: uuid.UUID
    instrument: str
    side: Side
    original_quantity: int
    order_type: OrderType
    agent_id: str
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    oco_group: Optional[str] = None
    explanation: Optional[str] = None
    is_short: bool = False
    is_short_cover: bool = False
    quantity: int = field(init=False)
    status: OrderStatus = field(default=OrderStatus.ACTIVE, init=False)

    def __post_init__(self):
        """Initialize quantity to original_quantity after object creation."""
        self.quantity = self.original_quantity

    def is_filled(self) -> bool:
        """
        Check if the order has been completely filled.

        Returns:
            bool: True if no quantity remains, False otherwise
        """
        return self.quantity == 0

    def fill(self, traded_qty: int):
        """
        Execute a portion of the order and update status accordingly.

        Args:
            traded_qty (int): Quantity being filled in this execution

        Raises:
            ValueError: If traded_qty exceeds remaining quantity
        """
        if traded_qty > self.quantity:
            raise ValueError(f"Cannot fill {traded_qty} when only {self.quantity} remains")

        self.quantity -= traded_qty
        if self.is_filled():
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def __repr__(self) -> str:
        """Return a detailed string representation of the order."""
        return (f"Order(id={self.order_id}, agent={self.agent_id}, side={self.side.value}, "
                f"type={self.order_type.value}, qty={self.quantity}/{self.original_quantity}, "
                f"price={'MKT' if self.price is None else self.price}, status={self.status.value}, "
                f"time={self.timestamp.isoformat() if self.timestamp else 'N/A'})")


class OrderBook:
    """
    A complete order book implementation with order matching engine.

    The OrderBook manages all orders for a single financial instrument, handles
    order matching according to price-time priority, and maintains trade history.
    It supports both limit and market orders with proper price discovery.

    Key Features:
        - Price-time priority matching
        - Support for limit, market, and stop orders
        - Real-time trade execution and recording
        - STOP order triggering based on market price movements
        - Best bid/ask price tracking
        - Volume-weighted average price (VWAP) calculation
        - Comprehensive logging and audit trail

    Attributes:
        instrument (str): The financial instrument symbol
        bids (Dict[float, deque[Order]]): Buy orders organized by price level
        asks (Dict[float, deque[Order]]): Sell orders organized by price level
        bid_levels (List[float]): Max-heap of bid prices (stored as negative values)
        ask_levels (List[float]): Min-heap of ask prices
        orders_by_id (Dict[uuid.UUID, Order]): Order lookup by ID
        stop_orders (Dict[uuid.UUID, Order]): Pending STOP orders awaiting trigger
        trade_history (SortedList): Chronological record of all trades
        trade_seq_counter (int): Sequential trade numbering
        cumulative_traded_volume (int): Total volume traded
        cumulative_turnover (float): Total value traded
        logger: Logging instance for this order book
    """

    def __init__(self, instrument: str):
        """
        Initialize the OrderBook for a specific instrument.

        Args:
            instrument (str): The financial instrument symbol (e.g., "AAPL")
        """
        self.instrument: str = instrument

        # Order storage: price level -> queue of orders at that price
        self.bids: Dict[float, deque[Order]] = defaultdict(deque)
        self.asks: Dict[float, deque[Order]] = defaultdict(deque)

        # Price level heaps for efficient best price lookup
        # Bids use negative prices for max-heap behavior
        self.bid_levels: List[float] = []  # max-heap (store negative prices)
        self.ask_levels: List[float] = []  # min-heap (store prices directly)

        # Order management and trade tracking
        self.orders_by_id: Dict[uuid.UUID, Order] = {}
        self.stop_orders: Dict[uuid.UUID, Order] = {}  # STOP orders waiting to be triggered
        self.trade_history: SortedList = SortedList(key=lambda trade: (trade["timestamp"], trade["seq"]))
        self.trade_seq_counter: int = 0

        # Market statistics
        self.cumulative_traded_volume: int = 0
        self.cumulative_turnover: float = 0.0

        # Logging setup
        log_dir = os.getenv("LOG_DIR", "logs")
        self.logger = setup_logger(f"OrderBook-{instrument}", f"{log_dir}/order_book_{instrument}.log")
        self.logger.info(f"Initialized OrderBook for {self.instrument}")

    def add_order(self, order: Order):
        """
        Add a new order to the book and attempt immediate matching.

        This method handles both limit and market orders, attempting to match
        them against existing orders in the book. For limit orders, any unmatched
        portion remains in the book. Market orders execute against available
        liquidity or are rejected if insufficient liquidity exists.

        Args:
            order (Order): The order to add to the book

        Raises:
            ValueError: If order already exists in the book
        """
        if order.order_id in self.orders_by_id:
            self.logger.error(f"Order {order.order_id} already exists in OrderBook for {self.instrument}")
            raise ValueError(f"Duplicate order ID: {order.order_id}")

        self.orders_by_id[order.order_id] = order
        self.logger.info(f"Adding order to OrderBook: {order}")

        if order.order_type == OrderType.STOP:
            # STOP orders are held separately until triggered
            self.stop_orders[order.order_id] = order
            self.logger.info(f"Added STOP order to pending queue: {order}")
            # Check if any stop orders should be triggered by current market prices
            self._check_stop_triggers()
        elif order.side == Side.BUY:
            if order.order_type == OrderType.MARKET:
                self.logger.info(f"Processing MARKET BUY order: {order}")
                self._match_market_order(order, opposite_side=Side.SELL)
            else:  # LIMIT order
                self.logger.info(f"Adding LIMIT BUY order: {order}")
                self._add_limit_order(order, side='bids', heap_ref=self.bid_levels, is_bid=True)
                self._match_limit_order(aggressor_side=Side.BUY)
                # Check stop orders after any price changes
                self._check_stop_triggers()
        else:  # Side.SELL
            if order.order_type == OrderType.MARKET:
                self.logger.info(f"Processing MARKET SELL order: {order}")
                self._match_market_order(order, opposite_side=Side.BUY)
            else:  # LIMIT order
                self.logger.info(f"Adding LIMIT SELL order: {order}")
                self._add_limit_order(order, side='asks', heap_ref=self.ask_levels, is_bid=False)
                self._match_limit_order(aggressor_side=Side.SELL)
                # Check stop orders after any price changes
                self._check_stop_triggers()

    def _check_stop_triggers(self):
        """
        Check if any pending STOP orders should be triggered by current market prices.
        
        STOP orders are triggered when:
        - STOP BUY: market price >= stop price (breakout above resistance)
        - STOP SELL: market price <= stop price (breakdown below support)
        
        When triggered, STOP orders become MARKET orders and execute immediately.
        """
        if not self.stop_orders:
            return
            
        # Get current market prices
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()
        
        # Use last trade price if available, otherwise use mid-price
        last_trade_price = None
        if self.trade_history:
            last_trade_price = self.trade_history[-1]["price"]
        
        # Determine current market price for trigger evaluation
        if last_trade_price is not None:
            current_price = last_trade_price
        elif best_bid is not None and best_ask is not None:
            current_price = (best_bid + best_ask) / 2
        elif best_bid is not None:
            current_price = best_bid
        elif best_ask is not None:
            current_price = best_ask
        else:
            # No market data available
            return
            
        triggered_orders = []
        
        for stop_order_id, stop_order in list(self.stop_orders.items()):

            if stop_order.side == Side.BUY:
                # STOP BUY: triggers when market price >= stop price
                should_trigger = current_price >= stop_order.price
            else:  # Side.SELL
                # STOP SELL: triggers when market price <= stop price
                should_trigger = current_price <= stop_order.price
                
            if should_trigger:
                triggered_orders.append(stop_order)
                del self.stop_orders[stop_order_id]
                self.logger.info(f"STOP order {stop_order_id} triggered at market price {current_price:.2f}, "
                               f"stop price was {stop_order.price:.2f}")
        
        # Convert triggered STOP orders to MARKET orders and execute them
        for stop_order in triggered_orders:
            # Convert to market order
            stop_order.order_type = OrderType.MARKET
            stop_order.price = None  # Market orders don't have a specific price
            
            # Execute as market order
            if stop_order.side == Side.BUY:
                self.logger.info(f"Executing triggered STOP BUY as MARKET order: {stop_order}")
                self._match_market_order(stop_order, opposite_side=Side.SELL)
            else:  # Side.SELL
                self.logger.info(f"Executing triggered STOP SELL as MARKET order: {stop_order}")
                self._match_market_order(stop_order, opposite_side=Side.BUY)
            
            # Clean up triggered STOP order from orders_by_id if fully filled
            if stop_order.is_filled():
                if stop_order.order_id in self.orders_by_id:
                    del self.orders_by_id[stop_order.order_id]
                    self.logger.info(f"Cleaned up filled triggered STOP order {stop_order.order_id}")
        
        # Check for additional stop triggers after executing triggered orders
        # This handles cascading stop triggers caused by the price movements
        if triggered_orders:
            self._check_stop_triggers()

    def cancel_order(self, order_id: uuid.UUID) -> bool:
        """
        Cancel an existing order in the order book.

        Removes the order from the appropriate price level and updates its status.
        If the price level becomes empty after removal, it is cleaned up from
        the heap structures.

        Args:
            order_id (uuid.UUID): Unique identifier of the order to cancel

        Returns:
            bool: True if cancellation was successful, False otherwise
        """
        order = self.orders_by_id.get(order_id)
        if not order:
            self.logger.warning(f"Attempted to cancel non-existent order ID: {order_id}")
            return False

        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED):
            self.logger.warning(f"Cannot cancel order {order_id} - status is {order.status.value}")
            return False

        # Check if this is a STOP order
        if order_id in self.stop_orders:
            del self.stop_orders[order_id]
            order.status = OrderStatus.CANCELED
            del self.orders_by_id[order_id]
            self.logger.info(f"STOP order canceled: {order}")
            return True

        # Remove from appropriate book side
        book_side = self.bids if order.side == Side.BUY else self.asks
        if order.price in book_side:
            orders_at_price = book_side[order.price]

            # Find and remove the specific order
            for i, o in enumerate(orders_at_price):
                if o.order_id == order_id:
                    del orders_at_price[i]
                    order.status = OrderStatus.CANCELED
                    del self.orders_by_id[order_id]
                    self.logger.info(f"Order {order_id} removed from "
                                   f"{'bids' if order.side == Side.BUY else 'asks'} at price {order.price}")

                    # Clean up empty price level
                    if not orders_at_price:
                        del book_side[order.price]
                        self._remove_price_level(order.side)
                    
                    self.logger.info(f"Order canceled: {order}")
                    return True
            else:
                self.logger.error(f"Order {order_id} not found in orders queue at price {order.price}")
                return False
        else:
            self.logger.error(f"Price level {order.price} not found for order {order_id}")
            return False


    def _remove_price_level(self, side: Side):
        """
        Clean up empty price levels from the heap structures.

        This method removes price levels that no longer have any orders,
        maintaining heap integrity for efficient price lookups.

        Args:
            side (Side): Which side's price levels to clean up
        """
        if side == Side.BUY:
            # Clean up bid levels (stored as negative values)
            while self.bid_levels and (-self.bid_levels[0] not in self.bids):
                removed_price = -heapq.heappop(self.bid_levels)
                self.logger.debug(f"Removed empty bid price level: {removed_price}")
        else:
            # Clean up ask levels
            while self.ask_levels and (self.ask_levels[0] not in self.asks):
                removed_price = heapq.heappop(self.ask_levels)
                self.logger.debug(f"Removed empty ask price level: {removed_price}")

    def _add_limit_order(self, order: Order, side: str, heap_ref: List[float], is_bid: bool):
        """
        Add a limit order to the specified side of the book.

        Creates a new price level if necessary and adds the order to the
        appropriate queue with price-time priority.

        Args:
            order (Order): The limit order to add
            side (str): 'bids' or 'asks'
            heap_ref (List[float]): Reference to the appropriate heap
            is_bid (bool): True for bid side, False for ask side
        """
        price = order.price
        book_side = getattr(self, side)

        # Create new price level if needed
        if price not in book_side:
            book_side[price] = deque()
            if is_bid:
                heapq.heappush(heap_ref, -price)  # Negative for max-heap
            else:
                heapq.heappush(heap_ref, price)   # Positive for min-heap
            self.logger.debug(f"Created new price level {price} on {side}")

        # Add order to price level queue (FIFO for time priority)
        book_side[price].append(order)
        self.logger.debug(f"Added order {order.order_id} to {side} at price {price}")

    def _match_market_order(self, market_order: Order, opposite_side: Side):
        """
        Match a market order against the opposite side of the book.

        Executes the market order against available liquidity at the best
        prices until the order is filled or no more liquidity exists.

        Args:
            market_order (Order): The market order to execute
            opposite_side (Side): The side to match against (BUY or SELL)
        """
        if opposite_side == Side.SELL:
            # Market buy order matches against asks
            heap = self.ask_levels
            book_side = self.asks
            get_best_price = lambda: heap[0] if heap else None
            buyer_is_market = True
        else:
            # Market sell order matches against bids
            heap = self.bid_levels
            book_side = self.bids
            get_best_price = lambda: -heap[0] if heap else None
            buyer_is_market = False

        # Execute against available liquidity
        while market_order.quantity > 0 and heap:
            best_price = get_best_price()
            if best_price is None:
                break

            orders_at_best = book_side[best_price]

            # Match against all orders at this price level
            while market_order.quantity > 0 and orders_at_best:
                resting_order = orders_at_best[0]
                traded_qty = min(market_order.quantity, resting_order.quantity)

                # Fill both orders
                resting_order.fill(traded_qty)
                market_order.fill(traded_qty)

                # Determine trade participants
                if buyer_is_market:
                    buy_agent_id, sell_agent_id = market_order.agent_id, resting_order.agent_id
                    buy_order_id, sell_order_id = market_order.order_id, resting_order.order_id
                    buy_status, sell_status = market_order.status, resting_order.status
                else:
                    buy_agent_id, sell_agent_id = resting_order.agent_id, market_order.agent_id
                    buy_order_id, sell_order_id = resting_order.order_id, market_order.order_id
                    buy_status, sell_status = resting_order.status, market_order.status

                # Record the trade
                self.record_trade(
                    buy_agent_id=buy_agent_id,
                    sell_agent_id=sell_agent_id,
                    price=best_price,
                    quantity=traded_qty,
                    timestamp=market_order.timestamp,
                    buy_order_id=buy_order_id,
                    sell_order_id=sell_order_id,
                    buy_order_status=buy_status,
                    sell_order_status=sell_status
                )

                # Remove filled resting order
                if resting_order.is_filled():
                    orders_at_best.popleft()
                    if resting_order.order_id in self.orders_by_id:
                        del self.orders_by_id[resting_order.order_id]
                    self.logger.info(f"Order {resting_order.order_id} fully filled and removed")

            # Clean up empty price level
            if not orders_at_best:
                del book_side[best_price]
                heapq.heappop(heap)
                self.logger.debug(f"Removed empty price level: {best_price}")

        # Log any unfilled market order quantity
        if market_order.quantity > 0:
            self.logger.warning(f"Market order {market_order.order_id} partially unfilled: "
                              f"{market_order.quantity} shares remaining")

    def _match_limit_order(self, aggressor_side: Side):
        """
        Check for crossed market conditions and execute matching trades.

        After adding a limit order, this method checks if the best bid price
        is greater than or equal to the best ask price. If so, it executes
        trades until the cross is resolved or one side is exhausted.

        Args:
            aggressor_side (Side): The side that triggered this matching attempt
        """
        while self.bid_levels and self.ask_levels:
            best_bid = -self.bid_levels[0]
            best_ask = self.ask_levels[0]

            # Check for crossed market
            if best_bid >= best_ask:
                bid_orders = self.bids[best_bid]
                ask_orders = self.asks[best_ask]

                # Execute trades at the crossed prices
                while bid_orders and ask_orders and best_bid >= best_ask:
                    bid_order = bid_orders[0]
                    ask_order = ask_orders[0]
                    traded_qty = min(bid_order.quantity, ask_order.quantity)

                    # Price determination: aggressor gets worse price
                    execution_price = best_ask if aggressor_side == Side.BUY else best_bid
                    trade_timestamp = (bid_order.timestamp if aggressor_side == Side.BUY
                                     else ask_order.timestamp)

                    # Fill orders
                    bid_order.fill(traded_qty)
                    ask_order.fill(traded_qty)

                    # Record trade
                    self.record_trade(
                        buy_agent_id=bid_order.agent_id,
                        sell_agent_id=ask_order.agent_id,
                        price=execution_price,
                        quantity=traded_qty,
                        timestamp=trade_timestamp,
                        buy_order_id=bid_order.order_id,
                        sell_order_id=ask_order.order_id,
                        buy_order_status=bid_order.status,
                        sell_order_status=ask_order.status
                    )

                    # Remove filled orders
                    if bid_order.is_filled():
                        bid_orders.popleft()
                        if bid_order.order_id in self.orders_by_id:
                            del self.orders_by_id[bid_order.order_id]
                        self.logger.info(f"Bid order {bid_order.order_id} fully filled")
                    if ask_order.is_filled():
                        ask_orders.popleft()
                        if ask_order.order_id in self.orders_by_id:
                            del self.orders_by_id[ask_order.order_id]
                        self.logger.info(f"Ask order {ask_order.order_id} fully filled")

                # Clean up empty price levels
                if not bid_orders:
                    del self.bids[best_bid]
                    heapq.heappop(self.bid_levels)
                    self.logger.debug(f"Removed empty bid price level: {best_bid}")
                if not ask_orders:
                    del self.asks[best_ask]
                    heapq.heappop(self.ask_levels)
                    self.logger.debug(f"Removed empty ask price level: {best_ask}")
            else:
                # No more crosses possible
                break

    def record_trade(
        self,
        buy_agent_id: str,
        sell_agent_id: str,
        price: float,
        quantity: int,
        timestamp: datetime,
        buy_order_id: Optional[uuid.UUID] = None,
        sell_order_id: Optional[uuid.UUID] = None,
        buy_order_status: Optional[OrderStatus] = None,
        sell_order_status: Optional[OrderStatus] = None
    ):
        """
        Record a completed trade and update market statistics.

        This method creates a permanent record of the trade with all relevant
        details and updates cumulative volume and turnover statistics.

        Args:
            buy_agent_id (str): ID of the buying agent
            sell_agent_id (str): ID of the selling agent
            price (float): Execution price
            quantity (int): Quantity traded
            timestamp (datetime): Trade execution time
            buy_order_id (Optional[uuid.UUID]): Buyer's order ID
            sell_order_id (Optional[uuid.UUID]): Seller's order ID
            buy_order_status (Optional[OrderStatus]): Buyer's order status after trade
            sell_order_status (Optional[OrderStatus]): Seller's order status after trade
        """
        trade = {
            "instrument": self.instrument,
            "buy_agent": buy_agent_id,
            "sell_agent": sell_agent_id,
            "price": price,
            "quantity": quantity,
            "timestamp": timestamp,
            "buy_order_id": buy_order_id,
            "sell_order_id": sell_order_id,
            "buy_order_status": buy_order_status.value if buy_order_status else None,
            "sell_order_status": sell_order_status.value if sell_order_status else None,
            "seq": self.trade_seq_counter
        }

        self.trade_seq_counter += 1
        self.trade_history.add(trade)
        self.cumulative_traded_volume += quantity
        self.cumulative_turnover += price * quantity

        self.logger.info(f"Trade executed: {buy_agent_id} bought {quantity} shares from "
                        f"{sell_agent_id} at ${price:.2f} (Total: ${price * quantity:.2f})")

    def get_best_bid(self) -> Tuple[Optional[float], int]:
        """
        Get the highest bid price and total quantity available at that price.

        Returns:
            Tuple[Optional[float], int]: (best_bid_price, total_quantity)
                                       or (None, 0) if no bids exist
        """
        if not self.bid_levels:
            return None, 0

        best_price = -self.bid_levels[0]
        orders = self.bids[best_price]
        total_qty = sum(order.quantity for order in orders)

        self.logger.debug(f"Best bid for {self.instrument}: ${best_price:.2f} ({total_qty} shares)")
        return best_price, total_qty

    def get_best_ask(self) -> Tuple[Optional[float], int]:
        """
        Get the lowest ask price and total quantity available at that price.

        Returns:
            Tuple[Optional[float], int]: (best_ask_price, total_quantity)
                                       or (None, 0) if no asks exist
        """
        if not self.ask_levels:
            return None, 0

        best_price = self.ask_levels[0]
        orders = self.asks[best_price]
        total_qty = sum(order.quantity for order in orders)

        self.logger.debug(f"Best ask for {self.instrument}: ${best_price:.2f} ({total_qty} shares)")
        return best_price, total_qty

    def get_spread(self) -> Optional[float]:
        """
        Calculate the bid-ask spread.

        Returns:
            Optional[float]: The spread (ask - bid) or None if no market exists
        """
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()

        if best_bid is None or best_ask is None:
            return None

        spread = best_ask - best_bid
        self.logger.debug(f"Spread for {self.instrument}: ${spread:.2f}")
        return spread

    def get_mid_price(self) -> Optional[float]:
        """
        Calculate the mid-market price.

        Returns:
            Optional[float]: The mid price ((bid + ask) / 2) or None if no market exists
        """
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()

        if best_bid is None or best_ask is None:
            return None

        mid_price = (best_bid + best_ask) / 2
        self.logger.debug(f"Mid price for {self.instrument}: ${mid_price:.2f}")
        return mid_price

    def get_vwap(self) -> Optional[float]:
        """
        Calculate the Volume-Weighted Average Price (VWAP) of all trades.

        VWAP is calculated as total turnover divided by total volume,
        providing the average price weighted by trade size.

        Returns:
            Optional[float]: VWAP rounded to 2 decimal places, or None if no trades
        """
        if self.cumulative_traded_volume == 0:
            return None

        vwap = round(self.cumulative_turnover / self.cumulative_traded_volume, 2)
        self.logger.debug(f"VWAP for {self.instrument}: ${vwap:.2f}")
        return vwap

    def get_market_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, int]]]:
        """
        Get market depth showing multiple price levels.

        Args:
            levels (int): Number of price levels to return on each side

        Returns:
            Dict[str, List[Tuple[float, int]]]: Dictionary with 'bids' and 'asks' keys,
                                              each containing list of (price, quantity) tuples
        """
        depth = {"bids": [], "asks": []}

        # Get bid levels (highest first)
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        for price in bid_prices:
            total_qty = sum(order.quantity for order in self.bids[price])
            depth["bids"].append((price, total_qty))

        # Get ask levels (lowest first)
        ask_prices = sorted(self.asks.keys())[:levels]
        for price in ask_prices:
            total_qty = sum(order.quantity for order in self.asks[price])
            depth["asks"].append((price, total_qty))

        return depth

    def get_order_count(self) -> Dict[str, int]:
        """
        Get count of orders by status.

        Returns:
            Dict[str, int]: Count of orders for each status
        """
        status_counts = defaultdict(int)
        for order in self.orders_by_id.values():
            status_counts[order.status.value] += 1
        return dict(status_counts)

    def get_pending_stop_orders(self) -> List[Order]:
        """
        Get all pending STOP orders that haven't been triggered yet.
        
        Returns:
            List[Order]: List of pending STOP orders
        """
        return list(self.stop_orders.values())

    def __repr__(self) -> str:
        """Return a summary representation of the order book."""
        best_bid, bid_qty = self.get_best_bid()
        best_ask, ask_qty = self.get_best_ask()

        return (f"OrderBook({self.instrument}: "
                f"Bid={best_bid}@{bid_qty}, Ask={best_ask}@{ask_qty}, "
                f"Volume={self.cumulative_traded_volume}, "
                f"VWAP={self.get_vwap()}, "
                f"PendingStops={len(self.stop_orders)})")
