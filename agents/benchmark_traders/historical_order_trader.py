"""
Historical Order Trader - replays pre-built historical orders at their original timestamps.
Used for backtesting and establishing baseline performance comparisons.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.benchmark_traders.trader import TraderAgent
from utils.time_utils import parse_datetime_utc


class HistoricalOrderTrader(TraderAgent):
    """
    Historical order replay trader - executes predefined orders at their original timestamps.
    """

    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        orders_file_path: Optional[str] = None,
        orders: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the Historical Order Trader.
        
        Args:
            instrument_exchange_map: Mapping of instrument symbols to exchange agent IDs
            orders_file_path: Path to JSON file containing historical orders data
            orders: Historical orders data keyed by instrument symbol (alternative to file path)
            agent_id: Unique identifier for this trader instance. If None, auto-generated.
            **kwargs: Additional arguments passed to parent TraderAgent class
            
        Raises:
            ValueError: If neither orders_file_path nor orders is provided, or if data is malformed
            FileNotFoundError: If orders_file_path points to non-existent file
            KeyError: If required order fields are missing
        """
        # Extract parameters that the parent TraderAgent expects
        trader_kwargs = {}
        for param in ['initial_cash', 'initial_positions', 'initial_cost_basis', 'action_interval_seconds', 'rabbitmq_host']:
            if param in kwargs:
                trader_kwargs[param] = kwargs[param]
        
        super().__init__(instrument_exchange_map=instrument_exchange_map, agent_id=agent_id, **trader_kwargs)
        
        # Load orders from file path or use provided orders
        if orders_file_path:
            self.orders = self._load_orders_from_file(orders_file_path)
            self.orders_source = f"file: {orders_file_path}"
        elif orders:
            self.orders = orders
            self.orders_source = "provided dictionary"
        else:
            raise ValueError("Either orders_file_path or orders must be provided")
        
        # Validate orders data structure
        if not self.orders:
            raise ValueError("Orders dictionary cannot be empty")
            
        self.executed_count: int = 0
        self._historical_queue: List[Dict[str, Any]] = []

        # Prepare and sort all orders by timestamp for efficient processing
        self._prepare_execution_queue()
        
        # Validate that we have orders for our configured instruments
        configured_instruments = set(self.instrument_exchange_map.keys())
        order_instruments = set(self.orders.keys())
        
        if not configured_instruments.intersection(order_instruments):
            self.logger.warning(
                f"No historical orders found for configured instruments {configured_instruments}. "
                f"Available instruments in orders: {order_instruments}"
            )
        
        # Log initialization summary
        total_orders = sum(len(order_list) for order_list in self.orders.values())
        instruments = list(self.instrument_exchange_map.keys())
        
        self.logger.info(
            f"HistoricalOrderTrader {self.agent_id} initialized for instruments {instruments} "
            f"with {total_orders} total historical orders from {self.orders_source}. "
            f"Prepared {len(self.pending_orders)} orders for execution."
        )

    def _load_orders_from_file(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load historical orders from a JSON file.
        
        Args:
            file_path: Path to JSON file containing orders data
            
        Returns:
            Dictionary mapping instrument symbols to lists of orders
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Orders file not found: {file_path}")
            
        try:
            with open(file_path, 'r') as f:
                orders_data = json.load(f)
            
            self.logger.info(f"Loaded orders from {file_path}")
            return orders_data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in orders file {file_path}: {e}")

    def _prepare_execution_queue(self) -> None:
        """
        Prepare a sorted queue of all orders for efficient execution during time ticks.
        
        Flattens the orders dictionary into a single sorted list and validates each order.
        """
        all_orders = []
        
        for instrument, order_list in self.orders.items():
            # Skip instruments not in our configuration
            if instrument not in self.instrument_exchange_map:
                self.logger.debug(
                    f"Skipping {len(order_list)} orders for unconfigured instrument {instrument}"
                )
                continue
                
            for order_index, order in enumerate(order_list):
                try:
                    # Validate and parse order timestamp
                    timestamp_str = order.get("timestamp")
                    if not timestamp_str:
                        raise ValueError("Missing timestamp field")
                        
                    order_time = parse_datetime_utc(timestamp_str)
                    
                    # Validate required order fields
                    self._validate_order_fields(order)
                    
                    # Add to execution queue with parsed timestamp
                    order_with_metadata = {
                        **order,
                        "instrument": instrument,
                        "parsed_timestamp": order_time,
                        "original_index": order_index
                    }
                    all_orders.append(order_with_metadata)
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to prepare order {order_index} for {instrument}: {e}. "
                        f"Order data: {order}. Skipping."
                    )
                    continue
        
        # Sort all orders by timestamp for efficient execution
        all_orders.sort(key=lambda x: x["parsed_timestamp"])
        self._historical_queue = all_orders
        
        self.logger.info(
            f"Prepared {len(self.pending_orders)} valid orders for execution "
            f"from {sum(len(order_list) for order_list in self.orders.values())} total orders"
        )

    async def initialize(self) -> None:
        """
        Initialize the trader.
        
        This method is called by the simulation framework after agent creation.
        The orders are already prepared during __init__.
        """
        await super().initialize()
        
        self.logger.info(
            f"HistoricalOrderTrader {self.agent_id} initialization complete. "
            f"Ready to execute {len(self.pending_orders)} orders."
        )

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        """
        Handle time tick events and execute orders that are due at the current time.
        
        Checks the pending orders queue for orders that should be executed at the
        current simulation time and executes them in chronological order.
        
        Args:
            payload: Time tick information from simulation clock
        """
        await super().handle_time_tick(payload)
        
        current_time = self.current_time
        if not current_time or not self._historical_queue:
            return
            
        # Execute all orders that are due at the current time
        orders_to_execute = []
        remaining_orders = []
        
        for order in self._historical_queue:
            order_time = order["parsed_timestamp"]
            
            # Check if this order should be executed now (with small tolerance for timing)
            if order_time <= current_time:
                orders_to_execute.append(order)
            else:
                remaining_orders.append(order)
        
        # Update pending orders list
        self._historical_queue = remaining_orders
        
        # Execute the due orders
        for order in orders_to_execute:
            await self._execute_historical_order(order)

    def _validate_order_fields(self, order: Dict[str, Any]) -> None:
        """
        Validate that an order contains all required fields with valid values.
        
        Args:
            order: Historical order dictionary to validate
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = ["side", "price", "quantity", "order_type"]
        
        for field in required_fields:
            if field not in order:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate side
        side = order.get("side", "").upper()
        if side not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side '{side}'. Must be 'BUY' or 'SELL'")
        
        # Validate price
        price = order.get("price")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError(f"Invalid price '{price}'. Must be a positive number")
        
        # Validate quantity
        quantity = order.get("quantity")
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            raise ValueError(f"Invalid quantity '{quantity}'. Must be a positive number")
        
        # Validate order type
        order_type = order.get("order_type", "").upper()
        valid_types = ["LIMIT", "MARKET", "STOP"]
        if order_type not in valid_types:
            raise ValueError(f"Invalid order_type '{order_type}'. Must be one of {valid_types}")

    async def _execute_historical_order(self, order: Dict[str, Any]) -> None:
        """
        Execute a single historical order.
        
        Args:
            order: Historical order data dictionary containing execution parameters
                   (includes instrument, side, price, quantity, order_type, etc.)
        """
        instrument = order.get("instrument")
        side = order.get("side", "").upper()
        quantity = order.get("quantity")
        price = order.get("price")
        order_type = order.get("order_type", "LIMIT").upper()
        
        try:
            # Submit order through parent TraderAgent's order placement API
            submitted_order_id = await self.place_order(
                instrument=instrument,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                explanation=f"Historical order replay from {self.orders_source}"
            )
            
            if submitted_order_id:
                self.executed_count += 1
                self.logger.info(
                    f"HistoricalOrderTrader {self.agent_id}: Executed {side} order "
                    f"for {instrument} at {price} with quantity {quantity} "
                    f"(order_id: {submitted_order_id}, #{self.executed_count})"
                )
            else:
                self.logger.error(
                    f"HistoricalOrderTrader {self.agent_id}: Failed to submit {side} order "
                    f"for {instrument} at {price} with quantity {quantity}"
                )
                
        except Exception as e:
            self.logger.error(
                f"HistoricalOrderTrader {self.agent_id}: Exception executing {side} order "
                f"for {instrument}: {e}"
            )

    async def on_trade_execution(self, msg: Dict[str, Any]) -> None:
        """
        Handle trade execution notifications from the exchange.
        
        Historical order traders typically don't need special trade execution logic
        beyond logging, but this can be overridden for custom behavior.
        
        Args:
            msg: Trade execution message from exchange
        """
        await super().on_trade_execution(msg)
        
        # Log trade execution for monitoring
        instrument = msg.get("instrument", "UNKNOWN")
        price = msg.get("price", "UNKNOWN")
        quantity = msg.get("quantity", "UNKNOWN")
        side = msg.get("role", "UNKNOWN")
        
        self.logger.debug(
            f"HistoricalOrderTrader {self.agent_id}: Trade executed for {instrument} "
            f"- {side} {quantity} at {price}"
        )

    async def on_market_data_update(self, instrument: str, snapshot: Dict[str, Any]) -> None:
        """
        Handle market data updates (no action taken for historical replay strategy).
        
        Historical order replay strategy ignores market data updates and executes
        orders purely based on their original timestamps.
        
        Args:
            instrument: Financial instrument symbol
            snapshot: Market data snapshot (ignored by this strategy)
        """
        # Historical replay strategy doesn't react to market data
        pass

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about prepared and executed orders.
        
        Returns:
            Dictionary containing execution statistics
        """
        total_orders = sum(len(order_list) for order_list in self.orders.values())
        
        return {
            "agent_id": self.agent_id,
            "total_historical_orders": total_orders,
            "prepared_orders": len(self.pending_orders) + self.executed_count,
            "executed_orders": self.executed_count,
            "pending_orders": len(self.pending_orders),
            "instruments": list(self.orders.keys()),
            "configured_instruments": list(self.instrument_exchange_map.keys()),
            "orders_source": self.orders_source
        }

    def __repr__(self) -> str:
        """Return string representation of the trader."""
        total_orders = sum(len(order_list) for order_list in self.orders.values())
        instruments = list(self.instrument_exchange_map.keys())
        
        return (
            f"HistoricalOrderTrader(agent_id='{self.agent_id}', "
            f"instruments={instruments}, total_orders={total_orders}, "
            f"executed={self.executed_count}, pending={len(self.pending_orders)})"
        )