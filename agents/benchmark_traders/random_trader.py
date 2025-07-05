"""
Random Trader - makes purely random buy/sell decisions for baseline comparison.
Provides statistical control group for evaluating strategy performance.
"""

import random
from typing import Dict, Any, Optional

from utils.messages import MessageType
from utils.orders import Side, OrderType
from agents.benchmark_traders.trader import TraderAgent


class RandomTrader(TraderAgent):
    """
    Random trading strategy - makes buy/sell decisions based on random probability.
    Serves as statistical baseline for performance comparison.
    """

    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        trade_probability: float = 0.1,
        position_size_pct: float = 0.05,
        random_seed: Optional[int] = None,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = 'localhost',
        **kwargs
    ) -> None:
        """
        Initialize the Random Trader.

        Args:
            instrument_exchange_map: Mapping of instrument symbols to exchange agent IDs
            trade_probability: Probability of making a trade at each decision point (0.0 to 1.0)
            position_size_pct: Percentage of portfolio to trade per signal (0.0 to 1.0)
            random_seed: Seed for random number generator (for reproducible results)
            agent_id: Unique identifier for this trader instance
            rabbitmq_host: RabbitMQ server hostname for message coordination
            **kwargs: Additional arguments passed to parent TraderAgent class
        """
        # Extract parameters that the parent TraderAgent expects
        trader_kwargs = {}
        for param in ['initial_cash', 'initial_positions', 'initial_cost_basis', 'action_interval_seconds']:
            if param in kwargs:
                trader_kwargs[param] = kwargs[param]
        
        super().__init__(
            instrument_exchange_map=instrument_exchange_map,
            agent_id=agent_id,
            rabbitmq_host=rabbitmq_host,
            **trader_kwargs
        )
        
        self.trade_probability = max(0.0, min(1.0, trade_probability))  # Clamp to [0, 1]
        self.position_size_pct = position_size_pct
        self.random_seed = random_seed
        
        # Initialize random number generator
        if random_seed is not None:
            random.seed(random_seed)
            self.logger.info(f"RandomTrader {self.agent_id} initialized with random seed: {random_seed}")
        
        # Track pending orders for resubmission
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        
        # Decision counters for statistics
        self.decision_counts = {
            "buy": 0,
            "sell": 0, 
            "hold": 0,
            "total": 0
        }

        self.first_request_sent: Dict[str, bool] = {}


        self.logger.info(
            f"RandomTrader {self.agent_id} initialized with "
            f"trade_probability={self.trade_probability:.1%}, "
            f"position_size_pct={self.position_size_pct:.1%} for instruments: "
            f"{list(self.instrument_exchange_map.keys())}"
        )

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        """
        Handle time tick events and coordinate random trading activities.
        
        Processes pending order resubmissions and makes random trading decisions
        for all configured instruments at regular intervals.
        
        Args:
            payload: Time tick information from simulation clock
        """
        await super().handle_time_tick(payload)
        
        current_time = self.current_time
        if self.next_action_time is None:
            self.next_action_time = current_time

        if current_time >= self.next_action_time:
            # Handle pending orders first
            await self._handle_pending_orders()

            # Make random trading decisions for all instruments
            for instrument in self.instrument_exchange_map.keys():
                await self._make_random_decision(instrument)
            
            self.next_action_time = current_time + self.action_interval

    async def _handle_pending_orders(self) -> None:
        """
        Handle resubmission of pending orders that failed initial placement.
        
        Attempts to resubmit orders that were not successfully placed in previous
        attempts, ensuring robust order execution in the face of temporary failures.
        """
        for order_id, order_info in list(self.pending_orders.items()):
            try:
                new_order_id = await self.place_order(
                    instrument=order_info['instrument'],
                    side=order_info['side'],
                    quantity=order_info['quantity'],
                    order_type=order_info['order_type'],
                    explanation=f"Resubmitting: {order_info['explanation']}"
                )

                if new_order_id:
                    # Update the pending orders with the new order ID
                    del self.pending_orders[order_id]
                    self.pending_orders[new_order_id] = order_info
                    self.logger.info(
                        f"Successfully resubmitted order {order_id} -> {new_order_id} for {order_info['instrument']} "
                        f"({order_info['side']} {order_info['quantity']})"
                    )
                else:
                    # Remove failed order from pending to avoid infinite resubmission
                    del self.pending_orders[order_id]
                    self.logger.warning(
                        f"Failed to resubmit order {order_id} for {order_info['instrument']} - removing from pending"
                    )
                    
            except Exception as e:
                # Remove order that caused exception to avoid infinite resubmission
                del self.pending_orders[order_id]
                self.logger.error(
                    f"Exception resubmitting order {order_id}: {e} - removing from pending"
                )

    async def _make_random_decision(self, instrument: str) -> None:
        """
        Make a random trading decision for the specified instrument.
        
        Generates random buy/sell/hold decisions based on trade_probability
        and current position status.
        
        Args:
            instrument: Financial instrument symbol to make decision for
        """
        self.decision_counts["total"] += 1
        
        # Get current position
        current_position = self.long_qty.get(instrument, 0)
        
        # Calculate position size for potential trades
        portfolio_value = self.portfolio_value
        if portfolio_value <= 0:
            self.logger.warning(f"Invalid portfolio value: {portfolio_value}")
            self.decision_counts["hold"] += 1
            return
            
        position_value = portfolio_value * self.position_size_pct
        
        # Generate random decision
        random_value = random.random()
        
        # Decision logic based on trade probability
        buy_threshold = self.trade_probability / 2
        sell_threshold = 1.0 - (self.trade_probability / 2)
        
        if random_value < buy_threshold and current_position <= 0:
            # Random BUY decision (only if not already long)
            await self._execute_random_buy(instrument, position_value)
            self.decision_counts["buy"] += 1
            
        elif random_value > sell_threshold and current_position > 0:
            # Random SELL decision (only if currently long)
            await self._execute_random_sell(instrument, current_position)
            self.decision_counts["sell"] += 1
            
        else:
            # HOLD decision (no action)
            self.decision_counts["hold"] += 1
            self.logger.debug(
                f"RandomTrader {self.agent_id} HOLD decision for {instrument} "
                f"(random: {random_value:.3f}, position: {current_position})"
            )

    async def _execute_random_buy(self, instrument: str, position_value: float) -> None:
        """
        Execute a random buy decision.
        
        Args:
            instrument: Financial instrument symbol
            position_value: Dollar value of position to establish
        """
        # Request current market data to get price for quantity calculation
        await self._request_market_snapshot(instrument)
        
        # Use a reasonable default price estimate if we don't have current data
        # This will be corrected when the market order executes at actual market price
        estimated_price = 100.0  # Default estimation
        quantity = max(1, int(position_value / estimated_price))
        
        explanation = f"Random BUY decision (prob: {self.trade_probability:.1%})"
        
        try:
            order_id = await self.place_order(
                instrument=instrument,
                side=Side.BUY.value,
                quantity=quantity,
                order_type=OrderType.MARKET.value,
                explanation=explanation
            )
            
            if order_id:
                self.pending_orders[order_id] = {
                    'instrument': instrument,
                    'side': Side.BUY.value,
                    'quantity': quantity,
                    'order_type': OrderType.MARKET.value,
                    'explanation': explanation
                }
                
                self.logger.info(
                    f"RandomTrader {self.agent_id} placed random BUY order {order_id} "
                    f"for {instrument}: {quantity} shares (estimated value: ${position_value:.2f})"
                )
            else:
                self.logger.error(f"Failed to place random BUY order for {instrument}")
                
        except Exception as e:
            self.logger.error(f"Exception placing random BUY order for {instrument}: {e}")

    async def _execute_random_sell(self, instrument: str, quantity: int) -> None:
        """
        Execute a random sell decision.
        
        Args:
            instrument: Financial instrument symbol
            quantity: Number of shares to sell (entire position)
        """
        explanation = f"Random SELL decision (prob: {self.trade_probability:.1%})"
        
        try:
            order_id = await self.place_order(
                instrument=instrument,
                side=Side.SELL.value,
                quantity=quantity,
                order_type=OrderType.MARKET.value,
                explanation=explanation
            )
            
            if order_id:
                self.pending_orders[order_id] = {
                    'instrument': instrument,
                    'side': Side.SELL.value,
                    'quantity': quantity,
                    'order_type': OrderType.MARKET.value,
                    'explanation': explanation
                }
                
                self.logger.info(
                    f"RandomTrader {self.agent_id} placed random SELL order {order_id} "
                    f"for {instrument}: {quantity} shares"
                )
            else:
                self.logger.error(f"Failed to place random SELL order for {instrument}")
                
        except Exception as e:
            self.logger.error(f"Exception placing random SELL order for {instrument}: {e}")

    async def _request_market_snapshot(self, instrument: str) -> None:
        """
        Request market data snapshot for the specified instrument.
        
        Args:
            instrument: Financial instrument symbol to request data for
        """
        if not self.current_time:
            return

        window_end = self.current_time
        window_start = window_end - self.action_interval

        exchange_id = self._get_exchange_for_instrument(instrument)
        if not exchange_id:
            self.logger.error(f"No exchange found for instrument '{instrument}'")
            return

        payload = {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat()
        }

        success = await self.send_message(exchange_id, MessageType.MARKET_DATA_SNAPSHOT_REQUEST, payload)
        if success:
            self.logger.debug(f"Requested market snapshot for {instrument}")
        else:
            self.logger.error(f"Failed to request market snapshot for {instrument}")

    async def on_market_data_update(self, instrument: str, snapshot: Dict[str, Any]) -> None:
        """
        Handle market data updates (no analysis performed for random strategy).
        
        Random trading strategy ignores market data and makes decisions based
        purely on random number generation.
        
        Args:
            instrument: Financial instrument symbol
            snapshot: Market data snapshot (ignored by this strategy)
        """
        await super().on_market_data_update(instrument, snapshot)
        
        # Random strategy doesn't use market data for decision making
        # This method is implemented for completeness and potential logging
        pass

    async def on_trade_execution(self, msg: Dict[str, Any]) -> None:
        """
        Handle trade execution notifications and clean up pending orders.
        
        Args:
            msg: Trade execution message containing order details
        """
        await super().on_trade_execution(msg)
        
        # Remove executed order from pending orders
        order_id = msg.get('order_id')
        if order_id and order_id in self.pending_orders:
            order_info = self.pending_orders[order_id]
            del self.pending_orders[order_id]
            
            self.logger.info(
                f"Random order {order_id} executed for {order_info['instrument']} "
                f"({order_info['side']} {order_info['quantity']}) - removed from pending"
            )

    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the random strategy configuration and state.
        
        Returns:
            Dictionary containing strategy parameters and decision statistics
        """
        total_decisions = self.decision_counts["total"]
        
        return {
            "agent_id": self.agent_id,
            "strategy": "random",
            "trade_probability": self.trade_probability,
            "position_size_pct": self.position_size_pct,
            "random_seed": self.random_seed,
            "instruments": list(self.instrument_exchange_map.keys()),
            "pending_orders_count": len(self.pending_orders),
            "decision_statistics": {
                "total_decisions": total_decisions,
                "buy_decisions": self.decision_counts["buy"],
                "sell_decisions": self.decision_counts["sell"],
                "hold_decisions": self.decision_counts["hold"],
                "buy_percentage": (self.decision_counts["buy"] / total_decisions * 100) if total_decisions > 0 else 0,
                "sell_percentage": (self.decision_counts["sell"] / total_decisions * 100) if total_decisions > 0 else 0,
                "hold_percentage": (self.decision_counts["hold"] / total_decisions * 100) if total_decisions > 0 else 0
            }
        }

    def __repr__(self) -> str:
        """Return string representation of the trader."""
        return (
            f"RandomTrader(agent_id='{self.agent_id}', "
            f"trade_probability={self.trade_probability:.1%}, "
            f"position_size_pct={self.position_size_pct:.1%}, "
            f"seed={self.random_seed}, "
            f"instruments={list(self.instrument_exchange_map.keys())})"
        )