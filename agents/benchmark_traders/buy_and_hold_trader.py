"""
Buy and Hold Trader - purchases and holds positions throughout simulation.
Serves as a passive investment baseline for comparing active trading strategies.
"""


from typing import Dict, Any, Optional

from utils.orders import Side, OrderType
from agents.benchmark_traders.trader import TraderAgent


class BuyAndHoldTrader(TraderAgent):
    """
    Buy and hold strategy - purchases fixed quantity at start, holds until end.
    Serves as passive investment baseline for comparing active strategies.
    """

    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        quantity_size: int = 100,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = 'localhost',
        **kwargs
    ) -> None:
        """
        Initialize the Buy and Hold Trader.

        Args:
            instrument_exchange_map: Mapping of instrument symbols to exchange agent IDs
            quantity_size: Number of shares to buy and hold for each instrument
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

        self.first_request_sent: Dict[str, bool] = {}
        
        self.quantity_size = quantity_size
        self.has_bought = False
        self.instruments_bought = set()  # Track which instruments have been purchased

        self.logger.info(
            f"BuyAndHoldTrader {self.agent_id} initialized with quantity_size={self.quantity_size} "
            f"for instruments: {list(self.instrument_exchange_map.keys())}"
        )

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        """
        Handle time tick events and place initial buy orders if not already done.
        
        On the first time tick, places market buy orders for all configured instruments.
        Subsequent time ticks are processed normally but no additional trades are made.
        
        Args:
            payload: Time tick information from simulation clock
        """
        await super().handle_time_tick(payload)
        
        # Only attempt to buy once and only if we haven't completed all purchases
        if not self.has_bought:
            current_time = self.current_time
            if self.next_action_time is None:
                self.next_action_time = current_time

            if current_time >= self.next_action_time:
                await self._place_initial_orders()
                self.next_action_time = current_time + self.action_interval

    async def _place_initial_orders(self) -> None:
        """
        Place initial buy orders for all configured instruments.
        
        Submits market buy orders for the specified quantity of each instrument.
        Orders are placed asynchronously and execution is confirmed through
        trade execution callbacks.
        """
        for instrument in self.instrument_exchange_map.keys():
            if instrument not in self.instruments_bought:
                try:
                    order_id = await self.place_order(
                        instrument=instrument,
                        side=Side.BUY.value,
                        quantity=self.quantity_size,
                        order_type=OrderType.MARKET.value,
                        explanation="Initial buy and hold position establishment"
                    )
                    
                    if order_id:
                        self.logger.info(
                            f"BuyAndHoldTrader {self.agent_id} placed initial BUY order {order_id} "
                            f"for {instrument}: {self.quantity_size} shares at market price"
                        )
                    else:
                        self.logger.error(
                            f"BuyAndHoldTrader {self.agent_id} failed to place BUY order for {instrument}"
                        )
                        
                except Exception as e:
                    self.logger.error(
                        f"BuyAndHoldTrader {self.agent_id} exception placing order for {instrument}: {e}"
                    )

    async def on_trade_execution(self, msg: Dict[str, Any]) -> None:
        """
        Handle trade execution notifications and track position establishment.
        
        Updates internal state when initial buy orders are executed, marking
        the completion of the buy-and-hold strategy implementation.
        
        Args:
            msg: Trade execution message from exchange containing execution details
        """
        await super().on_trade_execution(msg)
        
        # Check if this was our initial buy order
        instrument = msg.get("instrument")
        role = msg.get("role")
        quantity = msg.get("quantity")
        
        if (instrument and 
            role == "BUYER" and 
            quantity == self.quantity_size and
            instrument not in self.instruments_bought):
            
            self.instruments_bought.add(instrument)
            self.logger.info(
                f"BuyAndHoldTrader {self.agent_id} initial position established for {instrument}: "
                f"{self.quantity_size} shares at price {msg.get('price', 'UNKNOWN')}"
            )
            
            # Check if we've bought all instruments
            if len(self.instruments_bought) == len(self.instrument_exchange_map):
                self.has_bought = True
                self.logger.info(
                    f"BuyAndHoldTrader {self.agent_id} completed initial position establishment "
                    f"for all {len(self.instruments_bought)} instruments. "
                    f"Now holding positions until simulation end."
                )

    async def on_market_data_update(self, instrument: str, snapshot: Dict[str, Any]) -> None:
        """
        Handle market data updates (no action taken for buy-and-hold strategy).
        
        Buy-and-hold strategy ignores market data updates and maintains positions
        regardless of price movements or technical indicators.
        
        Args:
            instrument: Financial instrument symbol
            snapshot: Market data snapshot (ignored by this strategy)
        """
        # Buy-and-hold strategy doesn't react to market data
        # This method is implemented for completeness and potential logging
        pass

    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the buy-and-hold strategy execution.
        
        Returns:
            Dictionary containing strategy execution statistics and status
        """
        return {
            "agent_id": self.agent_id,
            "strategy": "buy_and_hold",
            "quantity_per_instrument": self.quantity_size,
            "total_instruments": len(self.instrument_exchange_map),
            "instruments_purchased": len(self.instruments_bought),
            "initial_buying_complete": self.has_bought,
            "instruments": list(self.instrument_exchange_map.keys()),
            "purchased_instruments": list(self.instruments_bought)
        }

    def __repr__(self) -> str:
        """Return string representation of the trader."""
        return (
            f"BuyAndHoldTrader(agent_id='{self.agent_id}', "
            f"quantity_size={self.quantity_size}, "
            f"instruments={list(self.instrument_exchange_map.keys())}, "
            f"buying_complete={self.has_bought})"
        )