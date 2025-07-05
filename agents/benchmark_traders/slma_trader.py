"""
SLMA Trader - dual moving average crossover strategy using short and long-term SMAs.
Buy when short SMA crosses above long SMA, sell when it crosses below.
"""
from datetime import timedelta
from typing import Dict, Any, Optional

from utils.messages import MessageType
from utils.orders import Side, OrderType
from agents.benchmark_traders.trader import TraderAgent


class SLMATrader(TraderAgent):
    """
    SLMA crossover strategy - buy when short SMA crosses above long SMA, sell when below.
    """

    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        short_window: int = 20,     # Short-term SMA window
        long_window: int = 50,      # Long-term SMA window
        position_size_pct: float = 0.05,  # Percentage of portfolio per trade
        agent_id: Optional[str] = None,
        rabbitmq_host: str = 'localhost',
        **kwargs
    ) -> None:
        """
        Initialize the SLMATrader.

        :param instrument_exchange_map: Mapping of instruments to their respective exchange IDs
        :param short_window: Window size for short-term SMA calculation
        :param long_window: Window size for long-term SMA calculation
        :param position_size_pct: Percentage of portfolio to trade per signal (0.0 to 1.0)
        :param agent_id: Unique identifier for the agent
        :param rabbitmq_host: Hostname for RabbitMQ server
        :param kwargs: Additional keyword arguments
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
        self.short_window = short_window
        self.long_window = long_window
        self.position_size_pct = position_size_pct
        
        # Track SMA values for each instrument
        self.short_sma_values = {}  # Short-term SMA values
        self.long_sma_values = {}   # Long-term SMA values
        self.last_prices = {}       # Last closing prices
        self.last_signals = {}      # Last trading signals (1 for long, 0 for neutral, -1 for short)
        
        # Track pending orders
        self.pending_orders = {}    # Track orders that need resubmission

        self.first_request_sent: Dict[str, bool] = {}

        self.logger.info(
            f"SLMATrader {self.agent_id} initialized with short_window={self.short_window}, "
            f"long_window={self.long_window}, position_size_pct={position_size_pct}"
        )

    async def update_market_snapshot(self, instrument: str) -> None:
        """
        Request a market snapshot for the given instrument.
        """
        if not self.current_time:
            return

        window_end = self.current_time
        if not self.first_request_sent.get(instrument, False):
            window_start = window_end - timedelta(days=30)
            self.first_request_sent[instrument] = True
        else:
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
            self.logger.debug(f"Requested market snapshot for {instrument} at {self.current_time.isoformat()}")
        else:
            self.logger.error(f"Failed to request market snapshot for {instrument}")

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        """
        On each time tick, request market data and handle pending orders.
        """
        await super().handle_time_tick(payload)
        
        current_time = self.current_time
        if self.next_action_time is None:
            self.next_action_time = current_time

        if current_time >= self.next_action_time:
            # Handle pending orders
            await self._handle_pending_orders()

            # Handle market data requests
            for instrument in self.instrument_exchange_map.keys():
                await self.update_market_snapshot(instrument)
            
            self.next_action_time = current_time + self.action_interval

    async def on_market_data_update(self, instrument: str, snapshot: Dict[str, Any]) -> None:
        """
        Handle market data updates. Make trading decisions based on SMA crossovers.
        
        Classic SLMA Crossover strategy:
        - Buy when short-term SMA crosses above long-term SMA (new uptrend) and we don't have a position
        - Sell entire position when short-term SMA crosses below long-term SMA (new downtrend)

        :param instrument: str, the financial instrument
        :param snapshot: dict, the market data snapshot
        """

        if not instrument:
            return

        # Get the latest price from market data
        data = snapshot.get("data", {})
        current_price = data.get("close")
        if not current_price:
            self.logger.error(f"Missing current price for {instrument} at {self.current_time.isoformat()}")
            return

        # Get SMA values from indicators
        indicators = snapshot.get("indicators", {})
        sma_values = indicators.get("trend_indicators").get("sma", {})
        current_short_sma = sma_values.get(str(self.short_window))
        current_long_sma = sma_values.get(str(self.long_window))

        self.logger.info(f"SMA values for {instrument}: short={current_short_sma}, long={current_long_sma}, current_price={current_price}")
        
        if not current_short_sma or not current_long_sma:
            self.logger.error(f"Missing SMA values for {instrument} at {self.current_time.isoformat()}")
            return

        # Initialize tracking if needed
        if instrument not in self.short_sma_values:
            self.short_sma_values[instrument] = current_short_sma
            self.long_sma_values[instrument] = current_long_sma
            self.last_prices[instrument] = current_price
            self.last_signals[instrument] = 0
            return

        # Get previous values
        last_short_sma = self.short_sma_values[instrument]
        last_long_sma = self.long_sma_values[instrument]
        current_position = self.long_qty[instrument]
        
        # Calculate position size based on portfolio percentage
        portfolio_value = self.portfolio_value
        position_value = portfolio_value * self.position_size_pct
        quantity = int(position_value / current_price) if current_price > 0 else 0
        
        if quantity <= 0:
            self.logger.warning(f"Calculated quantity too small for {instrument} at price {current_price}")
            return

        # Check for SMA crossover
        if last_short_sma <= last_long_sma and current_short_sma > current_long_sma:
            # Short SMA crossed above long SMA - Buy signal (only if we don't have a position)
            if current_position <= 0:
                order_id = await self.place_order(
                    instrument=instrument,
                    side=Side.BUY.value,
                    quantity=quantity,
                    order_type=OrderType.MARKET.value,
                    explanation=f"Short SMA {current_short_sma:.2f} crossed above long SMA {current_long_sma:.2f}"
                )
                
                if order_id:
                    self.pending_orders[order_id] = {
                        'instrument': instrument,
                        'side': Side.BUY.value,
                        'quantity': quantity,
                        'order_type': OrderType.MARKET.value,
                        'explanation': f"Short SMA {current_short_sma:.2f} crossed above long SMA {current_long_sma:.2f}"
                    }
                    
                    self.logger.info(
                        f"SLMATrader {self.agent_id} placed BUY order for {instrument} "
                        f"at price {current_price:.2f} (Short SMA: {current_short_sma:.2f}, "
                        f"Long SMA: {current_long_sma:.2f}, Quantity: {quantity})"
                    )
            else:
                self.logger.info(
                    f"SLMATrader {self.agent_id} skipped BUY signal for {instrument} - already have position of {current_position}"
                )
        
        elif last_short_sma >= last_long_sma and current_short_sma < current_long_sma:
            # Short SMA crossed below long SMA - Sell signal (sell entire position)
            if current_position > 0:
                order_id = await self.place_order(
                    instrument=instrument,
                    side=Side.SELL.value,
                    quantity=current_position,  # Sell entire position
                    order_type=OrderType.MARKET.value,
                    explanation=f"Short SMA {current_short_sma:.2f} crossed below long SMA {current_long_sma:.2f}"
                )
                
                if order_id:
                    self.pending_orders[order_id] = {
                        'instrument': instrument,
                        'side': Side.SELL.value,
                        'quantity': current_position,
                        'order_type': OrderType.MARKET.value,
                        'explanation': f"Short SMA {current_short_sma:.2f} crossed below long SMA {current_long_sma:.2f}"
                    }
                    
                    self.logger.info(
                        f"SLMATrader {self.agent_id} placed SELL order for {instrument} "
                        f"at price {current_price:.2f} (Short SMA: {current_short_sma:.2f}, "
                        f"Long SMA: {current_long_sma:.2f}, Quantity: {current_position})"
                    )
            else:
                self.logger.info(
                    f"SLMATrader {self.agent_id} skipped SELL signal for {instrument} - no position to sell"
                )

        # Update tracking values
        self.short_sma_values[instrument] = current_short_sma
        self.long_sma_values[instrument] = current_long_sma
        self.last_prices[instrument] = current_price

    async def on_trade_execution(self, trade: Dict[str, Any]) -> None:
        """
        Handle trade execution and clean up pending orders.
        """
        await super().on_trade_execution(trade)
        
        # Remove executed order from pending orders
        order_id = trade.get('order_id')
        if order_id and order_id in self.pending_orders:
            del self.pending_orders[order_id]
            self.logger.info(f"Order {order_id} executed, removed from pending orders")

    async def _handle_pending_orders(self) -> None:
        """
        Handle resubmission of pending orders.
        """
        for order_id, order_info in list(self.pending_orders.items()):
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
                    f"Resubmitted order {order_id} -> {new_order_id} for {order_info['instrument']} with quantity {order_info['quantity']}"
                )
            else:
                # Remove failed order from pending to avoid infinite resubmission
                del self.pending_orders[order_id]
                self.logger.error(
                    f"Failed to resubmit order {order_id} for {order_info['instrument']} - removing from pending"
                )