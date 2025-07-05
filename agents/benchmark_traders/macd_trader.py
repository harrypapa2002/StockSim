"""
MACD Trader - buy when MACD line crosses above signal line, sell when it crosses below.
Momentum strategy using Moving Average Convergence Divergence indicator.
"""
from datetime import timedelta
from typing import Dict, Any, Optional

from utils.messages import MessageType
from utils.orders import Side, OrderType
from agents.benchmark_traders.trader import TraderAgent


class MACDTrader(TraderAgent):
    """
    MACD crossover strategy - buy when MACD crosses above signal line, sell when below.
    """

    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        position_size_pct: float = 0.05,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = 'localhost',
        **kwargs
    ) -> None:
        """
        Initialize the MACD Trader.

        Args:
            instrument_exchange_map: Mapping of instrument symbols to exchange agent IDs
            fast_period: Fast EMA period for MACD calculation (default: 12)
            slow_period: Slow EMA period for MACD calculation (default: 26)
            signal_period: Signal line EMA period (default: 9)
            position_size_pct: Percentage of portfolio to trade per signal (0.0 to 1.0)
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
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.position_size_pct = position_size_pct
        
        # Track values for each instrument
        self.macd_values: Dict[str, float] = {}
        self.signal_values: Dict[str, float] = {}
        
        # Track pending orders for resubmission
        self.pending_orders: Dict[str, Dict[str, Any]] = {}

        self.first_request_sent: Dict[str, bool] = {}


        self.logger.info(
            f"MACDTrader {self.agent_id} initialized with periods "
            f"fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}, "
            f"position_size_pct={self.position_size_pct:.1%} for instruments: "
            f"{list(self.instrument_exchange_map.keys())}"
        )

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        """
        Handle time tick events and coordinate trading activities.
        
        Processes pending order resubmissions and requests market data snapshots
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

            # Request market data for all instruments
            for instrument in self.instrument_exchange_map.keys():
                await self._request_market_snapshot(instrument)
            
            self.next_action_time = current_time + self.action_interval

    async def _request_market_snapshot(self, instrument: str) -> None:
        """
        Request market data snapshot for the specified instrument.
        
        Args:
            instrument: Financial instrument symbol to request data for
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
            self.logger.debug(f"Requested market snapshot for {instrument}")
        else:
            self.logger.error(f"Failed to request market snapshot for {instrument}")

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
                self.logger.error(
                    f"Exception resubmitting order {order_id}: {e}"
                )

    async def on_market_data_update(self, instrument: str, snapshot: Dict[str, Any]) -> None:
        """
        Process market data updates and generate trading signals based on MACD crossovers.
        
        Analyzes current MACD and signal line values to detect crossovers and generates
        buy/sell signals when momentum shifts occur. Implements classic MACD strategy
        logic with position management and risk controls.
        
        Args:
            instrument: Financial instrument symbol
            snapshot: Market data snapshot containing price and indicator data
        """
        await super().on_market_data_update(instrument, snapshot)
        
        if not instrument:
            return

        # Extract current price from market data
        data = snapshot.get("data", {})
        current_price = data.get("close")
        if not current_price:
            self.logger.warning(f"No closing price available for {instrument}")
            return

        # Extract MACD values from indicators
        indicators = snapshot.get("indicators", {})
        macd_data = indicators.get("trend_indicators", {}).get("macd", {})

        current_macd = macd_data.get("line")
        current_signal = macd_data.get("signal")
        
        if current_macd is None or current_signal is None:
            self.logger.debug(
                f"MACD values not available for {instrument} "
                f"(periods: {self.fast_period}/{self.slow_period}/{self.signal_period})"
            )
            return

        # Initialize tracking for new instrument
        if instrument not in self.macd_values:
            self.macd_values[instrument] = current_macd
            self.signal_values[instrument] = current_signal
            self.logger.debug(
                f"Initialized MACD tracking for {instrument} "
                f"(MACD: {current_macd:.4f}, Signal: {current_signal:.4f})"
            )
            return

        # Get historical values for crossover detection
        last_macd = self.macd_values[instrument]
        last_signal = self.signal_values[instrument]
        current_position = self.long_qty.get(instrument, 0)
        
        # Calculate position size based on portfolio percentage
        portfolio_value = self.portfolio_value
        if portfolio_value <= 0:
            self.logger.warning(f"Invalid portfolio value: {portfolio_value}")
            return
            
        position_value = portfolio_value * self.position_size_pct
        quantity = int(position_value / current_price) if current_price > 0 else 0
        
        if quantity <= 0:
            self.logger.warning(
                f"Calculated quantity too small for {instrument} "
                f"(portfolio: {portfolio_value:.2f}, price: {current_price:.2f})"
            )
            # Update tracking values even if we can't trade
            self.macd_values[instrument] = current_macd
            self.signal_values[instrument] = current_signal
            return

        # Detect MACD crossover signals
        signal_generated = False
        
        # Buy signal: MACD crosses above signal line
        if last_macd <= last_signal and current_macd > current_signal:
            if current_position <= 0:  # Only buy if no existing position
                await self._execute_buy_signal(instrument, current_price, current_macd, current_signal, quantity)
                signal_generated = True
            else:
                self.logger.info(
                    f"MACDTrader {self.agent_id} skipped BUY signal for {instrument} "
                    f"- already have position of {current_position}"
                )
        
        # Sell signal: MACD crosses below signal line
        elif last_macd >= last_signal and current_macd < current_signal:
            if current_position > 0:  # Only sell if we have a position
                await self._execute_sell_signal(instrument, current_price, current_macd, current_signal, current_position)
                signal_generated = True
            else:
                self.logger.debug(
                    f"MACDTrader {self.agent_id} skipped SELL signal for {instrument} "
                    f"- no position to sell"
                )

        # Log current state for monitoring
        if signal_generated or self.logger.isEnabledFor(10):  # DEBUG level
            self.logger.debug(
                f"MACD analysis for {instrument}: Price {current_price:.2f}, "
                f"MACD {current_macd:.4f}, Signal {current_signal:.4f}, "
                f"Position {current_position}, Signal: {'YES' if signal_generated else 'NO'}"
            )

        # Update tracking values
        self.macd_values[instrument] = current_macd
        self.signal_values[instrument] = current_signal

    async def _execute_buy_signal(self, instrument: str, price: float, macd: float, signal: float, quantity: int) -> None:
        """
        Execute buy signal when MACD crosses above signal line.
        
        Args:
            instrument: Financial instrument symbol
            price: Current market price
            macd: Current MACD line value
            signal: Current signal line value
            quantity: Number of shares to purchase
        """
        explanation = f"MACD {macd:.4f} crossed above Signal {signal:.4f} (bullish momentum)"
        
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
                    f"MACDTrader {self.agent_id} placed BUY order {order_id} for {instrument}: "
                    f"{quantity} shares at {price:.2f} (MACD: {macd:.4f}, Signal: {signal:.4f})"
                )
            else:
                self.logger.error(f"Failed to place BUY order for {instrument}")
                
        except Exception as e:
            self.logger.error(f"Exception placing BUY order for {instrument}: {e}")

    async def _execute_sell_signal(self, instrument: str, price: float, macd: float, signal: float, quantity: int) -> None:
        """
        Execute sell signal when MACD crosses below signal line.
        
        Args:
            instrument: Financial instrument symbol
            price: Current market price
            macd: Current MACD line value
            signal: Current signal line value
            quantity: Number of shares to sell (entire position)
        """
        explanation = f"MACD {macd:.4f} crossed below Signal {signal:.4f} (bearish momentum)"
        
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
                    f"MACDTrader {self.agent_id} placed SELL order {order_id} for {instrument}: "
                    f"{quantity} shares at {price:.2f} (MACD: {macd:.4f}, Signal: {signal:.4f})"
                )
            else:
                self.logger.error(f"Failed to place SELL order for {instrument}")
                
        except Exception as e:
            self.logger.error(f"Exception placing SELL order for {instrument}: {e}")

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
                f"Order {order_id} executed for {order_info['instrument']} "
                f"({order_info['side']} {order_info['quantity']}) - removed from pending"
            )

    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the MACD strategy configuration and state.
        
        Returns:
            Dictionary containing strategy parameters and current state
        """
        return {
            "agent_id": self.agent_id,
            "strategy": "macd",
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
            "position_size_pct": self.position_size_pct,
            "instruments": list(self.instrument_exchange_map.keys()),
            "pending_orders_count": len(self.pending_orders),
            "tracked_instruments": list(self.macd_values.keys())
        }

    def __repr__(self) -> str:
        """Return string representation of the trader."""
        return (
            f"MACDTrader(agent_id='{self.agent_id}', "
            f"periods={self.fast_period}/{self.slow_period}/{self.signal_period}, "
            f"position_size_pct={self.position_size_pct:.1%}, "
            f"instruments={list(self.instrument_exchange_map.keys())})"
        )