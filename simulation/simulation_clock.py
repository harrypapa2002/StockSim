"""
Simulation Clock for StockSim Trading Environment

This module provides the central time management system for financial trading simulations.
The SimulationClock orchestrates the progression of simulation time, broadcasting synchronized
time ticks to all agents and managing the coordination of trading activities.

Key Features:
    - Centralized time management for distributed trading simulation
    - Barrier synchronization ensuring all agents process each time step
    - Support for agent evolution and assessment cycles
    - Graceful shutdown coordination across all simulation components
    - Comprehensive logging and monitoring of simulation progress

Architecture:
    The SimulationClock uses RabbitMQ message queues to coordinate timing across:
    - Exchange agents (market makers, order books)
    - Trading agents (AI traders, benchmark strategies)
    - Evolution agents (strategy refinement, assessment)

Example Usage:
    # Initialize simulation clock
    clock = SimulationClock(
        start_time=datetime(2024, 1, 1, 9, 30),
        end_time=datetime(2024, 1, 31, 16, 0),
        tick_interval_seconds=3600,  # 1 hour per tick
        expected_exchange_agent_count=1,
        expected_responses=5  # Number of trading agents
    )

    # Run the simulation
    await clock.run()
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Optional

import aio_pika
from aio_pika import ExchangeType
from aio_pika.abc import AbstractIncomingMessage
from aio_pika.exceptions import AMQPConnectionError, AMQPChannelError

from utils.logging_setup import setup_logger
from utils.messages import MessageType


class SimulationClock:
    """
    Central time management system for StockSim trading simulations.

    The SimulationClock serves as the authoritative timekeeper for the entire trading
    environment, ensuring all agents operate in perfect synchronization. It manages
    the progression from simulation start to end, broadcasting time ticks and waiting
    for acknowledgments to maintain barrier synchronization.

    The clock supports sophisticated coordination patterns including:
    - Two-phase time advancement (exchange first, then traders)
    - Periodic evolution cycles for strategy refinement
    - Graceful shutdown with proper cleanup

    Attributes:
        TIME_EXCHANGE_NAME (str): RabbitMQ exchange name for time coordination
        TIME_EXCHANGE_TYPE: Exchange type for message routing
        SIMULATION_CLOCK_QUEUE_NAME (str): Queue name for receiving responses
        SIMULATION_CLOCK_ROUTING_KEY (str): Routing key for clock messages
    """

    # RabbitMQ Configuration Constants
    TIME_EXCHANGE_NAME = "time_exchange"
    TIME_EXCHANGE_TYPE = ExchangeType.TOPIC
    SIMULATION_CLOCK_QUEUE_NAME = "simulation_clock_queue"
    SIMULATION_CLOCK_ROUTING_KEY = "simulation_clock"

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        tick_interval_seconds: int,
        rabbitmq_host: str = 'localhost',
        expected_exchange_agent_count: int = 1,
        expected_responses: int = 0,
    ):
        """
        Initialize the SimulationClock with timing and coordination parameters.

        Args:
            start_time (datetime): When the simulation begins
            end_time (datetime): When the simulation ends
            tick_interval_seconds (int): Real-world seconds between simulation ticks
            rabbitmq_host (str): RabbitMQ server hostname
            expected_exchange_agent_count (int): Number of exchange agents to wait for
            expected_responses (int): Number of trading agents to wait for
        """
        # Core timing configuration
        self.current_time: datetime = start_time
        self.end_time: datetime = end_time
        self.tick_interval: timedelta = timedelta(seconds=tick_interval_seconds)

        # Network and coordination setup
        self.rabbitmq_host: str = rabbitmq_host
        self.expected_exchange_agent_count: int = expected_exchange_agent_count
        self.expected_responses: int = expected_responses
        

        # Internal state management
        self.barrier_response_queue: asyncio.Queue = asyncio.Queue()
        self.trader_response_queue: asyncio.Queue[int] = asyncio.Queue()
        self._tick_count: int = 0
        self._should_stop: bool = False

        # RabbitMQ connection objects
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.RobustChannel] = None
        self.exchange: Optional[aio_pika.Exchange] = None
        self.clock_queue: Optional[aio_pika.Queue] = None

        # Logging setup
        self.log_dir = os.getenv("LOG_DIR", "logs_v33")
        self.logger = setup_logger("SimulationClock", f"{self.log_dir}/simulation_clock.log")

        self.logger.info(
            f"SimulationClock initialized: {self.current_time} → {self.end_time}, "
            f"tick_interval={self.tick_interval}, "
            f"exchange_agents={self.expected_exchange_agent_count}, trader_responses={self.expected_responses}"
        )

    async def setup_rabbitmq(self) -> None:
        """
        Establish RabbitMQ connection and declare necessary exchanges and queues.

        This method sets up the messaging infrastructure required for simulation
        coordination, including the time exchange and response queues.

        Raises:
            AMQPConnectionError: If connection to RabbitMQ fails
            AMQPChannelError: If exchange declaration fails
        """
        try:
            self.logger.info(f"Connecting to RabbitMQ server at {self.rabbitmq_host}...")
            self.connection = await aio_pika.connect_robust(host=self.rabbitmq_host)
            self.channel = await self.connection.channel()

            # Declare the main time coordination exchange
            self.exchange = await self.channel.declare_exchange(
                name=self.TIME_EXCHANGE_NAME,
                type=self.TIME_EXCHANGE_TYPE,
                durable=True
            )
            self.logger.info(f"Declared time exchange '{self.TIME_EXCHANGE_NAME}' ({self.TIME_EXCHANGE_TYPE})")

            # Set up the clock's response queue for barrier synchronization
            self.clock_queue = await self.channel.declare_queue(
                name=self.SIMULATION_CLOCK_QUEUE_NAME,
                durable=True,
                exclusive=True
            )

            await self.clock_queue.bind(self.exchange, routing_key=self.SIMULATION_CLOCK_ROUTING_KEY)
            self.logger.info(f"Clock queue bound to routing key '{self.SIMULATION_CLOCK_ROUTING_KEY}'")

            # Start consuming response messages
            await self.clock_queue.consume(self._on_message, no_ack=True)

        except AMQPConnectionError as e:
            self.logger.error(f"Failed to connect to RabbitMQ at '{self.rabbitmq_host}': {e}")
            raise
        except AMQPChannelError as e:
            self.logger.error(f"Failed to declare exchange '{self.TIME_EXCHANGE_NAME}': {e}")
            raise

    async def _on_message(self, message: AbstractIncomingMessage):
        """
        Process incoming response messages from agents.

        This method handles barrier synchronization by processing different types
        of response messages and routing them to appropriate queues.

        Args:
            message (AbstractIncomingMessage): Incoming RabbitMQ message
        """
        try:
            msg = json.loads(message.body.decode())
            msg_type_str = msg.get("type")

            if not msg_type_str:
                self.logger.warning("Received message without type field")
                return

            msg_type = MessageType(msg_type_str)
            payload = msg.get("payload", {})
            tick_id = payload.get("tick_id")

            if msg_type == MessageType.BARRIER_RESPONSE:
                self.logger.debug(f"Received BARRIER_RESPONSE for tick {tick_id}")
                await self.barrier_response_queue.put(tick_id)

            elif msg_type == MessageType.DECISION_RESPONSE:
                self.logger.debug(f"Received DECISION_RESPONSE for tick {tick_id}")
                await self.trader_response_queue.put(tick_id)

            else:
                self.logger.debug(f"Received unhandled message type: {msg_type_str}")

        except Exception as e:
            self.logger.error(f"Error processing clock queue message: {e}")

    async def broadcast_time_tick(
        self,
        tick_id: int = None,
        is_exchange: bool = False
    ) -> None:
        """
        Broadcast a TIME_TICK message to subscribed agents.

        This method sends synchronized time advancement signals to either exchange
        agents or trading agents, depending on the simulation phase.

        Args:
            tick_id (int): Unique identifier for this time tick
            is_exchange (bool): True to target exchange agents, False for traders

        Raises:
            AMQPChannelError: If message publishing fails
        """
        if not self.exchange:
            self.logger.error("Cannot broadcast TIME_TICK: exchange not initialized")
            return

        # Construct the time tick message
        message_payload = {
            "tick_id": tick_id,
            "current_time": self.current_time.isoformat()
        }


        message = {
            "type": MessageType.TIME_TICK.value,
            "payload": message_payload
        }

        try:
            routing_key = 'exchange' if is_exchange else 'trader'
            agent_type = "exchange" if is_exchange else "trading"

            self.logger.info(f"Broadcasting TIME_TICK to {agent_type} agents at {self.current_time}")

            await self.exchange.publish(
                aio_pika.Message(
                    body=json.dumps(message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=routing_key
            )

        except AMQPChannelError as e:
            self.logger.error(f"Failed to broadcast TIME_TICK: {e}")

    async def _wait_for_barrier_responses(self, tick_id: int, expected_count: int, agent_type: str):
        """
        Wait for barrier synchronization responses from agents.

        This method implements the barrier synchronization pattern, ensuring all
        agents have processed the current time tick before advancing.

        Args:
            tick_id (int): The tick ID we're waiting for responses to
            expected_count (int): Number of responses expected
            agent_type (str): Type of agents for logging purposes
        """
        received_count = 0

        while received_count < expected_count:
            response_tick_id = await self.barrier_response_queue.get()

            if response_tick_id == tick_id:
                received_count += 1
                self.logger.info(
                    f"Barrier sync: {received_count}/{expected_count} {agent_type} "
                    f"agents responded for tick {tick_id}"
                )
            else:
                self.logger.warning(
                    f"Received unexpected tick ID {response_tick_id} "
                    f"while waiting for {tick_id}. Ignoring."
                )

    async def _wait_for_trader_responses(self, tick_id: int):
        """
        Wait for decision responses from trading agents.

        Args:
            tick_id (int): The tick ID we're waiting for responses to
        """
        decisions_received = 0

        while decisions_received < self.expected_responses:
            response_tick_id = await self.trader_response_queue.get()

            if response_tick_id == tick_id:
                decisions_received += 1
                self.logger.info(
                    f"Trading decisions: {decisions_received}/{self.expected_responses} "
                    f"received for tick {tick_id}"
                )
            else:
                self.logger.warning(
                    f"Received unexpected tick ID {response_tick_id} "
                    f"while waiting for trading decisions on {tick_id}. Ignoring."
                )


    async def run(self) -> None:
        """
        Execute the main simulation loop.

        This method orchestrates the entire simulation from start to finish,
        managing time advancement, agent coordination, and evolution cycles.
        The simulation follows this pattern for each tick:

        1. Broadcast time tick to exchange agents
        2. Wait for exchange agent barrier responses
        3. Broadcast time tick to trading agents
        4. Handle evolution cycles (if scheduled)
        5. Wait for trading agent decision responses
        6. Advance simulation time

        The loop continues until the end time is reached or a stop signal is received.
        """
        try:
            await self.setup_rabbitmq()
            self.logger.info(
                f"Starting simulation: {self.current_time} → {self.end_time} "
                f"(interval: {self.tick_interval})"
            )

            while self.current_time < self.end_time and not self._should_stop:
                tick_id = self._tick_count

                # Phase 1: Exchange agents process the time tick
                await self.broadcast_time_tick(tick_id, is_exchange=True)
                await self._wait_for_barrier_responses(
                    tick_id, self.expected_exchange_agent_count, "exchange"
                )

                # Phase 2: Trading agents process the time tick
                await self.broadcast_time_tick(tick_id, is_exchange=False)

                # Phase 3: Wait for trading decisions
                await self._wait_for_trader_responses(tick_id)

                await asyncio.sleep(5)


                # Advance simulation time
                self.current_time += self.tick_interval
                self._tick_count += 1

                self.logger.info(
                    f"Completed tick {tick_id}: advanced to {self.current_time} "
                )

        except Exception as e:
            self.logger.error(f"Simulation encountered error: {e}", exc_info=True)
        finally:
            # Graceful shutdown sequence
            self.logger.info("Simulation completed. Initiating shutdown sequence...")
            await asyncio.sleep(2)  # Allow final messages to be processed
            await self.broadcast_stop_simulation()
            await asyncio.sleep(3)  # Allow stop messages to be delivered
            await self.teardown()
            self.logger.info("Simulation shutdown complete")

    async def broadcast_stop_simulation(self) -> None:
        """
        Broadcast STOP_SIMULATION message to all agents.

        This method signals all agents to gracefully shut down their operations
        and clean up resources.
        """
        stop_message = {
            "type": MessageType.STOP_SIMULATION.value,
            "payload": {}
        }

        if not self.exchange:
            self.logger.warning("Cannot broadcast STOP_SIMULATION: no exchange available")
            return

        try:
            self.logger.info("Broadcasting STOP_SIMULATION to all agents")
            await self.exchange.publish(
                aio_pika.Message(
                    body=json.dumps(stop_message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key='stop_simulation'
            )
        except Exception as e:
            self.logger.error(f"Failed to broadcast STOP_SIMULATION: {e}")

    async def teardown(self) -> None:
        """
        Clean up RabbitMQ connections and resources.

        This method ensures all network connections are properly closed
        and resources are released.
        """
        if self.connection and not self.connection.is_closed:
            try:
                await self.connection.close()
                self.logger.info("RabbitMQ connection closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing RabbitMQ connection: {e}")

    def stop(self) -> None:
        """
        Signal the simulation clock to stop at the next opportunity.

        This method provides a way to gracefully terminate the simulation
        before reaching the scheduled end time.
        """
        self._should_stop = True
        self.logger.info("Stop signal received - simulation will terminate after current tick")

    @property
    def progress_percentage(self) -> float:
        """
        Calculate the current simulation progress as a percentage.

        Returns:
            float: Progress percentage (0.0 to 100.0)
        """
        total_duration = self.end_time - (self.current_time - self.tick_interval * self._tick_count)
        elapsed_duration = self.current_time - (self.current_time - self.tick_interval * self._tick_count)

        if total_duration.total_seconds() <= 0:
            return 100.0

        return min(100.0, (elapsed_duration.total_seconds() / total_duration.total_seconds()) * 100.0)

    def __repr__(self) -> str:
        """Return a string representation of the simulation clock state."""
        return (
            f"SimulationClock(current={self.current_time}, "
            f"end={self.end_time}, tick={self._tick_count}, "
            f"progress={self.progress_percentage:.1f}%)"
        )
