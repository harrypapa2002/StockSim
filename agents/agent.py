"""
Base Agent Class - foundational agent architecture for StockSim simulation platform.
Provides RabbitMQ messaging, time synchronization, and extensible agent capabilities.
"""
import json
import os
import uuid
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

import aio_pika
from aio_pika import RobustConnection, RobustChannel, ExchangeType
from aio_pika.abc import AbstractIncomingMessage
from aio_pika.exceptions import AMQPConnectionError, AMQPChannelError

from utils.logging_setup import setup_logger
from utils.messages import MessageType
from utils.time_utils import parse_datetime_utc


class Agent(ABC):
    """
    Abstract base class for all agents in the StockSim simulation.
    """

    # Exchange configuration for simulation messaging
    SIMULATION_EXCHANGE_NAME = 'simulation_exchange'
    SIMULATION_EXCHANGE_TYPE = ExchangeType.DIRECT
    TIME_EXCHANGE_NAME = 'time_exchange'
    TIME_EXCHANGE_TYPE = ExchangeType.TOPIC

    def __init__(
        self,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = 'localhost',
        auto_ack: bool = False,
        reconnect_initial_delay: float = 5.0,
        reconnect_max_delay: float = 60.0
    ):
        """
        Initialize the Agent with RabbitMQ connection parameters.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID is generated.
            rabbitmq_host: Hostname for RabbitMQ server.
            auto_ack: Whether to automatically acknowledge messages.
            reconnect_initial_delay: Initial delay in seconds before attempting to reconnect.
            reconnect_max_delay: Maximum delay in seconds for reconnection attempts.
        """
        # Core agent properties
        self.LOG_DIR = os.getenv("LOG_DIR", "logs")
        self.current_time: Optional[datetime] = None
        self.agent_id = agent_id if agent_id else str(uuid.uuid4())
        self.current_tick_id: Optional[int] = None

        # RabbitMQ connection configuration
        self.rabbitmq_host = rabbitmq_host
        self.auto_ack = auto_ack
        self.reconnect_initial_delay = reconnect_initial_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.routing_key = f"agent.{self.agent_id}"

        # Connection state management
        self.connection: Optional[RobustConnection] = None
        self.channel: Optional[RobustChannel] = None
        self.exchange: Optional[aio_pika.Exchange] = None
        self.queue: Optional[aio_pika.Queue] = None
        self.consumer_tag: Optional[str] = None

        # Time synchronization components
        self.time_exchange: Optional[aio_pika.Exchange] = None
        self.time_queue: Optional[aio_pika.Queue] = None
        self.time_consumer_tag: Optional[str] = None

        # Agent lifecycle management
        self._stop_event = asyncio.Event()
        self._shutdown = False
        self._loop = asyncio.get_event_loop()

        # Initialize logging
        self.logger = setup_logger(
            self.agent_id,
            os.path.join(self.LOG_DIR, "agents", f"agent_{self.agent_id}.log")
        )
        self.logger.info(f"Initializing Agent {self.agent_id} with RabbitMQ host {self.rabbitmq_host}.")

    async def connect(self):
        """
        Establish connection to RabbitMQ and set up exchange and queue.

        Implements exponential backoff strategy for robust connection management,
        ensuring reliable operation in distributed simulation environments.
        Critical for multi-agent coordination research as described in the EMNLP paper.
        """
        delay = self.reconnect_initial_delay

        while not self._shutdown:
            try:
                self.logger.info(f"Attempting to connect to RabbitMQ at {self.rabbitmq_host}...")

                # Establish robust connection with publisher confirmations
                self.connection = await aio_pika.connect_robust(host=self.rabbitmq_host)
                self.channel = await self.connection.channel(publisher_confirms=True)
                await self.channel.set_qos(prefetch_count=1)

                # Setup simulation exchange for agent-to-agent communication
                self.exchange = await self.channel.declare_exchange(
                    name=self.SIMULATION_EXCHANGE_NAME,
                    type=self.SIMULATION_EXCHANGE_TYPE,
                    durable=True
                )
                self.logger.debug(f"Declared exchange '{self.SIMULATION_EXCHANGE_NAME}' of type '{self.SIMULATION_EXCHANGE_TYPE}'.")

                # Setup exclusive queue for this agent
                self.queue = await self.channel.declare_queue(
                    name=self.routing_key,
                    durable=True,
                    exclusive=True
                )
                self.logger.debug(f"Declared exclusive queue '{self.routing_key}'.")

                # Bind queue to exchange with agent-specific routing key
                await self.queue.bind(self.exchange, routing_key=self.routing_key)
                self.logger.debug(f"Bound queue '{self.routing_key}' to exchange '{self.SIMULATION_EXCHANGE_NAME}' "
                                  f"with routing key '{self.routing_key}'.")

                # Setup time synchronization exchange
                self.time_exchange = await self.channel.declare_exchange(
                    name=self.TIME_EXCHANGE_NAME,
                    type=self.TIME_EXCHANGE_TYPE,
                    durable=True
                )
                self.logger.debug(f"Declared time exchange '{self.TIME_EXCHANGE_NAME}' "
                                  f"of type '{self.TIME_EXCHANGE_TYPE}'.")

                # Setup time queue with appropriate routing
                self.time_queue = await self.channel.declare_queue(
                    name=f"time_queue.{self.agent_id}",
                    durable=True,
                    exclusive=True
                )

                # Configure routing key based on agent type
                if self.agent_id.startswith("candle") or self.agent_id.startswith("exchange"):
                    time_routing_key = f"exchange.#"
                else:
                    time_routing_key = f"trader.#"

                self.logger.info(f"Time routing key: {time_routing_key}")
                self.logger.debug(f"Declared exclusive time queue 'time_queue.{self.agent_id}'.")

                # Bind time queue to time exchange
                await self.time_queue.bind(self.time_exchange, routing_key=time_routing_key)
                await self.time_queue.bind(self.time_exchange, routing_key="stop_simulation")
                self.logger.info(f"Agent {self.agent_id} bound to time exchange '{self.TIME_EXCHANGE_NAME}'.")

                # Connection successful, exit retry loop
                return

            except (AMQPConnectionError, AMQPChannelError) as e:
                self.logger.error(f"Connection failed: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.reconnect_max_delay)
            except Exception as e:
                self.logger.error(f"Unexpected error during connection setup: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.reconnect_max_delay)

    async def publish_time(self, msg_type: MessageType, payload: dict, routing_key: str) -> None:
        """
        Publish a message on the TIME_EXCHANGE_NAME (topic exchange).

        Used for time coordination messages in the simulation environment.
        Critical for synchronizing multi-agent interactions and maintaining
        deterministic simulation progression.

        Args:
            msg_type: Type of the time message
            payload: Message payload containing time information
            routing_key: Routing key for message distribution
        """
        if not self.time_exchange:
            self.logger.error("Time exchange not initialized; cannot publish_time()")
            return

        body = {
            "sender": self.agent_id,
            "type": msg_type.value,
            "payload": payload
        }

        await self.time_exchange.publish(
            aio_pika.Message(
                body=json.dumps(body).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key=routing_key
        )

    async def send_message(
        self,
        recipient_id: str,
        message_type: MessageType,
        payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Publish a message to the simulation_exchange.

        Enables inter-agent communication for multi-agent coordination research.
        The recipient_id determines the routing key for message delivery.

        Args:
            recipient_id: The agent ID of the recipient.
            message_type: The type of the message.
            payload: The message payload.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        if not self.exchange:
            self.logger.error("Exchange not initialized. Cannot send message.")
            return False

        msg = {
            "sender": self.agent_id,
            "recipient": recipient_id,
            "type": message_type.value,
            "payload": payload
        }

        routing_key = f"agent.{recipient_id}"

        try:
            await self.exchange.publish(
                aio_pika.Message(
                    body=json.dumps(msg).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=routing_key
            )
            return True

        except (AMQPConnectionError, AMQPChannelError) as e:
            self.logger.error(f"Failed to send message to '{recipient_id}': {e}. Attempting to reconnect...")
            await self.reconnect()
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error while sending message to '{recipient_id}': {e}.")
            return False

    async def consume(self):
        """
        Start consuming messages asynchronously.

        Initiates message consumption for both regular agent communication
        and time synchronization messages. Critical for multi-agent
        coordination in the simulation environment.
        """
        if not self.queue:
            self.logger.error("Queue not initialized. Cannot start consuming.")
            return

        # Start consuming regular agent messages
        self.consumer_tag = await self.queue.consume(self.on_message)
        self.logger.info("Started consuming messages.")

        if not self.time_queue:
            self.logger.error("Time queue not initialized. Cannot start consuming TIME_TICK messages.")
            return

        # Start consuming time synchronization messages
        self.time_consumer_tag = await self.time_queue.consume(self.on_message)
        self.logger.info("Started consuming TIME_TICK messages.")

    async def on_message(self, message: AbstractIncomingMessage):
        """
        Callback to handle incoming messages.

        Processes both regular agent communication and time synchronization messages.
        Implements proper error handling and message acknowledgment for robust
        multi-agent coordination.

        Args:
            message: The incoming message from RabbitMQ.
        """
        async with message.process(ignore_processed=True):
            try:
                msg = json.loads(message.body.decode('utf-8'))
                msg_type = MessageType(msg.get("type"))

                if msg_type == MessageType.STOP_SIMULATION:
                    self.logger.info(f"Agent {self.agent_id} received STOP_SIMULATION. Shutting down.")
                    self.stop()
                elif msg_type == MessageType.TIME_TICK:
                    await self.process_time_tick(msg)
                else:
                    await self._handle_regular_message(msg)

            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error in message: {e}. Message: {message.body}")
            except Exception as e:
                self.logger.error(f"Error processing message: {e}. Message: {message.body}")

    @abstractmethod
    async def _handle_regular_message(self, msg: Dict[str, Any]):
        """
        Handle regular (non-time) messages from other agents.

        Must be implemented by concrete agent classes to define
        agent-specific message handling behavior.

        Args:
            msg: The decoded message dictionary.
        """
        pass

    async def reconnect(self):
        """
        Attempt to reconnect to RabbitMQ with exponential backoff.

        Ensures robust operation in distributed environments by automatically
        recovering from connection failures. Critical for long-running
        multi-agent simulations.
        """
        self.logger.info("Attempting to reconnect to RabbitMQ...")
        await self.connect()
        await self.consume()

    async def close(self):
        """
        Gracefully shut down the agent.

        Cancels consumers, allows time for pending messages to be confirmed,
        and cleanly closes the channel and connection. Ensures proper cleanup
        for research reproducibility.
        """
        # Cancel message consumers
        if self.queue and self.consumer_tag:
            await self.queue.cancel(self.consumer_tag)
            self.logger.info("Cancelled message consumption.")

        if self.time_queue and self.time_consumer_tag:
            await self.time_queue.cancel(self.time_consumer_tag)
            self.logger.info("Cancelled TIME_TICK message consumption.")

        # Allow time for pending confirmations
        await asyncio.sleep(1)
        self.logger.info("Waited briefly for pending publisher confirmations.")

        # Close connections
        if self.channel:
            await self.channel.close()
            self.logger.info("Channel closed.")

        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            self.logger.info("Closed RabbitMQ connection.")

        self.logger.info(f"Agent {self.agent_id} shut down.")

    async def run(self):
        """
        Main entry point to run the agent.

        Starts message consumption and waits for the stop event.
        Ensures proper cleanup on shutdown.
        """
        await self.consume()
        try:
            await self._stop_event.wait()
        finally:
            await self.close()

    async def initialize(self):
        """
        Initialize the agent before running.

        Establishes RabbitMQ connection and prepares the agent
        for simulation participation.
        """
        await self.connect()

    def stop(self):
        """
        Signal the agent to stop running.

        Sets shutdown flags and triggers the stop event
        for graceful agent termination.
        """
        self._shutdown = True
        self._stop_event.set()

    async def process_time_tick(self, msg: Dict[str, Any]):
        """
        Handle TIME_TICK messages for simulation synchronization.

        Delegates to the handle_time_tick method for agent-specific
        time processing logic.

        Args:
            msg: The TIME_TICK message dictionary.
        """
        payload = msg.get("payload", {})
        await self.handle_time_tick(payload)

    async def handle_time_tick(self, payload: Dict[str, Any]):
        """
        Process time tick events for simulation synchronization.

        Updates agent's current time and tick ID from the simulation coordinator.
        Critical for maintaining consistent time progression across all agents
        in deterministic simulation mode.

        Args:
            payload: Dictionary containing current_time and tick_id information.
        """
        current_time_str = payload.get("current_time")
        self.logger.info(f"Received TIME_TICK message: {current_time_str}")

        try:
            current_time = parse_datetime_utc(current_time_str)
        except ValueError:
            self.logger.error(f"Agent {self.agent_id} received invalid TIME_TICK format: {current_time_str}")
            return

        self.current_time = current_time
        self.current_tick_id = payload.get("tick_id")
