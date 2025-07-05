"""
StockSim LLM Trading Agent - core LLM trading agent with multi-instrument portfolio coordination.
Provides specialized analysts for market, news, and fundamental analysis.
"""

from __future__ import annotations

import asyncio
import os
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import jinja2

# Import specialized analyst agents
from agents.market_analyst import MarketAnalyst
from agents.news_analyst import NewsAnalyst
from agents.fundamental_analyst import FundamentalAnalyst

# Import base trader functionality
from agents.benchmark_traders.trader import TraderAgent

# Import utility modules
from utils.metrics import RationalityMetrics
from utils.indicators_tracker import IndicatorsTracker
from utils.parsing_utils import parse_model_output
from utils.messages import MessageType
from utils.orders import Side
from utils.polygon_client import PolygonClient
from utils.alpha_vantage_client import AlphaVantageClient
from utils.role import Role
from utils.time_utils import (
    parse_datetime_utc,
    seconds_to_human,
    parse_interval_to_timedelta,
)
from wrappers import LLMWrapperFactory, MultiWrapperManager


@dataclass
class DecisionRecord:
    """
    Structured record of LLM trading decisions for analysis.

    This enables systematic study of multi-agent coordination and
    decision quality tracking.
    """
    timestamp: str
    instrument: str
    decision_text: str
    parsed_actions: List[Dict[str, Any]]
    market_context: Dict[str, Any]
    analyst_inputs: Dict[str, Optional[str]]


class LLMTradingAgent(TraderAgent):
    """
    LLM Trading Agent with multi-instrument portfolio coordination and specialized analysts.
    """


    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        start_time: str,
        end_time: str,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = "localhost",

        # LLM Configuration
        models: Optional[Dict[str, Dict[str, Any]]] = None,
        use_llm_history: bool = True,

        # Analyst Feature Toggles
        enable_market_analyst: bool = True,
        enable_news_analyst: bool = True,
        enable_fundamental_analyst: bool = True,

        # Data Configuration
        extended_intervals: Optional[List[Dict[str, Any]]] = None,
        extended_warmup_candles: Optional[Dict[str, int]] = None,
        extended_indicator_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        data_source_config: Optional[Dict[str, Dict[str, str]]] = None,

        **kwargs,
    ) -> None:
        """
        Initialize LLM trading agent with multi-agent coordination capabilities.

        Args:
            instrument_exchange_map: Mapping of instruments to exchanges
            start_time: Simulation start time
            end_time: Simulation end time
            agent_id: Unique identifier for the agent
            rabbitmq_host: RabbitMQ broker host
            models: LLM model configurations for different roles
            enable_*_analyst: Toggles for specialized analyst agents
            data_source_config: Per-instrument data source configuration
            api_key_config: Environment variable names for API keys
            **kwargs: Additional configuration parameters
        """

        # Filter out LLM-specific parameters that TraderAgent doesn't accept
        llm_specific_params = {
            'use_thinking'
        }

        # Keep only parameters that TraderAgent accepts
        trader_kwargs = {k: v for k, v in kwargs.items() if k not in llm_specific_params}

        super().__init__(
            instrument_exchange_map=instrument_exchange_map,
            agent_id=agent_id,
            rabbitmq_host=rabbitmq_host,
            **trader_kwargs,
        )

        # Store LLM-specific parameters as instance attributes
        for param in llm_specific_params:
            setattr(self, param, kwargs.get(param, None))

        # Research feature configuration
        self.next_action_time = None
        self.enable_market_analyst = enable_market_analyst
        self.enable_news_analyst = enable_news_analyst
        self.enable_fundamental_analyst = enable_fundamental_analyst

        # Use the same metrics instance as TraderAgent for consistency
        self.research_metrics = self.metrics
        
        # Research data collection
        self.decision_history: List[DecisionRecord] = []
        
        self.models_config = self._process_model_configurations(models)
        
        # Create wrapper manager for modular LLM access
        self.wrapper_manager = self._create_wrapper_manager(use_llm_history)

        # Multi-instrument coordination state (set early for template initialization)
        self._multi_instrument_mode = len(self.instrument_exchange_map) > 1
        self._market_data_futures: Dict[str, asyncio.Future] = {}
        self._pending_market_snapshots: Dict[str, Dict[str, Any]] = {}

        # Initialize specialized analyst agents
        self._initialize_analysts(self.models_config)

        # Template configuration - load both single and multi-instrument templates
        self._initialize_prompt_templates()

        # Simulation parameters
        self.start_time: datetime = parse_datetime_utc(start_time)
        self.end_time: datetime = parse_datetime_utc(end_time)
        self.executed_orders: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Extended data configuration
        self.extended_intervals = extended_intervals or []
        self.extended_warmup_candles = extended_warmup_candles or {}
        self.extended_indicator_kwargs = extended_indicator_kwargs or {}
        self.data_source_config = data_source_config or {}
        self.all_candles_map: Dict[str, Dict[str, Any]] = {}

        # Analysis state - now supports multi-instrument coordination
        self.news_summary = None
        self._news_futures: Dict[str, asyncio.Future] = {}
        self._fundamentals_futures: Dict[str, asyncio.Future] = {}

        # Log data source configuration
        data_sources_info = []
        for instr in self.instrument_exchange_map.keys():
            config = self.data_source_config.get(instr, {})
            source = config.get("data_source", "polygon")
            symbol_type = config.get("symbol_type", "stock")
            data_sources_info.append(f"{instr} ({symbol_type} via {source})")
        
        self.logger.info(
            f"🧠 LLMTradingAgent {self.agent_id} initialized with features:\n"
            f"   • Market Analyst: {'✅' if self.enable_market_analyst else '❌'}\n"
            f"   • News Analyst: {'✅' if self.enable_news_analyst else '❌'}\n"
            f"   • Fundamental Analyst: {'✅' if self.enable_fundamental_analyst else '❌'}\n"
            f"   • Data Sources: {', '.join(data_sources_info)}"
        )

        # Load extended market data
        self.load_extended_data()

    def _process_model_configurations(self, models: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Process and validate model configurations.

        Args:
            models: Raw model configurations from config file

        Returns:
            Processed model configurations
        """
        if not models:
            # Provide default configuration
            default_config = {
                "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "model_type": "anthropic",
                "temperature": 0.0,
                "top_p": 0.95,
                "max_tokens": 20000,
                "budget_tokens": 16000,
                "use_thinking": True,
                "api_keys": {
                    "aws_access_key_id": "AWS_ACCESS_KEY_ID",
                    "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
                    "aws_region": "AWS_REGION"
                }
            }

            models = {
                "market_analysis": default_config.copy(),
                "news": default_config.copy(),
                "fundamental_analysis": default_config.copy(),
                "aggregator": default_config.copy()
            }

            self.logger.warning("No model configurations provided, using defaults")

        # Validate required model types
        required_models = ["aggregator"]
        for model_type in required_models:
            if model_type not in models:
                raise ValueError(f"Required model configuration missing: {model_type}")

        # Add analyst models if analysts are enabled
        if self.enable_market_analyst and "market_analysis" not in models:
            self.logger.warning("Market analyst enabled but no market_analysis model configured")

        if self.enable_news_analyst and "news" not in models:
            self.logger.warning("News analyst enabled but no news model configured")

        if self.enable_fundamental_analyst and "fundamental_analysis" not in models:
            self.logger.warning("Fundamental analyst enabled but no fundamental_analysis model configured")

        return models

    def _create_wrapper_manager(self, use_history: bool) -> MultiWrapperManager:
        """
        Create multi-wrapper manager with configured models.

        Args:
            use_history: Default history setting (fallback for models without specific setting)

        Returns:
            Configured MultiWrapperManager instance
        """
        try:
            manager = LLMWrapperFactory.create_multi_wrapper_manager(
                models_config=self.models_config,
                agent_id=self.agent_id,
                use_history=use_history,
                agent=self
            )

            self.logger.info(f"✅ Multi-wrapper manager created with models: {list(self.models_config.keys())}")
            return manager

        except Exception as e:
            self.logger.error(f"❌ Failed to create wrapper manager: {e}")
            raise

    def _initialize_analysts(self, models: Dict[str, Dict[str, Any]]) -> None:
        """Initialize specialized analyst agents using the wrapper manager."""

        # Market Analyst
        if self.enable_market_analyst and "market_analysis" in self.models_config:
            try:
                self.market_analyst = MarketAnalyst(
                    agent=self,
                    wrapper_manager=self.wrapper_manager,
                    wrapper_type="market_analysis"
                )
                self.logger.info("📊 Market Analyst: Initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Market Analyst: {e}")
                self.enable_market_analyst = False

        # News Analyst
        if self.enable_news_analyst and "news" in self.models_config:
            try:
                self.news_analyst = NewsAnalyst(
                    agent=self,
                    wrapper_manager=self.wrapper_manager,
                    wrapper_type="news"
                )
                self.logger.info("📰 News Analyst: Initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize News Analyst: {e}")
                self.enable_news_analyst = False

        # Fundamental Analyst
        if self.enable_fundamental_analyst and "fundamental_analysis" in self.models_config:
            try:
                self.fundamental_analyst = FundamentalAnalyst(
                    agent=self,
                    wrapper_manager=self.wrapper_manager,
                    wrapper_type="fundamental_analysis"
                )
                self.logger.info("📈 Fundamental Analyst: Initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Fundamental Analyst: {e}")
                self.enable_fundamental_analyst = False

    def _log_initialization_summary(self) -> None:
        """Log comprehensive initialization summary."""
        # Extract data source information
        data_sources_info = []
        for instr in self.instrument_exchange_map.keys():
            config = self.data_source_config.get(instr, {})
            source = config.get("data_source", "polygon")
            symbol_type = config.get("symbol_type", "stock")
            data_sources_info.append(f"{instr} ({symbol_type} via {source})")

        # Extract model information
        model_info = []
        for model_type, config in self.models_config.items():
            model_id = config.get("model_id", "unknown")
            provider = config.get("model_type", "unknown")
            temp = config.get("temperature", "default")
            thinking = config.get("use_thinking", False)
            thinking_str = " +thinking" if thinking else ""
            model_info.append(f"{model_type}: {provider}:{model_id} (T={temp}){thinking_str}")

        self.logger.info(
            f"🧠 LLMTradingAgent {self.agent_id} initialized with:\n"
            f"   Features:\n"
            f"     • Market Analyst: {'✅' if self.enable_market_analyst else '❌'}\n"
            f"     • News Analyst: {'✅' if self.enable_news_analyst else '❌'}\n"
            f"     • Fundamental Analyst: {'✅' if self.enable_fundamental_analyst else '❌'}\n"
            f"   Models:\n" +
            "\n".join(f"     • {info}" for info in model_info) + "\n" +
            f"   Data Sources:\n" +
            "\n".join(f"     • {info}" for info in data_sources_info)
        )

    def _initialize_prompt_templates(self) -> None:
        """Initialize Jinja2 prompt templates for both single and multi-instrument trading."""
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        tpl_dir = os.path.join(base_dir, "templates")
        
        # Load single-instrument templates (existing)
        with open(os.path.join(tpl_dir, "trading_prompt_template_full.j2"), "r") as f:
            base_full_tpl: str = f.read()

        with open(os.path.join(tpl_dir, "trading_prompt_template_delta.j2"), "r") as f:
            self._delta_tpl: str = f.read()
            
        # Load multi-instrument templates if they exist
        multi_full_path = os.path.join(tpl_dir, "trading_prompt_template_multi_full.j2")
        multi_delta_path = os.path.join(tpl_dir, "trading_prompt_template_multi_delta.j2")
        
        if os.path.exists(multi_full_path) and os.path.exists(multi_delta_path):
            with open(multi_full_path, "r") as f:
                self._multi_full_tpl: str = f.read()
            with open(multi_delta_path, "r") as f:
                self._multi_delta_tpl: str = f.read()
        else:
            # Fallback to single-instrument templates
            self._multi_full_tpl = base_full_tpl
            self._multi_delta_tpl = self._delta_tpl
            self.logger.warning("Multi-instrument templates not found, using single-instrument templates")
        
        # Initialize template state
        if self._multi_instrument_mode:
            # Multi-instrument mode: one shared template state
            self._sent_full_prompt = False
            self._active_full_tpl = self._multi_full_tpl
        else:
            # Single-instrument mode: simple variables for single instrument
            self._sent_full_prompt = False
            self._active_full_tpl = base_full_tpl

        self._initial_ctx: Dict[str, Dict[str, Any]] = {}

        # Configure Jinja2 environment
        self._jinja_env = jinja2.Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
            undefined=jinja2.StrictUndefined,
        )
        
        self.logger.info(f"📝 Templates initialized for {'multi-instrument' if self._multi_instrument_mode else 'single-instrument'} mode")

    def load_extended_data(self) -> None:
        """
        Load extended historical market data for enhanced analysis.
        
        This provides additional context for LLM decision-making and enables
        more comprehensive market analysis as described in the paper.
        """
        result = {}
        
        for instr in self.instrument_exchange_map.keys():
            result[instr] = {}
            warmup_candles = self.extended_warmup_candles.get(instr, 250)
            indicator_kwargs = self.extended_indicator_kwargs.get(instr, {})
            
            for interval_def in self.extended_intervals:
                days = interval_def["days"]
                label = interval_def["label"]
                resolution = interval_def["resolution"]

                resolution_timedelta = parse_interval_to_timedelta(resolution)
                warmup_duration = warmup_candles * resolution_timedelta
                interval_start = self.start_time - timedelta(days=days)
                raw_start = interval_start - warmup_duration
                interval_end = self.start_time - resolution_timedelta

                # Choose data source based on configuration
                instr_config = self.data_source_config.get(instr, {})
                data_source = instr_config.get("data_source", "polygon")
                symbol_type = instr_config.get("symbol_type", "stock")
                
                # Initialize appropriate data client
                if data_source == "alpha_vantage":
                    loader = AlphaVantageClient()
                else:
                    loader = PolygonClient()
                
                # Load data based on symbol type and data source
                if symbol_type == "crypto" and data_source == "alpha_vantage":
                    full_candles = loader.load_crypto_aggregates(
                        symbol=instr,
                        interval=resolution,
                        start_date=raw_start.isoformat(),
                        end_date=interval_end.isoformat(),
                        market="USD",
                        sort="asc",
                        limit=10000,
                        use_cache=True
                    )
                elif symbol_type == "crypto" and data_source == "polygon":
                    full_candles = loader.load_crypto_aggregates(
                        symbol=instr,
                        interval=resolution,
                        start_date=raw_start.isoformat(),
                        end_date=interval_end.isoformat(),
                        market="USD",
                        sort="asc",
                        limit=10000,
                        use_cache=True
                    )
                else:
                    # Stock data for both Polygon and Alpha Vantage
                    full_candles = loader.load_aggregates(
                        symbol=instr,
                        interval=resolution,
                        start_date=raw_start.isoformat(),
                        end_date=interval_end.isoformat(),
                        adjusted=True,
                        sort="asc",
                        limit=10000,
                        use_cache=True
                    )

                # Calculate technical indicators
                indicator_tracker = IndicatorsTracker(**indicator_kwargs)
                for candle in full_candles:
                    indicator_tracker.update(candle)

                # Trim to requested window
                trimmed_candles = [c for c in full_candles if c["timestamp"] >= interval_start.isoformat()]
                num_trimmed = len(trimmed_candles)

                # Process indicators
                full_indicators = indicator_tracker.get_full_history()
                time_series = full_indicators["time_series"]
                aggregate = full_indicators.get("aggregate", {})

                trimmed_time_series = {}
                for key, value in time_series.items():
                    if isinstance(value, dict):
                        trimmed_time_series[key] = {
                            subkey: subval[-num_trimmed:] if isinstance(subval, list) else subval
                            for subkey, subval in value.items()
                        }
                    elif isinstance(value, list):
                        trimmed_time_series[key] = value[-num_trimmed:]
                    else:
                        trimmed_time_series[key] = value

                result[instr][label] = {
                    "candles": trimmed_candles,
                    "indicators": {
                        "time_series": trimmed_time_series,
                        "aggregate": aggregate
                    }
                }
                
        self.all_candles_map = result
        self.logger.info(f"📚 Extended data loaded for {len(result)} instruments")

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        """
        Handle simulation time ticks and coordinate market data requests.
        
        This method orchestrates the multi-agent decision-making process
        that enables the research capabilities described in the paper.
        """
        await super().handle_time_tick(payload)

        current_time = self.current_time
        if self.next_action_time is None:
            self.next_action_time = current_time

        # Execute trading decision cycle
        if current_time >= self.next_action_time:
            if self._multi_instrument_mode:
                # Multi-instrument mode: coordinate all market data collection first
                await self.coordinate_multi_instrument_decisions()
            else:
                # Single-instrument mode: process individually
                for instrument in self.instrument_exchange_map.keys():
                    asyncio.create_task(self.update_market_snapshot(instrument))
            self.next_action_time = current_time + self.action_interval
        else:
            # Signal completion to simulation clock
            await self.publish_time(
                msg_type=MessageType.DECISION_RESPONSE,
                payload={"tick_id": self.current_tick_id},
                routing_key="simulation_clock"
            )
            return
    
    async def coordinate_multi_instrument_decisions(self) -> None:
        """
        Coordinate trading decisions across all instruments in multi-instrument mode.
        
        This method ensures all market data is collected before making decisions,
        enabling comprehensive portfolio management across multiple instruments.
        """
        instruments = list(self.instrument_exchange_map.keys())
        
        # Clear pending snapshots
        self._pending_market_snapshots.clear()
        
        # Initialize futures for market data collection
        self._market_data_futures = {
            instrument: asyncio.get_event_loop().create_future()
            for instrument in instruments
        }
        
        self.logger.info(f"📊 Initiating multi-instrument coordination for {len(instruments)} instruments: {', '.join(instruments)}")
        
        # Request market data for all instruments in parallel
        market_data_tasks = []
        for instrument in instruments:
            task = asyncio.create_task(self.update_market_snapshot(instrument))
            market_data_tasks.append(task)
        
        # Wait for all market data requests to be sent
        await asyncio.gather(*market_data_tasks)
        
        # Wait for all market data responses to arrive
        try:
            market_snapshots = await asyncio.gather(*self._market_data_futures.values(), return_exceptions=True)
            
            self.logger.info(f"✅ Collected market data for {len(market_snapshots)} instruments")
            
            # Process all decisions with complete market context
            await self.make_multi_instrument_decision()
            
        except Exception as e:
            self.logger.error(f"❌ Error in multi-instrument coordination: {e}")
            # Fallback to individual processing
            for instrument in instruments:
                if instrument in self._pending_market_snapshots:
                    await self.on_market_data_update(instrument, self._pending_market_snapshots[instrument])
        
        # Signal completion to simulation clock
        await self.publish_time(
            msg_type=MessageType.DECISION_RESPONSE,
            payload={"tick_id": self.current_tick_id},
            routing_key="simulation_clock"
        )
    
    async def make_multi_instrument_decision(self) -> None:
        """
        Make trading decisions considering all instruments simultaneously.
        
        This enables portfolio-level optimization and cross-instrument analysis.
        """
        instruments = list(self.instrument_exchange_map.keys())
        
        # Collect all analyst data and market snapshots
        all_market_data = {}
        all_analysts = {}
        all_executed_orders = {}
        
        # Collect market snapshots and executed orders for all instruments
        for instrument in instruments:
            snapshot = self._pending_market_snapshots.get(instrument, {})
            all_market_data[instrument] = snapshot
            all_executed_orders[instrument] = self.executed_orders.pop(instrument, [])
        
        # Prepare data structures for multi-instrument analyst calls
        instruments_market_data = {}
        instruments_news_data = {}
        instruments_fundamentals_data = {}
        
        # Collect news and fundamental data from futures
        for instrument in instruments:
            snapshot = all_market_data[instrument]
            
            # Prepare market data for multi-instrument analysis
            if snapshot.get("data"):
                instruments_market_data[instrument] = {
                    "open_price": snapshot["data"].get("open"),
                    "high_price": snapshot["data"].get("high"),
                    "low_price": snapshot["data"].get("low"),
                    "close_price": snapshot["data"].get("close"),
                    "volume": snapshot["data"].get("volume"),
                    "indicators": snapshot.get("indicators", {}),
                    "vwap": snapshot["data"].get("vwap"),
                    "transactions": snapshot["data"].get("transactions", 0)
                }
            
            # Collect news data if available
            if self.enable_news_analyst and instrument in self._news_futures:
                try:
                    news_data = await self._news_futures[instrument]
                    if news_data:
                        instruments_news_data[instrument] = news_data
                    self._news_futures.pop(instrument, None)
                except Exception as e:
                    self.logger.error(f"❌ News data collection error for {instrument}: {e}")
                    self._news_futures.pop(instrument, None)
            
            # Collect fundamental data if available
            if self.enable_fundamental_analyst and instrument in self._fundamentals_futures:
                try:
                    fundamental_data = await self._fundamentals_futures[instrument]
                    if fundamental_data:
                        instruments_fundamentals_data[instrument] = fundamental_data
                    self._fundamentals_futures.pop(instrument, None)
                except Exception as e:
                    self.logger.error(f"❌ Fundamental data collection error for {instrument}: {e}")
                    self._fundamentals_futures.pop(instrument, None)
        
        # Now call multi-instrument analyst methods in parallel
        analyst_tasks = []
        
        # Create tasks for enabled analysts
        if self.enable_market_analyst and instruments_market_data:
            task = asyncio.create_task(
                self.market_analyst.get_multi_instrument_market_analysis(instruments_market_data)
            )
            analyst_tasks.append(("market", task))
        
        if self.enable_news_analyst and instruments_news_data:
            task = asyncio.create_task(
                self.news_analyst.get_multi_instrument_news_analysis(instruments_news_data)
            )
            analyst_tasks.append(("news", task))
        
        if self.enable_fundamental_analyst and instruments_fundamentals_data:
            task = asyncio.create_task(
                self.fundamental_analyst.get_multi_instrument_fundamental_analysis(instruments_fundamentals_data)
            )
            analyst_tasks.append(("fundamental", task))
        
        # Execute all analyst tasks in parallel
        market_analysis = None
        news_analysis = None
        fundamentals_analysis = None
        
        if analyst_tasks:
            self.logger.info(f"🔄 Running {len(analyst_tasks)} multi-instrument analysts in parallel")
            
            # Wait for all tasks to complete
            for analyst_type, task in analyst_tasks:
                try:
                    result = await task
                    if analyst_type == "market":
                        market_analysis = result
                    elif analyst_type == "news":
                        news_analysis = result
                    elif analyst_type == "fundamental":
                        fundamentals_analysis = result
                except Exception as e:
                    self.logger.error(f"❌ Multi-instrument {analyst_type} analysis failed: {e}")
        
        # Create unified analyst results structure
        all_analysts = {
            "multi_instrument": {
                "market": market_analysis,
                "news": news_analysis,
                "fund": fundamentals_analysis
            }
        }
        
        # Build comprehensive context for multi-instrument decision
        ctx = self._build_multi_instrument_context(all_market_data, all_analysts, all_executed_orders)
        
        # Render prompt using multi-instrument template
        template_src = self._multi_full_tpl if not self._sent_full_prompt else self._multi_delta_tpl
        template = self._jinja_env.from_string(template_src)
        rendered = template.render(**ctx)
        
        # Log decision context
        self.logger.info(f"🧠 Generating multi-instrument decision for {len(instruments)} instruments")
        
        # Generate LLM decision
        if not self._sent_full_prompt:
            self._sent_full_prompt = True
        
        raw = await self._generate_final_decision(rendered)
        
        if raw:
            # Parse and execute decisions for all instruments
            await self._process_multi_instrument_response(raw, instruments, market_analysis, news_analysis, fundamentals_analysis)
        else:
            self.logger.warning("⚠️ No multi-instrument decision generated")
    
    def _build_multi_instrument_context(self, market_data: Dict[str, Dict], analysts: Dict[str, Dict], executed_orders: Dict[str, List]) -> Dict[str, Any]:
        """
        Build context for multi-instrument trading decisions.
        
        Args:
            market_data: Market snapshots for all instruments
            analysts: Analyst outputs dictionary with multi-instrument structure
            executed_orders: Recent executed orders for all instruments
        
        Returns:
            Context dictionary for template rendering
        """
        instruments_context = []
        
        # Extract multi-instrument analyst results
        multi_analysts = analysts.get("multi_instrument", {})
        
        for instrument in market_data.keys():
            snapshot = market_data[instrument]
            data = snapshot.get("data", {}) or {}
            executed_block = executed_orders[instrument]
            
            executed_lines = (
                "\n".join(
                    f"- {o['action']} {o['quantity']} @ ${o['price']:.2f} ({o['orderType']})"
                    for o in executed_block
                )
                if executed_block
                else "None"
            )
            
            long_sh = self.long_qty[instrument]
            short_sh = self.short_qty[instrument]
            
            instrument_ctx = {
                "instrument": instrument,
                "market_open": bool(data),
                "open": data.get("open"),
                "high": data.get("high"),
                "low": data.get("low"),
                "close": data.get("close"),
                "volume": data.get("volume"),
                "shares_long": long_sh,
                "shares_short": short_sh,
                "shares_net": long_sh - short_sh,
                "executed_orders": executed_lines,
                "market_analysis": multi_analysts.get("market") if self.enable_market_analyst else None,
                "news_analysis": multi_analysts.get("news") if self.enable_news_analyst else None,
                "fund_analysis": multi_analysts.get("fund") if self.enable_fundamental_analyst else None,
            }
            
            instruments_context.append(instrument_ctx)
        
        return {
            "instruments": instruments_context,
            "window_start": self.start_time.isoformat(),
            "window_end": self.end_time.isoformat(),
            "action_interval": seconds_to_human(int(self.action_interval.total_seconds())),
            "now": self.current_time.isoformat(),
            "portfolio_cash": f"{self.cash:.2f}",
            "total_instruments": len(instruments_context),
        }
    
    async def _process_multi_instrument_response(self, response: str, instruments: List[str], 
                                               market_analysis: Optional[str] = None, 
                                               news_analysis: Optional[str] = None, 
                                               fundamentals_analysis: Optional[str] = None) -> None:
        """
        Process LLM response for multi-instrument decisions.
        
        Args:
            response: Raw LLM response
            instruments: List of instruments being traded
            market_analysis: Multi-instrument market analysis result
            news_analysis: Multi-instrument news analysis result
            fundamentals_analysis: Multi-instrument fundamental analysis result
        """
        try:
            # Parse the multi-instrument response
            parsed_actions = parse_model_output(response)
            
            if not parsed_actions:
                self.logger.info("📝 Multi-instrument decision: No actions recommended")
                return
            
            # Record decision for research - create records for each instrument
            for instrument in instruments:
                # Get market context for this instrument
                market_snapshot = self._pending_market_snapshots.get(instrument, {})
                market_data = market_snapshot.get("data", {}) or {}
                
                market_context = {
                    "open": market_data.get("open"),
                    "high": market_data.get("high"),
                    "low": market_data.get("low"),
                    "close": market_data.get("close"),
                    "volume": market_data.get("volume"),
                    "market_open": bool(market_data)
                }
                
                # Extract analyst inputs from the multi-instrument results
                analyst_inputs = {
                    "market": market_analysis if self.enable_market_analyst else None,
                    "news": news_analysis if self.enable_news_analyst else None,
                    "fund": fundamentals_analysis if self.enable_fundamental_analyst else None,
                }
                
                # Filter actions for this specific instrument
                instrument_actions = [
                    action for action in parsed_actions 
                    if action.get("instrument") == instrument
                ]
                
                decision_record = DecisionRecord(
                    timestamp=self.current_time.isoformat(),
                    instrument=instrument,
                    decision_text=response,  # Full multi-instrument response
                    parsed_actions=instrument_actions,  # Only actions for this instrument
                    market_context=market_context,
                    analyst_inputs=analyst_inputs
                )
                self.decision_history.append(decision_record)
            
            # Execute actions for each instrument
            instrument_actions = {}
            for action in parsed_actions:
                instrument = action.get("instrument")
                if instrument and instrument in instruments:
                    if instrument not in instrument_actions:
                        instrument_actions[instrument] = []
                    instrument_actions[instrument].append(action)
                else:
                    self.logger.warning(f"⚠️ Action without valid instrument: {action}")
            
            # Execute actions for each instrument
            execution_tasks = []
            for instrument, actions in instrument_actions.items():
                task = asyncio.create_task(self._execute_instrument_actions(instrument, actions))
                execution_tasks.append(task)
            
            if execution_tasks:
                await asyncio.gather(*execution_tasks, return_exceptions=True)
                self.logger.info(f"✅ Executed multi-instrument decisions for {len(instrument_actions)} instruments")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process multi-instrument response: {e}")
    
    async def _execute_instrument_actions(self, instrument: str, actions: List[Dict[str, Any]]) -> None:
        """
        Execute trading actions for a specific instrument.
        
        Args:
            instrument: Instrument symbol
            actions: List of trading actions for this instrument
        """
        for action in actions:
            try:
                await self.process_decision(
                    instrument=instrument,
                    action=action.get("action", "HOLD").upper(),
                    order_type=(action.get("orderType") or "LIMIT").upper(),
                    price=action.get("price"),
                    quantity=action.get("quantity"),
                    oco_group=action.get("ocoGroup"),
                    explanation=action.get("explanation")
                )
                
            except Exception as e:
                self.logger.error(f"❌ Failed to execute action for {instrument}: {e}")

    async def update_market_snapshot(self, instrument: str) -> None:
        """Request market data snapshots for multi-modal analysis."""
        if not self.current_time:
            return
            
        window_end = self.current_time
        
        # Determine analysis window based on agent state
        if not (hasattr(self, 'market_analyst') and 
                self.market_analyst.has_sent_first_prompt.get(instrument, False)):
            window_start = window_end - timedelta(days=30)
        else:
            window_start = window_end - self.action_interval

        # Configure fundamentals window based on data source and symbol type
        fundamentals_window_start = None
        should_request_fundamentals = False

        if hasattr(self, 'fundamental_analyst') and self.enable_fundamental_analyst:
            # Get data source configuration for this instrument
            instr_config = self.data_source_config.get(instrument, {})
            data_source = instr_config.get("data_source", "polygon")
            symbol_type = instr_config.get("symbol_type", "stock")

            # Only request fundamentals for Polygon stocks
            if symbol_type == "stock" and data_source == "polygon":
                should_request_fundamentals = True

                # Set window start only if first prompt has been sent
                if self.fundamental_analyst.has_sent_first_prompt[instrument]:
                    fundamentals_window_start = window_start
                # For first request, leave fundamentals_window_start as None

                self.logger.debug(f"📊 Requesting Polygon fundamentals for {instrument} ({symbol_type}) - window_start: {fundamentals_window_start}")
            else:
                # Skip fundamentals for Alpha Vantage or crypto
                self.logger.debug(f"📊 Skipping fundamentals for {data_source} {instrument} ({symbol_type}) - not supported")

        exchange_id = self._get_exchange_for_instrument(instrument)
        if exchange_id is None:
            self.logger.error(f"❌ No exchange found for {instrument}")
            return

        payload = {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat()
        }

        # Prepare futures for async coordination only when analysts are enabled
        if self.enable_news_analyst:
            self._news_futures[instrument] = asyncio.get_event_loop().create_future()
        if self.enable_fundamental_analyst and should_request_fundamentals:
            self._fundamentals_futures[instrument] = asyncio.get_event_loop().create_future()

        # Request data types based on enabled analysts and data source capabilities
        requests = [
            self.send_message(exchange_id, MessageType.MARKET_DATA_SNAPSHOT_REQUEST, payload)
        ]

        if self.enable_news_analyst:
            requests.append(self.send_message(exchange_id, MessageType.NEWS_SNAPSHOT_REQUEST, payload))

        if self.enable_fundamental_analyst and should_request_fundamentals:
            requests.append(self.send_message(exchange_id, MessageType.FUNDAMENTALS_REQUEST, {
                "window_start": fundamentals_window_start.isoformat() if fundamentals_window_start else None,
                "window_end": window_end.isoformat()
            }))

        # Request all data types in parallel
        await asyncio.gather(*requests)

    async def _handle_news_snapshot_response(self, payload: Dict[str, Any]) -> None:
        """Process news data for sentiment analysis."""
        instrument = payload["instrument"]
        news = payload.get("news", [])

        if not news or not self.enable_news_analyst:
            if instrument in self._news_futures and not self._news_futures[instrument].done():
                self._news_futures[instrument].set_result(None)
            return

        try:
            if self._multi_instrument_mode:
                # Multi-instrument mode: just store the raw data, analysis will be done later
                if instrument in self._news_futures and not self._news_futures[instrument].done():
                    self._news_futures[instrument].set_result(news)
            else:
                # Single-instrument mode: analyze immediately
                summary = await self.news_analyst.get_news_analysis(instrument, news)
                if instrument in self._news_futures and not self._news_futures[instrument].done():
                    self._news_futures[instrument].set_result(summary)
        except Exception as e:
            self.logger.error(f"❌ News analysis failed for {instrument}: {e}")
            if instrument in self._news_futures and not self._news_futures[instrument].done():
                self._news_futures[instrument].set_result(None)

    async def _handle_fundamentals_response(self, payload: Dict[str, Any]) -> None:
        """Process fundamental data for financial analysis."""
        instrument = payload["instrument"]
        fundamentals = payload.get("fundamentals", {})

        if not self.enable_fundamental_analyst:
            # Only set result if future exists
            if instrument in self._fundamentals_futures and not self._fundamentals_futures[instrument].done():
                self._fundamentals_futures[instrument].set_result(None)
            return

        # Check for meaningful fundamental data
        has_data = False
        if fundamentals:
            ipos = fundamentals.get("ipos", [])
            splits = fundamentals.get("splits", [])
            dividends = fundamentals.get("dividends", [])
            ticker_events = fundamentals.get("ticker_events", {}).get("events", [])
            financials = fundamentals.get("financials", [])
            has_data = bool(ipos or splits or dividends or ticker_events or financials)

        if not has_data:
            self.logger.info(f"📊 No fundamentals data available for {instrument}")
            if instrument in self._fundamentals_futures and not self._fundamentals_futures[instrument].done():
                self._fundamentals_futures[instrument].set_result(None)
            return

        try:
            if self._multi_instrument_mode:
                # Multi-instrument mode: just store the raw data, analysis will be done later
                if instrument in self._fundamentals_futures and not self._fundamentals_futures[instrument].done():
                    self._fundamentals_futures[instrument].set_result(fundamentals)
            else:
                # Single-instrument mode: analyze immediately
                summary = await self.fundamental_analyst.get_fundamental_analysis(instrument, fundamentals)
                if instrument in self._fundamentals_futures and not self._fundamentals_futures[instrument].done():
                    self._fundamentals_futures[instrument].set_result(summary)
        except Exception as e:
            self.logger.error(f"❌ Fundamental analysis failed for {instrument}: {e}")
            if instrument in self._fundamentals_futures and not self._fundamentals_futures[instrument].done():
                self._fundamentals_futures[instrument].set_result(None)

    def _render_prompt(self, instrument: str, ctx: Dict[str, Any]) -> str:
        """Render trading prompt using Jinja2 templates."""
        try:
            if self._multi_instrument_mode:
                # Multi-instrument mode: this method shouldn't be called, but handle gracefully
                self.logger.warning(f"⚠️ _render_prompt called in multi-instrument mode for {instrument} - using fallback")
                template_src = self._active_full_tpl if not self._sent_full_prompt else self._multi_delta_tpl
            else:
                # Single-instrument mode: simple template state (no dictionaries needed)
                template_src = self._active_full_tpl if not self._sent_full_prompt else self._delta_tpl
            
            template = self._jinja_env.from_string(template_src)
            return template.render(**ctx)
        except Exception as exc:
            self.logger.error(f"❌ Template render failed for {instrument}: {exc}")
            self.logger.debug(f"Multi-instrument mode: {self._multi_instrument_mode}")
            self.logger.debug(f"Active template type: {type(self._active_full_tpl)}")
            self.logger.debug(f"Sent full prompt type: {type(self._sent_full_prompt)}")
            self.logger.debug(f"Context keys: {list(ctx.keys())}")
            raise

    def get_performance_metrics(self, risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """Get performance metrics for analysis."""
        return self.research_metrics.compute_all_metrics(risk_free_rate)

    def _ctx_for_tick(
        self,
        instrument: str,
        snapshot: Dict[str, Any],
        analysts: Dict[str, Optional[str]],
        executed_block: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build context for prompt rendering with multi-agent analysis inputs."""

        data = snapshot.get("data", {}) or {}
        executed_lines = (
            "\n".join(
                f"- {o['action']} {o['quantity']} @ ${o['price']:.2f} ({o['orderType']})"
                for o in executed_block
            )
            if executed_block
            else "None"
        )

        long_sh = self.long_qty[instrument]
        short_sh = self.short_qty[instrument]

        # Context with multi-agent analysis inputs
        return {
            "instrument": instrument,
            "window_start": self.start_time.isoformat(),
            "window_end": self.end_time.isoformat(),
            "action_interval": seconds_to_human(int(self.action_interval.total_seconds())),
            "now": self.current_time.isoformat(),

            # Multi-agent analysis inputs
            "market_analysis": analysts.get("market") if self.enable_market_analyst else None,
            "news_analysis": analysts.get("news") if self.enable_news_analyst else None,
            "fund_analysis": analysts.get("fund") if self.enable_fundamental_analyst else None,

            # Market data
            "open": data.get("open"),
            "high": data.get("high"),
            "low": data.get("low"),
            "close": data.get("close"),
            "volume": data.get("volume"),

            # Portfolio state
            "shares_long": long_sh,
            "shares_short": short_sh,
            "shares_net": long_sh - short_sh,
            "portfolio_cash": f"{self.cash:.2f}",
            "executed_orders": executed_lines,

            # Market state
            "market_open": bool(data),
            "market_data_note": (
                "Current interval market data (available AFTER agent's decision)"
                if bool(data)
                else "Market closed - no new trading data available"
            )
        }

    async def _generate_final_decision(self, prompt: str) -> Optional[str]:
        """
        Generate final trading decision using the aggregator wrapper.

        Args:
            prompt: Trading decision prompt

        Returns:
            Generated decision text or None if failed
        """
        try:
            # Log the complete decision-making prompt
            self.logger.debug(f"🤖 FINAL DECISION PROMPT:\n" + "="*100 + f"\n{prompt}\n" + "="*100)

            response = await self.wrapper_manager.generate_with_wrapper(
                "aggregator",
                prompt
            )

            # Log the complete response
            if response:
                self.logger.debug(f"🤖 FINAL DECISION RESPONSE:\n" + "="*100 + f"\n{response}\n" + "="*100)
                self.logger.info(f"🤖 Generated final decision - Response length: {len(response)} chars")
            else:
                self.logger.warning("🤖 No response received from aggregator wrapper")

            return response
        except Exception as e:
            self.logger.error(f"❌ Final decision generation failed: {e}")
            return None

    async def on_market_data_update(self, instrument: str, snapshot: Dict[str, Any]) -> None:
        """
        Main decision-making coordination method implementing multi-agent capabilities.

        This method orchestrates the specialized analysts and demonstrates the
        multi-agent coordination described in the EMNLP paper.
        
        In multi-instrument mode, this stores data and signals completion.
        In single-instrument mode, this processes the decision immediately.
        """
        
        # In multi-instrument mode, store the snapshot and signal completion
        if self._multi_instrument_mode:
            self._pending_market_snapshots[instrument] = snapshot
            
            # Signal that market data for this instrument is ready
            if instrument in self._market_data_futures and not self._market_data_futures[instrument].done():
                self._market_data_futures[instrument].set_result(snapshot)
            
            return
        
        # Single-instrument mode: process immediately (existing behavior)

        data = snapshot.get("data", {})
        self.logger.debug(f"Data received: {data}")
        self.logger.debug(f"Indicators received: {snapshot.get('indicators', {})}")

        # Market analysis (if market is open and analyst enabled)
        market_task = None
        if data and self.enable_market_analyst:
            self.logger.debug(f"📊 Market data received for {instrument}: {data}")
            market_task = asyncio.create_task(
                self.market_analyst.get_market_analysis(
                    instrument,
                    data.get("open"),
                    data.get("high"),
                    data.get("low"),
                    data.get("close"),
                    data.get("volume"),
                    snapshot.get("indicators", {}),
                    data.get("vwap", None),
                    data.get("transactions")
                )
            )
        else:
            if not data:
                self.logger.info(f"🔒 Market CLOSED for {instrument} - proceeding with available analysis")

        # Execute market analysis task
        market_text = None
        if market_task is not None:
            market_text = await market_task

        # Wait for news and fundamentals analysis only if analysts are enabled and futures exist
        news_output = None
        if self.enable_news_analyst and instrument in self._news_futures:
            news_output = await self._news_futures[instrument]
            self._news_futures.pop(instrument, None)

        fundamentals_output = None
        if self.enable_fundamental_analyst and instrument in self._fundamentals_futures:
            fundamentals_output = await self._fundamentals_futures[instrument]
            self._fundamentals_futures.pop(instrument, None)

        # Collect executed orders
        executed_block = self.executed_orders.pop(instrument, [])

        # Consolidate analyst results
        analysts = {
            "market": market_text,
            "news": news_output,
            "fund": fundamentals_output
        }

        # Check if we have any meaningful data to make a decision
        has_analyst_data = any([
            market_text,  # Market analysis available
            news_output,  # News analysis available
            fundamentals_output  # Fundamental analysis available
        ])

        # Skip LLM generation if no market data and no analyst insights
        if not has_analyst_data:
            self.logger.info(f"⏭️ Skipping LLM generation for {instrument} - no market data or analyst insights available")

            # Signal completion to simulation clock
            await self.publish_time(
                msg_type=MessageType.DECISION_RESPONSE,
                payload={"tick_id": self.current_tick_id},
                routing_key="simulation_clock"
            )
            return

        # Generate trading decision prompt
        ctx = self._ctx_for_tick(instrument, snapshot, analysts, executed_block)
        if instrument not in self._initial_ctx:
            self._initial_ctx[instrument] = ctx

        rendered = self._render_prompt(instrument, ctx)

        # Log decision context
        market_status = "OPEN" if data else "CLOSED"
        analyst_status = []
        if market_text:
            analyst_status.append("Market ✅")
        if news_output:
            analyst_status.append("News ✅")
        if fundamentals_output:
            analyst_status.append("Fund ✅")

        analyst_summary = f" | Analysts: {', '.join(analyst_status) if analyst_status else 'None'}"
        self.logger.info(f"🧠 Generating decision for {instrument} - Market: {market_status}{analyst_summary}")

        if data:
            ohlcv = f"OHLCV({data.get('open')}, {data.get('high')}, {data.get('low')}, {data.get('close')}, {data.get('volume')})"
            self.logger.info(f"📊 Market data: {ohlcv}")

        # Generate LLM decision
        if not self._sent_full_prompt:
            self._sent_full_prompt = True

        raw = await self._generate_final_decision(rendered)

        if not raw:
            self.logger.error(f"❌ Failed to generate decision for {instrument}")
            await self.publish_time(
                msg_type=MessageType.DECISION_RESPONSE,
                payload={"tick_id": self.current_tick_id},
                routing_key="simulation_clock"
            )
            return

        # Process and execute decisions
        try:
            decisions = parse_model_output(raw)

            # Record decision for research
            self._record_decision_for_research(instrument, raw, decisions, ctx, analysts)

            # Execute trading decisions
            for d in decisions:
                await self.process_decision(
                    instrument,
                    d.get("action", "HOLD").upper(),
                    (d.get("orderType") or "LIMIT").upper(),
                    d.get("price"),
                    d.get("quantity"),
                    oco_group=d.get("ocoGroup"),
                    explanation=d.get("explanation")
                )

        except Exception as exc:
            self.logger.error(f"❌ Decision parsing error: {exc}\nRaw output: {raw}")

        finally:
            # Log completion
            word_count = len(raw.split())
            self.logger.info(f"✅ Decision cycle completed for {instrument} - Market: {market_status}, Response: {word_count} words")

            # Signal completion to simulation clock
            await self.publish_time(
                msg_type=MessageType.DECISION_RESPONSE,
                payload={"tick_id": self.current_tick_id},
                routing_key="simulation_clock"
            )

    def _record_decision_for_research(self, instrument: str, decision_text: str,
                                    parsed_actions: List[Dict[str, Any]],
                                    context: Dict[str, Any],
                                    analysts: Dict[str, Optional[str]]) -> None:
        """Record decision data for research analysis."""

        # Extract market context
        market_context = {
            "open": context.get("open"),
            "high": context.get("high"),
            "low": context.get("low"),
            "close": context.get("close"),
            "volume": context.get("volume"),
            "market_open": context.get("market_open")
        }

        # Create structured decision record
        decision_record = DecisionRecord(
            timestamp=self.current_time.isoformat(),
            instrument=instrument,
            decision_text=decision_text,
            parsed_actions=parsed_actions,
            market_context=market_context,
            analyst_inputs=analysts
        )

        # Store for research analysis
        self.decision_history.append(decision_record)

    async def process_decision(
        self,
        instrument: str,
        action: str,
        order_type: str,
        price: Optional[float],
        quantity: int,
        oco_group: Optional[str] = None,
        explanation: Optional[str] = None
    ) -> None:
        """Process and execute trading decisions."""

        # Execute trading actions
        if action == "BUY":
            order_id = await self.place_order(instrument, Side.BUY.value, quantity, order_type, price, oco_group, explanation)
            if order_id:
                self.logger.info(f"💰 BUY Order Placed: {instrument} qty={quantity}, order_id={order_id}")

        elif action == "SELL":
            order_id = await self.place_order(
                instrument, Side.SELL.value, quantity, order_type, price, oco_group, explanation, is_short=False
            )
            if order_id:
                self.logger.info(f"💸 SELL Order Placed: {instrument} qty={quantity}, order_id={order_id}")

        elif action == "SHORT":
            order_id = await self.place_order(
                instrument, Side.SELL.value, quantity, order_type, price, oco_group, explanation, is_short=True
            )
            if order_id:
                self.logger.info(f"📉 SHORT Order Placed: {instrument} qty={quantity}, order_id={order_id}")

        elif action == "SHORT_COVER":
            order_id = await self.place_order(
                instrument, Side.BUY.value, quantity, order_type, price, oco_group, explanation, is_short_cover=True
            )
            if order_id:
                self.logger.info(f"📈 SHORT_COVER Order Placed: {instrument} qty={quantity}, order_id={order_id}")

        elif action == "HOLD":
            self.logger.info(f"⏸️ HOLD decision: {instrument} - no trade executed")
            return
        else:
            self.logger.warning(f"❓ Unknown action '{action}' for {instrument} - ignoring")
            return

    async def on_trade_execution(self, trade_info: Dict[str, Any]) -> None:
        """Handle trade execution with research metrics tracking."""

        sym = trade_info["instrument"]
        req_qty = int(trade_info["quantity"])
        px = float(trade_info["price"])
        role = trade_info["role"]
        otype = trade_info.get("order_type")
        is_short = trade_info.get("is_short", False)
        is_short_cover = trade_info.get("is_short_cover", False)

        # Calculate actual execution quantity based on holdings
        if role == Role.BUYER.value and is_short_cover:
            held_short = self.short_qty[sym]
            exec_qty = min(req_qty, held_short)
            if exec_qty == 0:
                return
        elif role == Role.SELLER.value and not is_short:
            held_long = self.long_qty[sym]
            exec_qty = min(req_qty, held_long)
            if exec_qty == 0:
                return
        else:
            exec_qty = req_qty

        # Record execution for research
        execution_record = {
            "action": "SHORT_COVER" if is_short_cover else
                     "BUY" if role == Role.BUYER.value else
                     "SHORT" if is_short else "SELL",
            "price": px,
            "quantity": exec_qty,
            "orderType": otype
        }

        self.executed_orders[sym].append(execution_record)

        await super().on_trade_execution(trade_info)

    def get_research_summary(self) -> Dict[str, Any]:
        """
        Generate research summary for multi-agent coordination analysis.

        This provides data for evaluating the multi-agent capabilities described in the paper.
        """

        summary = {
            "agent_id": self.agent_id,
            "simulation_period": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
                "current": self.current_time.isoformat() if self.current_time else None
            },
            "configuration": {
                "market_analyst_enabled": self.enable_market_analyst,
                "news_analyst_enabled": self.enable_news_analyst,
                "fundamental_analyst_enabled": self.enable_fundamental_analyst
            },
            "performance_metrics": self.get_performance_metrics(),
            "decision_statistics": {
                "total_decisions": len(self.decision_history),
                "instruments_traded": list(self.instrument_exchange_map.keys())
            }
        }

        # Analyze multi-agent coordination patterns
        if self.decision_history:
            coordination_stats = {
                "decisions_with_market_analysis": 0,
                "decisions_with_news_analysis": 0,
                "decisions_with_fundamental_analysis": 0,
                "decisions_with_all_analysts": 0
            }

            for record in self.decision_history:
                inputs = record.analyst_inputs
                if inputs.get("market"):
                    coordination_stats["decisions_with_market_analysis"] += 1
                if inputs.get("news"):
                    coordination_stats["decisions_with_news_analysis"] += 1
                if inputs.get("fund"):
                    coordination_stats["decisions_with_fundamental_analysis"] += 1
                if all(inputs.get(key) for key in ["market", "news", "fund"]):
                    coordination_stats["decisions_with_all_analysts"] += 1

            summary["multi_agent_coordination"] = coordination_stats

        return summary

    async def save_research_data(self, output_dir: str = "research_data") -> None:
        """Save research data for post-simulation analysis."""

        os.makedirs(output_dir, exist_ok=True)

        # Save decision history
        decision_data = [asdict(record) for record in self.decision_history]
        with open(os.path.join(output_dir, f"decisions_{self.agent_id}.json"), "w") as f:
            json.dump(decision_data, f, indent=2)

        # Save research summary
        summary = self.get_research_summary()
        with open(os.path.join(output_dir, f"research_summary_{self.agent_id}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Save performance metrics
        metrics = self.get_performance_metrics()
        with open(os.path.join(output_dir, f"performance_metrics_{self.agent_id}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"💾 Research data saved to {output_dir}")

