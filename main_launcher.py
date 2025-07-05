"""
StockSim Main Launcher

This module serves as the entry point for the StockSim simulation platform. It orchestrates
the initialization and execution of all system components including exchange agents,
trading agents, and the simulation clock based on YAML configuration files.

Key Features:
- Multi-instrument trading simulation with historical candle data
- Dual-market support: stocks (via Polygon.io or Alpha Vantage) and cryptocurrency 
- Multi-agent coordination with LLM and traditional algorithmic traders
- Asynchronous RabbitMQ-based communication between components
- YAML-based configuration for zero-code experiment setup
- Multi-instrument analyst coordination for portfolio-level decision making
- Comprehensive validation and error handling for production environments
- Automatic chart and report generation post-simulation
- Performance metrics collection and analysis
"""
import logging
import os
import json
import time
import random
import sys
import signal
import traceback
from datetime import datetime

import yaml
import asyncio
from multiprocessing import Process
from typing import Dict, Any, List

from agents.benchmark_traders.historical_order_trader import HistoricalOrderTrader
from exchanges.candle_based_exchange_agent import CandleBasedExchangeAgent
from exchanges.exchange_agent import ExchangeAgent
from agents.llm_agent import LLMTradingAgent
from agents.benchmark_traders.buy_and_hold_trader import BuyAndHoldTrader
from agents.benchmark_traders.sma_trader import SMATrader
from agents.benchmark_traders.macd_trader import MACDTrader
from agents.benchmark_traders.random_trader import RandomTrader
from agents.benchmark_traders.bollinger_bands_trader import BollingerBandsTrader
from agents.benchmark_traders.slma_trader import SLMATrader
from simulation.simulation_clock import SimulationClock
from utils.logging_setup import setup_logger
from utils.time_utils import parse_datetime_utc, interval_to_seconds
from dotenv import load_dotenv
load_dotenv()

AGENT_TYPE_MAPPING = {
    "LLMTradingAgent": LLMTradingAgent,
    "Buy_And_Hold_Trader": BuyAndHoldTrader,
    "SMA_Trader": SMATrader,
    "MACD_Trader": MACDTrader,
    "Random_Trader": RandomTrader,
    "Bollinger_Bands_Trader": BollingerBandsTrader,
    "SLMA_Trader": SLMATrader,
    "HistoricalOrderTrader": HistoricalOrderTrader
}

log_dir = os.getenv("LOG_DIR", "logs")
launcher_logger = setup_logger("Launcher", os.path.join(log_dir, "launcher.log"))

def load_json_file(file_path: str) -> Any:
    """Load JSON file with error handling."""
    if not file_path:
        return {}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file '{file_path}' does not exist.")
    with open(file_path, 'r') as file:
        return json.load(file)

def agent_runner(agent_class, parameters):
    """Run trading agent in async context."""
    async def async_agent_runner():
        agent = agent_class(**parameters)
        await agent.initialize()
        await agent.run()
    asyncio.run(async_agent_runner())

def exchange_agent_runner(agent_class, parameters):
    """Run exchange agent in async context."""
    async def async_exchange_agent_runner():
        agent = agent_class(**parameters)
        await agent.initialize()
        await agent.run()
    asyncio.run(async_exchange_agent_runner())

def simulation_clock_runner(simulation_config, rabbitmq_host, expected_responses):
    """Run simulation clock with proper time management."""
    start_time_str = simulation_config["start_time"]
    end_time_str = simulation_config["end_time"]
    tick_interval_raw = simulation_config.get("tick_interval", "1d")
    if isinstance(tick_interval_raw, str):
        tick_interval_seconds = interval_to_seconds(tick_interval_raw)
    else:
        tick_interval_seconds = tick_interval_raw
    expected_exchange_agent_count = simulation_config.get("expected_exchange_agent_count", 1)

    try:
        start_time = parse_datetime_utc(start_time_str)
        end_time = parse_datetime_utc(end_time_str)
    except ValueError as e:
        launcher_logger.error(f"Invalid time format in simulation config: {e}")
        sys.exit(1)
        
    simulation_clock = SimulationClock(
        start_time=start_time,
        end_time=end_time,
        tick_interval_seconds=tick_interval_seconds,
        rabbitmq_host=rabbitmq_host,
        expected_exchange_agent_count=expected_exchange_agent_count,
        expected_responses=expected_responses
    )
    asyncio.run(simulation_clock.run())

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration with validation."""
    if not os.path.exists(config_path):
        launcher_logger.error(f"Config file '{config_path}' does not exist.")
        sys.exit(1)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def validate_configuration(config: Dict[str, Any]) -> List[str]:
    """
    Comprehensive configuration validation for demo environments.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required sections validation
    required_sections = ["instruments", "agents", "simulation"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: '{section}'")

    # Instruments validation
    instruments = config.get("instruments", [])
    if not instruments:
        errors.append("At least one instrument must be specified")

    # Exchange configuration validation
    exchanges_config = config.get("exchanges", {})
    for instrument in instruments:
        if instrument not in exchanges_config:
            errors.append(f"Missing exchange configuration for instrument: '{instrument}'")
        else:
            inst_cfg = exchanges_config[instrument]
            # Validate required fields
            required_fields = ["data_source", "symbol_type", "candle_interval"]
            for field in required_fields:
                if field not in inst_cfg:
                    errors.append(f"Missing required field '{field}' for instrument '{instrument}'")

            # Validate data source and symbol type combinations
            data_source = inst_cfg.get("data_source", "")
            symbol_type = inst_cfg.get("symbol_type", "")

            if symbol_type == "crypto" and data_source not in ["alpha_vantage", "polygon"]:
                errors.append(f"Invalid data source '{data_source}' for crypto symbol '{instrument}'. Use 'alpha_vantage' or 'polygon'")

            if symbol_type == "stock" and data_source not in ["polygon", "alpha_vantage"]:
                errors.append(f"Invalid data source '{data_source}' for stock symbol '{instrument}'. Use 'polygon' or 'alpha_vantage'")

    # Agent configuration validation
    agents_config = config.get("agents", {})
    if not agents_config:
        errors.append("At least one agent must be configured")

    for agent_name, agent_details in agents_config.items():
        agent_type = agent_details.get("type")
        if not agent_type:
            errors.append(f"Missing agent type for agent '{agent_name}'")
        elif agent_type not in AGENT_TYPE_MAPPING:
            errors.append(f"Unsupported agent type '{agent_type}' for agent '{agent_name}'")

        # Validate LLM agent specific requirements
        if agent_type == "LLMTradingAgent":
            parameters = agent_details.get("parameters", {})
            models = parameters.get("models", {})

            # Check for required analyst configurations
            enable_market = parameters.get("enable_market_analyst", True)
            enable_news = parameters.get("enable_news_analyst", False)
            enable_fundamental = parameters.get("enable_fundamental_analyst", False)

            if enable_market and "market_analysis" not in models:
                errors.append(f"LLM agent '{agent_name}' has market analyst enabled but no market_analysis model configured")

            if enable_news and "news" not in models:
                errors.append(f"LLM agent '{agent_name}' has news analyst enabled but no news model configured")

            if enable_fundamental and "fundamental_analysis" not in models:
                errors.append(f"LLM agent '{agent_name}' has fundamental analyst enabled but no fundamental_analysis model configured")

            if "aggregator" not in models:
                errors.append(f"LLM agent '{agent_name}' missing required aggregator model")

    # Simulation configuration validation
    simulation_config = config.get("simulation", {})
    required_sim_fields = ["start_time", "end_time"]
    for field in required_sim_fields:
        if field not in simulation_config:
            errors.append(f"Missing required simulation field: '{field}'")

    # Validate time formats and ranges
    try:
        start_time = parse_datetime_utc(simulation_config.get("start_time", ""))
        end_time = parse_datetime_utc(simulation_config.get("end_time", ""))

        if end_time <= start_time:
            errors.append("Simulation end_time must be after start_time")

        # Check for reasonable simulation duration (not too long for demo)
        duration = end_time - start_time
        if duration.days > 365:
            errors.append("Simulation duration exceeds 1 year - consider shorter period for demo")

    except ValueError as e:
        errors.append(f"Invalid time format in simulation config: {e}")

    return errors

def check_dependencies() -> List[str]:
    """
    Check system dependencies and API keys for demo readiness.

    Returns:
        List of dependency issues (empty if all good)
    """
    issues = []

    # Check environment variables
    required_env_vars = {
        "RABBITMQ_HOST": "RabbitMQ message broker",
        "LOG_DIR": "Logging directory"
    }

    for var, description in required_env_vars.items():
        if not os.getenv(var):
            issues.append(f"Missing required environment variable: {var} ({description})")

    # Check for at least one data source API key
    data_api_keys = ["POLYGON_API_KEY", "ALPHA_VANTAGE_API_KEY"]
    if not any(os.getenv(key) for key in data_api_keys):
        issues.append("At least one data source API key required: POLYGON_API_KEY or ALPHA_VANTAGE_API_KEY")

    # Check if output directories exist (for potential future use)
    charts_dir = "charts"
    reports_dir = "reports"

    if not os.path.exists(charts_dir):
        issues.append(f"Info: Charts directory '{charts_dir}' does not exist - will be created if needed")

    if not os.path.exists(reports_dir):
        issues.append(f"Info: Reports directory '{reports_dir}' does not exist - will be created if needed")

    return issues

def print_demo_banner():
    """Log an attractive demo banner with system information."""
    banner_lines = [
        "╔══════════════════════════════════════════════════════════════════════════════════╗",
        "║                                   StockSim                                       ║",
        "║              Multi-Agent Financial Market Simulation Platform                    ║",
        "║                                                                                  ║",
        "╠══════════════════════════════════════════════════════════════════════════════════╣",
        "║  Features:                                                                       ║",
        "║    • Multi-instrument portfolio coordination                                      ║",
        "║    • LLM-powered trading agents with specialized analysts                         ║",
        "║    • Comprehensive technical analysis and backtesting                            ║",
        "║    • Automatic chart and report generation                                       ║",
        "║                                                                                  ║",
        "║  Research Paper: \"StockSim: A Multi-Agent Framework for Financial Research\"      ║",
        "║  Version: 2.0                                                                    ║",
        "║                                                                                  ║",
        "║  Repository: https://github.com/StockSim/StockSim                                ║",
        "║  License: MIT                                                                    ║",
        "╚══════════════════════════════════════════════════════════════════════════════════╝"
    ]

    for line in banner_lines:
        launcher_logger.info(line)

def generate_post_simulation_artifacts(config: Dict[str, Any]):
    """
    Generate charts, reports, and analysis artifacts after simulation completion.
    This supports comprehensive evaluation and analysis of trading performance.
    """
    try:
        from utils.plot_charts import make_chart_dropdown, generate_demo_report, ensure_output_directories
        from utils.polygon_client import PolygonClient
        from utils.alpha_vantage_client import AlphaVantageClient

        launcher_logger.info("🎨 Generating post-simulation artifacts...")

        # Ensure output directories exist
        charts_dir, reports_dir = ensure_output_directories()

        instruments = config.get("instruments", [])
        exchanges_config = config.get("exchanges", {})
        simulation_config = config.get("simulation", {})

        simulation_start_str = simulation_config["start_time"]
        simulation_end_str = simulation_config["end_time"]

        # Generate charts and reports for each instrument
        for instrument in instruments:
            try:
                inst_cfg = exchanges_config.get(instrument, {})
                data_source = inst_cfg.get("data_source", "polygon")
                symbol_type = inst_cfg.get("symbol_type", "stock")
                interval = inst_cfg.get("candle_interval", "1d")
                indicator_kwargs = inst_cfg.get("indicator_kwargs", {})

                launcher_logger.info(f"📊 Generating artifacts for {instrument} ({symbol_type})...")

                # Initialize appropriate data client
                if data_source == "alpha_vantage":
                    client = AlphaVantageClient()
                else:
                    client = PolygonClient()

                # Load market data
                if symbol_type == "crypto":
                    candles = client.load_crypto_aggregates(
                        symbol=instrument,
                        interval=interval,
                        start_date=simulation_start_str,
                        end_date=simulation_end_str,
                        market="USD",
                        sort="asc",
                        limit=10000,
                        use_cache=True
                    )
                else:
                    candles = client.load_aggregates(
                        symbol=instrument,
                        interval=interval,
                        start_date=simulation_start_str,
                        end_date=simulation_end_str,
                        adjusted=True,
                        sort="asc",
                        limit=10000,
                        use_cache=True
                    )

                if candles:
                    # Generate interactive chart
                    chart_filename = f"{instrument}_demo_chart.html"
                    make_chart_dropdown(
                        candles=candles,
                        instrument=instrument,
                        scales_seconds=[3600, 14400, 86400],  # 1h, 4h, 1d
                        out_html=chart_filename,
                        indicator_kwargs=indicator_kwargs,
                        symbol_type=symbol_type
                    )

                    # Generate analysis report
                    report = generate_demo_report(instrument, candles, indicator_kwargs)
                    if report:
                        report_filename = os.path.join(reports_dir, f"{instrument}_demo_report.json")
                        with open(report_filename, 'w') as f:
                            json.dump(report, f, indent=2)
                        launcher_logger.info(f"📈 Generated report: {report_filename}")

            except Exception as e:
                launcher_logger.error(f"❌ Failed to generate artifacts for {instrument}: {e}")
                continue

        # Generate summary report
        summary_report = {
            "simulation_info": {
                "start_time": simulation_start_str,
                "end_time": simulation_end_str,
                "duration_days": (parse_datetime_utc(simulation_end_str) - parse_datetime_utc(simulation_start_str)).days,
                "instruments": instruments,
                "total_agents": sum(agent_config.get("count", 1) for agent_config in config.get("agents", {}).values()),
                "exchange_mode": config.get("exchange_mode", "candle")
            },
            "generated_artifacts": {
                "charts_directory": charts_dir,
                "reports_directory": reports_dir,
                "timestamp": datetime.now().isoformat()
            },
            "research_metrics": {
                "llm_agents": sum(1 for agent in config.get("agents", {}).values() if agent.get("type") == "LLMTradingAgent"),
                "benchmark_agents": sum(1 for agent in config.get("agents", {}).values() if agent.get("type") != "LLMTradingAgent"),
                "multi_market": len(set(exchanges_config.get(inst, {}).get("symbol_type", "stock") for inst in instruments)) > 1
            }
        }

        summary_file = os.path.join(reports_dir, "simulation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_report, f, indent=2)

        launcher_logger.info(f"✅ Post-simulation artifacts generated successfully!")
        launcher_logger.info(f"📁 Charts available in: {charts_dir}/")
        launcher_logger.info(f"📊 Reports available in: {reports_dir}/")

    except Exception as e:
        launcher_logger.error(f"❌ Failed to generate post-simulation artifacts: {e}")
        launcher_logger.error(traceback.format_exc())

def main():
    """Main launcher function with comprehensive validation and monitoring."""
    # Log demo banner
    print_demo_banner()

    # Load configuration
    config_file_path = sys.argv[1] if len(sys.argv) > 1 else "configs/demo_config.yaml"
    launcher_logger.info(f"Loading configuration from: {config_file_path}")

    try:
        config = load_config(config_file_path)
    except Exception as e:
        launcher_logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Validate configuration
    validation_errors = []
    launcher_logger.info("Validating configuration...")
    if config.get("exchange_mode", "candle").lower() == "candle":
        validation_errors = validate_configuration(config)
    if validation_errors:
        launcher_logger.error("Configuration validation failed:")
        for error in validation_errors:
            launcher_logger.error(f"   • {error}")
        sys.exit(1)

    # Check dependencies
    launcher_logger.info("Checking system dependencies...")
    dependency_issues = check_dependencies()
    if dependency_issues:
        launcher_logger.warning("Dependency check results:")
        for issue in dependency_issues:
            launcher_logger.warning(f"   • {issue}")

        # Check if any critical issues would prevent running
        critical_issues = [issue for issue in dependency_issues if not issue.startswith("Warning:") and not issue.startswith("Info:")]
        if critical_issues:
            launcher_logger.error("Critical issues detected. Please resolve before continuing.")
            sys.exit(1)
        else:
            launcher_logger.info("All critical dependencies satisfied. Warnings can be ignored for basic demo.")
    else:
        launcher_logger.info("All dependencies satisfied.")

    # Extract configuration sections
    instruments = config.get("instruments", [])
    exchanges_config = config.get("exchanges", {})
    agents_config = config.get("agents", {})
    simulation_config = config.get("simulation", {})

    launcher_logger.info("Demo Configuration Summary:")
    launcher_logger.info(f"   • Instruments: {', '.join(instruments)}")
    launcher_logger.info(f"   • Agents: {len(agents_config)} configured")
    launcher_logger.info(f"   • Exchange Mode: {config.get('exchange_mode', 'candle')}")
    launcher_logger.info(f"   • Simulation Period: {simulation_config.get('start_time')} to {simulation_config.get('end_time')}")

    # Count different agent types for demo insights
    agent_type_counts = {}
    llm_agents_with_analysts = []

    for agent_name, agent_details in agents_config.items():
        agent_type = agent_details.get("type", "Unknown")
        agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + agent_details.get("count", 1)

        if agent_type == "LLMTradingAgent":
            params = agent_details.get("parameters", {})
            analysts = []
            if params.get("enable_market_analyst", True):
                analysts.append("Market")
            if params.get("enable_news_analyst", False):
                analysts.append("News")
            if params.get("enable_fundamental_analyst", False):
                analysts.append("Fundamental")

            if analysts:
                llm_agents_with_analysts.append(f"{agent_name} ({', '.join(analysts)})")

    launcher_logger.info("Agent Composition:")
    for agent_type, count in agent_type_counts.items():
        launcher_logger.info(f"   • {agent_type}: {count}")

    if llm_agents_with_analysts:
        launcher_logger.info("LLM Agent Analyst Configuration:")
        for agent_info in llm_agents_with_analysts:
            launcher_logger.info(f"   • {agent_info}")

    simulation_start_timestamp = datetime.now()

    launcher_logger.info(f"Starting StockSim Simulation at {simulation_start_timestamp}")
    launcher_logger.info(f"Configuration: {config_file_path}")
    launcher_logger.info(f"Simulation config: {simulation_config}")

    rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
    launcher_logger.info("Starting the launcher...")

    exchange_mode = config.get("exchange_mode", "orderbook").lower()
    simulation_start_time = simulation_config["start_time"]
    simulation_end_time = simulation_config["end_time"]

    exchange_agents = []
    indicator_kwargs_map = {instrument: inst_cfg.get("indicator_kwargs", {}) for instrument, inst_cfg in exchanges_config.items()}
    warmup_candles_map = {instrument: inst_cfg.get("warmup_candles", 250) for instrument, inst_cfg in exchanges_config.items()}
    data_source_map = {
        instrument: {
            "data_source": inst_cfg.get("data_source", "polygon"),
            "symbol_type": inst_cfg.get("symbol_type", "stock")
        }
        for instrument, inst_cfg in exchanges_config.items()
    }

    llm_count = sum(
        details.get("count", 1)
        for _, details in agents_config.items()
        if details.get("type") == "LLMTradingAgent"
    )

    launcher_logger.info("Launching Exchange Agents...")

    if exchange_mode == "candle":
        # Dual-market candle-based exchange with configurable data sources
        instrument_exchange_map = {instrument: f"candle_exchange_{instrument.lower()}" for instrument in instruments}

        launcher_logger.info("Launching CandleBasedExchangeAgents with dual-market support...")
        for instrument, exchange_id in instrument_exchange_map.items():
            inst_cfg = exchanges_config.get(instrument, {})

            interval = inst_cfg.get("candle_interval", "1d")
            data_source = inst_cfg.get("data_source", "polygon")
            symbol_type = inst_cfg.get("symbol_type", "stock")
            spread_factor = inst_cfg.get("spread_factor", 0.001)

            launcher_logger.info(f"   • {instrument} ({symbol_type}) via {data_source} - {interval} intervals")

            news_kwargs = inst_cfg.get("news", {})
            tickers = news_kwargs.get("tickers", [instrument])
            news_max_results = news_kwargs.get("max_results", 50)

            exchange_params = {
                "instrument": instrument,
                "resolution": interval,
                "start_date": simulation_start_time,
                "end_date": simulation_end_time,
                "warmup_candles": warmup_candles_map[instrument],
                "agent_id": exchange_id,
                "rabbitmq_host": rabbitmq_host,
                "tickers": tickers,
                "spread_factor": spread_factor,
                "limit_news": news_max_results,
                "indicator_kwargs": indicator_kwargs_map[instrument],
                "data_source": data_source,
                "symbol_type": symbol_type
            }

            p = Process(
                target=exchange_agent_runner,
                args=(CandleBasedExchangeAgent, exchange_params),
                name=exchange_id
            )
            p.start()
            exchange_agents.append(p)
            launcher_logger.info(f"Started CandleBasedExchangeAgent for instrument '{instrument}' using {data_source} data source.")

    else:
        # Real-time order book mode
        launcher_logger.info("   • Using real-time order book mode")
        launcher_logger.info("Launching standard ExchangeAgents...")
        instrument_exchange_map = {instrument: f"exchange_{instrument.lower()}" for instrument in instruments}

        for instrument, exchange_id in instrument_exchange_map.items():
            inst_cfg = exchanges_config.get(instrument, {})
            trades_outfile = inst_cfg.get("trades_outfile", "")
            data_source = inst_cfg.get("data_source", "polygon")
            symbol_type = inst_cfg.get("symbol_type", "stock")
            
            # News configuration
            news_kwargs = inst_cfg.get("news", {})
            tickers = news_kwargs.get("tickers", [instrument])
            news_max_results = news_kwargs.get("max_results", 50)
            
            # Warmup configuration
            warmup_start_date = inst_cfg.get("warmup_start_date")
            warmup_end_date = inst_cfg.get("warmup_end_date")
            warmup_resolution = inst_cfg.get("warmup_resolution", "1d")
            warmup_candles = inst_cfg.get("warmup_candles", 250)
            
            # Base resolution configuration (like candle-based exchange)
            base_resolution = inst_cfg.get("candle_interval", "1m")

            exchange_params = {
                "instrument": instrument,
                "agent_id": exchange_id,
                "rabbitmq_host": rabbitmq_host,
                "trades_output_file": trades_outfile,
                "tickers": tickers,
                "limit_news": news_max_results,
                "indicator_kwargs": indicator_kwargs_map[instrument],
                "data_source": data_source,
                "symbol_type": symbol_type,
                "data_start_date": warmup_start_date,
                "data_end_date": warmup_end_date or simulation_end_time,
                "warmup_resolution": warmup_resolution,
                "warmup_candles": warmup_candles,
                "resolution": base_resolution
            }

            p = Process(
                target=exchange_agent_runner,
                args=(ExchangeAgent, exchange_params),
                name=exchange_id
            )
            p.start()
            exchange_agents.append(p)
            launcher_logger.info(
                f"Started ExchangeAgent '{exchange_id}' for instrument '{instrument}' "
                f"({symbol_type}) via {data_source} with warmup: {warmup_start_date} to {warmup_end_date}"
            )

    launcher_logger.info("Waiting for exchange agents to initialize...")
    time.sleep(10)

    launcher_logger.info("Launching Trading Agents...")

    # Agent parameter customization
    agent_custom_params = {
        "Random_Trader": lambda params: {
            **params,
            "action_interval_seconds": interval_to_seconds(params["action_interval"]) if "action_interval" in params else params.get("action_interval_seconds", 86400)
        },
        "LLMTradingAgent": lambda params: {
            **{k: v for k, v in params.items() if k != "action_interval"},
            "start_time": simulation_start_time,
            "end_time": simulation_end_time,
            "extended_intervals": params.get("extended_intervals"),
            "extended_warmup_candles": warmup_candles_map,
            "extended_indicator_kwargs": indicator_kwargs_map,
            "data_source_config": data_source_map,
            "action_interval_seconds": interval_to_seconds(params["action_interval"]) if "action_interval" in params else params.get("action_interval_seconds", 86400)
        },
        "HistoricalOrderTrader": lambda params: {
            **params,
            "orders": load_json_file(params.get("orders", "")) if "orders" in params else None
        },
        "Buy_And_Hold_Trader": lambda params: {
            **params,
            "quantity_size": params.get("quantity_size", 100),
            "action_interval_seconds": interval_to_seconds(params["action_interval"]) if "action_interval" in params else params.get("action_interval_seconds", 86400)
        },
        "SMA_Trader": lambda params: {
            **params,
            "window": params.get("window", 20),
            "position_size_pct": params.get("position_size_pct", 0.05),
            "action_interval_seconds": interval_to_seconds(params["action_interval"]) if "action_interval" in params else params.get("action_interval_seconds", 86400)
        },
        "SLMA_Trader": lambda params: {
            **params,
            "short_window": params.get("short_window", 20),
            "long_window": params.get("long_window", 50),
            "position_size_pct": params.get("position_size_pct", 0.05),
            "action_interval_seconds": interval_to_seconds(params["action_interval"]) if "action_interval" in params else params.get("action_interval_seconds", 86400)
        },
        "MACD_Trader": lambda params: {
            **params,
            "fast_period": params.get("fast_period", 12),
            "slow_period": params.get("slow_period", 26),
            "signal_period": params.get("signal_period", 9),
            "position_size_pct": params.get("position_size_pct", 0.05),
            "action_interval_seconds": interval_to_seconds(params["action_interval"]) if "action_interval" in params else params.get("action_interval_seconds", 86400)
        },
        "Bollinger_Bands_Trader": lambda params: {
            **params,
            "sma_period": params.get("sma_period", 20),
            "std_dev_multiplier": params.get("std_dev_multiplier", 2.0),
            "position_size_pct": params.get("position_size_pct", 0.05),
            "action_interval_seconds": interval_to_seconds(params["action_interval"]) if "action_interval" in params else params.get("action_interval_seconds", 86400)
        },
    }

    agent_processes = []

    for agent_name, agent_details in agents_config.items():
        agent_type = agent_details.get("type")
        parameters = agent_details.get("parameters", {})
        agent_class = AGENT_TYPE_MAPPING.get(agent_type)

        if not agent_class:
            launcher_logger.error(f"Unsupported agent type '{agent_type}' for agent '{agent_name}'. Skipping.")
            continue

        count = agent_details.get("count", 1)
        launcher_logger.info(f"   • {agent_name} ({agent_type}) - {count} instance(s)")

        for i in range(count):
            unique_agent_id = f"{agent_name}_{i + 1}" if count > 1 else agent_name

            if agent_type in agent_custom_params:
                try:
                    parameters = agent_custom_params[agent_type](parameters)
                except (FileNotFoundError, ValueError) as e:
                    launcher_logger.error(f"Error loading parameters for agent '{agent_name}': {e}")
                    continue

            parameters["agent_id"] = unique_agent_id
            if "instrument_exchange_map" not in parameters:
                parameters["instrument_exchange_map"] = instrument_exchange_map
            parameters["rabbitmq_host"] = rabbitmq_host

            if agent_type == "Random_Trader":
                parameters["seed"] = random.randint(0, 10 ** 6)


            launcher_logger.debug(f"Parameters for agent '{agent_name}': {parameters}")
            p = Process(target=agent_runner, args=(agent_class, parameters), name=unique_agent_id)
            p.start()
            agent_processes.append(p)
            launcher_logger.info(f"Started Agent '{unique_agent_id}' of type '{agent_type}'.")

    launcher_logger.info("Waiting for trading agents to initialize...")
    time.sleep(20)

    launcher_logger.info("Starting Simulation Clock...")
    # Start simulation clock
    p_clock = Process(
        target=simulation_clock_runner,
        args=(simulation_config, rabbitmq_host, llm_count),
        name="SimulationClock"
    )
    p_clock.start()
    launcher_logger.info("Started SimulationClock.")

    all_processes = exchange_agents + agent_processes + [p_clock]

    def shutdown(signum, frame):
        """Graceful shutdown handler."""
        launcher_logger.info(f"Received shutdown signal ({signum}). Terminating all processes...")
        launcher_logger.info("Shutting down StockSim simulation...")

        for process in all_processes:
            if process.is_alive():
                process.terminate()
                launcher_logger.info(f"Terminated process '{process.name}' (PID: {process.pid}).")

        launcher_logger.info("Stockim simulation session completed.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    launcher_logger.info("StockSim Simulation is now running!")
    launcher_logger.info(f"Monitor progress in the logs directory: {log_dir}")
    launcher_logger.info(f"Simulation will run from {simulation_start_time} to {simulation_end_time}")
    launcher_logger.info("Press Ctrl+C to stop the simulation gracefully")
    launcher_logger.info("="*80)


    launcher_logger.info("Simulation is running. Press Ctrl+C to stop.")

    try:
        for proc in all_processes:
            proc.join()

        # Normal completion - generate reports
        launcher_logger.info("Simulation completed successfully!")
        generate_post_simulation_artifacts(config)

    except KeyboardInterrupt:
        launcher_logger.info("Simulation interrupted by user")

        shutdown(signal.SIGINT, None)

    except Exception as e:
        launcher_logger.error(f"Simulation failed: {e}")
        launcher_logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()