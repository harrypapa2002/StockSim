"""
Market Analyst Agent for StockSim Demo

This module implements the technical market analysis component of the multi-agent
LLM coordination system. The MarketAnalyst provides sophisticated technical analysis
using multiple timeframes, indicators, and volume profile analysis.

Key Features:
- Multi-timeframe technical analysis with extended intervals
- Advanced indicator calculations (SMA, RSI, MACD, ATR, Volume Profile)
- Support/resistance level identification
- Professional-grade prompt engineering for LLM analysis
- Integration with the modular wrapper system
- Comprehensive logging for research analysis

This agent demonstrates the specialized expertise approach described in the paper,
where different LLMs can be assigned to different analytical roles within a
coordinated trading system.
"""

import os
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Optional

import numpy as np
import jinja2

from utils.logging_setup import setup_logger
from utils.time_utils import parse_datetime_utc, seconds_to_human

def _flatten_indicators(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert IndicatorsTracker output into the flat structure expected by
    `_format_indicators()`. Handles both new IndicatorsTracker format and legacy formats.
    """
    # Check if this is the new IndicatorsTracker format
    if "trend_indicators" in raw:
        # New format
        return {
            # trend
            "sma":   raw.get("trend_indicators", {}).get("sma", {}),
            "macd": {
                "value":   raw.get("trend_indicators", {}).get("macd", {}).get("line"),
                "signal":  raw.get("trend_indicators", {}).get("macd", {}).get("signal"),
            },
            # momentum
            "rsi":   raw.get("momentum_indicators", {}).get("rsi", {}),
            # volatility
            "atr":   raw.get("volatility_indicators", {}).get("atr", {}),
            # support / resistance
            "support_levels":    raw.get("support_resistance", {}).get("support_levels", []),
            "resistance_levels": raw.get("support_resistance", {}).get("resistance_levels", []),
            # volume
            "volume_profile":    raw.get("volume_analysis", {}).get("volume_profile", {}),
        }
    else:
        # Legacy format or direct format - return as is with safe defaults
        return {
            "sma": raw.get("sma", {}),
            "macd": raw.get("macd", {}),
            "rsi": raw.get("rsi", {}),
            "atr": raw.get("atr", {}),
            "support_levels": raw.get("support_levels", []),
            "resistance_levels": raw.get("resistance_levels", []),
            "volume_profile": raw.get("volume_profile", {}),
        }

def _format_indicators(indicators: Dict[str, Any]) -> str:
    """Enhanced indicator formatting with better structure and interpretations"""
    indicators = _flatten_indicators(indicators)
    sma = indicators.get("sma", {})
    rsi = indicators.get("rsi", {})
    macd = indicators.get("macd", {})
    atr = indicators.get("atr", {})
    support = indicators.get("support_levels", [])
    resistance = indicators.get("resistance_levels", [])
    volume_profile = indicators.get("volume_profile", {})

    # Enhanced SMA formatting with trend context
    sma_entries = []
    for period, value in sma.items():
        if value is not None:
            sma_entries.append(f"SMA-{period}: ${value:.2f}")
    sma_str = " | ".join(sma_entries) if sma_entries else "N/A"

    # Enhanced RSI with interpretation
    rsi_val = rsi.get("value")
    if rsi_val:
        rsi_str = f"{rsi_val:.1f}"
    else:
        rsi_str = "N/A"

    # Enhanced MACD with momentum direction
    macd_val = macd.get("value")
    signal_val = macd.get("signal")
    if macd_val is not None and signal_val is not None:
        histogram = macd_val - signal_val
        macd_str = f"Value: {macd_val:.3f} | Signal: {signal_val:.3f} | Histogram: {histogram:.3f}"
    else:
        macd_str = "N/A"

    # ATR with volatility context
    atr_val = atr.get("value")
    if atr_val is not None:
        atr_str = f"{atr_val:.4f} (Current volatility measure)"
    else:
        atr_str = "N/A"

    support_str = ", ".join(f"${s:.2f}" for s in support) if support else "None identified"
    resistance_str = ", ".join(f"${r:.2f}" for r in resistance) if resistance else "None identified"

    # Enhanced volume profile
    poc = volume_profile.get("poc_price", None)
    va = volume_profile.get("value_area", (None, None))
    bins = volume_profile.get("bin_centers", [])
    volumes = volume_profile.get("volume_distribution", [])

    if isinstance(poc, (float, int)) and all(isinstance(x, (float, int)) for x in va):
        volume_str = f"POC: ${poc:.2f} | Value Area: ${va[0]:.2f}-${va[1]:.2f}"
        if bins and volumes:
            top_volume_indices = np.argsort(volumes)[-3:][::-1]
            significant_levels = "; ".join(
                f"${bins[idx]:.2f} ({volumes[idx]:.0f})" for idx in top_volume_indices
            )
            volume_str += f" | Key Levels: {significant_levels}"
    else:
        volume_str = "Insufficient volume profile data"

    return (
        f"**Moving Averages**: {sma_str}\n"
        f"**RSI**: {rsi_str}\n"
        f"**MACD**: {macd_str}\n"
        f"**ATR**: {atr_str}\n"
        f"**Support Levels**: {support_str}\n"
        f"**Resistance Levels**: {resistance_str}\n"
        f"**Volume Profile**: {volume_str}"
    )

class MarketAnalyst:
    """
    Market Analyst providing comprehensive technical analysis for trading decisions.

    This agent implements sophisticated technical analysis capabilities including:
    - Multi-timeframe analysis across different resolutions
    - Advanced technical indicators (SMA, RSI, MACD, ATR, Volume Profile)
    - Support/resistance level identification
    - Professional prompt engineering for LLM analysis

    The analyst maintains conversation history and adapts its analysis style
    based on market conditions and available data.
    """

    def __init__(
        self,
        agent,
        wrapper_manager=None,
        wrapper_type: str = "market_analysis"
    ):
        """
        Initialize the MarketAnalyst with centralized wrapper manager.

        Args:
            agent: Parent trading agent instance
            wrapper_manager: Centralized wrapper manager from the main agent
            wrapper_type: Type of wrapper to use from the manager
        """
        self.agent = agent
        self.wrapper_manager = wrapper_manager or getattr(agent, 'wrapper_manager', None)
        self.wrapper_type = wrapper_type
        
        self.logger = setup_logger(
            f"market_{self.agent.agent_id}",
            f"{self.agent.LOG_DIR}/market/market_{self.agent.agent_id}.log"
        )
        self.has_sent_first_prompt: Dict[str, bool] = defaultdict(lambda: False)

        if self.wrapper_manager is None:
            raise ValueError("No wrapper manager provided - MarketAnalyst requires centralized wrapper management")

        self._initialize_prompt_templates()
        self.logger.info(f"MarketAnalyst for {self.agent.agent_id} initialized with centralized wrapper manager.")

    def analyze_extended_intervals(self, instrument: str) -> str:
        """
        Analyze extended timeframe data for comprehensive market context.

        This method processes multi-timeframe data to provide context across
        different time horizons, enabling more robust trading decisions.

        Args:
            instrument: Financial instrument symbol to analyze

        Returns:
            Formatted string containing multi-timeframe analysis
        """
        if instrument not in self.agent.all_candles_map:
            return f"[No extended data available for {instrument}]"

        lines = []

        for interval_def in self.agent.extended_intervals:
            label = interval_def.get("label", "Unknown Interval")
            resolution = interval_def.get("resolution", "unknown")
            interval_data = self.agent.all_candles_map[instrument].get(label, {})

            candles = interval_data.get("candles", [])
            indicators = interval_data.get("indicators", {})
            
            # Use the new IndicatorsTracker format
            flattened = _flatten_indicators(indicators)

            if len(candles) < 2:
                continue

            closes = [c["close"] for c in candles]
            volumes = [c["volume"] for c in candles]

            first, last = candles[0], candles[-1]
            first_close = first["close"]
            last_close = last["close"]
            net_pct = (last_close - first_close) / first_close * 100.0
            avg_close = mean(closes)
            avg_vol = mean(volumes)
            min_low = min(c['low'] for c in candles)
            max_high = max(c['high'] for c in candles)

            # Calculate momentum streaks
            up, down = 0, 0
            max_up, max_down = 0, 0
            for i in range(1, len(closes)):
                if closes[i] > closes[i - 1]:
                    up += 1
                    down = 0
                elif closes[i] < closes[i - 1]:
                    down += 1
                    up = 0
                else:
                    up = down = 0
                max_up = max(max_up, up)
                max_down = max(max_down, down)

            num_candles = len(candles)
            start_time = parse_datetime_utc(candles[0]["timestamp"]).strftime("%Y-%m-%d, %H:%M:%S")
            end_time = parse_datetime_utc(candles[-1]["timestamp"]).strftime("%Y-%m-%d, %H:%M:%S")

            # RSI analysis
            rsi_data = flattened.get("rsi", {})
            rsi_latest = rsi_data.get("value") if rsi_data else None
            rsi_avg = rsi_latest  # For extended intervals, we typically have one value per timeframe

            # SMA trend analysis
            sma_summary_strs = []
            sma_dict = flattened.get("sma", {})
            for period, value in sma_dict.items():
                if value is None:
                    continue
                # For extended intervals, we have single values, so we can't calculate trends
                # We'll just report the current SMA value
                sma_summary_strs.append(f"SMA {period}: {value:.2f}")

            # Volume profile analysis
            volume_profile = flattened.get("volume_profile", {})
            poc = volume_profile.get("poc_price", None)
            va = volume_profile.get("value_area", (None, None))
            bins = volume_profile.get("bin_centers", [])
            volumes_per_bin = volume_profile.get("volume_distribution", [])

            formatted_poc = f"{poc:.2f}" if isinstance(poc, (int, float)) else "N/A"
            formatted_va = (
                f"{va[0]:.2f}–{va[1]:.2f}" if all(isinstance(x, (int, float)) for x in va) else "N/A"
            )

            # Enhanced volume profile description
            if bins and volumes_per_bin:
                top_volume_indices = np.argsort(volumes_per_bin)[-3:][::-1]
                significant_levels = "; ".join(
                    f"{bins[idx]:.2f} ({volumes_per_bin[idx]:.0f})" for idx in top_volume_indices
                )
                volume_profile_str = (
                    f"POC={formatted_poc}, VA={formatted_va}, Key volume levels: {significant_levels}"
                )
            else:
                volume_profile_str = f"POC={formatted_poc}, VA={formatted_va}, Key volume levels: N/A"

            bullish = sum(1 for c in candles if c["close"] > c["open"])
            bearish = num_candles - bullish

            description_lines = [
                f"  - {label} | Resolution: {resolution}",
                f"  - Time Range: {start_time} → {end_time}",
                f"  - Net Price Change: {first_close:.2f} → {last_close:.2f} ({net_pct:.2f}%)",
                f"  - High: {max_high:.2f}, Low: {min_low:.2f}, Avg Close: {avg_close:.2f}, Avg Volume: {avg_vol:.2f}",
                f"  - Bullish candles: {bullish}, Bearish candles: {bearish}",
                f"  - SMA Summary: {'; '.join(sma_summary_strs)}" if sma_summary_strs else "  - SMA: unavailable",
                (
                    f"  - RSI Summary: latest={rsi_latest:.1f}, avg={rsi_avg:.1f}"
                    if rsi_avg is not None else "  - RSI: unavailable"
                ),
                f"  - Volume Profile: {volume_profile_str}"
            ]

            lines.append("\n".join(description_lines))

            # Include recent OHLCV data for context
            ohlcv_lines = [f" - OHLCV Data:"]
            for c in candles:
                ts = parse_datetime_utc(c["timestamp"]).strftime("%Y-%m-%d %H:%M")
                vwap_str = f"{c.get('vwap', 'N/A'):.2f}" if c.get('vwap') else "N/A"
                ohlcv_lines.append(
                    f" {ts} | O={c['open']:.2f} H={c['high']:.2f} L={c['low']:.2f} C={c['close']:.2f} V={c['volume']:.2f} VWAP={vwap_str}"
                )

            if ohlcv_lines:
                lines.append("\n" + "\n".join(ohlcv_lines))

        return "\n\n".join(lines)

    def _initialize_prompt_templates(self) -> None:
        """Initialize Jinja2 prompt templates for market analysis."""
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        tpl_dir = os.path.join(base_dir, "templates")
        
        # Load single-instrument templates
        with open(os.path.join(tpl_dir, "market_analyst_first_time.j2"), "r") as f:
            self._first_time_template = f.read()

        with open(os.path.join(tpl_dir, "market_analyst_subsequent.j2"), "r") as f:
            self._subsequent_template = f.read()
        
        # Load multi-instrument templates
        with open(os.path.join(tpl_dir, "market_analyst_multi_first_time.j2"), "r") as f:
            self._multi_first_time_template = f.read()

        with open(os.path.join(tpl_dir, "market_analyst_multi_subsequent.j2"), "r") as f:
            self._multi_subsequent_template = f.read()
        
        # Configure Jinja2 environment
        self._jinja_env = jinja2.Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
            undefined=jinja2.StrictUndefined,
        )

    def construct_first_time_prompt(
        self,
        instrument: str,
        open_price: Any,
        high_price: Any,
        low_price: Any,
        close_price: Any,
        volume: Any,
        indicators: Dict[str, Any],
        vwap: Optional[float] = None,
        transactions: int = 0
    ) -> str:
        """
        Construct the initial comprehensive prompt for market analysis using Jinja2 template.
        """
        vwap_str = f"${vwap:.4f}" if vwap is not None else "N/A"
        
        context = {
            "instrument": instrument,
            "session_start": self.agent.start_time.isoformat(),
            "session_end": self.agent.end_time.isoformat(),
            "current_time": self.agent.current_time.isoformat(),
            "action_interval": seconds_to_human(int(self.agent.action_interval.total_seconds())),
            "extended_intervals_analysis": self.analyze_extended_intervals(instrument),
            "open_price": f"{float(open_price):.4f}",
            "high_price": f"{float(high_price):.4f}",
            "low_price": f"{float(low_price):.4f}",
            "close_price": f"{float(close_price):.4f}",
            "volume": f"{float(volume):,.0f}",
            "vwap_str": vwap_str,
            "transactions": transactions,
            "formatted_indicators": _format_indicators(indicators)
        }
        
        template = self._jinja_env.from_string(self._first_time_template)
        prompt = template.render(**context)
        
        self.logger.debug(f"Constructed first-time market analysis prompt at {self.agent.current_time} for {instrument}")
        return prompt

    def construct_subsequent_prompt(
        self,
        instrument: str,
        open_price: Any,
        high_price: Any,
        low_price: Any,
        close_price: Any,
        volume: Any,
        indicators: Dict[str, Any],
        vwap: Optional[float] = None,
        transactions: int = 0
    ) -> str:
        """
        Construct follow-up prompts for continued market analysis using Jinja2 template.
        """
        vwap_str = f"${vwap:.4f}" if vwap is not None else "N/A"
        
        context = {
            "instrument": instrument,
            "current_time": self.agent.current_time.isoformat(),
            "open_price": f"{float(open_price):.4f}",
            "high_price": f"{float(high_price):.4f}",
            "low_price": f"{float(low_price):.4f}",
            "close_price": f"{float(close_price):.4f}",
            "volume": f"{float(volume):,.0f}",
            "vwap_str": vwap_str,
            "transactions": transactions,
            "formatted_indicators": _format_indicators(indicators)
        }
        
        template = self._jinja_env.from_string(self._subsequent_template)
        prompt = template.render(**context)
        
        self.logger.debug(f"Constructed subsequent market analysis prompt at {self.agent.current_time} for {instrument}")
        return prompt

    def construct_multi_instrument_first_time_prompt(
        self,
        instruments_data: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Construct initial comprehensive prompt for multi-instrument market analysis.
        """
        instruments_context = []
        
        for instrument, data in instruments_data.items():
            vwap_str = f"${data.get('vwap'):.4f}" if data.get('vwap') is not None else "N/A"
            
            instrument_ctx = {
                "instrument": instrument,
                "open_price": f"{float(data.get('open_price', 0)):.4f}",
                "high_price": f"{float(data.get('high_price', 0)):.4f}",
                "low_price": f"{float(data.get('low_price', 0)):.4f}",
                "close_price": f"{float(data.get('close_price', 0)):.4f}",
                "volume": f"{float(data.get('volume', 0)):,.0f}",
                "vwap_str": vwap_str,
                "transactions": data.get('transactions', 0),
                "formatted_indicators": _format_indicators(data.get('indicators', {})),
                "extended_intervals_analysis": self.analyze_extended_intervals(instrument)
            }
            instruments_context.append(instrument_ctx)
        
        context = {
            "instruments": instruments_context,
            "num_instruments": len(instruments_context),
            "session_start": self.agent.start_time.isoformat(),
            "session_end": self.agent.end_time.isoformat(),
            "current_time": self.agent.current_time.isoformat(),
            "action_interval": seconds_to_human(int(self.agent.action_interval.total_seconds()))
        }
        
        # Use multi-instrument template (will create this next)
        template = self._jinja_env.from_string(self._multi_first_time_template)
        prompt = template.render(**context)
        
        instruments_list = list(instruments_data.keys())
        self.logger.debug(f"Constructed multi-instrument first-time market analysis prompt at {self.agent.current_time} for {', '.join(instruments_list)}")
        return prompt

    def construct_multi_instrument_subsequent_prompt(
        self,
        instruments_data: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Construct follow-up prompt for multi-instrument market analysis.
        """
        instruments_context = []
        
        for instrument, data in instruments_data.items():
            vwap_str = f"${data.get('vwap'):.4f}" if data.get('vwap') is not None else "N/A"
            
            instrument_ctx = {
                "instrument": instrument,
                "open_price": f"{float(data.get('open_price', 0)):.4f}",
                "high_price": f"{float(data.get('high_price', 0)):.4f}",
                "low_price": f"{float(data.get('low_price', 0)):.4f}",
                "close_price": f"{float(data.get('close_price', 0)):.4f}",
                "volume": f"{float(data.get('volume', 0)):,.0f}",
                "vwap_str": vwap_str,
                "transactions": data.get('transactions', 0),
                "formatted_indicators": _format_indicators(data.get('indicators', {}))
            }
            instruments_context.append(instrument_ctx)
        
        context = {
            "instruments": instruments_context,
            "num_instruments": len(instruments_context),
            "current_time": self.agent.current_time.isoformat()
        }
        
        # Use multi-instrument template (will create this next)
        template = self._jinja_env.from_string(self._multi_subsequent_template)
        prompt = template.render(**context)
        
        instruments_list = list(instruments_data.keys())
        self.logger.debug(f"Constructed multi-instrument subsequent market analysis prompt at {self.agent.current_time} for {', '.join(instruments_list)}")
        return prompt

    async def get_market_analysis(
        self,
        instrument: str,
        open_price: Any,
        high_price: Any,
        low_price: Any,
        close_price: Any,
        volume: Any,
        indicators: Dict[str, Any],
        vwap: Optional[float] = None,
        transactions: int = 0
    ) -> str:
        """
        Generate comprehensive market analysis using the configured LLM.

        This method orchestrates the analysis process, including prompt construction,
        LLM interaction, and response processing.

        Args:
            instrument: Financial instrument symbol
            open_price: Opening price for the current period
            high_price: Highest price for the current period
            low_price: Lowest price for the current period
            close_price: Closing price for the current period
            volume: Trading volume for the current period
            indicators: Dictionary of technical indicators
            vwap: Volume-weighted average price
            transactions: Number of transactions in the period

        Returns:
            Generated market analysis text
        """
        try:
            # Determine prompt type based on conversation history
            if not self.has_sent_first_prompt[instrument]:
                prompt = self.construct_first_time_prompt(
                    instrument, open_price, high_price, low_price, close_price,
                    volume, indicators, vwap, transactions
                )
                self.has_sent_first_prompt[instrument] = True
                prompt_type = "FIRST_TIME"
            else:
                prompt = self.construct_subsequent_prompt(
                    instrument, open_price, high_price, low_price,
                    close_price, volume, indicators, vwap, transactions
                )
                prompt_type = "SUBSEQUENT"

            self.logger.info(f"📊 Sending {prompt_type} market analysis prompt for {instrument}")

            # Log detailed prompt for research tracking
            self.logger.debug(f"📊 MARKET ANALYST PROMPT [{prompt_type}] for {instrument}:\n" + "="*80 + f"\n{prompt}\n" + "="*80)

            # Generate analysis using the LLM wrapper
            result = await self.wrapper_manager.generate_with_wrapper(self.wrapper_type, prompt)

            if result:
                self.logger.info(f"📊 Received market analysis for {instrument} - Response length: {len(result)} chars")

                # Log detailed response for research tracking
                self.logger.debug(f"📊 MARKET ANALYST RESPONSE for {instrument}:\n" + "="*80 + f"\n{result}\n" + "="*80)

                return result
            else:
                self.logger.warning(f"📊 No result received from market analysis for {instrument}")
                return f"Market analysis unavailable for {instrument} at {self.agent.current_time}"

        except Exception as e:
            self.logger.error(f"📊 Error generating market analysis for {instrument}: {e}")
            return f"Market analysis error for {instrument}: {str(e)}"

    async def get_multi_instrument_market_analysis(
        self,
        instruments_data: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive market analysis for multiple instruments simultaneously.
        
        This enables cross-instrument analysis and portfolio-level market insights.
        
        Args:
            instruments_data: Dictionary mapping instrument symbols to their market data
                Format: {
                    "AAPL": {
                        "open_price": 150.0,
                        "high_price": 152.0,
                        "low_price": 148.0,
                        "close_price": 151.0,
                        "volume": 1000000,
                        "indicators": {...},
                        "vwap": 150.5,
                        "transactions": 5000
                    },
                    "NVDA": {...}
                }
        
        Returns:
            Generated multi-instrument market analysis text
        """
        try:
            instruments = list(instruments_data.keys())
            self.logger.info(f"📊 Generating multi-instrument market analysis for {len(instruments)} instruments: {', '.join(instruments)}")
            
            # Determine if this is first time for any instrument
            is_first_time = any(not self.has_sent_first_prompt.get(instr, False) for instr in instruments)
            
            if is_first_time:
                prompt = self.construct_multi_instrument_first_time_prompt(instruments_data)
                # Mark all instruments as having received first prompt
                for instr in instruments:
                    self.has_sent_first_prompt[instr] = True
                prompt_type = "MULTI_FIRST_TIME"
            else:
                prompt = self.construct_multi_instrument_subsequent_prompt(instruments_data)
                prompt_type = "MULTI_SUBSEQUENT"
            
            self.logger.info(f"📊 Sending {prompt_type} market analysis prompt for {len(instruments)} instruments")
            
            # Log detailed prompt for research tracking
            self.logger.debug(f"📊 MULTI-INSTRUMENT MARKET ANALYST PROMPT [{prompt_type}]:\n" + "="*80 + f"\n{prompt}\n" + "="*80)
            
            # Generate analysis using the LLM wrapper
            result = await self.wrapper_manager.generate_with_wrapper(self.wrapper_type, prompt)
            
            if result:
                self.logger.info(f"📊 Received multi-instrument market analysis - Response length: {len(result)} chars")
                
                # Log detailed response for research tracking
                self.logger.debug(f"📊 MULTI-INSTRUMENT MARKET ANALYST RESPONSE:\n" + "="*80 + f"\n{result}\n" + "="*80)
                
                return result
            else:
                self.logger.warning(f"📊 No result received from multi-instrument market analysis")
                return f"Multi-instrument market analysis unavailable at {self.agent.current_time}"
                
        except Exception as e:
            self.logger.error(f"📊 Error generating multi-instrument market analysis: {e}")
            return f"Multi-instrument market analysis error: {str(e)}"
