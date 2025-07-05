"""
Specialized LLM Wrappers for Different Analyst Types

This module contains concrete implementations of specialized wrappers
for different types of financial analysis agents.
"""

from typing import Dict, Any, Optional
from .base_wrapper import BaseLLMWrapper, ModelConfig


class MarketAnalystWrapper(BaseLLMWrapper):
    """Wrapper for market analysis tasks with technical indicators."""

    def get_wrapper_name(self) -> str:
        return "MarketAnalyst"

    def get_system_prompt(self) -> str:
        return """You are an expert market analyst specializing in technical analysis."""


class NewsAnalystWrapper(BaseLLMWrapper):
    """Wrapper for news sentiment analysis tasks."""

    def get_wrapper_name(self) -> str:
        return "NewsAnalyst"

    def get_system_prompt(self) -> str:
        return """You are an expert financial news analyst specializing in sentiment analysis and market impact assessment."""


class FundamentalAnalystWrapper(BaseLLMWrapper):
    """Wrapper for fundamental analysis tasks."""

    def get_wrapper_name(self) -> str:
        return "FundamentalAnalyst"

    def get_system_prompt(self) -> str:
        return """You are an expert fundamental analyst specializing in financial data analysis."""


class AggregatorWrapper(BaseLLMWrapper):
    """Wrapper for aggregating multiple analysis inputs into final decisions."""

    def get_wrapper_name(self) -> str:
        return "Aggregator"

    def get_system_prompt(self) -> str:
        return """You are a senior portfolio manager responsible for making final trading decisions."""