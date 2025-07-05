"""
LLM Wrappers Module - Modular Implementation for StockSim Demo

This module provides a comprehensive, modular system for managing LLM wrappers
across different providers and specialized analysis tasks.

Key Features:
- Dynamic model registration and factory pattern
- Full configuration-driven parameter management via config files
- Support for AWS (Anthropic, Llama, DeepSeek) and OpenAI
- Specialized wrappers for different analysis types
- Extensible architecture for adding new model providers
- Clean separation of concerns and comprehensive error handling

Usage Examples:

    # Simple wrapper creation from config
    from wrappers import create_llm_wrapper

    wrapper = create_llm_wrapper(
        model_config={
            "model_id": "gpt-4",
            "model_type": "openai",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 16000,
            "api_keys": {
                "openai_api_key": "OPENAI_API_KEY"
            }
        },
        wrapper_type="market_analysis"
    )

    # Multi-wrapper manager for complex agents
    from wrappers import LLMWrapperFactory

    manager = LLMWrapperFactory.create_multi_wrapper_manager(
        models_config={
            "market_analysis": {
                "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "model_type": "anthropic",
                "temperature": 0.0,
                "use_thinking": True,
                "api_keys": {
                    "aws_access_key_id": "AWS_ACCESS_KEY_ID",
                    "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
                    "aws_region": "AWS_REGION"
                }
            },
            "news": {
                "model_id": "gpt-4",
                "model_type": "openai",
                "temperature": 0.3,
                "api_keys": {
                    "openai_api_key": "OPENAI_API_KEY"
                }
            }
        }
    )

    # Generate responses
    market_analysis = await manager.generate_with_wrapper(
        "market_analysis",
        "Analyze NVDA price action..."
    )

    # Dynamic provider registration
    from wrappers import ProviderRegistry
    from wrappers.base_wrapper import BaseModelProvider

    class CustomProvider(BaseModelProvider):
        async def generate_response(self, config, messages, system_prompt, **kwargs):
            # Custom implementation
            return "Custom response"

        def validate_config(self, config):
            return []  # No validation errors

    ProviderRegistry.register_provider("custom", CustomProvider())
"""

from .base_wrapper import (
    BaseLLMWrapper,
    ModelConfig,
    ConversationManager,
    BaseModelProvider,
    ProviderRegistry
)
from .specialized_wrappers import (
    MarketAnalystWrapper,
    NewsAnalystWrapper,
    FundamentalAnalystWrapper,
    AggregatorWrapper
)
from .wrapper_factory import (
    WrapperRegistry,
    ConfigurationManager,
    LLMWrapperFactory,
    MultiWrapperManager
)

# Version information
__version__ = "2.0.0"
__description__ = "Modular LLM wrapper system for financial analysis agents"

# Public API
__all__ = [
    # Core classes
    "BaseLLMWrapper",
    "ModelConfig",
    "ConversationManager",
    "BaseModelProvider",
    "ProviderRegistry",

    # Specialized wrappers
    "MarketAnalystWrapper",
    "NewsAnalystWrapper",
    "FundamentalAnalystWrapper",
    "AggregatorWrapper",

    # Factory and management
    "WrapperRegistry",
    "ConfigurationManager",
    "LLMWrapperFactory",
    "MultiWrapperManager",

    # Module metadata
    "__version__",
    "__description__"
]


# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"LLM Wrappers module v{__version__} initialized")
logger.info(f"Available wrappers: {WrapperRegistry.list_available_wrappers()}")
logger.info(f"Supported providers: {ProviderRegistry.list_providers()}")
