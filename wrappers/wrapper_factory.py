"""
LLM Wrapper Factory and Configuration Management

This module provides a factory pattern for creating specialized LLM wrappers
with configuration-driven parameter management.
"""

from typing import Dict, Any, Optional, Type, Union
from .base_wrapper import BaseLLMWrapper, ModelConfig
from .specialized_wrappers import (
    MarketAnalystWrapper,
    NewsAnalystWrapper,
    FundamentalAnalystWrapper,
    AggregatorWrapper
)


class WrapperRegistry:
    """Registry for managing wrapper types and their configurations."""

    _wrappers: Dict[str, Type[BaseLLMWrapper]] = {
        "market_analysis": MarketAnalystWrapper,
        "news": NewsAnalystWrapper,
        "fundamental_analysis": FundamentalAnalystWrapper,
        "aggregator": AggregatorWrapper,
    }

    @classmethod
    def register_wrapper(cls, name: str, wrapper_class: Type[BaseLLMWrapper]) -> None:
        """Register a new wrapper type."""
        cls._wrappers[name] = wrapper_class

    @classmethod
    def get_wrapper_class(cls, name: str) -> Optional[Type[BaseLLMWrapper]]:
        """Get wrapper class by name."""
        return cls._wrappers.get(name)

    @classmethod
    def list_available_wrappers(cls) -> list[str]:
        """List all available wrapper types."""
        return list(cls._wrappers.keys())


class ConfigurationManager:
    """Manages configuration loading and parameter resolution."""

    DEFAULT_CONFIGS = {
        "anthropic": {
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": 20000,
            "budget_tokens": 16000,
            "use_thinking": True
        },
        "openai": {
            "temperature": 0.0,
            "top_p": 0.95,
            "reasoning_effort": "high"
        },
        "google": {
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 30000,
            "use_thinking": False
        },
        "llama": {
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 2048,
        },
        "deepseek": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 30000
        }
    }

    ANALYST_SPECIFIC_CONFIGS = {
        "market_analysis": {
            "anthropic": {
                "temperature": 0.0,
                "top_p": 0.95,
                "max_tokens": 20000,
                "budget_tokens": 16000,
                "use_thinking": True
            },
            "openai": {
                "temperature": 0.0,
                "top_p": 0.95,
                "reasoning_effort": "high"
            },
            "google": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 30000,
                "use_thinking": False
            },
            "llama": {
                "temperature": 0.3,
                "top_p": 0.95,
                "max_tokens": 2048,
            },
            "deepseek": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 30000
            }

        },
        "news": {
            "anthropic": {
                "temperature": 0.0,
                "top_p": 0.95,
                "max_tokens": 20000,
                "budget_tokens": 16000,
                "use_thinking": True
            },
            "openai": {
                "temperature": 0.0,
                "top_p": 0.95,
                "reasoning_effort": "high"
            },
            "google": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 30000,
                "use_thinking": False
            },
            "llama": {
                "temperature": 0.3,
                "top_p": 0.95,
                "max_tokens": 2048,
            },
            "deepseek": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 30000
            }

        },
        "fundamental_analysis": {
            "anthropic": {
                "temperature": 0.0,
                "top_p": 0.95,
                "max_tokens": 20000,
                "budget_tokens": 16000,
                "use_thinking": True
            },
            "openai": {
                "temperature": 0.0,
                "top_p": 0.95,
                "reasoning_effort": "high"
            },
            "google": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 30000,
                "use_thinking": False
            },
            "llama": {
                "temperature": 0.3,
                "top_p": 0.95,
                "max_tokens": 2048,
            },
            "deepseek": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 30000
            }
        },
        "aggregator": {
            "anthropic": {
                "temperature": 0.0,
                "top_p": 0.95,
                "max_tokens": 20000,
                "budget_tokens": 16000,
                "use_thinking": True
            },
            "openai": {
                "temperature": 0.0,
                "top_p": 0.95,
                "reasoning_effort": "high"
            },
            "google": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 30000,
                "use_thinking": False
            },
            "llama": {
                "temperature": 0.3,
                "top_p": 0.95,
                "max_tokens": 2048,
            },
            "deepseek": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 30000
            }

        }
    }

    # Standard API key environment variable names
    DEFAULT_API_KEY_MAPPING = {
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_region": "AWS_REGION",
        "openai_api_key": "OPENAI_API_KEY",
        "google_api_key": "GOOGLE_API_KEY",
        "huggingface_token": "HUGGINGFACE_TOKEN"
    }

    @classmethod
    def create_model_config(
            cls,
            model_id: str,
            model_type: str,
            analyst_type: str = "aggregator",
            custom_config: Optional[Dict[str, Any]] = None,
            api_key_mapping: Optional[Dict[str, str]] = None
    ) -> ModelConfig:
        """
        Create a model configuration with appropriate defaults and overrides.

        Args:
            model_id: Model identifier (e.g., "gpt-4", "claude-3-sonnet")
            model_type: Model provider type (e.g., "openai", "anthropic")
            analyst_type: Type of analyst (affects default parameters)
            custom_config: Custom configuration overrides
            api_key_mapping: Custom API key environment variable mapping

        Returns:
            ModelConfig object with resolved parameters
        """
        # Start with provider defaults
        base_config = cls.DEFAULT_CONFIGS.get(model_type, {}).copy()

        # Apply analyst-specific overrides
        analyst_config = cls.ANALYST_SPECIFIC_CONFIGS.get(analyst_type, {})
        provider_analyst_config = analyst_config.get(model_type, {})
        base_config.update(provider_analyst_config)

        # Apply custom overrides
        if custom_config:
            # Filter out use_thinking for non-Anthropic models
            filtered_config = custom_config.copy()
            base_config.update(filtered_config)

        # Handle API key mapping
        api_keys = {}
        key_mapping = api_key_mapping or cls.DEFAULT_API_KEY_MAPPING

        # Extract API key configuration from custom_config if present
        if custom_config and "api_keys" in custom_config:
            api_keys = custom_config["api_keys"].copy()

        # Apply default environment variable mappings
        for key_name, env_var in key_mapping.items():
            if key_name not in api_keys:
                api_keys[key_name] = env_var

        base_config["api_keys"] = api_keys

        return ModelConfig(
            model_id=model_id,
            model_type=model_type,
            **base_config
        )

    @classmethod
    def load_from_config_dict(
            cls,
            config_dict: Dict[str, Any],
            analyst_type: str = "aggregator"
    ) -> ModelConfig:
        """
        Load model configuration from a dictionary (e.g., from YAML config).

        Args:
            config_dict: Configuration dictionary with model_id, model_type, etc.
            analyst_type: Type of analyst for parameter defaults

        Returns:
            ModelConfig object
        """
        model_id = config_dict.get("model_id")
        model_type = config_dict.get("model_type")

        if not model_id or not model_type:
            raise ValueError("model_id and model_type are required in config")

        # Extract custom parameters
        custom_config = {k: v for k, v in config_dict.items()
                         if k not in ["model_id", "model_type"]}

        return cls.create_model_config(
            model_id=model_id,
            model_type=model_type,
            analyst_type=analyst_type,
            custom_config=custom_config
        )


class LLMWrapperFactory:
    """
    Factory for creating specialized LLM wrappers with dynamic configuration.

    This factory supports:
    - Dynamic wrapper registration
    - Configuration-driven parameter management
    - Per-agent and per-analyst parameter customization
    - Multiple model provider support
    """

    @classmethod
    def create_wrapper(
            cls,
            wrapper_type: str,
            model_config: Union[Dict[str, Any], ModelConfig],
            agent_id: str = "default",
            use_history: bool = True,
            agent=None,
            **kwargs
    ) -> BaseLLMWrapper:
        """
        Create a specialized wrapper instance.

        Args:
            wrapper_type: Type of wrapper (e.g., "market_analysis", "news")
            model_config: Model configuration (dict or ModelConfig object)
            agent_id: Unique identifier for the agent
            use_history: Whether to maintain conversation history
            agent: Parent agent instance
            **kwargs: Additional configuration parameters

        Returns:
            Specialized wrapper instance

        Raises:
            ValueError: If wrapper_type is not supported
            TypeError: If model_config is invalid
        """
        # Get wrapper class
        wrapper_class = WrapperRegistry.get_wrapper_class(wrapper_type)
        if not wrapper_class:
            available = WrapperRegistry.list_available_wrappers()
            raise ValueError(f"Unsupported wrapper type: {wrapper_type}. Available: {available}")

        # Convert config dict to ModelConfig if needed
        if isinstance(model_config, dict):
            config = ConfigurationManager.load_from_config_dict(
                model_config,
                analyst_type=wrapper_type
            )
        elif isinstance(model_config, ModelConfig):
            config = model_config
        else:
            raise TypeError("model_config must be a dict or ModelConfig object")

        # Create wrapper instance
        return wrapper_class(
            model_config=config,
            agent_id=agent_id,
            wrapper_type=wrapper_type,
            use_history=use_history,
            agent=agent,
            **kwargs
        )

    @classmethod
    def create_multi_wrapper_manager(
            cls,
            models_config: Dict[str, Dict[str, Any]],
            agent_id: str = "default",
            use_history: bool = True,
            agent=None,
            **kwargs
    ) -> "MultiWrapperManager":
        """
        Create a multi-wrapper manager for handling multiple specialized wrappers.

        Args:
            models_config: Dict mapping wrapper types to model configurations
            agent_id: Unique identifier for the agent
            use_history: Whether to maintain conversation history
            agent: Parent agent instance
            **kwargs: Additional configuration parameters

        Returns:
            MultiWrapperManager instance
        """
        return MultiWrapperManager(
            models_config=models_config,
            agent_id=agent_id,
            use_history=use_history,
            agent=agent,
            **kwargs
        )


class MultiWrapperManager:
    """
    Manager class for handling multiple specialized wrappers.

    This class provides a unified interface for managing multiple
    specialized wrappers with different configurations.
    """

    def __init__(
            self,
            models_config: Dict[str, Dict[str, Any]],
            agent_id: str = "default",
            use_history: bool = True,
            agent=None,
            **kwargs
    ):
        """
        Initialize the multi-wrapper manager.

        Args:
            models_config: Dict mapping wrapper types to model configurations
            agent_id: Unique identifier for the agent
            use_history: Whether to maintain conversation history
            agent: Parent agent instance
            **kwargs: Additional configuration parameters
        """
        self.agent = agent
        self.agent_id = agent_id
        self.use_history = use_history

        # Initialize wrappers for each configured model type
        self.wrappers: Dict[str, BaseLLMWrapper] = {}

        for wrapper_type, model_config in models_config.items():
            if wrapper_type in WrapperRegistry.list_available_wrappers():
                # Check if model has specific history setting, otherwise use default
                model_use_history = model_config.get('use_history', use_history)
                
                # Log history configuration for this wrapper
                if agent and hasattr(agent, 'logger'):
                    agent.logger.info(f"🧠 {wrapper_type} wrapper: history={'ON' if model_use_history else 'OFF'}")
                
                self.wrappers[wrapper_type] = LLMWrapperFactory.create_wrapper(
                    wrapper_type=wrapper_type,
                    model_config=model_config,
                    agent_id=f"{agent_id}_{wrapper_type}",
                    use_history=model_use_history,
                    agent=agent,
                    **kwargs
                )

    def get_wrapper(self, wrapper_type: str) -> BaseLLMWrapper:
        """Get a specific wrapper instance."""
        if self.agent and hasattr(self.agent, 'logger'):
            self.agent.logger.debug(f"🔍 MultiWrapperManager.get_wrapper() called for {wrapper_type}")
            self.agent.logger.debug(f"🔍 Available wrappers: {list(self.wrappers.keys())}")
        
        if wrapper_type not in self.wrappers:
            error_msg = f"Wrapper type {wrapper_type} not configured. Available: {list(self.wrappers.keys())}"
            if self.agent and hasattr(self.agent, 'logger'):
                self.agent.logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        wrapper = self.wrappers[wrapper_type]
        if self.agent and hasattr(self.agent, 'logger'):
            self.agent.logger.debug(f"✅ Retrieved wrapper {wrapper_type}: {type(wrapper).__name__}")
        
        return wrapper

    async def generate_with_wrapper(self, wrapper_type: str, prompt: str, **kwargs) -> Optional[str]:
        """Generate response using a specific wrapper."""
        try:
            wrapper = self.get_wrapper(wrapper_type)
            if self.agent and hasattr(self.agent, 'logger'):
                self.agent.logger.debug(f"🔄 MultiWrapperManager calling {wrapper_type} wrapper with prompt length: {len(prompt)}")
            
            result = await wrapper.generate(prompt, **kwargs)
            
            if self.agent and hasattr(self.agent, 'logger'):
                if result:
                    self.agent.logger.debug(f"✅ MultiWrapperManager received response from {wrapper_type} wrapper: {len(result)} chars")
                else:
                    self.agent.logger.warning(f"❌ MultiWrapperManager received no response from {wrapper_type} wrapper")
            
            return result
        except Exception as e:
            if self.agent and hasattr(self.agent, 'logger'):
                self.agent.logger.error(f"❌ MultiWrapperManager error with {wrapper_type} wrapper: {e}")
            return None

    def set_system_prompt(self, wrapper_type: str, system_prompt: str) -> None:
        """Set system prompt for a specific wrapper."""
        wrapper = self.get_wrapper(wrapper_type)
        wrapper.set_system_prompt(system_prompt)

    def clear_system_prompt_override(self, wrapper_type: str) -> None:
        """Clear system prompt override for a specific wrapper."""
        wrapper = self.get_wrapper(wrapper_type)
        wrapper.clear_system_prompt_override()

    def list_configured_wrappers(self) -> list[str]:
        """List all configured wrapper types."""
        return list(self.wrappers.keys())
