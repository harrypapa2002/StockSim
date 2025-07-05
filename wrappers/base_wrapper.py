"""
Base LLM Wrapper - Modular Implementation for StockSim Demo

This module provides a clean, modular base class for LLM wrappers that supports
dynamic model registration and configuration-driven parameter management.

Key Features:
- Dynamic model registration and factory pattern
- Full configuration-driven parameter management via config files
- Support for AWS (Anthropic, Llama, DeepSeek) and OpenAI
- Modular conversation history management
- Clean separation of concerns
- Extensible architecture for adding new model providers
- Comprehensive performance monitoring and API usage tracking
"""

from __future__ import annotations

from pathlib import Path

import asyncio
import contextlib
import io
import os
import json
import random
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union


# Core dependencies
from openai import AsyncOpenAI

from aiobotocore.config import AioConfig
from aiohttp.client_exceptions import SocketTimeoutError
from botocore.exceptions import ClientError
from aiobotocore.session import get_session

# Tokenizer imports (with silence)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from transformers import AutoTokenizer, logging
    logging.set_verbosity_error()

from dotenv import load_dotenv
from utils.logging_setup import setup_logger

# Web search imports
from utils.web_search_tool import WebSearchTool


def pretty_claude_response(raw: str | dict) -> str:
    """
    Turn the `content` array Claude returns into a compact, human-readable
    block – one logical line per item.

    Example output:

        [text]   Here is a summary
        [tool]   web_search  id=call_01  args={'query': '…'}
        [image]  (512×512 PNG)
    """
    import json, textwrap

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            # if caller passes only the "content" array already parsed:
            pass

    content = raw if isinstance(raw, list) else raw.get("content", [])
    lines: list[str] = []

    for itm in content:
        t = itm.get("type")
        if t == "text":
            txt = itm.get("text", "").strip().replace("\n", " ⏎ ")
            lines.append(f"[text]   {txt}")
        elif t == "tool_use":
            name = itm.get("name")
            tid = itm.get("id")
            args = json.dumps(itm.get("input", {}), ensure_ascii=False)
            lines.append(f"[tool]   {name:<12} id={tid}  args={args}")
        elif t == "image":
            meta = itm.get("source", {})
            lines.append(
                f"[image]  ({meta.get('width', '?')}×{meta.get('height', '?')} {meta.get('mime_type', 'file')})")
        else:
            lines.append(f"[{t}]    {json.dumps(itm, ensure_ascii=False)}")
    return textwrap.indent("\n".join(lines), prefix="  ")


class ModelConfig:
    """Configuration container for model parameters."""

    def __init__(
        self,
        model_id: str,
        model_type: str,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 20000,
        budget_tokens: int = 16000,
        use_thinking: bool = False,
        reasoning_effort: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.budget_tokens = budget_tokens
        self.use_thinking = use_thinking
        self.reasoning_effort = reasoning_effort or "high"
        self.api_keys = api_keys or {}
        self.extra_params = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "budget_tokens": self.budget_tokens,
            "use_thinking": self.use_thinking,
            "reasoning_effort": self.reasoning_effort,
            "api_keys": self.api_keys,
            **self.extra_params
        }


class ConversationManager:
    """Manages conversation history with format normalization."""

    def __init__(self, provider: BaseModelProvider, use_history: bool = True):
        self.provider = provider
        self.use_history = use_history
        self.history: List[Dict[str, Any]] = []
        self.lock = asyncio.Lock()

    def add_message(self, role: str, content: Union[str, List[Dict[str, Any]]]) -> None:
        """Add a message to conversation history."""
        if not self.use_history:
            return

        message = self.provider.format_message(role, content)
        if message:
            self.history.append(message)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get formatted conversation history."""
        return self.history.copy()


def resolve_credential(
    key_name: str,
    mapping: Dict[str, str],
    *,
    env_fallback: bool = True,
) -> Optional[str]:
    """
    Return a usable secret value.

    1. Look in `mapping[key_name]`.
       • If it is *itself* an env-var **label** (UPPER-CASE, no spaces), read that env-var.
         Example: mapping = {"openai_api_key": "OPENAI_API_KEY"} ⇒
         we return os.getenv("OPENAI_API_KEY").

       • Otherwise treat the value as the secret itself and return it.

    2. If still empty and `env_fallback` is True, fall back to
       os.getenv(key_name.upper()).

    Parameters
    ----------
    key_name : the logical name (e.g. "openai_api_key")
    mapping  : the api_keys mapping block
    env_fallback : set False if you *never* want fallback

    Returns
    -------
    str | None
    """
    raw = mapping.get(key_name)
    if raw and raw.isupper() and " " not in raw:
        # Looks like an env-var label
        return os.getenv(raw)
    return raw or (os.getenv(key_name.upper()) if env_fallback else None)

class BaseModelProvider(ABC):
    """Abstract base class for model providers."""

    @abstractmethod
    async def generate_response(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        **kwargs
    ) -> Optional[str]:
        """Generate response using the model provider."""
        pass

    @abstractmethod
    def validate_config(self, config: ModelConfig) -> List[str]:
        """Validate model configuration for this provider."""
        pass

    @abstractmethod
    def format_message(self, role: str, content: Union[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Format message according to provider's expected format."""
        pass

    def _extract_text_from_content(self, content: List[Dict[str, Any]]) -> str:
        """Extract text from content array - common utility method."""
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                text_parts.append(item["text"])
        return "\n".join(text_parts)


class AnthropicProvider(BaseModelProvider):
    """Anthropic/Claude provider implementation."""

    def _extract_text_from_claude(self, raw: str) -> str:
        try:
            output_json = json.loads(raw)
            content = output_json.get("content", [])
            return "\n".join(
                item["text"] for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        except Exception:
            return raw  # fallback


    def format_message(self, role: str, content: Union[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Format message for Anthropic API."""
        if not content or (isinstance(content, str) and not content.strip()):
            return None

        if isinstance(content, str):
            return {"role": role, "content": [{"type": "text", "text": content}]}
        elif isinstance(content, list):
            return {"role": role, "content": content}
        else:
            return None

    async def generate_response(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        **kwargs
    ) -> Optional[str]:
        """Generate response using Anthropic models."""
        # Extract web search parameters
        urls_to_search = kwargs.get("urls_to_search")
        logger = kwargs.get("logger")
        agent_id = kwargs.get("agent_id", "unknown")

        # Build payload
        payload = {
            "system": system_prompt,
            "messages": messages,
            "max_tokens": model_config.max_tokens,
            "anthropic_version": "bedrock-2023-05-31"
        }

        # Add web search tool if URLs are provided
        if urls_to_search and logger:
            web_search_tool = WebSearchTool(urls_to_search, logger)
            tool_def = web_search_tool.get_tool_definition()
            payload["tools"] = [tool_def]
            logger.info(f"Added web search tool with {len(urls_to_search)} available URLs")

        if model_config.use_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": model_config.budget_tokens}
        else:
            payload["temperature"] = model_config.temperature
            payload["top_p"] = model_config.top_p

        # Log full API request
        if logger:
            logger.debug(f"🔄 ANTHROPIC API REQUEST:\n" + "="*80 + f"\nModel: {model_config.model_id}\nPayload: {json.dumps(payload, indent=2)}\n" + "="*80)


        try:
            # Make API call
            raw = await self._make_aws_api_call(model_config, payload, logger, strip= not bool(urls_to_search))

            if raw:
                pass
        except Exception as e:
            if logger:
                logger.error(f"❌ Anthropic API call failed: {e}")
        finally:
            pass

        if not raw:
            return None

        # Handle tool calls if present
        if urls_to_search and logger and raw:
            return await self._handle_anthropic_tool_calls(model_config, raw, payload, urls_to_search, logger)

        return self._extract_text_from_claude(raw)

    def validate_config(self, config: ModelConfig) -> List[str]:
        """Validate Anthropic model configuration."""
        errors = []

        # Check for required API keys
        aws_access_key = config.api_keys.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = config.api_keys.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key:
            errors.append("AWS_ACCESS_KEY_ID is required for Anthropic models")
        if not aws_secret_key:
            errors.append("AWS_SECRET_ACCESS_KEY is required for Anthropic models")

        return errors

    async def _make_aws_api_call(self, model_config: ModelConfig, payload: Dict[str, Any], logger, strip: bool = True) -> Optional[str]:
        """Make AWS Bedrock API call with retry logic."""
        max_retries = 10
        base_delay = 3.0

        aws_access_key = resolve_credential("aws_access_key_id", model_config.api_keys)
        aws_secret_key = resolve_credential("aws_secret_access_key", model_config.api_keys)
        aws_region = resolve_credential("aws_region", model_config.api_keys) or "us-east-1"

        for attempt in range(max_retries):
            try:
                session = get_session()
                aws_config = AioConfig(read_timeout=3600)

                async with session.create_client(
                    "bedrock-runtime",
                    region_name=aws_region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    config=aws_config
                ) as client:
                    response = await client.invoke_model(
                        modelId=model_config.model_id,
                        contentType="application/json",
                        body=json.dumps(payload),
                    )
                    output_binary = await response["body"].read()
                    output_str = output_binary.decode("utf-8")

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "📥 CLAUDE RAW RESPONSE\n" + "=" * 80 +
                            f"\n{pretty_claude_response(output_str)}\n" + "=" * 80
                        )

                    if not strip:
                        return output_str

                        # Parse response
                    try:
                        output_json = json.loads(output_str)
                        content = output_json.get("content", [])
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item["text"])
                        return "\n".join(text_parts).strip()
                    except json.JSONDecodeError:
                        return None

            except (ClientError, SocketTimeoutError) as e:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)

                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    return None
            except Exception:
                return None

        return None

    async def _handle_anthropic_tool_calls(self, model_config: ModelConfig, result: str, original_payload: Dict[str, Any], urls_to_search: List[str], logger) -> str:
        """Handle tool calls from Anthropic models and get final response"""
        try:
            output_json = json.loads(result)
            content = output_json.get("content", [])

            # Check if there are tool calls
            has_tool_calls = any(
                isinstance(item, dict) and item.get("type") == "tool_use"
                for item in content
            )

            if not has_tool_calls:
                # No tool calls, extract text response
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                        text_parts.append(item["text"])
                return "\n".join(text_parts).strip()

            # Build conversation with the assistant's tool use
            conversation = original_payload["messages"].copy()
            assistant_message = {
                "role": "assistant",
                "content": content,
            }
            conversation.append(assistant_message)

            # Process each tool call
            tool_results = []
            web_search_tool = WebSearchTool(urls_to_search, logger)

            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_name = item.get("name")
                    tool_input = item.get("input", {})
                    tool_use_id = item.get("id")

                    logger.info(f"Processing tool call: {tool_name} with input: {tool_input}")

                    if tool_name == "web_search":
                        # Execute the web search tool
                        search_result = await web_search_tool.execute_search(tool_input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": [search_result]
                        })
                    else:
                        # Unknown tool
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]
                        })

            # Add tool results to conversation
            if tool_results:
                tool_message = {
                    "role": "user",
                    "content": tool_results
                }
                conversation.append(tool_message)

                # Make another API call to get the final response
                final_payload = {
                    "system": original_payload["system"],
                    "messages": conversation,
                    "max_tokens": model_config.max_tokens,
                    "anthropic_version": "bedrock-2023-05-31"
                }

                if model_config.use_thinking:
                    final_payload["thinking"] = {"type": "enabled", "budget_tokens": model_config.budget_tokens}
                else:
                    final_payload["temperature"] = model_config.temperature

                logger.info(f"Making follow-up API call after tool execution")
                if logger.isEnabledFor(logging.DEBUG):
                    pretty_fp = json.dumps(final_payload, indent=2, ensure_ascii=False)
                    logger.debug("📤 FOLLOW-UP PAYLOAD → Claude\n" + pretty_fp)
                final_result = await self._make_aws_api_call(model_config, final_payload, logger)

                final_text = self._extract_text_from_claude(final_result)
                if final_text:
                    return final_text
                logger.warning("Tool call resulted in empty final response")
                return ""
            return ""

        except Exception as e:
            logger.error(f"Tool call handling failed: {e}")
            return ""


class OpenAIProvider(BaseModelProvider):
    """OpenAI provider implementation."""

    def format_message(self, role: str, content: Union[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Format message for OpenAI API."""
        if not content or (isinstance(content, str) and not content.strip()):
            return None

        if isinstance(content, str):
            return {"role": role, "content": content}
        else:
            text = self._extract_text_from_content(content)
            return {"role": role, "content": text} if text else None

    async def generate_response(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        **kwargs
    ) -> Optional[str]:
        """Generate response using OpenAI models."""
        api_key = resolve_credential("openai_api_key", model_config.api_keys)
        if not api_key:
            raise RuntimeError("OpenAI API key not found")

        # Extract web search parameters
        urls_to_search = kwargs.get("urls_to_search")
        logger = kwargs.get("logger")
        agent_id = kwargs.get("agent_id", "unknown")

        client = AsyncOpenAI(api_key=api_key)


        try:
            formatted_messages = []
            if system_prompt:
                formatted_messages.append({"role": "developer", "content": system_prompt})

            formatted_messages.extend(messages)

            # Build request parameters
            request_params = {
                "model": model_config.model_id,
                "messages": formatted_messages
            }

            # Add web search tool if URLs are provided
            if urls_to_search and logger:
                web_search_tool = WebSearchTool(urls_to_search, logger)
                tool_def = self._get_openai_tool_definition(web_search_tool)
                request_params["tools"] = [tool_def]
                request_params["tool_choice"] = "required"
                logger.info(f"Added web search tool with {len(urls_to_search)} available URLs")

            # Add parameters based on model type
            if not (model_config.model_id.startswith("o1") or
                   model_config.model_id.startswith("o3") or
                   model_config.model_id.startswith("o4")):
                request_params.update({
                    "temperature": model_config.temperature,
                    "top_p": model_config.top_p,
                    "seed": 42
                })

            if (model_config.model_id.startswith("o3") or
                model_config.model_id.startswith("o4") or
                model_config.model_id.startswith("o1")):
                request_params["reasoning_effort"] = model_config.reasoning_effort

            # Log full API request
            if logger:
                logger.debug(f"🔄 OPENAI API REQUEST:\n" + "="*80 + f"\nModel: {model_config.model_id}\nParameters: {json.dumps(request_params, indent=2)}\n" + "="*80)

            response = await client.chat.completions.create(**request_params)

            if response.choices:
                choice = response.choices[0]

                # Check if there are tool calls
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls and urls_to_search and logger:
                    return await self._handle_openai_tool_calls(
                        client, formatted_messages, choice.message, request_params, urls_to_search, logger
                    )

                return choice.message.content
            return None

        except Exception as e:
            if logger:
                logger.error(f"❌ OpenAI API call failed: {e}")
            return None
        finally:
            await client.close()

    def _get_openai_tool_definition(self, web_search_tool) -> Dict[str, Any]:
        """Get OpenAI tool definition from web search tool."""
        tool_def = web_search_tool.get_tool_definition()
        return {
            "type": "function",
            "function": {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "parameters": tool_def["input_schema"]
            }
        }

    async def _handle_openai_tool_calls(
        self,
        client: AsyncOpenAI,
        messages: List[Dict[str, Any]],
        assistant_message,
        request_params: Dict[str, Any],
        urls_to_search: List[str],
        logger
    ) -> str:
        """Handle OpenAI tool calls and get final response"""
        try:
            # Create a copy of messages for this tool call sequence
            tool_messages = messages.copy()

            # Add assistant message with tool calls to the working conversation
            assistant_msg = {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in assistant_message.tool_calls
                ]
            }
            tool_messages.append(assistant_msg)

            # Process each tool call and add tool results
            web_search_tool = WebSearchTool(urls_to_search, logger)

            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "web_search":
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                        logger.info(f"Processing OpenAI tool call: web_search with input: {tool_input}")

                        search_result = await web_search_tool.execute_search(tool_input)

                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": search_result["text"]
                        }
                        tool_messages.append(tool_message)

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool arguments: {e}")
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Failed to parse tool arguments"
                        }
                        tool_messages.append(tool_message)
                else:
                    # Unknown tool
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Unknown tool: {tool_call.function.name}"
                    }
                    tool_messages.append(tool_message)

            # Get final response after tool execution
            final_params = request_params.copy()
            final_params["messages"] = tool_messages
            final_params.pop("tools", None)  # Remove tools for final call
            final_params.pop("tool_choice", None)  # Remove tool_choice if present

            logger.info(f"Making follow-up OpenAI call after tool execution")
            final_response = await client.chat.completions.create(**final_params)

            if final_response.choices and len(final_response.choices) > 0:
                final_content = final_response.choices[0].message.content
                return final_content or ""
            else:
                logger.warning("No final response from OpenAI after tool calls")
                return ""

        except Exception as e:
            logger.error(f"Error handling OpenAI tool calls: {e}")
            return ""

    def validate_config(self, config: ModelConfig) -> List[str]:
        """Validate OpenAI model configuration."""
        errors = []

        api_key = config.api_keys.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            errors.append("OPENAI_API_KEY is required for OpenAI models")

        return errors


class LlamaProvider(BaseModelProvider):
    """Llama provider implementation (via AWS Bedrock)."""

    def format_message(self, role: str, content: Union[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Format message for Llama models."""
        if not content or (isinstance(content, str) and not content.strip()):
            return None

        if isinstance(content, str):
            return {"role": role, "text": content}
        else:
            text = self._extract_text_from_content(content)
            return {"role": role, "text": text} if text else None

    async def generate_response(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        **kwargs
    ) -> Optional[str]:
        """Generate response using Llama models."""
        # Extract logger from kwargs
        logger = kwargs.get("logger")
        agent_id = kwargs.get("agent_id", "unknown")
        
        # Build tokenized prompt
        tokenizer = self._get_tokenizer(model_config)
        if not tokenizer:
            if logger:
                logger.error("❌ Failed to load tokenizer for Llama model")
            return None

        # Convert messages to text format and apply chat template
        text_messages = []
        if system_prompt:
            text_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if "text" in msg:
                text_messages.append({"role": msg["role"], "content": msg["text"]})

        formatted_prompt = tokenizer.apply_chat_template(
            conversation=text_messages,
            add_generation_prompt=True,
            tokenize=False
        )

        payload = {
            "prompt": formatted_prompt,
            "temperature": model_config.temperature,
            "max_gen_len": model_config.max_tokens,
        }

        # Log full API request
        if logger:
            logger.debug(f"🔄 LLAMA API REQUEST:\n" + "="*80 + f"\nModel: {model_config.model_id}\nPayload: {json.dumps(payload, indent=2)}\n" + "="*80)

        return await self._make_aws_api_call(model_config, payload, logger)

    def validate_config(self, config: ModelConfig) -> List[str]:
        """Validate Llama model configuration."""
        errors = []

        aws_access_key = config.api_keys.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = config.api_keys.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key:
            errors.append("AWS_ACCESS_KEY_ID is required for Llama models")
        if not aws_secret_key:
            errors.append("AWS_SECRET_ACCESS_KEY is required for Llama models")

        return errors

    def _get_tokenizer(self, config: ModelConfig):
        """Get tokenizer for Llama models."""
        try:
            if config.model_id.startswith("meta.llama3-1-405b"):
                return AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-3.1-405B-Instruct",
                    token=os.getenv("HUGGINGFACE_TOKEN")
                )
        except Exception:
            pass
        return None

    async def _make_aws_api_call(self, model_config: ModelConfig, payload: Dict[str, Any], logger=None) -> Optional[str]:
        """Make AWS API call for Llama models."""
        aws_access_key = resolve_credential("aws_access_key_id", model_config.api_keys)
        aws_secret_key = resolve_credential("aws_secret_access_key", model_config.api_keys)
        aws_region = resolve_credential("aws_region", model_config.api_keys) or "us-east-1"

        try:
            session = get_session()
            aws_config = AioConfig(read_timeout=3600)

            async with session.create_client(
                "bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                config=aws_config
            ) as client:
                response = await client.invoke_model(
                    modelId=model_config.model_id,
                    contentType="application/json",
                    body=json.dumps(payload),
                )
                output_binary = await response["body"].read()
                output_str = output_binary.decode("utf-8")

                # Log raw response
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "📥 LLAMA RAW RESPONSE\n" + "=" * 80 +
                        f"\n{output_str}\n" + "=" * 80
                    )

                try:
                    output_json = json.loads(output_str)
                    text = output_json.get("generation", "")
                    # Clean llama response
                    text = re.sub(r"^<\|start_header_id\|>assistant<\|end_header_id\|>\s*", "", text)
                    return text.strip()
                except json.JSONDecodeError:
                    if logger:
                        logger.error("❌ Failed to parse Llama response JSON")
                    return None

        except Exception as e:
            if logger:
                logger.error(f"❌ Llama API call failed: {e}")
            return None


class DeepSeekProvider(BaseModelProvider):
    """DeepSeek provider implementation (via AWS Bedrock)."""

    def format_message(self, role: str, content: Union[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Format message for DeepSeek models."""
        if not content or (isinstance(content, str) and not content.strip()):
            return None

        if isinstance(content, str):
            return {"role": role, "text": content}
        else:
            text = self._extract_text_from_content(content)
            return {"role": role, "text": text} if text else None

    async def generate_response(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        **kwargs
    ) -> Optional[str]:
        """Generate response using DeepSeek models."""
        # Extract logger from kwargs
        logger = kwargs.get("logger")
        agent_id = kwargs.get("agent_id", "unknown")
        
        # Build tokenized prompt
        tokenizer = self._get_tokenizer(model_config)
        if not tokenizer:
            if logger:
                logger.error("❌ Failed to load tokenizer for DeepSeek model")
            return None

        # Convert messages and apply chat template
        text_messages = []

        # For DeepSeek, integrate system prompt into the first user message
        for i, msg in enumerate(messages):
            if "text" in msg:
                if i == 0 and system_prompt and msg["role"] == "user":
                    # Integrate system prompt into first user message
                    combined_content = f"{system_prompt}\n\n{msg['text']}"
                    text_messages.append({"role": msg["role"], "content": combined_content})
                else:
                    text_messages.append({"role": msg["role"], "content": msg["text"]})

        formatted_prompt = tokenizer.apply_chat_template(
            conversation=text_messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Add reasoning instruction
        if "Please reason step by step" not in formatted_prompt:
            formatted_prompt += "\n\nPlease reason step by step."

        payload = {
            "prompt": formatted_prompt,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "stop": []
        }

        # Log full API request
        if logger:
            logger.debug(f"🔄 DEEPSEEK API REQUEST:\n" + "="*80 + f"\nModel: {model_config.model_id}\nPayload: {json.dumps(payload, indent=2)}\n" + "="*80)

        return await self._make_aws_api_call(model_config, payload, logger)

    def validate_config(self, config: ModelConfig) -> List[str]:
        """Validate DeepSeek model configuration."""
        errors = []

        aws_access_key = config.api_keys.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = config.api_keys.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key:
            errors.append("AWS_ACCESS_KEY_ID is required for DeepSeek models")
        if not aws_secret_key:
            errors.append("AWS_SECRET_ACCESS_KEY is required for DeepSeek models")

        return errors

    def _get_tokenizer(self, config: ModelConfig):
        """Get tokenizer for DeepSeek models."""
        try:
            return AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-R1",
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
        except Exception:
            pass
        return None

    async def _make_aws_api_call(self, model_config: ModelConfig, payload: Dict[str, Any], logger=None) -> Optional[str]:
        """Make AWS API call for DeepSeek models."""
        aws_access_key = resolve_credential("aws_access_key_id", model_config.api_keys)
        aws_secret_key = resolve_credential("aws_secret_access_key", model_config.api_keys)
        aws_region = resolve_credential("aws_region", model_config.api_keys) or "us-east-1"

        try:
            session = get_session()
            aws_config = AioConfig(read_timeout=3600)

            async with session.create_client(
                "bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                config=aws_config
            ) as client:
                response = await client.invoke_model(
                    modelId=model_config.model_id,
                    contentType="application/json",
                    body=json.dumps(payload),
                )
                output_binary = await response["body"].read()
                output_str = output_binary.decode("utf-8")

                # Log raw response
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "📥 DEEPSEEK RAW RESPONSE\n" + "=" * 80 +
                        f"\n{output_str}\n" + "=" * 80
                    )

                try:
                    output_json = json.loads(output_str)
                    choices = output_json.get("choices", [])
                    if choices:
                        text = choices[0].get("text", "")
                        # Clean DeepSeek thinking tags
                        text = re.sub(r'^.*?</think>\n+', '', text, flags=re.DOTALL)
                        return text.strip()
                    return ""
                except json.JSONDecodeError:
                    if logger:
                        logger.error("❌ Failed to parse DeepSeek response JSON")
                    return None

        except Exception as e:
            if logger:
                logger.error(f"❌ DeepSeek API call failed: {e}")
            return None


class ProviderRegistry:
    """Registry for model providers."""

    _providers: Dict[str, BaseModelProvider] = {
        "anthropic": AnthropicProvider(),
        "openai": OpenAIProvider(),
        "llama": LlamaProvider(),
        "deepseek": DeepSeekProvider()
    }

    @classmethod
    def register_provider(cls, name: str, provider: BaseModelProvider) -> None:
        """Register a new model provider."""
        cls._providers[name] = provider

    @classmethod
    def get_provider(cls, name: str) -> Optional[BaseModelProvider]:
        """Get provider by name."""
        return cls._providers.get(name)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all available providers."""
        return list(cls._providers.keys())


class BaseLLMWrapper(ABC):
    """
    Base class for all LLM wrappers with modular design.

    Features:
    - Dynamic provider registration
    - Configuration-driven parameters
    - Pluggable conversation management
    - Provider-agnostic interface
    """

    def __init__(
        self,
        model_config: ModelConfig,
        agent_id: str = "default",
        use_history: bool = True,
        agent=None,
        **kwargs
    ):
        """
        Initialize base LLM wrapper.

        Args:
            config: Model configuration object
            agent_id: Unique identifier for the agent
            use_history: Whether to maintain conversation history
            agent: Parent agent instance (for logging)
            **kwargs: Additional configuration parameters
        """
        load_dotenv()

        self.config = model_config
        self.agent_id = agent_id
        self.agent = agent

        # Initialize logging
        self._setup_logging()

        # Get provider first (before validation and conversation manager)
        self.provider = ProviderRegistry.get_provider(model_config.model_type)
        if not self.provider:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")

        # Validate configuration
        self._validate_configuration()

        # Initialize conversation manager (now that provider is available)
        self.conversation = ConversationManager(
            provider=self.provider,
            use_history=use_history
        )

        # System prompt management
        self._system_prompt_override: Optional[str] = None

        self.logger.info(f"{self.get_wrapper_name()} initialized for {model_config.model_type}:{model_config.model_id}")

    def _setup_logging(self) -> None:
        root = Path(os.getenv("LOG_DIR", "logs"))

        file_path = (
                root / "wrappers" / f"{self.agent_id}.log"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(
            f"{self.get_wrapper_name()}[{self.agent_id}]",
            file_path,
            log_to_stdout=False
        )

    def _validate_configuration(self) -> None:
        """Validate the model configuration."""
        errors = self.provider.validate_config(self.config)
        if errors:
            error_msg = f"Configuration validation failed: {'; '.join(errors)}"
            raise ValueError(error_msg)

    # System prompt management
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set a custom system prompt override."""
        self._system_prompt_override = system_prompt
        self.logger.debug("System prompt override set")

    def clear_system_prompt_override(self) -> None:
        """Clear the system prompt override."""
        self._system_prompt_override = None
        self.logger.debug("System prompt override cleared")

    def get_effective_system_prompt(self) -> str:
        """Get the effective system prompt (override if set, otherwise default)."""
        return self._system_prompt_override or self.get_system_prompt()

    # Main interface methods
    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Main interface for generating responses.

        Args:
            prompt: User prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response or None if failed
        """
        if not prompt or not prompt.strip():
            self.logger.warning("Empty prompt provided")
            return None

        self.logger.debug(f"🔄 {self.get_wrapper_name()} starting generation (use_history={self.conversation.use_history})")

        async with self.conversation.lock:
            try:
                # Get conversation history and system prompt
                system_prompt = self.get_effective_system_prompt()
                self.logger.debug(f"📝 {self.get_wrapper_name()} system prompt length: {len(system_prompt)}")
                
                if self.conversation.use_history:
                    # Add user message to persistent history
                    self.conversation.add_message("user", prompt)
                    messages = self.conversation.get_history()
                    self.logger.debug(f"📚 {self.get_wrapper_name()} using history with {len(messages)} messages")
                else:
                    # Create temporary message list without persisting
                    user_message = self.provider.format_message("user", prompt)
                    messages = [user_message] if user_message else []
                    self.logger.debug(f"📝 {self.get_wrapper_name()} using temporary message (no history)")

                self.logger.debug(f"🚀 {self.get_wrapper_name()} calling provider.generate_response...")

                # Generate response using provider (include agent_id for performance tracking)
                response = await self.provider.generate_response(
                    model_config=self.config,
                    messages=messages,
                    system_prompt=system_prompt,
                    logger=self.logger,
                    agent_id=self.agent_id,
                    urls_to_search=kwargs.get("urls_to_search", [])
                )

                self.logger.debug(f"📥 {self.get_wrapper_name()} provider returned response: {len(response) if response else 0} chars")

                # Add assistant response to history only if using history
                if response and response.strip() and self.conversation.use_history:
                    self.conversation.add_message("assistant", response)

                return response

            except Exception as e:
                self.logger.error(f"❌ {self.get_wrapper_name()} generation failed: {e}")
                import traceback
                self.logger.error(f"❌ {self.get_wrapper_name()} traceback: {traceback.format_exc()}")
                return None

    # Abstract methods that subclasses must implement
    @abstractmethod
    def get_wrapper_name(self) -> str:
        """Return the name of this wrapper for logging purposes."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent type."""
        pass
