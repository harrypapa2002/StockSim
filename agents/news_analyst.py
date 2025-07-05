"""
News Analyst Agent for StockSim Demo

This module implements the news sentiment analysis component of the multi-agent
LLM coordination system described in the EMNLP 2025 paper. The NewsAnalyst
provides sophisticated analysis of financial news, market sentiment, and
information flows that could impact trading decisions.

Key Features:
- Real-time news sentiment analysis and trend identification
- Source credibility assessment and consensus analysis
- Market relevance scoring for news events
- Integration with web search capabilities for detailed article analysis
- Professional-grade prompt engineering for financial news interpretation
- Comprehensive logging for research analysis

This agent demonstrates the multi-modal information processing capabilities
described in the paper, where news analysis is coordinated with technical
and fundamental analysis for comprehensive trading decisions.
"""

import os
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any

import jinja2

from utils.logging_setup import setup_logger
from utils.time_utils import seconds_to_human


def format_joined_news(news_items: List[Dict[str, Any]]) -> str:
    """
    Format news items into a structured, readable format for LLM analysis.

    Args:
        news_items: List of news item dictionaries with metadata

    Returns:
        Formatted string containing structured news information
    """
    formatted_news = []

    for item in news_items:
        timestamp = item.get("timestamp", "unknown")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()

        title = item.get("headline", "No Headline")
        source = item.get("source", "Unknown Source")
        description = item.get("description", "").strip()
        url = item.get("url", "N/A")
        keywords = item.get("keywords", [])

        formatted = f"[{timestamp}] {title} - {source}\nURL: {url}"
        formatted += f"\n{description}"
        if keywords:
            formatted += f"\nKeywords: {', '.join(keywords)}"

        formatted_news.append(formatted)

    return "\n\n".join(formatted_news)


def extract_urls_from_news(news_items: List[Dict[str, Any]]) -> List[str]:
    """
    Extract valid URLs from news items for potential web search integration.

    Args:
        news_items: List of news item dictionaries

    Returns:
        List of valid HTTP/HTTPS URLs
    """
    urls = []
    for item in news_items:
        url = item.get("url", "")
        if url and url != "N/A" and url.startswith(("http://", "https://")):
            urls.append(url)
    return urls

class NewsAnalyst:
    """
    News Analyst providing comprehensive sentiment and information analysis for trading decisions.

    This agent implements sophisticated news analysis capabilities including:
    - Financial news sentiment analysis and trend identification
    - Source credibility assessment and consensus evaluation
    - Market relevance scoring for news events and developments
    - Integration with web search for detailed article analysis
    - Professional prompt engineering for financial news interpretation

    The analyst maintains conversation history and tracks evolving news narratives
    to provide context-aware sentiment analysis.
    """

    def __init__(
        self,
        agent,
        wrapper_manager=None,
        wrapper_type: str = "news"
    ):
        """
        Initialize the NewsAnalyst with centralized wrapper manager.

        Args:
            agent: Parent trading agent instance
            wrapper_manager: Centralized wrapper manager from the main agent
            wrapper_type: Type of wrapper to use from the manager
        """
        self.agent = agent
        self.wrapper_manager = wrapper_manager or getattr(agent, 'wrapper_manager', None)
        self.wrapper_type = wrapper_type
        
        self.logger = setup_logger(
            f"news_{self.agent.agent_id}",
            f"{self.agent.LOG_DIR}/news/news_{self.agent.agent_id}.log"
        )
        self.has_sent_first_prompt: Dict[str, bool] = defaultdict(lambda: False)

        if self.wrapper_manager is None:
            raise ValueError("No wrapper manager provided - NewsAnalyst requires centralized wrapper management")

        self._initialize_prompt_templates()
        self.logger.info(f"NewsAnalyst for {self.agent.agent_id} initialized with centralized wrapper manager.")

    def _initialize_prompt_templates(self) -> None:
        """Initialize Jinja2 prompt templates for news analysis."""
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        tpl_dir = os.path.join(base_dir, "templates")
        
        # Load single-instrument templates
        with open(os.path.join(tpl_dir, "news_analyst_first_time.j2"), "r") as f:
            self._first_time_template = f.read()

        with open(os.path.join(tpl_dir, "news_analyst_subsequent.j2"), "r") as f:
            self._subsequent_template = f.read()
        
        # Load multi-instrument templates
        with open(os.path.join(tpl_dir, "news_analyst_multi_first_time.j2"), "r") as f:
            self._multi_first_time_template = f.read()

        with open(os.path.join(tpl_dir, "news_analyst_multi_subsequent.j2"), "r") as f:
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
        news_items: List[Dict[str, Any]]
    ) -> str:
        """
        Construct the initial comprehensive prompt for news analysis using Jinja2 template.
        """
        joined_news = format_joined_news(news_items)
        
        context = {
            "instrument": instrument,
            "session_start": self.agent.start_time.isoformat(),
            "session_end": self.agent.end_time.isoformat(),
            "current_time": self.agent.current_time.isoformat(),
            "joined_news": joined_news
        }
        
        template = self._jinja_env.from_string(self._first_time_template)
        prompt = template.render(**context)
        
        self.logger.debug(f"Constructed first-time news prompt at {self.agent.current_time} for {instrument}")
        return prompt

    def construct_subsequent_prompt(
        self,
        instrument: str,
        news_items: List[Dict[str, Any]]
    ) -> str:
        """
        Construct follow-up prompts for continued news analysis using Jinja2 template.
        """
        joined_news = format_joined_news(news_items)
        
        context = {
            "instrument": instrument,
            "current_time": self.agent.current_time.isoformat(),
            "joined_news": joined_news
        }
        
        template = self._jinja_env.from_string(self._subsequent_template)
        prompt = template.render(**context)
        
        self.logger.debug(f"Constructed subsequent news prompt at {self.agent.current_time} for {instrument}")
        return prompt

    def construct_multi_instrument_first_time_prompt(
        self,
        instruments_news: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Construct initial comprehensive prompt for multi-instrument news analysis.
        """
        instruments_context = []
        
        for instrument, news_items in instruments_news.items():
            joined_news = format_joined_news(news_items)
            
            instrument_ctx = {
                "instrument": instrument,
                "joined_news": joined_news,
                "news_count": len(news_items)
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
        
        template = self._jinja_env.from_string(self._multi_first_time_template)
        prompt = template.render(**context)
        
        instruments_list = list(instruments_news.keys())
        self.logger.debug(f"Constructed multi-instrument first-time news prompt at {self.agent.current_time} for {', '.join(instruments_list)}")
        return prompt

    def construct_multi_instrument_subsequent_prompt(
        self,
        instruments_news: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Construct follow-up prompt for multi-instrument news analysis.
        """
        instruments_context = []
        
        for instrument, news_items in instruments_news.items():
            joined_news = format_joined_news(news_items)
            
            instrument_ctx = {
                "instrument": instrument,
                "joined_news": joined_news,
                "news_count": len(news_items)
            }
            instruments_context.append(instrument_ctx)
        
        context = {
            "instruments": instruments_context,
            "num_instruments": len(instruments_context),
            "current_time": self.agent.current_time.isoformat()
        }
        
        template = self._jinja_env.from_string(self._multi_subsequent_template)
        prompt = template.render(**context)
        
        instruments_list = list(instruments_news.keys())
        self.logger.debug(f"Constructed multi-instrument subsequent news prompt at {self.agent.current_time} for {', '.join(instruments_list)}")
        return prompt

    async def get_news_analysis(
        self,
        instrument: str,
        news_items: List[Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive news sentiment analysis using the configured LLM.

        This method orchestrates the news analysis process, including prompt construction,
        LLM interaction, and response processing. It also handles URL extraction for
        potential web search integration.

        Args:
            instrument: Financial instrument symbol
            news_items: List of news item dictionaries containing headlines, sources, etc.

        Returns:
            Generated news analysis text with sentiment assessment
        """
        try:
            # Determine prompt type based on conversation history
            if not self.has_sent_first_prompt[instrument]:
                prompt = self.construct_first_time_prompt(instrument, news_items)
                self.has_sent_first_prompt[instrument] = True
                prompt_type = "FIRST_TIME"
            else:
                prompt = self.construct_subsequent_prompt(instrument, news_items)
                prompt_type = "SUBSEQUENT"

            # Check if web search is enabled for this model
            model_config = self.agent.models_config.get(self.wrapper_type, {})
            use_web_search = model_config.get("use_web_search", True)  # Default to True for backward compatibility
            
            urls_to_search = []
            if use_web_search:
                urls_to_search = extract_urls_from_news(news_items)
                self.logger.debug(f"Found {len(urls_to_search)} URLs to expose to web_search")
            else:
                self.logger.debug(f"Web search disabled for {self.wrapper_type} - analyzing news summaries only")

            self.logger.info(f"📰 Sending {prompt_type} news analysis prompt for {instrument}")

            # Log detailed prompt for research tracking
            self.logger.debug(f"📰 NEWS ANALYST PROMPT [{prompt_type}] for {instrument}:\n" + "="*80 + f"\n{prompt}\n" + "="*80)

            # Generate analysis using the LLM wrapper
            result = await self.wrapper_manager.generate_with_wrapper(self.wrapper_type, prompt, urls_to_search=urls_to_search)

            if result:
                self.logger.info(f"📰 Received news analysis result for {instrument} - Response length: {len(result)} chars")

                # Log detailed response for research tracking
                self.logger.debug(f"📰 NEWS ANALYST RESPONSE for {instrument}:\n" + "="*80 + f"\n{result}\n" + "="*80)

                return result
            else:
                self.logger.warning(f"📰 No result received from news analysis for {instrument}")
                return f"News analysis unavailable for {instrument} at {self.agent.current_time}"

        except Exception as e:
            self.logger.error(f"📰 Error generating news analysis for {instrument}: {e}")
            return f"News analysis error for {instrument}: {str(e)}"

    async def get_multi_instrument_news_analysis(
        self,
        instruments_news: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Generate comprehensive news sentiment analysis for multiple instruments simultaneously.
        
        This enables cross-instrument news correlation analysis and portfolio-level sentiment insights.
        
        Args:
            instruments_news: Dictionary mapping instrument symbols to their news items
                Format: {
                    "AAPL": [
                        {
                            "timestamp": "2024-01-15T10:30:00",
                            "headline": "Apple Reports Strong Q4 Earnings",
                            "source": "Reuters",
                            "description": "Apple Inc. reported...",
                            "url": "https://example.com/news/apple-earnings",
                            "keywords": ["earnings", "revenue", "iPhone"]
                        },
                        ...
                    ],
                    "NVDA": [...]
                }
        
        Returns:
            Generated multi-instrument news analysis text
        """
        try:
            instruments = list(instruments_news.keys())
            total_news = sum(len(news_list) for news_list in instruments_news.values())
            self.logger.info(f"📰 Generating multi-instrument news analysis for {len(instruments)} instruments with {total_news} total news items")
            
            # Determine if this is first time for any instrument
            is_first_time = any(not self.has_sent_first_prompt.get(instr, False) for instr in instruments)
            
            if is_first_time:
                prompt = self.construct_multi_instrument_first_time_prompt(instruments_news)
                # Mark all instruments as having received first prompt
                for instr in instruments:
                    self.has_sent_first_prompt[instr] = True
                prompt_type = "MULTI_FIRST_TIME"
            else:
                prompt = self.construct_multi_instrument_subsequent_prompt(instruments_news)
                prompt_type = "MULTI_SUBSEQUENT"
            
            # Check if web search is enabled for this model
            model_config = self.agent.models_config.get(self.wrapper_type, {})
            use_web_search = model_config.get("use_web_search", True)
            
            urls_to_search = []
            if use_web_search:
                # Collect URLs from all instruments
                for news_items in instruments_news.values():
                    urls_to_search.extend(extract_urls_from_news(news_items))
                self.logger.debug(f"Found {len(urls_to_search)} URLs across all instruments for web_search")
            else:
                self.logger.debug(f"Web search disabled for {self.wrapper_type} - analyzing news summaries only")
            
            self.logger.info(f"📰 Sending {prompt_type} news analysis prompt for {len(instruments)} instruments")
            
            # Log detailed prompt for research tracking
            self.logger.debug(f"📰 MULTI-INSTRUMENT NEWS ANALYST PROMPT [{prompt_type}]:\n" + "="*80 + f"\n{prompt}\n" + "="*80)
            
            # Generate analysis using the LLM wrapper
            result = await self.wrapper_manager.generate_with_wrapper(self.wrapper_type, prompt, urls_to_search=urls_to_search)
            
            if result:
                self.logger.info(f"📰 Received multi-instrument news analysis - Response length: {len(result)} chars")
                
                # Log detailed response for research tracking
                self.logger.debug(f"📰 MULTI-INSTRUMENT NEWS ANALYST RESPONSE:\n" + "="*80 + f"\n{result}\n" + "="*80)
                
                return result
            else:
                self.logger.warning(f"📰 No result received from multi-instrument news analysis")
                return f"Multi-instrument news analysis unavailable at {self.agent.current_time}"
                
        except Exception as e:
            self.logger.error(f"📰 Error generating multi-instrument news analysis: {e}")
            return f"Multi-instrument news analysis error: {str(e)}"
