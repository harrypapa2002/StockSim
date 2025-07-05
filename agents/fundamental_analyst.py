"""
Fundamental Analyst Agent - analyzes financial statements, corporate events, and fundamental metrics.
Provides comprehensive fundamental analysis for trading decisions in StockSim.
"""

import os
from collections import defaultdict
from typing import Dict, Any

import jinja2

from utils.logging_setup import setup_logger
from utils.time_utils import seconds_to_human


def format_corporate_events_section(fundamentals: Dict[str, Any]) -> str:
    """
    Format corporate events (IPOs, splits, dividends, ticker events) for analysis.

    Args:
        fundamentals: Dictionary containing fundamental data including corporate events

    Returns:
        Formatted string containing structured corporate events information
    """
    sections = []

    # IPO Events
    ipos = fundamentals.get("ipos", [])
    if ipos:
        sections.append("IPOs:")
        for ipo in ipos[:5]:  # Limit to most recent 5
            offer_size = ipo.get('total_offer_size', 0)
            sections.append(
                f"- {ipo.get('announced_date')}: Status={ipo.get('ipo_status')}, "
                f"Offer Size=${offer_size:,.0f}"
            )

    # Stock Splits
    splits = fundamentals.get("splits", [])
    if splits:
        sections.append("Stock Splits:")
        for split in splits[:5]:  # Limit to most recent 5
            sections.append(
                f"- {split.get('execution_date')}: "
                f"{split.get('split_from')}:{split.get('split_to')} split"
            )

    # Dividend Events
    dividends = fundamentals.get("dividends", [])
    if dividends:
        sections.append("Dividends:")
        for div in dividends[:8]:  # Show more dividend history
            cash_amount = div.get('cash_amount', 0)
            sections.append(
                f"- Ex-date: {div.get('ex_dividend_date')}, "
                f"Pay-date: {div.get('pay_date')}, "
                f"Amount: ${cash_amount:.3f}, "
                f"Frequency: {div.get('frequency')}"
            )

    # Ticker Events
    ticker_events = fundamentals.get("ticker_events", {})
    events = ticker_events.get("events", [])
    if events:
        sections.append("Ticker Events:")
        for event in events[:10]:  # Show recent ticker events
            sections.append(
                f"- {event.get('date')}: {event.get('type')} - "
                f"{event.get('details', {})}"
            )

    return "\n".join(sections) if sections else "No corporate events in this period."


def format_financial_statements_section(fundamentals: Dict[str, Any]) -> str:
    """
    Format financial statements with key metrics and ratio analysis.

    Args:
        fundamentals: Dictionary containing financial statements data

    Returns:
        Formatted string containing structured financial analysis
    """
    financials = fundamentals.get("financials", [])
    if not financials:
        return "No financial statements in this period."

    sections = []

    for filing in financials[:3]:  # Limit to 3 most recent filings
        filing_date = filing.get("filing_date")
        timeframe = filing.get("timeframe", "")
        fiscal_period = filing.get("fiscal_period", "")
        fiscal_year = filing.get("fiscal_year", "")

        sections.append(
            f"\n=== {timeframe.upper()} {fiscal_period} {fiscal_year} "
            f"(Filed: {filing_date}) ==="
        )

        fin_data = filing.get("financials", {})

        # Income Statement Analysis
        income = fin_data.get("income_statement", {})
        if income:
            sections.append("Income Statement:")
            revenue = income.get("revenues", {}).get("value", 0)
            net_income = income.get("net_income_loss", {}).get("value", 0)
            gross_profit = income.get("gross_profit", {}).get("value", 0)
            operating_income = income.get("operating_income_loss", {}).get("value", 0)
            eps_diluted = income.get("diluted_earnings_per_share", {}).get("value", 0)

            # Calculate margins
            gross_margin = (gross_profit / revenue * 100) if revenue else 0
            operating_margin = (operating_income / revenue * 100) if revenue else 0
            net_margin = (net_income / revenue * 100) if revenue else 0

            sections.extend([
                f"- Revenue: ${revenue:,.0f}",
                f"- Gross Profit: ${gross_profit:,.0f} (Margin: {gross_margin:.1f}%)",
                f"- Operating Income: ${operating_income:,.0f} (Margin: {operating_margin:.1f}%)",
                f"- Net Income: ${net_income:,.0f} (Margin: {net_margin:.1f}%)",
                f"- Diluted EPS: ${eps_diluted:.2f}"
            ])

        # Balance Sheet Analysis
        balance = fin_data.get("balance_sheet", {})
        if balance:
            sections.append("Balance Sheet:")
            total_assets = balance.get("assets", {}).get("value", 0)
            current_assets = balance.get("current_assets", {}).get("value", 0)
            current_liabilities = balance.get("current_liabilities", {}).get("value", 0)
            long_term_debt = balance.get("long_term_debt", {}).get("value", 0)
            equity = balance.get("equity", {}).get("value", 0)

            # Calculate key ratios
            current_ratio = (current_assets / current_liabilities) if current_liabilities else 0
            debt_to_equity = (long_term_debt / equity) if equity else 0
            asset_turnover = (revenue / total_assets) if total_assets and revenue else 0

            sections.extend([
                f"- Total Assets: ${total_assets:,.0f}",
                f"- Current Ratio: {current_ratio:.2f}",
                f"- Long-term Debt: ${long_term_debt:,.0f}",
                f"- Debt-to-Equity: {debt_to_equity:.2f}",
                f"- Total Equity: ${equity:,.0f}",
                f"- Asset Turnover: {asset_turnover:.2f}x"
            ])

        # Cash Flow Analysis
        cash_flow = fin_data.get("cash_flow_statement", {})
        if cash_flow:
            sections.append("Cash Flow:")
            operating_cf = cash_flow.get("net_cash_flow_from_operating_activities", {}).get("value", 0)
            investing_cf = cash_flow.get("net_cash_flow_from_investing_activities", {}).get("value", 0)
            financing_cf = cash_flow.get("net_cash_flow_from_financing_activities", {}).get("value", 0)
            net_cf = cash_flow.get("net_cash_flow", {}).get("value", 0)

            # Calculate cash flow metrics
            fcf = operating_cf + investing_cf  # Free cash flow approximation
            ocf_margin = (operating_cf / revenue * 100) if revenue and operating_cf else 0

            sections.extend([
                f"- Operating Cash Flow: ${operating_cf:,.0f} (Margin: {ocf_margin:.1f}%)",
                f"- Investing Cash Flow: ${investing_cf:,.0f}",
                f"- Financing Cash Flow: ${financing_cf:,.0f}",
                f"- Net Cash Flow: ${net_cf:,.0f}",
                f"- Free Cash Flow (approx): ${fcf:,.0f}"
            ])

    return "\n".join(sections)


class FundamentalAnalyst:
    """
    Fundamental Analyst providing comprehensive financial analysis for trading decisions.
    """

    def __init__(
        self,
        agent,
        wrapper_manager=None,
        wrapper_type: str = "fundamental_analysis"
    ):
        """
        Initialize the FundamentalAnalyst with centralized wrapper manager.

        Args:
            agent: Parent trading agent instance
            wrapper_manager: Centralized wrapper manager from the main agent
            wrapper_type: Type of wrapper to use from the manager
        """
        self.agent = agent
        self.wrapper_manager = wrapper_manager or getattr(agent, 'wrapper_manager', None)
        self.wrapper_type = wrapper_type
        
        self.logger = setup_logger(
            f"fundamental_{self.agent.agent_id}",
            f"{self.agent.LOG_DIR}/fundamentals/fundamental_{self.agent.agent_id}.log"
        )
        self.has_sent_first_prompt: Dict[str, bool] = defaultdict(lambda: False)

        if self.wrapper_manager is None:
            raise ValueError("No wrapper manager provided - FundamentalAnalyst requires centralized wrapper management")

        self._initialize_prompt_templates()
        self.logger.info(f"FundamentalAnalyst for {self.agent.agent_id} initialized with centralized wrapper manager.")

    def _initialize_prompt_templates(self) -> None:
        """Initialize Jinja2 prompt templates for fundamental analysis."""
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        tpl_dir = os.path.join(base_dir, "templates")
        
        # Load single-instrument templates
        with open(os.path.join(tpl_dir, "fundamental_analyst_first_time.j2"), "r") as f:
            self._first_time_template = f.read()

        with open(os.path.join(tpl_dir, "fundamental_analyst_subsequent.j2"), "r") as f:
            self._subsequent_template = f.read()
        
        # Load multi-instrument templates
        with open(os.path.join(tpl_dir, "fundamental_analyst_multi_first_time.j2"), "r") as f:
            self._multi_first_time_template = f.read()

        with open(os.path.join(tpl_dir, "fundamental_analyst_multi_subsequent.j2"), "r") as f:
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
        fundamentals: Dict[str, Any]
    ) -> str:
        """
        Construct the initial comprehensive prompt for fundamental analysis using Jinja2 template.
        """
        # Format the fundamentals structure
        corporate_events = format_corporate_events_section(fundamentals)
        financial_statements = format_financial_statements_section(fundamentals)
        body = f"{corporate_events}\n\n{financial_statements}"
        
        context = {
            "instrument": instrument,
            "session_start": self.agent.start_time.isoformat(),
            "session_end": self.agent.end_time.isoformat(),
            "current_time": self.agent.current_time.isoformat(),
            "fundamental_data": body
        }
        
        template = self._jinja_env.from_string(self._first_time_template)
        prompt = template.render(**context)
        
        self.logger.debug(f"Constructed first-time fundamentals prompt at {self.agent.current_time} for {instrument}")
        return prompt

    def construct_subsequent_prompt(
        self,
        instrument: str,
        fundamentals: Dict[str, Any]
    ) -> str:
        """
        Construct follow-up prompts for continued fundamental analysis using Jinja2 template.
        """
        # Format the fundamentals structure for delta updates
        corporate_events = format_corporate_events_section(fundamentals)
        financial_statements = format_financial_statements_section(fundamentals)
        body = f"{corporate_events}\n\n{financial_statements}"
        
        context = {
            "instrument": instrument,
            "current_time": self.agent.current_time.isoformat(),
            "fundamental_data": body
        }
        
        template = self._jinja_env.from_string(self._subsequent_template)
        prompt = template.render(**context)
        
        self.logger.debug(f"Constructed subsequent fundamentals prompt at {self.agent.current_time} for {instrument}")
        return prompt

    def construct_multi_instrument_first_time_prompt(
        self,
        instruments_fundamentals: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Construct initial comprehensive prompt for multi-instrument fundamental analysis.
        """
        instruments_context = []
        
        for instrument, fundamentals in instruments_fundamentals.items():
            corporate_events = format_corporate_events_section(fundamentals)
            financial_statements = format_financial_statements_section(fundamentals)
            fundamental_data = f"{corporate_events}\n\n{financial_statements}"
            
            instrument_ctx = {
                "instrument": instrument,
                "fundamental_data": fundamental_data
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
        
        instruments_list = list(instruments_fundamentals.keys())
        self.logger.debug(f"Constructed multi-instrument first-time fundamental prompt at {self.agent.current_time} for {', '.join(instruments_list)}")
        return prompt

    def construct_multi_instrument_subsequent_prompt(
        self,
        instruments_fundamentals: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Construct follow-up prompt for multi-instrument fundamental analysis.
        """
        instruments_context = []
        
        for instrument, fundamentals in instruments_fundamentals.items():
            corporate_events = format_corporate_events_section(fundamentals)
            financial_statements = format_financial_statements_section(fundamentals)
            fundamental_data = f"{corporate_events}\n\n{financial_statements}"
            
            instrument_ctx = {
                "instrument": instrument,
                "fundamental_data": fundamental_data
            }
            instruments_context.append(instrument_ctx)
        
        context = {
            "instruments": instruments_context,
            "num_instruments": len(instruments_context),
            "current_time": self.agent.current_time.isoformat()
        }
        
        template = self._jinja_env.from_string(self._multi_subsequent_template)
        prompt = template.render(**context)
        
        instruments_list = list(instruments_fundamentals.keys())
        self.logger.debug(f"Constructed multi-instrument subsequent fundamental prompt at {self.agent.current_time} for {', '.join(instruments_list)}")
        return prompt

    async def get_fundamental_analysis(
        self,
        instrument: str,
        fundamentals: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive fundamental analysis using the configured LLM.

        This method orchestrates the fundamental analysis process, including prompt
        construction, LLM interaction, and response processing.

        Args:
            instrument: Financial instrument symbol
            fundamentals: Dictionary containing financial statements and corporate events

        Returns:
            Generated fundamental analysis text
        """
        try:
            # Determine prompt type based on conversation history
            if not self.has_sent_first_prompt[instrument]:
                prompt = self.construct_first_time_prompt(instrument, fundamentals)
                self.has_sent_first_prompt[instrument] = True
                prompt_type = "FIRST_TIME"
            else:
                prompt = self.construct_subsequent_prompt(instrument, fundamentals)
                prompt_type = "SUBSEQUENT"

            self.logger.info(f"📈 Sending {prompt_type} fundamental analysis prompt for {instrument}")

            # Log detailed prompt for research tracking
            self.logger.debug(f"📈 FUNDAMENTAL ANALYST PROMPT [{prompt_type}] for {instrument}:\n" + "="*80 + f"\n{prompt}\n" + "="*80)

            # Generate analysis using the LLM wrapper
            result = await self.wrapper_manager.generate_with_wrapper(self.wrapper_type, prompt)

            if result:
                self.logger.info(f"📈 Received fundamental analysis result for {instrument} - Response length: {len(result)} chars")

                # Log detailed response for research tracking
                self.logger.debug(f"📈 FUNDAMENTAL ANALYST RESPONSE for {instrument}:\n" + "="*80 + f"\n{result}\n" + "="*80)

                return result
            else:
                self.logger.warning(f"📈 No result received from fundamental analysis for {instrument}")
                return f"Fundamental analysis unavailable for {instrument} at {self.agent.current_time}"

        except Exception as e:
            self.logger.error(f"📈 Error generating fundamental analysis for {instrument}: {e}")
            return f"Fundamental analysis error for {instrument}: {str(e)}"

    async def get_multi_instrument_fundamental_analysis(
        self,
        instruments_fundamentals: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive fundamental analysis for multiple instruments simultaneously.
        
        This enables cross-instrument fundamental comparison and portfolio-level financial insights.
        
        Args:
            instruments_fundamentals: Dictionary mapping instrument symbols to their fundamental data
                Format: {
                    "AAPL": {
                        "financials": [...],
                        "ipos": [...],
                        "splits": [...],
                        "dividends": [...],
                        "ticker_events": {"events": [...]}
                    },
                    "NVDA": {...}
                }
        
        Returns:
            Generated multi-instrument fundamental analysis text
        """
        try:
            instruments = list(instruments_fundamentals.keys())
            self.logger.info(f"📈 Generating multi-instrument fundamental analysis for {len(instruments)} instruments: {', '.join(instruments)}")
            
            # Determine if this is first time for any instrument
            is_first_time = any(not self.has_sent_first_prompt.get(instr, False) for instr in instruments)
            
            if is_first_time:
                prompt = self.construct_multi_instrument_first_time_prompt(instruments_fundamentals)
                # Mark all instruments as having received first prompt
                for instr in instruments:
                    self.has_sent_first_prompt[instr] = True
                prompt_type = "MULTI_FIRST_TIME"
            else:
                prompt = self.construct_multi_instrument_subsequent_prompt(instruments_fundamentals)
                prompt_type = "MULTI_SUBSEQUENT"
            
            self.logger.info(f"📈 Sending {prompt_type} fundamental analysis prompt for {len(instruments)} instruments")
            
            # Log detailed prompt for research tracking
            self.logger.debug(f"📈 MULTI-INSTRUMENT FUNDAMENTAL ANALYST PROMPT [{prompt_type}]:\n" + "="*80 + f"\n{prompt}\n" + "="*80)
            
            # Generate analysis using the LLM wrapper
            result = await self.wrapper_manager.generate_with_wrapper(self.wrapper_type, prompt)
            
            if result:
                self.logger.info(f"📈 Received multi-instrument fundamental analysis - Response length: {len(result)} chars")
                
                # Log detailed response for research tracking
                self.logger.debug(f"📈 MULTI-INSTRUMENT FUNDAMENTAL ANALYST RESPONSE:\n" + "="*80 + f"\n{result}\n" + "="*80)
                
                return result
            else:
                self.logger.warning(f"📈 No result received from multi-instrument fundamental analysis")
                return f"Multi-instrument fundamental analysis unavailable at {self.agent.current_time}"
                
        except Exception as e:
            self.logger.error(f"📈 Error generating multi-instrument fundamental analysis: {e}")
            return f"Multi-instrument fundamental analysis error: {str(e)}"
