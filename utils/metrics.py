"""
StockSim Trading Performance Metrics

This module provides comprehensive trading performance analysis for the StockSim simulation platform.

The RationalityMetrics class tracks portfolio performance and trading behavior to enable
systematic evaluation of LLM trading agents against traditional strategies.

Key Features:
- Portfolio value tracking with time series support
- Comprehensive risk-adjusted performance metrics
- Trading efficiency analysis (win rate, profit factor)
- Statistical measures for research validation
- ROIC calculation for capital efficiency assessment

Research Applications:
- LLM vs baseline performance comparison
- Multi-agent coordination effectiveness
- Market regime adaptation analysis
- Decision consistency measurement
"""

from typing import List, Dict, Any, Optional
import numpy as np


class RationalityMetrics:
    """
    Trading performance metrics calculator for StockSim research platform.
    
    This class provides comprehensive performance analysis capabilities for evaluating
    LLM trading agents, enabling the research methodologies described in the EMNLP paper.
    
    Attributes:
        portfolio_history: List of portfolio values over time
        trade_history: List of all executed trades with metadata
        portfolio_time_series: Time-stamped portfolio values for temporal analysis
    """
    
    def __init__(self):
        """Initialize metrics tracking with empty state."""
        self.portfolio_history: List[float] = []
        self.trade_history: List[Dict[str, Any]] = []
        self._last_portfolio_value: Optional[float] = None
        self.portfolio_time_series: List[Dict[str, Any]] = []
        self._returns: List[float] = []

    def record_portfolio_value(self, value: float, timestamp: Optional[str] = None) -> None:
        """
        Record portfolio value and calculate period return.
        
        This method enables real-time tracking of portfolio performance and
        automatic calculation of period-to-period returns for statistical analysis.
        
        Args:
            value: Current total portfolio value
            timestamp: Optional timestamp for time series analysis
        """
        if self._last_portfolio_value is not None:
            ret = value / self._last_portfolio_value - 1
            self._returns.append(ret)
        
        self._last_portfolio_value = value
        self.portfolio_history.append(value)
        
        if timestamp:
            self.portfolio_time_series.append({
                "timestamp": timestamp, 
                "value": round(value, 2)
            })

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Record executed trade for analysis.
        
        Args:
            trade: Trade dictionary containing action, quantity, price, etc.
        """
        self.trade_history.append(trade)

    def compute_roi(self) -> float:
        """
        Calculate Return on Investment.
        
        ROI = (Final Portfolio Value - Initial Portfolio Value) / Initial Portfolio Value
        
        Returns:
            ROI as decimal (e.g., 0.15 = 15% return)
        """
        if len(self.portfolio_history) < 2:
            return 0.0
        
        initial_value = self.portfolio_history[0]
        final_value = self.portfolio_history[-1]
        
        if initial_value == 0:
            return 0.0
        
        return round((final_value - initial_value) / initial_value, 4)

    def compute_returns(self) -> List[float]:
        """
        Get list of period returns for statistical analysis.
        
        Returns:
            List of decimal returns between periods
        """
        return self._returns

    def compute_sharpe_ratio(self, risk_free_rate: float = 0.0, annualize: bool = False) -> float:
        """
        Calculate Sharpe ratio for risk-adjusted return analysis.
        
        The Sharpe ratio measures excess return per unit of risk, essential for
        comparing LLM performance against benchmarks on a risk-adjusted basis.
        
        Args:
            risk_free_rate: Risk-free rate for excess return calculation
            annualize: Whether to annualize the ratio (assumes daily returns)
            
        Returns:
            Sharpe ratio (higher values indicate better risk-adjusted performance)
        """
        returns = self.compute_returns()
        n = len(returns)
        
        if n < 2:
            return 0.0

        if all(r == 0 for r in returns):
            return 0.0

        excess_returns = [r - risk_free_rate for r in returns]
        avg_excess: float = float(np.mean(excess_returns))
        std_excess: float = float(np.std(excess_returns, ddof=1))

        if std_excess == 0 or np.isnan(std_excess) or np.isinf(std_excess):
            return 0.0

        sharpe = avg_excess / std_excess

        # Annualize if requested (assuming daily returns)
        if annualize:
            sharpe *= np.sqrt(252)  # 252 trading days per year

        return round(sharpe, 4)

    def compute_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio focusing on downside risk.
        
        The Sortino ratio improves on the Sharpe ratio by considering only downside
        volatility, providing better insight into LLM risk management capabilities.
        
        Args:
            risk_free_rate: Risk-free rate for excess return calculation
            
        Returns:
            Sortino ratio (higher values indicate better downside risk management)
        """
        returns = self.compute_returns()
        n = len(returns)
        
        if n < 2:
            return 0.0

        if all(r == 0 for r in returns):
            return 0.0

        excess_returns = [r - risk_free_rate for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]

        # If no downside returns, return large positive number to indicate excellent performance
        if not downside_returns:
            avg_excess = np.mean(excess_returns)
            if avg_excess > 0:
                return 999.0  # Large positive number indicates excellent performance
            else:
                return 0.0

        # Calculate downside deviation (standard Sortino formula)
        downside_deviation = np.sqrt(np.mean([r**2 for r in downside_returns]))
        
        # Return 0 if downside deviation is zero or invalid
        if downside_deviation == 0 or np.isnan(downside_deviation) or np.isinf(downside_deviation):
            return 0.0

        avg_excess = np.mean(excess_returns)
        return round(avg_excess / downside_deviation, 4)

    def compute_win_rate(self) -> float:
        """
        Calculate win rate from profitable closing trades.
        
        Win Rate = Number of Profitable Trades / Total Closing Trades
        
        This metric provides insight into the frequency of successful LLM decisions.
        
        Returns:
            Win rate as decimal (e.g., 0.65 = 65% win rate)
        """
        closing_trades = [t for t in self.trade_history if t["action"] in {"SELL", "SHORT_COVER"}]
        
        if not closing_trades:
            return 0.0
        
        win_trades = [t for t in closing_trades if t.get("realized_profit", 0) > 0]
        return round(len(win_trades) / len(closing_trades), 4)

    def compute_profit_factor(self) -> float:
        """
        Calculate profit factor for trading efficiency analysis.
        
        Profit Factor = Total Gross Profit / Total Gross Loss
        
        Values > 1.0 indicate profitable trading. This metric is crucial for
        evaluating LLM trading effectiveness across different market conditions.
        
        Returns:
            Profit factor (values > 1.0 indicate profitability)
        """
        closing_trades = [t for t in self.trade_history if t["action"] in {"SELL", "SHORT_COVER"}]
        
        if not closing_trades:
            return 0.0

        total_profit = sum(t.get("realized_profit", 0) for t in closing_trades if t.get("realized_profit", 0) > 0)
        total_loss = abs(sum(t.get("realized_profit", 0) for t in closing_trades if t.get("realized_profit", 0) < 0))

        if total_loss == 0:
            # If there are profits but no losses, return large number
            if total_profit > 0:
                return 999.0
            else:
                return 0.0

        return round(total_profit / total_loss, 4)

    def compute_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown for risk assessment.
        
        Max Drawdown = Maximum(Peak Value - Trough Value) / Peak Value
        
        This critical risk metric measures the largest peak-to-trough decline,
        essential for evaluating LLM risk management under adverse conditions.
        
        Returns:
            Maximum drawdown as decimal (e.g., 0.15 = 15% maximum drawdown)
        """
        if not self.portfolio_history:
            return 0.0

        peak = self.portfolio_history[0] # Start with first value
        max_dd = 0.0
        
        for value in self.portfolio_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                
        return round(max_dd, 4)

    def get_last_portfolio_value(self) -> float:
        """
        Get the most recent portfolio value.
        
        Returns:
            Last recorded portfolio value, rounded to 4 decimal places
        """
        return round(self._last_portfolio_value, 4) if self._last_portfolio_value is not None else 0.0

    def compute_total_traded_volume(self) -> float:
        """
        Calculate total dollar volume traded across all transactions.
        
        Returns:
            Total traded volume in dollars
        """
        return round(sum(t["price"] * t["quantity"] for t in self.trade_history), 4)

    def compute_average_trade_size(self) -> float:
        """
        Calculate average trade size for activity analysis.
        
        Returns:
            Average trade size in dollars
        """
        if not self.trade_history:
            return 0.0
        return round(self.compute_total_traded_volume() / len(self.trade_history), 4)

    def compute_roic(self) -> float:
        """
        Calculate Return on Invested Capital.
        
        ROIC = (Final Portfolio Value - Initial Portfolio Value) / Total Capital Used for Entries
        
        This metric measures the total change in portfolio value relative to the total
        capital invested in opening positions, providing insight into capital efficiency.
        
        Returns:
            ROIC as decimal
        """
        if len(self.portfolio_history) < 2:
            return 0.0

        initial_value = self.portfolio_history[0]
        final_value = self.portfolio_history[-1]
        portfolio_change = final_value - initial_value

        # Only count opening positions (BUY and SHORT) for capital calculation
        total_capital_used_for_entries = sum(
            t["price"] * t["quantity"]
            for t in self.trade_history
            if t["action"] in {"BUY", "SHORT"}
        )

        if total_capital_used_for_entries == 0:
            # If no opening trades were made, ROIC is undefined
            # Return the simple ROI instead
            if initial_value == 0:
                return 0.0
            return round(portfolio_change / initial_value, 4)

        return round(portfolio_change / total_capital_used_for_entries, 4)

    def compute_all_metrics(self, risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Generate comprehensive performance metrics summary.
        
        This method provides all key metrics needed for LLM trading evaluation
        as described in the EMNLP paper research methodology.
        
        Args:
            risk_free_rate: Risk-free rate for risk-adjusted calculations
            
        Returns:
            Dictionary containing all calculated performance metrics
        """
        # Get all closing trades for profit calculation
        closing_trades = [t for t in self.trade_history if t["action"] in {"SELL", "SHORT_COVER"}]

        profit_per_trade = 0.0
        if closing_trades:
            total_profit = sum(t.get("realized_profit", 0) for t in closing_trades)
            profit_per_trade = round(total_profit / len(closing_trades), 4)

        return {
            "ROI": self.compute_roi(),
            "Sharpe Ratio": self.compute_sharpe_ratio(risk_free_rate),
            "Annualized Sharpe Ratio": self.compute_sharpe_ratio(risk_free_rate, annualize=True),
            "Sortino Ratio": self.compute_sortino_ratio(risk_free_rate),
            "Win Rate": self.compute_win_rate(),
            "Profit Factor": self.compute_profit_factor(),
            "Max Drawdown": self.compute_max_drawdown(),
            "Num Trades": len(self.trade_history),
            "Num Closed Trades": len(closing_trades),
            "Total Traded Volume": self.compute_total_traded_volume(),
            "Average Trade Size": self.compute_average_trade_size(),
            "ROIC": self.compute_roic(),
            "Profit per Trade": profit_per_trade,
            "Last Portfolio Value": self._last_portfolio_value
        }