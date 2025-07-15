# StockSim: Multi-Agent LLM Financial Market Simulation Platform

[![StockSim Screenshot](https://github.com/user-attachments/assets/8fc30ca7-2746-4d91-a483-9e98a0fcff35)](https://harrypapa2002.github.io/StockSim/)

🔗 **Live Demo**: [harrypapa2002.github.io/StockSim](https://harrypapa2002.github.io/StockSim/)

📄 **Paper**: [StockSim: Multi-Agent LLM Financial Market Simulation Platform (arXiv:2507.09255)](https://arxiv.org/abs/2507.09255)

StockSim is a comprehensive, open-source simulation platform designed for systematic evaluation of Large Language Models (LLMs) in realistic financial decision-making environments. The platform provides unprecedented capabilities for studying LLM behavior in dynamic, multi-agent trading scenarios.

## 🌟 Key Features

### Core Capabilities
- **Dual Simulation Modes**: Real-time order book simulation and high-throughput historical backtesting
- **Multi-Agent Coordination**: Support for heterogeneous LLM and traditional algorithmic traders
- **Multi-Instrument Trading**: Portfolio-level coordination across stocks, cryptocurrencies, and other assets
- **Advanced Market Simulation**: Realistic market microstructure with latency, slippage, and market impact modeling
- **Comprehensive Analysis**: Automatic chart generation, performance metrics, and research data export

### LLM Integration
- **Specialized Analysts**: Market analyst, news analyst, and fundamental analyst with independent reasoning
- **Multi-Modal Information Processing**: Technical indicators, news sentiment, and fundamental data
- **Heterogeneous Model Support**: OpenAI, Anthropic, and other LLM providers
- **Advanced Prompting**: Role-specific templates and chain-of-thought reasoning
- **Decision Traceability**: Complete conversation logs and explanation tracking

### Technical Architecture
- **Production-Grade Infrastructure**: RabbitMQ-based asynchronous messaging
- **Scalable Design**: Supports 500+ concurrent agents with real-time coordination
- **Professional Visualization**: Interactive charts with technical analysis overlays
- **Data Source Flexibility**: Polygon.io, Alpha Vantage integration for real market data
- **Research-Ready Output**: Comprehensive metrics, logs, and exportable results

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- API keys for data sources (Polygon.io or Alpha Vantage)
- Optional: LLM API keys (OpenAI, AWS) for advanced agents

### 1. Setup Environment
```bash
# Clone the repository
git clone <repository-url>
cd StockSim

# Copy environment template (create .env file)
cp .env.example .env  # or create .env manually

# Edit .env with your API keys
nano .env
```

Required environment variables:
```bash
# Message Broker
RABBITMQ_HOST=localhost

# Data Sources (at least one required)
POLYGON_API_KEY=your_polygon_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# LLM Providers (optional)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# AWS (for Anthropic Bedrock)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-west-2

# Directories
LOG_DIR=logs
METRICS_OUTPUT_DIR=metrics
```

### 2. Run with Docker (Recommended)
```bash
# Build and start the platform
make build && make up

# Run simulation with default config
make run

# Or specify custom config
make run CONFIG=my_config.yaml

# Monitor in real-time
make monitor

# View logs
make logs
```

### 3. Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start RabbitMQ (separately)
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# Run simulation
python main_launcher.py configs/config.yaml
```

### 4. View Results
```bash
# Access interactive charts
open charts/NVDA_stocksim_chart.html

# View performance reports
cat reports/simulation_summary.json

# Access detailed logs
tail -f logs/launcher.log
```

## 📊 Configuration Example

Create or modify `configs/config.yaml`:

```yaml
# Instruments to trade
instruments:
  - "NVDA"
  - "BTCUSDT"

# Exchange configuration for each instrument
exchanges:
  NVDA:
    data_source: "polygon"        # or "alpha_vantage"
    symbol_type: "stock"          # or "crypto"
    candle_interval: "1h"
    warmup_candles: 250
    indicator_kwargs:
      sma_periods: [20, 50, 200]
      rsi_period: 14
    news:
      tickers: ["NVDA"]
      max_results: 50
  
  BTCUSDT:
    data_source: "alpha_vantage"
    symbol_type: "crypto"
    candle_interval: "1h"
    warmup_candles: 100

# Trading agents configuration
agents:
  llm_trader_1:
    type: "LLMTradingAgent"
    count: 1
    parameters:
      initial_cash: 100000
      action_interval: "1h"
      
      # Enable/disable analyst modules
      enable_market_analyst: true
      enable_news_analyst: true
      enable_fundamental_analyst: true
      
      # LLM model configurations
      # Note: API key values reference environment variable names (from .env file)
      # This allows using different API keys for different models/agents
      models:
        aggregator:
          model_id: "gpt-4o"
          model_type: "openai"
          temperature: 0.0
          api_keys:
            openai_api_key: "OPENAI_API_KEY"  # References .env variable name
        
        market_analysis:
          model_id: "claude-sonnet-4"
          model_type: "anthropic"
          temperature: 0.1
          use_thinking: true
          api_keys:
            aws_access_key_id: "AWS_ACCESS_KEY_ID"     # References .env variable name
            aws_secret_access_key: "AWS_SECRET_ACCESS_KEY"  # References .env variable name
            aws_region: "AWS_REGION"                   # References .env variable name
        
        news:
          model_id: "gpt-4o-mini"
          model_type: "openai"
          temperature: 0.0
          # Uses default OPENAI_API_KEY from .env when api_keys not specified
        
        fundamental_analysis:
          model_id: "claude-sonnet-4"
          model_type: "anthropic"
          temperature: 0.0
          # Uses default AWS credentials from .env when api_keys not specified

  benchmark_sma:
    type: "SMA_Trader"
    count: 1
    parameters:
      initial_cash: 100000
      window: 20
      position_size_pct: 0.05
      action_interval: "1h"

  benchmark_buy_hold:
    type: "Buy_And_Hold_Trader"
    count: 1
    parameters:
      initial_cash: 100000
      quantity_size: 100
      action_interval: "1d"

# Simulation parameters
simulation:
  start_time: "2024-01-01T00:00:00Z"
  end_time: "2024-12-31T23:59:59Z"
  tick_interval: "1h"
  expected_exchange_agent_count: 2

# Exchange mode: "candle" or "orderbook"
exchange_mode: "candle"

```

## 🏗️ System Architecture

### Core Components

1. **Exchange Simulation Engine**
   - **Candle-based Exchange** (`exchanges/candle_based_exchange_agent.py`): Historical backtesting with OHLCV data
   - **Order Book Exchange** (`exchanges/exchange_agent.py`): Real-time order matching with market microstructure
   - Market data integration, technical indicators, and news feeds

2. **Agent Framework**
   - **Base Trader** (`agents/benchmark_traders/trader.py`): Foundation class with portfolio management
   - **LLM Trading Agent** (`agents/llm_agent.py`): Advanced multi-agent LLM coordination
   - **Specialized Analysts**: Market, news, and fundamental analysis modules

3. **Communication Infrastructure**
   - RabbitMQ message broker for asynchronous coordination
   - Simulation clock for deterministic time progression
   - Comprehensive logging and metrics collection

4. **Analysis & Visualization**
   - Interactive charts (`utils/plot_charts.py`) with technical indicators
   - Performance metrics computation and export
   - Research data collection and analysis tools

### Multi-Agent Coordination

StockSim supports sophisticated multi-agent scenarios:
- **Competitive Trading**: Multiple agents competing in the same market
- **Specialized Roles**: Different agents focusing on different analysis types
- **Portfolio Coordination**: Multi-instrument portfolio management
- **Research Evaluation**: Systematic comparison of LLM approaches

## 📈 Supported Trading Strategies

### LLM-Based Agents
- **Market Analyst**: Technical analysis using price patterns and indicators
- **News Analyst**: Sentiment analysis of market news and events  
- **Fundamental Analyst**: Analysis of corporate events, earnings, and fundamentals
- **Multi-Agent Coordinator**: Combines insights from specialized analysts

### Benchmark Agents
- **Buy and Hold** (`Buy_And_Hold_Trader`): Simple buy-and-hold strategy
- **SMA Trader** (`SMA_Trader`): Simple moving average crossover
- **SLMA Trader** (`SLMA_Trader`): Short/Long moving average crossover
- **MACD Trader** (`MACD_Trader`): MACD-based momentum trading
- **Bollinger Bands** (`Bollinger_Bands_Trader`): Mean reversion strategy
- **Random Trader** (`Random_Trader`): Random trading for baseline comparison

## 🔧 Advanced Usage

### Custom Agent Development
```python
from agents.benchmark_traders.trader import TraderAgent

class MyCustomAgent(TraderAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your strategy parameters
        
    async def on_market_data_update(self, instrument: str, snapshot: dict):
        """Called when new market data is received"""
        data = snapshot.get("data", {})
        
        if data:
            current_price = data.get("close")
            
            # Implement your trading logic here
            if self.should_buy(current_price):
                await self.place_order(
                    instrument=instrument,
                    side="BUY", 
                    quantity=100, 
                    order_type="MARKET"
                )
    
    def should_buy(self, price):
        # Your custom logic here
        return True
```

### Multi-Timeframe Analysis
```bash
# Generate charts with multiple timeframes
python utils/plot_charts.py \
  --symbol NVDA \
  --interval 1h \
  --start_date 2024-01-01 \
  --end_date 2024-12-31 \
  --scales 3600 14400 86400 \  # 1h, 4h, 1d
  --data_source polygon \
  --symbol_type stock \
  --output nvda_analysis.html \
  --generate_report
```

## 📋 Docker Commands Reference

| Command | Description |
|---------|-------------|
| `make build` | Build Docker containers |
| `make up` | Start all services |
| `make run` | Run simulation with default config |
| `make run CONFIG=file.yaml` | Run with specific config |
| `make shell` | Access simulation container |
| `make monitor` | Monitor resource usage |
| `make logs` | View simulation logs |
| `make rabbitmq-logs` | View RabbitMQ logs |
| `make charts` | Generate analysis charts |
| `make edit-config` | Edit configuration file |
| `make down` | Stop all services |
| `make clean` | Clean up containers and volumes |
| `make stats` | Show current resource usage |

## 🎯 Research Applications

StockSim enables research in multiple areas:

### NLP & LLM Research
- Sequential decision-making under uncertainty
- Multi-modal information processing (text + numerical data)
- Prompt engineering and reasoning evaluation
- Multi-agent coordination patterns
- Chain-of-thought reasoning in financial contexts

### Financial Technology
- Algorithmic trading strategy development
- Market microstructure analysis
- Risk management evaluation
- Portfolio optimization research
- High-frequency trading simulation

### AI Safety & Evaluation
- LLM reliability in high-stakes environments
- Decision consistency and robustness testing
- Explanation quality assessment
- Behavioral analysis under market stress
- Model calibration and uncertainty quantification

## 📊 Performance Metrics

StockSim automatically computes comprehensive performance metrics:

### Financial Metrics
- **ROI**: Return on Investment
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses

### Trading Metrics
- **Total Trades**: Number of executed trades
- **Average Trade Size**: Mean trade value
- **ROIC**: Return on Invested Capital
- **Profit per Trade**: Average profit per closed trade
- **Trading Volume**: Total value traded

### Research Metrics (LLM Agents)
- **Decision Consistency**: Variance in similar scenarios
- **Analyst Utilization**: Usage of different analyst modules
- **Response Quality**: Explanation coherence and accuracy
- **Coordination Effectiveness**: Multi-agent collaboration patterns

## 📚 File Structure

```
StockSim/
├── agents/
│   ├── llm_agent.py                    # Main LLM trading agent
│   ├── market_analyst.py               # Technical analysis specialist
│   ├── news_analyst.py                 # News sentiment analysis
│   ├── fundamental_analyst.py          # Fundamental analysis
│   └── benchmark_traders/
│       ├── trader.py                   # Base trader class
│       ├── buy_and_hold_trader.py      # Buy and hold strategy
│       ├── sma_trader.py               # SMA crossover strategy
│       └── ...                         # Other benchmark strategies
├── exchanges/
│   ├── candle_based_exchange_agent.py  # Historical backtesting
│   └── exchange_agent.py               # Real-time order book
├── simulation/
│   └── simulation_clock.py             # Time management
├── utils/
│   ├── plot_charts.py                  # Chart generation
│   ├── polygon_client.py               # Polygon.io integration
│   ├── alpha_vantage_client.py         # Alpha Vantage integration
│   └── indicators_tracker.py          # Technical indicators
├── wrappers/
│   ├── base_wrapper.py                 # LLM wrapper base
│   ├── specialized_wrappers.py         # Provider-specific wrappers
│   └── wrapper_factory.py             # LLM wrapper factory
├── configs/
│   └── config.yaml                     # Main configuration
├── templates/
│   ├── trading_prompt_template_full.j2 # LLM prompts
│   └── trading_prompt_template_delta.j2
├── main_launcher.py                    # Main entry point
├── docker-compose.yml                  # Docker configuration
├── Makefile                           # Docker commands
└── README.md                          # This file
```

## 🤝 Contributing

We welcome contributions! Please:

1. **Fork the repository** and create a feature branch
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Submit a pull request** with clear description

### Development Areas
- New trading strategies and agents
- Additional data source integrations
- Enhanced visualization features
- Performance optimizations
- Research analysis tools

## 📄 License

StockSim is released under the MIT License. See LICENSE file for details.

---

*StockSim: Bridging the gap between NLP research and realistic financial simulation*
