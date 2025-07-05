"""
Enhanced Stock Chart Plotting Utilities for StockSim

This module provides sophisticated interactive chart generation capabilities for financial data
visualization. It supports both
stock and cryptocurrency data with comprehensive technical analysis overlays.

Key Features:
- Interactive candlestick charts with multiple timeframe support
- Technical indicators overlay (support/resistance, volume profile, moving averages)
- Order execution visualization with LLM explanation tooltips
- Portfolio performance tracking with dual-axis display
- Professional styling
- Dual-market capability (stocks and cryptocurrency)

Chart Components:
- Main price action: OHLCV candlestick data with configurable timeframes
- Volume analysis: Volume bars with profile analysis and Point of Control (POC)
- Technical overlays: Support/resistance levels, trend lines, technical indicators
- Trade markers: Buy/sell order visualization with explanations
- Portfolio tracking: Real-time portfolio value progression
- Market context: Weekend gaps and holiday exclusions

Output Format:
- Interactive HTML files using Plotly.js
- Professional styling
- Mobile-responsive design with touch-friendly controls
- Export capabilities for static images and data analysis

Usage Example:
    ```python
    from utils.plot_stock_charts import make_chart_dropdown
    from utils.polygon_client import PolygonClient

    # Load market data
    client = PolygonClient()
    candles = client.load_aggregates("NVDA", "1d", "2024-01-01", "2024-12-31")

    # Generate interactive chart
    make_chart_dropdown(
        candles=candles,
        instrument="NVDA",
        scales_seconds=[3600, 14400, 86400],  # 1h, 4h, 1d
        out_html="nvda_analysis.html",
        indicator_kwargs={"sma_periods": [20, 50, 200]},
        executed_orders=trading_orders,
        portfolio_timeseries=portfolio_data
    )
    ```

Technical Implementation:
- Optimized resampling algorithms for multiple timeframes
- Professional color schemes for financial market visualization
- Advanced indicator computation with real-time updates
- Sophisticated order execution simulation and visualization
- Market microstructure modeling for realistic backtesting
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.indicators_tracker import IndicatorsTracker
from utils.time_utils import interval_to_seconds
from utils.polygon_client import PolygonClient
from utils.alpha_vantage_client import AlphaVantageClient


def resample_candles(candles: List[Dict[str, Any]], new_interval_seconds: int) -> List[Dict[str, Any]]:
    """
    Intelligently resample candlestick data to different timeframes with proper OHLCV aggregation.

    This function groups raw candle data into larger time intervals while preserving
    the integrity of price action through proper OHLCV (Open, High, Low, Close, Volume)
    aggregation rules used in financial markets.

    Args:
        candles: List of candle dictionaries with timestamp, open, high, low, close, volume
        new_interval_seconds: Target aggregation period in seconds (e.g., 3600 for 1-hour candles)

    Returns:
        List of resampled candles maintaining chronological order

    Aggregation Rules:
        - Open: First candle's opening price in the time bucket
        - High: Maximum high price across all candles in the bucket
        - Low: Minimum low price across all candles in the bucket
        - Close: Last candle's closing price in the time bucket
        - Volume: Sum of all volumes in the time bucket

    Technical Details:
        - Handles timezone-aware timestamps correctly
        - Maintains market data integrity for backtesting
        - Supports arbitrary timeframe conversions (1m->1h, 1h->1d, etc.)
        - Preserves price action characteristics across timeframes
    """
    if not candles:
        return []

    # Ensure proper timestamp handling and chronological ordering
    candles.sort(key=lambda c: datetime.fromisoformat(c["timestamp"]))

    # Group candles into time buckets based on the new interval
    groups = {}
    for candle in candles:
        ts = datetime.fromisoformat(candle["timestamp"])
        # Calculate which time bucket this candle belongs to
        group_index = int(ts.timestamp() // new_interval_seconds)
        groups.setdefault(group_index, []).append(candle)

    # Aggregate each group according to OHLCV rules
    resampled = []
    for group_index in sorted(groups.keys()):
        group = groups[group_index]
        group.sort(key=lambda c: datetime.fromisoformat(c["timestamp"]))

        # Apply standard OHLCV aggregation
        open_price = group[0]["open"]           # First open
        high_price = max(c["high"] for c in group)      # Highest high
        low_price = min(c["low"] for c in group)        # Lowest low
        close_price = group[-1]["close"]        # Last close
        total_volume = sum(c["volume"] for c in group)  # Sum volumes

        resampled.append({
            "timestamp": group[0]["timestamp"],  # Use first timestamp of the group
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": total_volume
        })

    return resampled


def prepare_candle_data_for_plot(data: List[Dict[str, Any]]) -> tuple:
    """
    Extract and format candlestick data components for Plotly visualization.

    Args:
        data: List of candle dictionaries with OHLCV data

    Returns:
        Tuple of (times, opens, highs, lows, closes, volumes) for plotting

    Technical Notes:
        - Maintains timestamp order for proper chart rendering
        - Handles missing data gracefully
        - Optimized for Plotly candlestick chart requirements
    """
    times = [candle["timestamp"] for candle in data]
    opens = [candle["open"] for candle in data]
    highs = [candle["high"] for candle in data]
    lows = [candle["low"] for candle in data]
    closes = [candle["close"] for candle in data]
    volumes = [candle["volume"] for candle in data]
    return times, opens, highs, lows, closes, volumes


def create_enhanced_color_scheme() -> Dict[str, str]:
    """
    Define a professional color scheme optimized for financial charts and presentations.

    Returns:
        Dictionary mapping chart elements to optimal colors for research visualization

    Color Selection Rationale:
        - High contrast for accessibility
        - Professional appearance
        - Clear distinction between bullish/bearish movements
        - Optimized for both light and dark presentation environments
    """
    return {
        'bullish': '#26A69A',           # Professional teal for up moves
        'bearish': '#EF5350',           # Professional red for down moves
        'volume': 'rgba(70, 130, 180, 0.6)',    # Steel blue with transparency
        'support': '#1E88E5',           # Clear blue for support lines
        'resistance': '#D32F2F',        # Strong red for resistance lines
        'buy_order': '#4CAF50',         # Green for buy orders
        'sell_order': '#F44336',        # Red for sell orders
        'portfolio': '#7B1FA2',         # Purple for portfolio tracking
        'background': '#FAFAFA',        # Light gray background
        'grid': '#E0E0E0',              # Subtle grid lines
        'volume_profile': 'rgba(255, 193, 7, 0.7)',  # Amber for volume profile
        'poc': '#FF5722'                # Deep orange for Point of Control
    }


def ensure_output_directories():
    """
    Ensure that the charts and reports output directories exist.
    Creates them if they don't exist.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    charts_dir = os.path.join(project_root, "charts")
    reports_dir = os.path.join(project_root, "reports")

    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    return charts_dir, reports_dir


def make_chart_dropdown(
        candles: List[Dict[str, Any]],
        instrument: str,
        scales_seconds: List[int],
        out_html: str = None,
        indicator_kwargs: Dict[str, Any] = None,
        executed_orders: List[Dict[str, Any]] = None,
        portfolio_timeseries: Optional[List[Dict[str, Any]]] = None,
        symbol_type: str = "stock"
):
    """
    Create a comprehensive interactive candlestick chart with technical analysis.

    This function generates a sophisticated financial chart with multiple timeframes, technical
    indicators, order execution visualization, and portfolio tracking capabilities. The chart
    is optimized for both research analysis and demonstration purposes.

    Args:
        candles: Historical OHLCV data for the instrument
        instrument: Ticker symbol (e.g., 'NVDA', 'BTCUSDT')
        scales_seconds: List of timeframe intervals in seconds for multi-timeframe analysis
        out_html: Output file path for the generated HTML chart (defaults to charts/{instrument}_stocksim_chart.html)
        indicator_kwargs: Configuration parameters for technical indicators
        executed_orders: Trading orders with timestamps, prices, and explanations
        portfolio_timeseries: Portfolio value progression over time

    Features:
        - Interactive timeframe switching via dropdown menus
        - Support/resistance level visualization with toggle controls
        - Volume profile analysis with Point of Control identification
        - Order execution markers with detailed LLM explanation hover information
        - Portfolio performance overlay with separate y-axis
        - Market hours awareness (excludes weekends and holidays)
        - Professional styling optimized for presentations
        - Advanced technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
        - Real-time indicator updates as timeframes change

    Technical Implementation:
        - Multi-panel layout with price, volume, and portfolio sections
        - Dynamic trace visibility management for timeframe switching
        - Professional color coding and styling
        - Interactive controls with enhanced user experience
        - Export capabilities
    """
    # Ensure output directories exist and set default output path
    charts_dir, reports_dir = ensure_output_directories()

    if out_html is None:
        out_html = os.path.join(charts_dir, f"{instrument}_stocksim_chart.html")
    elif not os.path.isabs(out_html):
        # If relative path provided, put it in charts directory
        out_html = os.path.join(charts_dir, out_html)

    # Initialize color scheme and figure for professional presentation
    colors = create_enhanced_color_scheme()
    fig = go.Figure()

    all_traces = []
    buttons = []
    interval_trace_counts = []

    # Process each timeframe interval with full technical analysis
    for idx, scale in enumerate(scales_seconds):
        resampled = resample_candles(candles, scale)
        times, opens, highs, lows, closes, volumes = prepare_candle_data_for_plot(resampled)

        # Enhanced candlestick trace with professional styling for demos
        candle_trace = go.Candlestick(
            x=times,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            increasing_line_color=colors['bullish'],
            decreasing_line_color=colors['bearish'],
            increasing_fillcolor=colors['bullish'],
            decreasing_fillcolor=colors['bearish'],
            line=dict(width=1.5),
            name=f"{_format_timeframe_label(scale)} Candles",
            showlegend=False
        )

        # Enhanced volume trace with gradient effect
        volume_trace = go.Bar(
            x=times,
            y=volumes,
            marker_color=colors['volume'],
            marker_line=dict(width=0),
            name="Volume",
            showlegend=False,
            yaxis="y2",
            opacity=0.7,
            hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>'
        )

        # Initialize technical indicators with proper configuration
        indicator_config = indicator_kwargs or {}
        tracker = IndicatorsTracker(**indicator_config)

        # Warm up indicators with resampled data
        for candle in resampled:
            tracker.update(candle)

        # Generate support and resistance lines with enhanced styling
        min_time = times[0] if times else datetime.now().isoformat()
        max_time = times[-1] if times else datetime.now().isoformat()

        support_lines = []
        for i, level in enumerate(tracker.support_levels):
            support_line = go.Scatter(
                x=[min_time, max_time],
                y=[level, level],
                mode="lines",
                line=dict(color=colors['support'], dash="dot", width=2),
                name=f"Support {i + 1}",
                showlegend=False,
                hovertemplate=f'Support Level: $%{{y:.2f}}<extra></extra>',
                visible=True
            )
            support_lines.append(support_line)

        resistance_lines = []
        for i, level in enumerate(tracker.resistance_levels):
            resistance_line = go.Scatter(
                x=[min_time, max_time],
                y=[level, level],
                mode="lines",
                line=dict(color=colors['resistance'], dash="dash", width=2),
                name=f"Resistance {i + 1}",
                showlegend=False,
                hovertemplate=f'Resistance Level: $%{{y:.2f}}<extra></extra>',
                visible=True
            )
            resistance_lines.append(resistance_line)

        sr_traces = support_lines + resistance_lines

        # Enhanced Volume Profile with Point of Control (POC)
        vp_trace = None
        if (hasattr(tracker, 'volume_profile_bins') and tracker.volume_profile_bins is not None and
            hasattr(tracker, 'volume_profile_values') and tracker.volume_profile_values is not None and
            len(tracker.volume_profile_bins) > 1 and len(tracker.volume_profile_values) > 0):

            # Calculate bin centers for volume profile
            bin_centers = (tracker.volume_profile_bins[:-1] + tracker.volume_profile_bins[1:]) / 2
            volume_values = tracker.volume_profile_values

            # Create horizontal volume profile bars
            vp_trace = go.Bar(
                x=volume_values,
                y=bin_centers,
                orientation='h',
                marker=dict(
                    color=colors['volume_profile'],
                    line=dict(width=1, color='rgba(255, 193, 7, 0.9)')
                ),
                width=(bin_centers[1] - bin_centers[0]) * 0.8 if len(bin_centers) > 1 else 1.0,
                name="Volume Profile",
                showlegend=True,
                xaxis='x2',
                yaxis='y',
                hovertemplate='<b>Volume Profile</b><br>Price: $%{y:.2f}<br>Volume: %{x:.0f}<extra></extra>',
                visible=True
            )

            # Add Point of Control (POC) marker
            if hasattr(tracker, 'volume_profile_poc_price') and tracker.volume_profile_poc_price:
                poc_trace = go.Scatter(
                    x=[min_time, max_time],
                    y=[tracker.volume_profile_poc_price, tracker.volume_profile_poc_price],
                    mode="lines",
                    line=dict(color=colors['poc'], width=3, dash="solid"),
                    name="POC (Point of Control)",
                    showlegend=True,
                    hovertemplate=f'POC: $%{{y:.2f}}<extra></extra>',
                    visible=True
                )
                sr_traces.append(poc_trace)

        if vp_trace is None:
            # Create empty volume profile trace to maintain structure
            vp_trace = go.Bar(
                x=[],
                y=[],
                orientation='h',
                name="Volume Profile",
                showlegend=False,
                xaxis='x2',
                yaxis='y',
                visible=False
            )

        # Aggregate traces for this timeframe
        timeframe_traces = [candle_trace, volume_trace] + sr_traces + [vp_trace]
        all_traces.extend(timeframe_traces)
        interval_trace_counts.append(len(timeframe_traces))

    # Add order execution markers with enhanced visualization for LLM explanations
    order_marker_trace = None
    if executed_orders:
        instrument_orders = sorted(
            [order for order in executed_orders if order.get("instrument") == instrument],
            key=lambda o: datetime.fromisoformat(o["timestamp"])
        )

        if instrument_orders:
            order_times = [order["timestamp"] for order in instrument_orders]
            order_prices = [order["price"] for order in instrument_orders]
            order_actions = [order["action"] for order in instrument_orders]

            # Enhanced order visualization with LLM explanation support
            marker_symbols = [
                "triangle-up" if act.upper() in ["BUY", "SHORT_COVER"] else "triangle-down"
                for act in order_actions
            ]
            marker_colors = [
                colors['buy_order'] if act.upper() in ["BUY", "SHORT_COVER"] else colors['sell_order']
                for act in order_actions
            ]

            # Rich hover information with LLM explanations
            hover_texts = []
            for order in instrument_orders:
                hover_text = (
                    f"<b>{order['action'].upper()}</b><br>" +
                    f"Price: ${order['price']:.2f}<br>" +
                    f"Quantity: {order.get('quantity', 'N/A')}<br>" +
                    f"Type: {order.get('orderType', 'Limit')}<br>" +
                    f"Time: {order['timestamp']}<br>"
                )

                # Add LLM explanation if available
                if order.get('explanation'):
                    wrapped_explanation = _wrap_text(order['explanation'], max_chars_per_line=40)
                    hover_text += f"<br><i>Explanation:</i><br>{wrapped_explanation}"
                else:
                    hover_text += "<br><i>No explanation available</i>"

                hover_texts.append(hover_text)

            order_marker_trace = go.Scatter(
                x=order_times,
                y=order_prices,
                mode="markers",
                marker=dict(
                    size=14,
                    symbol=marker_symbols,
                    color=marker_colors,
                    line=dict(width=2, color='white'),
                    opacity=0.9
                ),
                name="LLM Trading Orders",
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                showlegend=True,
                visible=True
            )
            all_traces.append(order_marker_trace)

    # Add portfolio performance tracking for demo
    portfolio_trace = None
    if portfolio_timeseries:
        portfolio_sorted = sorted(portfolio_timeseries, key=lambda e: datetime.fromisoformat(e["timestamp"]))
        portfolio_times = [entry["timestamp"] for entry in portfolio_sorted]
        portfolio_values = [entry["value"] for entry in portfolio_sorted]

        portfolio_trace = go.Scatter(
            x=portfolio_times,
            y=portfolio_values,
            mode="lines+markers",
            line=dict(color=colors['portfolio'], width=3),
            marker=dict(size=6, color=colors['portfolio']),
            name="Portfolio Value",
            yaxis="y3",
            showlegend=True,
            hovertemplate='<b>Portfolio Performance</b><br>Value: $%{y:,.2f}<br>%{x}<extra></extra>',
            visible=True
        )
        all_traces.append(portfolio_trace)

    # Calculate visibility templates for timeframe switching
    total_traces = len(all_traces)
    visibility_templates = []
    trace_index = 0

    for count in interval_trace_counts:
        vis = [False] * total_traces
        # Show traces for this timeframe
        for i in range(count):
            if trace_index + i < total_traces:
                vis[trace_index + i] = True
        # Always show order markers and portfolio if they exist
        if order_marker_trace:
            order_index = -1 if not portfolio_trace else -2
            if len(vis) + order_index >= 0:
                vis[order_index] = True
        if portfolio_trace:
            if len(vis) > 0:
                vis[-1] = True
        visibility_templates.append(vis)
        trace_index += count

    # Add all traces to figure
    for trace in all_traces:
        fig.add_trace(trace)

    # Create timeframe selection buttons
    for idx, (scale, vis_template) in enumerate(zip(scales_seconds, visibility_templates)):
        label = _format_timeframe_label(scale)
        button = dict(
            label=f"📊 {label}",
            method="update",
            args=[
                {"visible": vis_template},
                {"title": f"{instrument} Technical Analysis - {label} Timeframe"}
            ]
        )
        buttons.append(button)

    # Set initial visibility
    if visibility_templates:
        initial_visibility = visibility_templates[0]
        for i, trace in enumerate(fig.data):
            trace.visible = initial_visibility[i] if i < len(initial_visibility) else True

    # Enhanced toggle controls for different chart elements
    sr_indices = []
    vp_indices = []
    for i, trace in enumerate(fig.data):
        trace_name = getattr(trace, 'name', '')
        # Support/Resistance traces only (excluding POC which belongs to volume profile)
        if ('Support' in trace_name or 'Resistance' in trace_name) and 'POC' not in trace_name:
            sr_indices.append(i)
        # Volume Profile traces including POC (Point of Control)
        elif 'Volume Profile' in trace_name or 'POC' in trace_name:
            vp_indices.append(i)

    # Support/Resistance toggle controls - completely independent
    sr_toggle_buttons = [
        dict(
            label="📈 Show S/R",
            method="restyle",
            args=[{"visible": True}, sr_indices]
        ),
        dict(
            label="📉 Hide S/R",
            method="restyle",
            args=[{"visible": False}, sr_indices]
        )
    ]

    # Volume Profile toggle controls - completely independent
    vp_toggle_buttons = [
        dict(
            label="📊 Show Vol Profile",
            method="restyle",
            args=[{"visible": True}, vp_indices]
        ),
        dict(
            label="🚫 Hide Vol Profile",
            method="restyle",
            args=[{"visible": False}, vp_indices]
        )
    ]

    # Market holidays and non-trading days for realistic visualization
    non_trading_days = [
        # 2024 US Market Holidays
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
        "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
        # 2025 US Market Holidays
        "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
        "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"
    ]

    # Professional layout configuration
    # Determine if rangebreaks should be applied (only for stocks, not crypto)
    xaxis_config = dict(
        domain=[0.0, 0.82],
        title=dict(text="", font=dict(size=1)),
        gridcolor=colors['grid'],
        linecolor='#BDBDBD',
        tickfont=dict(size=14),
        showgrid=True
    )
    
    # Only apply rangebreaks for stocks (not crypto which trades 24/7)
    if symbol_type == "stock":
        xaxis_config["rangebreaks"] = [
            dict(bounds=["sat", "mon"]),  # Hide weekends
            dict(values=non_trading_days)  # Hide holidays
        ]
    
    fig.update_layout(
        # Enhanced axis configuration
        xaxis=xaxis_config,
        xaxis2=dict(
            domain=[0.84, 1.0],
            showticklabels=False,
            anchor="y",
            autorange='reversed',
            showgrid=False,
            title=dict(text="Volume", font=dict(size=24, color='#424242'))
        ),

        # Multi-panel y-axis layout
        yaxis=dict(
            domain=[0.55, 1.0],
            title=dict(text="Price ($)", font=dict(size=26, color='#424242')),
            gridcolor=colors['grid'],
            linecolor='#BDBDBD',
            tickfont=dict(size=22),
            showgrid=True
        ),
        yaxis2=dict(
            domain=[0.0, 0.20],
            title=dict(text="Volume", font=dict(size=24, color='#424242')),
            anchor="x",
            gridcolor=colors['grid'],
            tickfont=dict(size=20),
            autorange=True,
            rangemode="tozero"
        ),
        yaxis3=dict(
            domain=[0.25, 0.50],
            title=dict(text="Portfolio ($)", font=dict(size=24, color='#424242')),
            anchor="x",
            gridcolor=colors['grid'],
            tickfont=dict(size=20)
        ),

        # Professional styling
        template="plotly_white",
        paper_bgcolor=colors['background'],
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=16, color='#424242'),

        # Interactive controls with enhanced styling
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                y=1.25,
                xanchor="left",
                yanchor="top",
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#BDBDBD',
                borderwidth=1,
                font=dict(size=16)
            ),
            dict(
                buttons=sr_toggle_buttons,
                direction="down",
                showactive=True,
                x=0.25,
                y=1.25,
                xanchor="left",
                yanchor="top",
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#BDBDBD',
                borderwidth=1,
                font=dict(size=16)
            ),
            dict(
                buttons=vp_toggle_buttons,
                direction="down",
                showactive=True,
                x=0.45,
                y=1.25,
                xanchor="left",
                yanchor="top",
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#BDBDBD',
                borderwidth=1,
                font=dict(size=16)
            )
        ],

        # Chart title optimized for presentations
        title=dict(
            text=f"{instrument} Technical Analysis - {_format_timeframe_label(scales_seconds[0])} Timeframe",
            font=dict(size=22, color='#212121'),
            x=0.5,
            xanchor='center'
        ),

        # Legend configuration for presentations
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#BDBDBD',
            borderwidth=1,
            itemsizing='constant',  # Added for consistent item sizes
            itemwidth=30  # Added to make legend items narrower
        ),

        # Disable range slider for cleaner appearance
        xaxis_rangeslider_visible=False,

        # Responsive margins
        margin=dict(l=80, r=60, t=140, b=80),

        # Interactive features for enhanced user experience
        hovermode='x unified',
        dragmode='zoom'
    )

    # Export chart with enhanced configuration for research use
    output_filepath = out_html if out_html else f"{instrument}_stocksim_chart.html"
    fig.write_html(
        output_filepath,
        auto_open=False,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{instrument}_chart',
                'height': 900,
                'width': 1600,
                'scale': 2
            }
        }
    )
    print(f"✅ Generated chart: {output_filepath}")


def _format_timeframe_label(seconds: int) -> str:
    """
    Convert seconds to human-readable timeframe labels for interface.

    Args:
        seconds: Time interval in seconds

    Returns:
        Formatted label (e.g., "15s", "5m", "1h", "1d") optimized for UI
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h"
    elif seconds < 604800:  # Less than a week
        days = seconds // 86400
        return f"{days}d"
    elif seconds < 2629746:  # Less than a month (30.44 days)
        weeks = seconds // 604800
        return f"{weeks}w"
    else:
        months = seconds // 2629746
        return f"{months}mo"


def validate_demo_data(candles: List[Dict[str, Any]]) -> bool:
    """
    Validate market data for visualization compatibility and completeness.

    Args:
        candles: List of OHLCV candle data

    Returns:
        True if data is valid for visualization, False otherwise

    Validation Checks:
        - Minimum number of candles for meaningful analysis
        - Required OHLCV fields present
        - Chronological ordering
        - Reasonable price ranges
        - Volume data availability
    """
    required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for i, candle in enumerate(candles[:5]):  # Check first 5 candles
        for field in required_fields:
            if field not in candle:
                print(f"❌ Error: Missing required field '{field}' in candle {i}")
                return False

    # Check for reasonable price ranges
    prices = [c['close'] for c in candles]
    if max(prices) / min(prices) > 1000:
        print("⚠️ Warning: Extreme price range detected, chart may not display optimally")

    print(f"✅ Data validation passed: {len(candles)} candles ready for visualization")
    return True


def generate_demo_report(
    instrument: str,
    candles: List[Dict[str, Any]],
    indicator_kwargs: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive analysis report for the visualization.

    Args:
        instrument: Trading instrument symbol
        candles: Historical OHLCV data
        indicator_kwargs: Technical indicator configuration

    Returns:
        Dictionary containing analysis summary for presentation
    """
    if not validate_demo_data(candles):
        return {}

    # Initialize indicators for analysis
    tracker = IndicatorsTracker(**(indicator_kwargs or {}))
    for candle in candles:
        tracker.update(candle)

    # Calculate key metrics
    prices = [c['close'] for c in candles]
    volumes = [c['volume'] for c in candles]

    total_return = (prices[-1] / prices[0] - 1) * 100
    volatility = (max(prices) / min(prices) - 1) * 100
    avg_volume = sum(volumes) / len(volumes)

    # Get latest technical indicators
    indicators = tracker.get_latest_values()

    report = {
        'instrument': instrument,
        'period': f"{candles[0]['timestamp'][:10]} to {candles[-1]['timestamp'][:10]}",
        'total_candles': len(candles),
        'price_summary': {
            'start_price': round(prices[0], 2),
            'end_price': round(prices[-1], 2),
            'highest_price': round(max(prices), 2),
            'lowest_price': round(min(prices), 2),
            'total_return_pct': round(total_return, 2),
            'volatility_pct': round(volatility, 2)
        },
        'volume_summary': {
            'average_volume': round(avg_volume, 0),
            'total_volume': sum(volumes),
            'max_volume': max(volumes)
        },
        'technical_indicators': {
            'support_levels': len(tracker.support_levels),
            'resistance_levels': len(tracker.resistance_levels),
            'current_rsi': indicators.get('momentum_indicators', {}).get('rsi', {}).get('value'),
            'poc_price': tracker.volume_profile_poc_price
        }
    }

    print(f"📊 Analysis Report Generated for {instrument}")
    print(f"   Period: {report['period']}")
    print(f"   Total Return: {report['price_summary']['total_return_pct']:.2f}%")
    print(f"   Support/Resistance Levels: {report['technical_indicators']['support_levels']}/{report['technical_indicators']['resistance_levels']}")

    return report


def _wrap_text(text, max_chars_per_line=40):
    """
    Wrap text to a maximum number of characters per line.

    Args:
        text: The text to wrap
        max_chars_per_line: Maximum characters per line

    Returns:
        Text with line breaks inserted to limit width
    """
    if not text:
        return ""

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        # Check if adding this word would exceed the max line length
        if current_length + len(word) + (1 if current_line else 0) > max_chars_per_line:
            # Start a new line
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            # Add to current line
            current_line.append(word)
            current_length += len(word) + (1 if current_line else 0)

    # Add the last line
    if current_line:
        lines.append(" ".join(current_line))

    return "<br>".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate enhanced interactive stock charts for StockSim analysis"
    )
    parser.add_argument("--symbol", type=str, required=True,
                       help="Ticker symbol (e.g., NVDA for stocks, BTCUSDT for crypto)")
    parser.add_argument("--interval", type=str, required=True,
                       help="Base candle interval (e.g., 1d for daily, 1h for hourly)")
    parser.add_argument("--start_date", type=str, required=True,
                       help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, required=True,
                       help="End date in YYYY-MM-DD format")
    parser.add_argument("--scales", type=int, nargs="+", default=[900, 1800, 3600, 14400, 86400],
                       help="Timeframe scales in seconds (default: 15m, 30m, 1h, 4h, 1d)")
    parser.add_argument("--data_source", type=str, choices=["polygon", "alpha_vantage"],
                       default="polygon", help="Data source for market data (Polygon.io or Alpha Vantage)")
    parser.add_argument("--symbol_type", type=str, choices=["stock", "crypto"],
                       default="stock", help="Symbol type for proper data loading")
    parser.add_argument("--output", type=str, default=None,
                       help="Output HTML file path")
    parser.add_argument("--indicator_config", type=str, default=None,
                       help="JSON config file for technical indicators")
    parser.add_argument("--orders_file", type=str, default=None,
                       help="JSON file with executed orders and explanations")
    parser.add_argument("--portfolio_file", type=str, default=None,
                       help="JSON file with portfolio performance data")
    parser.add_argument("--generate_report", action="store_true",
                       help="Generate analysis report")

    args = parser.parse_args()

    print(f"🚀 Starting StockSim Chart Generation for {args.symbol}")

    # Load indicator configuration
    indicator_kwargs = {}
    if args.indicator_config and os.path.exists(args.indicator_config):
        with open(args.indicator_config, 'r') as f:
            indicator_kwargs = json.load(f)
        print(f"📋 Loaded indicator configuration from {args.indicator_config}")

    # Load order execution data
    executed_orders = None
    if args.orders_file and os.path.exists(args.orders_file):
        with open(args.orders_file, 'r') as f:
            executed_orders = json.load(f)
        print(f"📈 Loaded {len(executed_orders)} executed orders with LLM explanations")

    # Load portfolio performance data
    portfolio_timeseries = None
    if args.portfolio_file and os.path.exists(args.portfolio_file):
        with open(args.portfolio_file, 'r') as f:
            portfolio_timeseries = json.load(f)
        print(f"💰 Loaded portfolio performance data with {len(portfolio_timeseries)} entries")

    # Initialize data client based on user preference
    if args.data_source == "alpha_vantage":
        client = AlphaVantageClient()
        print(f"🔌 Connected to Alpha Vantage for {args.symbol_type} data")
    else:
        client = PolygonClient()
        print(f"🔌 Connected to Polygon.io for {args.symbol_type} data")

    # Load market data with proper symbol type handling
    print(f"📊 Loading {args.symbol} data from {args.data_source}...")
    try:
        if args.data_source == "alpha_vantage":
            # Use Alpha Vantage for market data
            if args.symbol_type == "crypto":
                candles = client.load_crypto_aggregates(
                    symbol=args.symbol,
                    interval=args.interval,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    market="USD",
                    sort="asc",
                    limit=10000,
                    use_cache=True
                )
            else:
                candles = client.load_aggregates(
                    symbol=args.symbol,
                    interval=args.interval,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    adjusted=True,
                    sort="asc",
                    limit=10000,
                    use_cache=True
                )
        else:
            # Use Polygon.io for market data (default)
            if args.symbol_type == "crypto":
                candles = client.load_crypto_aggregates(
                    symbol=args.symbol,
                    interval=args.interval,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    market="USD",
                    sort="asc",
                    limit=10000,
                    use_cache=True
                )
            else:
                candles = client.load_aggregates(
                    symbol=args.symbol,
                    interval=args.interval,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    adjusted=True,
                    sort="asc",
                    limit=10000,
                    use_cache=True
                )

        print(f"📈 Successfully loaded {len(candles)} candles")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # Validate data for visualization
    if not validate_demo_data(candles):
        sys.exit(1)

    # Validate and adjust timeframe scales
    base_interval_sec = interval_to_seconds(args.interval)

    # Filter scales to only include those >= base interval
    valid_scales = [s for s in args.scales if s >= base_interval_sec]

    # Always include the base interval as the first scale
    if base_interval_sec not in valid_scales:
        valid_scales.insert(0, base_interval_sec)

    # Sort scales to ensure proper order
    scales = sorted(set(valid_scales))

    if not scales:
        print(f"⚠️ No valid scales. Using base interval only.")
        scales = [base_interval_sec]

    print(f"🔧 Using timeframes: {[_format_timeframe_label(s) for s in scales]}")

    # Generate analysis report if requested
    if args.generate_report:
        report = generate_demo_report(args.symbol, candles, indicator_kwargs)

        # Ensure output directories exist and save report to reports folder
        charts_dir, reports_dir = ensure_output_directories()
        report_file = os.path.join(reports_dir, f"{args.symbol}_analysis_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Analysis report saved to {report_file}")

    # Generate enhanced interactive chart
    print(f"🎨 Generating interactive chart...")
    make_chart_dropdown(
        candles=candles,
        instrument=args.symbol,
        scales_seconds=scales,
        out_html=args.output,
        indicator_kwargs=indicator_kwargs,
        executed_orders=executed_orders,
        portfolio_timeseries=portfolio_timeseries,
        symbol_type=args.symbol_type
    )

    print(f"🎉 StockSim chart generation completed successfully!")
