"""
Time Utilities for StockSim

This module provides time and date parsing utilities for the StockSim simulation platform.
It handles timezone-aware datetime parsing, resolution conversions, and interval calculations
required for candle-based backtesting and time series analysis.

Key Features:
- UTC-aware datetime parsing with automatic timezone handling
- Resolution string parsing for candle intervals (1m, 5m, 1h, 1d, etc.)
- Interval to seconds/timedelta conversions
- Support for multiple time formats used by different data providers
- Human-readable time formatting utilities
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing import Optional


def parse_datetime_utc(dt_str: str) -> Optional[datetime]:
    """
    Parse an ISO formatted datetime string and ensure the result is UTC offset-aware.
    If the input datetime string is offset-naive, it assumes UTC.
    """
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def resolution_to_seconds(res_str: str) -> Optional[int]:
    """
    Converts a resolution string (e.g., "5m", "15m", "1h", "1d") to seconds.
    """
    try:
        if res_str.endswith("m"):
            return int(res_str[:-1]) * 60
        elif res_str.endswith("h"):
            return int(res_str[:-1]) * 3600
        elif res_str.endswith("d"):
            return int(res_str[:-1]) * 86400
        elif res_str.endswith("w"):
            return int(res_str[:-1]) * 604800
    except (ValueError, AttributeError):
        return None
    return None

def seconds_to_human(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    elif seconds < 3600:
        minutes = seconds // 60
        rem = seconds % 60
        return f"{minutes} minute{'s' if minutes != 1 else ''}" + (f" {rem} second{'s' if rem != 1 else ''}" if rem > 0 else "")
    elif seconds < 86400:
        hours = seconds // 3600
        rem = seconds % 3600
        minutes = rem // 60
        return f"{hours} hour{'s' if hours != 1 else ''}" + (f" {minutes} minute{'s' if minutes != 1 else ''}" if minutes > 0 else "")
    else:
        days = seconds // 86400
        rem = seconds % 86400
        hours = rem // 3600
        return f"{days} day{'s' if days != 1 else ''}" + (f" {hours} hour{'s' if hours != 1 else ''}" if hours > 0 else "")


def parse_resolution_alpha_vantage(interval: str) -> str:
    """
    Converts our flexible syntax to Alpha-Vantage syntax.
    '60m' | '60min'  → '60min'
    '1h'             → '60min'   (AV only supports minute intraday)
    '1d'             → 'DAILY'
    '1w'             → 'WEEKLY'
    '1mo'            → 'MONTHLY'
    """
    n, u = _parse_parts(interval)
    if u == "m":   return f"{n}min"
    if u == "h":   return f"{n*60}min"
    if u == "d":   return "DAILY"
    if u == "w":   return "WEEKLY"
    if u == "mo":  return "MONTHLY"
    raise ValueError(f"Unsupported interval: {interval}")

def parse_resolution_polygon(interval: str) -> tuple[int, str]:
    """
    Convert our flexible syntax to Polygon API format.
    Returns (multiplier, timespan) tuple.
    
    Examples:
    '1m' -> (1, 'minute')
    '5m' -> (5, 'minute') 
    '1h' -> (1, 'hour')
    '1d' -> (1, 'day')
    '1w' -> (1, 'week')
    '1mo' -> (1, 'month')
    """
    n, u = _parse_parts(interval)
    
    # Map to Polygon timespan
    if u == "m":
        timespan = 'minute'
    elif u == "h":
        timespan = 'hour'  
    elif u == "d":
        timespan = 'day'
    elif u == "w":
        timespan = 'week'
    elif u == "mo":
        timespan = 'month'
    else:
        raise ValueError(f"Unsupported time unit: {u}")
    
    return n, timespan

_INTERVAL_RX = re.compile(r"^(\d+)\s*(min|m|h|d|w|mo)$", re.I)

def _parse_parts(interval: str) -> tuple[int, str]:
    """
    Returns (numeric_value, unit) from strings like '60min', '60m', '1h', '1d', '2w', '3mo'.
    Raises ValueError if the format is unsupported.
    """
    m = _INTERVAL_RX.match(interval.strip())
    if not m:
        raise ValueError(f"Unsupported interval format: {interval}")
    value = int(m.group(1))
    unit  = m.group(2).lower()
    if unit == "min":   unit = "m"       # normalise: 'min' ➜ 'm'
    return value, unit

def parse_interval_to_timedelta(interval: str) -> timedelta:
    n, u = _parse_parts(interval)
    return (
        timedelta(minutes=n) if u == "m" else
        timedelta(hours=n)   if u == "h" else
        timedelta(days=n)    if u == "d" else
        timedelta(weeks=n)   if u == "w" else
        timedelta(days=n*30) if u == "mo" else
        None
    )

def interval_to_seconds(interval: str) -> int:
    if interval.lower().endswith("mo"):
        return int(interval[:-2]) * 2592000
    unit = interval[-1].lower()
    try:
        value = int(interval[:-1])
    except ValueError:
        raise ValueError(f"Invalid interval format: {interval}")
    if unit == 's':
        return value
    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    elif unit == 'd':
        return value * 86400
    elif unit == 'w':
        return value * 604800
    else:
        raise ValueError(f"Unsupported interval format: {interval}")




