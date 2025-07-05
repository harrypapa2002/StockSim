"""
Date and Time Parsing Utilities for StockSim

This module provides common date and time parsing functions used across
exchange agents for consistent handling of ISO format dates and timestamps.

Functions:
    parse_iso_date: Parse ISO date strings (YYYY-MM-DD)
    parse_iso_datetime: Parse ISO datetime strings with timezone support
"""

from datetime import datetime, date
from typing import Optional


def parse_iso_date(s: Optional[str]) -> Optional[date]:
    """
    Parse ISO date string (YYYY-MM-DD) with error handling.
    
    Args:
        s: ISO date string
        
    Returns:
        Date object or None if invalid
    """
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_iso_datetime(s: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO datetime string with timezone support.
    
    Args:
        s: ISO datetime string (e.g., "2024-08-01T22:03:34Z")
        
    Returns:
        Timezone-aware datetime object or None if invalid
    """
    if not s:
        return None
    try:
        # Replace trailing 'Z' with '+00:00' for fromisoformat compatibility
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        return None