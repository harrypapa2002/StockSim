"""
Data Validation Utilities for StockSim

This module provides common data validation and parsing functions used across
exchange agents for consistent handling of various input types.

Functions:
    parse_quantity: Parse and validate quantity values from various input types
"""

import re
from typing import Union


def parse_quantity(value: Union[str, int, float, None], default: int = 0) -> int:
    """
    Parse quantity from various input types with error handling.
    
    Args:
        value: Input value to convert to integer
        default: Default value if conversion fails
        
    Returns:
        Integer quantity or default value
    """
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default

    return default