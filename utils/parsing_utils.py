"""
Text Parsing Utilities for StockSim Trading Simulation

This module provides the core parse_model_output function for processing LLM outputs
in financial trading applications, along with its essential dependencies.
"""

import json
import re
from typing import Any, Dict, List


def remove_trailing_commas(json_str: str) -> str:
    """Remove trailing commas from JSON strings to fix common formatting issues."""
    pattern = r',\s*(?=[}\]])'
    return re.sub(pattern, '', json_str)


def extract_json_array(text: str) -> str:
    """Extract a JSON array from text, handling code block formatting."""
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'\s*```', '', text, flags=re.DOTALL)
    text = text.strip()

    start = text.rfind('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in model output")
    return text[start:end + 1].strip()


def _fix_json_text(text: str) -> str:
    """Fix common issues in raw LLM JSON outputs."""
    in_string = False
    escape_next = False
    fixed_chars = []

    i = 0
    while i < len(text):
        char = text[i]

        if escape_next:
            if char in '"\\/:bfnrt':
                fixed_chars.append(char)
            elif char == 'u' and i + 4 < len(text):
                unicode_part = text[i:i + 5]
                if re.match(r'u[0-9a-fA-F]{4}', unicode_part):
                    fixed_chars.append(unicode_part)
                    i += 4
                else:
                    fixed_chars.append('\\' + char)
            else:
                fixed_chars.append('\\' + char)
            escape_next = False

        elif char == '"' and not escape_next:
            in_string = not in_string
            fixed_chars.append(char)

        elif char == '\\' and in_string:
            fixed_chars.append(char)
            escape_next = True

        elif char == '\n' and in_string:
            fixed_chars.append('\\n')

        elif char == '\r' and in_string:
            fixed_chars.append('\\r')

        elif char == '\t' and in_string:
            fixed_chars.append('\\t')

        else:
            fixed_chars.append(char)

        i += 1

    return ''.join(fixed_chars)


def parse_model_output(text: str) -> List[Dict[str, Any]]:
    """
    Parse LLM output to extract trading decisions as a list of dictionaries.

    Args:
        text (str): Raw LLM output text containing trading decisions

    Returns:
        List[Dict[str, Any]]: List of parsed trading decision dictionaries
    """
    json_part = extract_json_array(text)
    json_part = remove_trailing_commas(json_part)
    # Remove comments from JSON
    json_part = re.sub(r'//.*?\n', '', json_part)
    # Fix invalid escape sequences
    json_part = _fix_json_text(json_part)
    return json.loads(json_part)
