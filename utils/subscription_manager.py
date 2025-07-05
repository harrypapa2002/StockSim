"""
Subscription Management Utilities for StockSim

This module provides common subscription management functionality for
exchange agents, including status enums and subscription handling patterns.

Classes:
    SubscriptionStatus: Enumeration for agent subscription status
    
Functions:
    create_subscription_confirmation: Create standardized subscription confirmation
    create_unsubscription_confirmation: Create standardized unsubscription confirmation
"""

from enum import Enum
from typing import Dict, Any


class SubscriptionStatus(Enum):
    """Enumeration for agent subscription status to market data feeds."""
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


def create_subscription_confirmation(instrument: str, status: SubscriptionStatus) -> Dict[str, Any]:
    """
    Create a standardized subscription confirmation message.

    Args:
        instrument: Financial instrument symbol
        status: Subscription status (SUBSCRIBED or UNSUBSCRIBED)

    Returns:
        Dictionary containing confirmation message payload
    """
    return {
        "instrument": instrument,
        "status": status.value
    }


def create_subscription_response(instrument: str) -> Dict[str, Any]:
    """
    Create a subscription confirmation response.

    Args:
        instrument: Financial instrument symbol

    Returns:
        Dictionary containing subscription confirmation
    """
    return create_subscription_confirmation(instrument, SubscriptionStatus.SUBSCRIBED)


def create_unsubscription_response(instrument: str) -> Dict[str, Any]:
    """
    Create an unsubscription confirmation response.

    Args:
        instrument: Financial instrument symbol

    Returns:
        Dictionary containing unsubscription confirmation
    """
    return create_subscription_confirmation(instrument, SubscriptionStatus.UNSUBSCRIBED)