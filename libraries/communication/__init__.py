"""
Communication infrastructure for EggAI agents.

This module provides messaging, channels, transport, and streaming functionality.
"""

from .channels import channels
from .messaging import subscribe
from .streaming import get_conversation_string, stream_dspy_response, validate_conversation
from .transport import create_kafka_transport

__all__ = [
    "channels",
    "subscribe",
    "create_kafka_transport",
    "get_conversation_string",
    "stream_dspy_response",
    "validate_conversation",
]
