"""
Configuration module for the DerivOmniModel trading system.

This module provides configuration management through Pydantic models
and application constants.

Public API:
    - Settings: Main configuration class
    - load_settings: Factory function to load settings from environment
    - CONTRACT_TYPES: Enumeration of supported contract types
    - SIGNAL_TYPES: Enumeration of signal classification types
    - Constants: MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH, DEFAULT_SEED
"""

from config.constants import (
    CONTRACT_TYPES,
    DEFAULT_SEED,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    SIGNAL_TYPES,
    ContractType,
    SignalType,
)
from config.settings import Settings, load_settings

__all__ = [
    "Settings",
    "load_settings",
    "CONTRACT_TYPES",
    "SIGNAL_TYPES",
    "MIN_SEQUENCE_LENGTH",
    "MAX_SEQUENCE_LENGTH",
    "DEFAULT_SEED",
    "ContractType",
    "SignalType",
]
