"""
Configuration management using Pydantic v2 (Colab compatible).

This module provides type-safe configuration management for the trading system
using Pydantic v2. Configuration is loaded from environment variables and
validated on initialization.

Environment variables use double underscore for nesting:
    TRADING__SYMBOL=R_100
    THRESHOLDS__CONFIDENCE_THRESHOLD_HIGH=0.75
    HYPERPARAMS__LEARNING_RATE=0.0005

Example:
    >>> from config.settings import load_settings
    >>> settings = load_settings()
    >>> print(settings.trading.symbol)
    >>> device = settings.get_device()
"""

import logging
from typing import Literal

import torch

# Pydantic v2 imports
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.constants import DEFAULT_SEED, MIN_SEQUENCE_LENGTH
from utils.device import resolve_device

logger = logging.getLogger(__name__)


class Trading(BaseModel):
    """Trading configuration for live execution.

    Attributes:
        symbol: Deriv trading symbol (e.g., 'R_100', 'R_75', '1HZ100V')
        timeframe: Timeframe for candle data (e.g., '1m', '5m')
        stake_amount: Amount to stake per trade in account currency
        payout_ratio: Binary options payout ratio (industry standard: 0.95 for Deriv)
    """

    symbol: str = Field(..., description="Deriv trading symbol")
    timeframe: str = Field(default="1m", description="Candle timeframe")
    stake_amount: float = Field(..., description="Stake amount per trade", gt=0)
    payout_ratio: float = Field(
        default=0.95, 
        description="Binary options payout ratio (0.95 = 95% payout on win)",
        gt=0.0,
        le=1.0
    )


class Thresholds(BaseModel):
    """Probability thresholds for signal classification.

    Signals are classified based on model output probabilities:
    - prob >= confidence_threshold_high: REAL_TRADE
    - learning_threshold_max <= prob < confidence_threshold_high: SHADOW_TRADE
    - learning_threshold_min <= prob < learning_threshold_max: Learning zone
    - prob < learning_threshold_min: IGNORE

    Attributes:
        confidence_threshold_high: Minimum probability for real trade execution
        learning_threshold_min: Minimum probability to track as shadow trade
        learning_threshold_max: Maximum probability for learning zone
    """

    confidence_threshold_high: float = Field(
        ..., description="High confidence threshold for real trades", gt=0.5, le=1.0
    )
    learning_threshold_min: float = Field(
        ..., description="Minimum threshold for shadow trades", ge=0.0, le=1.0
    )
    learning_threshold_max: float = Field(
        ..., description="Maximum threshold for learning zone", ge=0.0, le=1.0
    )

    @model_validator(mode="after")
    def validate_thresholds_logic(self) -> "Thresholds":
        """Ensure thresholds are in ascending order."""
        if not (
            self.learning_threshold_min
            < self.learning_threshold_max
            < self.confidence_threshold_high
        ):
            raise ValueError(
                f"Thresholds must satisfy: learning_min < learning_max < confidence_high. "
                f"Got: {self.learning_threshold_min} < {self.learning_threshold_max} < {self.confidence_threshold_high}"
            )
        return self


class ModelHyperparams(BaseModel):
    """Neural network hyperparameters.

    Attributes:
        learning_rate: Learning rate for optimizer (typical range: 1e-5 to 1e-2)
        batch_size: Mini-batch size for training (must be positive)
        dropout_rate: Dropout probability for regularization (range: [0, 1))
        lstm_hidden_size: Hidden size for BiLSTM layers
        cnn_filters: Number of filters for CNN layers
        latent_dim: Dimensionality of latent space for volatility autoencoder
    """

    learning_rate: float = Field(..., description="Optimizer learning rate", gt=0, le=1.0)
    batch_size: int = Field(..., description="Training batch size", gt=0)
    dropout_rate: float = Field(default=0.1, description="Dropout probability", ge=0.0, lt=1.0)
    lstm_hidden_size: int = Field(..., description="BiLSTM hidden size", gt=0)
    cnn_filters: int = Field(..., description="CNN filter count", gt=0)
    latent_dim: int = Field(..., description="Autoencoder latent dimension", gt=0)
    use_tft: bool = Field(default=True, description="Use Temporal Fusion Transformer (TFT) instead of BiLSTM")
    regime_caution_threshold: float = Field(default=0.2, description="Regime caution threshold", gt=0)
    regime_veto_threshold: float = Field(default=0.5, description="Regime veto threshold", gt=0)

    @field_validator("learning_rate", "batch_size")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        """Validate that critical hyperparameters are positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator("dropout_rate")
    @classmethod
    def valid_dropout(cls, v: float) -> float:
        """Validate dropout rate is in valid range [0, 1)."""
        if not (0 <= v < 1):
            raise ValueError(f"Dropout must be in [0, 1), got {v}")
        return v


class DataShapes(BaseModel):
    """Input sequence shapes for model training and inference.

    Attributes:
        sequence_length_ticks: Number of tick prices in sequence
        sequence_length_candles: Number of OHLCV candles in sequence
    """

    sequence_length_ticks: int = Field(
        ..., description="Tick sequence length", ge=MIN_SEQUENCE_LENGTH
    )
    sequence_length_candles: int = Field(
        ..., description="Candle sequence length", ge=MIN_SEQUENCE_LENGTH
    )
    feature_dim_candles: int = Field(
        default=10, description="Number of feature channels per candle", ge=1
    )
    feature_dim_ticks: int = Field(
        default=1, description="Number of feature channels per tick", ge=1
    )
    warmup_steps: int = Field(
        default=50, description="Warmup steps for technical indicators", ge=0
    )

    @field_validator("sequence_length_ticks", "sequence_length_candles")
    @classmethod
    def validate_length(cls, v: int) -> int:
        """Ensure sequence lengths meet minimum requirements."""
        if v < MIN_SEQUENCE_LENGTH:
            raise ValueError(f"Sequence length must be >= {MIN_SEQUENCE_LENGTH}, got {v}")
        return v


class ExecutionSafety(BaseModel):
    """Execution safety controls for production trading.

    These settings configure the SafeTradeExecutor wrapper that provides
    critical protections for live trading.

    Attributes:
        max_trades_per_minute: Maximum global trades per minute
        max_trades_per_minute_per_symbol: Maximum trades per symbol per minute
        max_daily_loss: Stop trading when daily losses exceed this (account currency)
        max_stake_per_trade: Maximum stake for any single trade
        max_retry_attempts: Number of retries for broker errors
        retry_base_delay: Base delay in seconds for exponential backoff
        kill_switch_enabled: If True, blocks ALL trade execution
    """

    max_trades_per_minute: int = Field(
        default=5, description="Maximum trades per minute (global)", ge=1, le=60
    )
    max_trades_per_minute_per_symbol: int = Field(
        default=3, description="Maximum trades per symbol per minute", ge=1, le=30
    )
    max_daily_loss: float = Field(
        default=50.0, description="Maximum daily loss before trading halts", gt=0
    )
    max_stake_per_trade: float = Field(
        default=10.0, description="Maximum stake per individual trade", gt=0
    )
    max_retry_attempts: int = Field(
        default=3, description="Number of broker error retries", ge=1, le=10
    )
    retry_base_delay: float = Field(
        default=1.0, description="Base delay in seconds for exponential backoff", gt=0, le=30
    )
    kill_switch_enabled: bool = Field(
        default=False, description="Emergency halt - blocks ALL trading"
    )


class CalibrationConfig(BaseModel):
    """Calibration monitoring configuration.
    
    Controls the CalibrationMonitor that tracks reconstruction errors
    for graceful degradation to shadow-only mode.
    
    Attributes:
        error_threshold: Reconstruction errors above this are considered "high"
        consecutive_threshold: Number of consecutive high errors to trigger shadow-only mode
        window_size: Rolling window size for error statistics
    """
    
    error_threshold: float = Field(
        default=1.0, description="High error threshold for reconstruction", gt=0
    )
    consecutive_threshold: int = Field(
        default=5, description="Consecutive high errors to activate shadow-only", ge=1, le=100
    )
    window_size: int = Field(
        default=20, description="Rolling window for error statistics", ge=5, le=200
    )


class ShadowTradeConfig(BaseModel):
    """Shadow trade configuration.
    
    Controls shadow trade tracking and resolution behavior.
    
    Attributes:
        duration_minutes: Duration in minutes for shadow trade contracts
        min_probability_track: Minimum probability to track as shadow trade
    """
    
    duration_minutes: int = Field(
        default=1, description="Shadow trade duration in minutes", ge=1, le=60
    )
    min_probability_track: float = Field(
        default=0.45, description="Minimum probability to track shadow trade", ge=0.0, le=1.0
    )


class HeartbeatConfig(BaseModel):
    """Heartbeat and monitoring configuration.
    
    Controls periodic status logging and stale data detection.
    
    Attributes:
        interval_seconds: Heartbeat interval in seconds
        stale_data_threshold_seconds: Seconds without ticks before warning
    """
    
    interval_seconds: int = Field(
        default=60, description="Heartbeat logging interval", ge=10, le=600
    )
    stale_data_threshold_seconds: int = Field(
        default=30, description="Stale data warning threshold", ge=10, le=300
    )


class ObservabilityConfig(BaseModel):
    """Observability and telemetry configuration.
    
    Controls metrics export, health checks, and monitoring outputs.
    
    Attributes:
        enable_prometheus: Enable Prometheus metrics export
        health_check_file_path: Path to system health JSON file
    """
    
    enable_prometheus: bool = Field(
        default=True, description="Enable Prometheus metrics"
    )
    health_check_file_path: str = Field(
        default="logs/system_health.json", description="Health check output file"
    )


class Settings(BaseSettings):
    """
    Main application settings loaded from environment variables.

    Configuration is automatically loaded from .env file and environment
    variables. Use double underscore for nested configuration:
        TRADING__SYMBOL=R_100
        HYPERPARAMS__LEARNING_RATE=0.0005

    Attributes:
        trading: Trading configuration (symbol, timeframe, stake)
        thresholds: Probability thresholds for signal classification
        hyperparams: Model hyperparameters
        data_shapes: Input sequence dimensions
        environment: Deployment environment (development/production)
        seed: Random seed for reproducibility
        device_preference: Compute device preference (cpu/cuda/mps/auto)
        deriv_api_token: Deriv API authentication token
        deriv_app_id: Deriv application ID
    """

    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=".env", extra="ignore")

    trading: Trading
    thresholds: Thresholds
    hyperparams: ModelHyperparams
    data_shapes: DataShapes
    execution_safety: ExecutionSafety = Field(default_factory=ExecutionSafety)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    shadow_trade: ShadowTradeConfig = Field(default_factory=ShadowTradeConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    environment: str = Field(default="development", description="Deployment environment")
    seed: int = Field(default=DEFAULT_SEED, description="Random seed for reproducibility")
    device_preference: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto", description="Compute device preference"
    )

    deriv_api_token: str = Field(default="", description="Deriv API token for authentication")
    deriv_app_id: int = Field(default=1089, description="Deriv application ID")

    def get_device(self) -> torch.device:
        """Resolve and return the compute device based on preference."""
        return resolve_device(self.device_preference)

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def validate_api_credentials(self) -> bool:
        """Check if API credentials are configured."""
        return bool(self.deriv_api_token and len(self.deriv_api_token) > 0)


def load_settings() -> Settings:
    """
    Load and validate settings from environment variables and .env file.

    Returns:
        Validated Settings instance

    Raises:
        ValidationError: If configuration is invalid

    Example:
        >>> settings = load_settings()
        >>> print(f"Trading {settings.trading.symbol} on {settings.get_device()}")
    """
    try:
        settings = Settings()  # type: ignore[call-arg]
        logger.info(f"Settings loaded successfully. Environment: {settings.environment}")
        logger.debug(f"Device preference: {settings.device_preference}")
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        raise
