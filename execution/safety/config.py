from dataclasses import dataclass

@dataclass
class ExecutionSafetyConfig:
    """Configuration for safety limits."""
    max_trades_per_minute: int = 5
    max_trades_per_minute_per_symbol: int = 2
    max_daily_loss: float = 50.0
    max_stake_per_trade: float = 20.0
    max_retry_attempts: int = 3
    retry_base_delay: float = 1.0
    kill_switch_enabled: bool = True  # C2 Fix: Safe-by-default
    
    # H4: Warmup Veto Configuration
    min_warmup_steps: int = 120  # Matches ID-004 requirements
