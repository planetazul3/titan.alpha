"""
Trading execution and decision-making logic.

Modules:
    - decision: DecisionEngine for processing model outputs
    - signals: TradeSignal dataclass and utilities
    - filters: Signal filtering logic
    - shadow_logger: Shadow trade logging with versioning
    - shadow_store: NDJSON-backed shadow trade store (legacy)
    - sqlite_shadow_store: SQLite-backed shadow trade store (recommended)
    - regime: RegimeVeto authority for blocking trades during anomalous conditions
    - executor: TradeExecutor abstraction for broker-isolated execution
    - safety: SafeTradeExecutor wrapper with rate limiting, P&L caps, and kill switch
    - position_sizer: Kelly Criterion position sizing with safety margins

Example:
    >>> from execution import DecisionEngine, ShadowLogger
    >>> from execution.regime import RegimeVeto
    >>> from execution.executor import DerivTradeExecutor
    >>> from execution.safety import SafeTradeExecutor, ExecutionSafetyConfig
    >>>
    >>> # Wrap executor with safety controls for production
    >>> raw_executor = DerivTradeExecutor(client, settings)
    >>> config = ExecutionSafetyConfig(max_trades_per_minute=5, max_daily_loss=50.0)
    >>> executor = SafeTradeExecutor(raw_executor, config)
    >>>
    >>> regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
    >>> engine = DecisionEngine(settings, regime_veto=regime_veto, shadow_logger=shadow_logger)
    >>> signals = engine.process_model_output(probs, reconstruction_error)
"""
