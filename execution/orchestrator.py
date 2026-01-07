"""
Inference Orchestrator Module.

Encapsulates the complex logic of the main trading loop:
1. Feature Engineering
2. Model Inference
3. Regime Veto Assessment
4. Challenger Inference
5. Execution
6. Observability

This refactor addresses Audit Issue #6 (Responsibility Leakage).
"""

import logging
import time
import asyncio
import torch
import numpy as np
from typing import Any, Dict, Optional, List
from functools import partial

from config.settings import Settings
from models.core import DerivOmniModel
from execution.executor import DerivTradeExecutor
from observability import TradingMetrics
from observability.calibration import CalibrationMonitor
from execution.real_trade_tracker import RealTradeTracker
from execution.strategy_adapter import StrategyAdapter
from data.buffer import MarketDataBuffer

logger = logging.getLogger(__name__)

class InferenceOrchestrator:
    """
    Coordinator for the live trading inference cycle.
    """
    
    def __init__(
        self,
        model: DerivOmniModel,
        engine: "DecisionEngine",  # From execution.decision
        executor: "SafeTradeExecutor",  # From execution.safety
        feature_builder: "FeatureBuilder",  # From data.features
        device: torch.device,
        settings: Settings,
        metrics: TradingMetrics,
        calibration_monitor: Optional[CalibrationMonitor] = None,
        trade_tracker: Optional[RealTradeTracker] = None,
        strategy_adapter: Optional[StrategyAdapter] = None,
        tracer: Optional["Tracer"] = None  # From opentelemetry.trace
    ):
        self.model = model
        self.engine = engine
        self.executor = executor
        self.feature_builder = feature_builder
        self.device = device
        self.settings = settings
        self.metrics = metrics
        self.calibration_monitor = calibration_monitor
        self.trade_tracker = trade_tracker
        self.strategy_adapter = strategy_adapter
        self.tracer = tracer
        
        self.inference_count = 0
        
        # Concurrency control for challengers
        # Limit concurrent challenger tasks to prevent event loop starvation
        self.challenger_semaphore = asyncio.Semaphore(2)

    async def run_cycle(
        self,
        market_snapshot: Dict[str, Any],
        challengers: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[float]:
        """
        Execute a single inference cycle.
        
        Args:
            market_snapshot: Dictionary containing 'ticks', 'candles', etc.
            challengers: Optional list of challenger model stacks for A/B testing
            
        Returns:
            float: Reconstruction error (if calculated), else None
        """
        self.inference_count += 1
        inference_start = time.perf_counter()
        
        span = None
        if self.tracer:
            span = self.tracer.start_span("run_inference")
            span.set_attribute("device", str(self.device))
            
        reconstruction_error = 0.0

        try:
            # 1. Unpack Snapshot
            t_np = market_snapshot.get('ticks')
            c_np = market_snapshot.get('candles')
            
            if t_np is None or c_np is None or len(t_np) == 0 or len(c_np) == 0:
                logger.warning("[ORCHESTRATOR] Empty snapshot. Skipping inference.")
                return None

            # 2. Build Canonical Features
            features = self.feature_builder.build(
                ticks=t_np, 
                candles=c_np, 
                timestamp=time.time()
            )

            t_tensor = features["ticks"].unsqueeze(0).to(self.device)
            c_tensor = features["candles"].unsqueeze(0).to(self.device)
            v_tensor = features["vol_metrics"].unsqueeze(0).to(self.device)

            # 3. Model Inference (Offloaded to Executor)
            loop = asyncio.get_running_loop()
            
            with torch.no_grad():
                # Predict Probabilities
                probs = await loop.run_in_executor(
                    None, 
                    partial(self.model.predict_probs, t_tensor, c_tensor, v_tensor)
                )

                # Predict Reconstruction Error (Regime Assessment)
                reconstruction_error = await loop.run_in_executor(
                    None,
                    lambda: self.model.get_volatility_anomaly_score(v_tensor).item()
                )

            if span:
                span.set_attribute("reconstruction_error", reconstruction_error)

            # 4. Observability: Record Metrics
            inf_latency = time.perf_counter() - inference_start
            self.metrics.record_inference_latency(inf_latency)
            self.metrics.set_reconstruction_error(reconstruction_error)

            sample_probs = {k: v.item() for k, v in probs.items()}
            logger.info(f"Predictions: {sample_probs}")
            logger.info(f"Reconstruction error: {reconstruction_error:.4f} (latency: {inf_latency * 1000:.1f}ms)")

            # 5. Calibration Monitor Update
            if self.calibration_monitor:
                self.calibration_monitor.record(reconstruction_error)
                self.calibration_monitor.recover_if_healthy()

            # 6. Run Challengers (Concurrent, Bounded)
            if challengers:
                # Issue 3: Unbounded Fire-and-Forget Fixed
                # We do NOT await this, but we use semaphore inside the task or check before launch
                # Ideally, we launch a task that acquires semaphore.
                
                # Filter strictly for available slots to avoid queue buildup? 
                # Or just let them queue on semaphore? 
                # Queuing on semaphore is safer than unbounded spawning.
                # However, if we spawn 1000 tasks that all wait on semaphore, we still have 1000 tasks objects.
                # Better: Check if semaphore is available, or use a limited queue.
                # Simple approach: If semaphore is locked, skip challenger execution for this cycle.
                
                if not self.challenger_semaphore.locked():
                     asyncio.create_task(self._run_challengers(
                         challengers, t_tensor, c_tensor, v_tensor, 
                         t_np, c_np, market_snapshot
                    ))
                else:
                    logger.debug("[ORCHESTRATOR] Challenger execution skipped (busy)")

            # 7. Decision Engine
            decision_start = time.perf_counter()
            
            entry_price = float(c_np[-1, 3]) # Close of last candle
            
            real_trades = await self.engine.process_with_context(
                probs=sample_probs,
                reconstruction_error=reconstruction_error,
                tick_window=t_np,
                candle_window=c_np,
                entry_price=entry_price,
                market_data={
                    "ticks_count": len(t_np), 
                    "candles_count": len(c_np)
                },
            )
            
            decision_latency = time.perf_counter() - decision_start
            self.metrics.record_decision_latency(decision_latency)

            # Record Veto Status
            stats = self.engine.get_statistics()
            if stats.get("vetoed_count", 0) > 0:
                 # Heuristic: if global veto count increased? No, that's aggregated.
                 # The engine doesn't return per-call metadata easily. 
                 # Assuming real_trades is empty if vetoed.
                 pass
            
            # Shadow-Only Check
            if self.calibration_monitor and self.calibration_monitor.should_skip_real_trades():
                if real_trades:
                    logger.warning(f"[SHADOW-ONLY] Blocking {len(real_trades)} trades due to calibration.")
                    self.metrics.record_trade_attempt(outcome="blocked_shadow_only", contract_type="all")
                real_trades = []

            # 8. Execution
            if real_trades and self.executor and self.trade_tracker:
                await self._execute_trades(real_trades, entry_price, reconstruction_error, c_np)

            return reconstruction_error

        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Inference cycle failed: {e}", exc_info=True)
            self.metrics.record_error("inference_failure")
            return None
        finally:
            if span:
                span.end()

    async def _execute_trades(self, trades, entry_price, reconstruction_error, c_np):
        """Handle trade execution logic."""
        # Calculate Volatility
        closes = c_np[:, 3]
        if hasattr(closes, "cpu"): closes = closes.cpu().numpy()
        elif hasattr(closes, "numpy"): closes = closes.numpy()
        
        volatility = 0.0
        if len(closes) > 20:
             log_returns = np.diff(np.log(closes[-20:]))
             volatility = float(np.std(log_returns) * np.sqrt(365 * 24 * 60))

        for signal in trades:
            # Metadata injection
            if signal.metadata is None: signal.metadata = {}
            signal.metadata['volatility'] = volatility
            
            if not self.strategy_adapter:
                logger.error("No StrategyAdapter. Cannot execute.")
                continue

            try:
                # Convert Signal -> Request
                assessment = self.engine.get_regime_assessment(reconstruction_error)
                regime_str = self.engine.get_regime_state_string(assessment)
                
                req = await self.strategy_adapter.convert_signal(
                    signal, 
                    reconstruction_error=reconstruction_error,
                    regime_state=regime_str
                )
            except Exception as e:
                logger.error(f"Signal conversion failed: {e}")
                continue

            # Execute with Context
            async with self.trade_tracker.intent(
                direction=signal.direction,
                entry_price=entry_price,
                stake=req.stake,
                probability=signal.probability,
                contract_type=req.contract_type
            ) as intent_id:
                
                start_exec = time.perf_counter()
                result = await self.executor.execute(req)
                lat = time.perf_counter() - start_exec
                self.metrics.record_execution_latency(lat)

                if result.success:
                    logger.info(f"Trade executed: {result.contract_id}")
                    self.trade_tracker.confirm_intent(intent_id, result.contract_id)
                    await self.trade_tracker.register_trade(
                        contract_id=result.contract_id,
                        direction=signal.direction,
                        entry_price=entry_price,
                        stake=req.stake,
                        probability=signal.probability,
                        contract_type=str(req.contract_type)
                    )
                else:
                    logger.error(f"Trade failed: {result.error}")
                    self.metrics.record_trade_attempt(outcome="failed", contract_type=req.contract_type)
                    # Helper cleanup if needed
                    self.trade_tracker.cleanup_intent(intent_id)

    async def _run_challengers(
        self, 
        challengers, 
        t_tensor, c_tensor, v_tensor, 
        t_np, c_np, 
        snapshot
    ):
        """Run challenger models with semaphore protection."""
        async with self.challenger_semaphore:
            for c_stack in challengers:
                try:
                    model = c_stack["model"]
                    engine = c_stack["engine"]
                    loop = asyncio.get_running_loop()
                    
                    # 1. Predictions
                    probs = await loop.run_in_executor(
                        None, 
                        partial(model.predict_probs, t_tensor, c_tensor, v_tensor)
                    )
                    sample_probs = {k: v.item() for k, v in probs.items()}
                    
                    # 2. Recon Error
                    reconstruction_error = await loop.run_in_executor(
                        None,
                        lambda: model.get_volatility_anomaly_score(v_tensor).item()
                    )
                    
                    # 3. Shadow Process
                    await engine.process_with_context(
                        probs=sample_probs,
                        reconstruction_error=reconstruction_error,
                        tick_window=t_np,
                        candle_window=c_np,
                        entry_price=float(c_np[-1, 3]),
                        market_data={"ticks_count": len(t_np), "candles_count": len(c_np)}
                    )
                except Exception as e:
                    logger.error(f"[CHALLENGER {c_stack['version']}] Failed: {e}")
