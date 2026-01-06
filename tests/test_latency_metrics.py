
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import time
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.orchestrator import InferenceOrchestrator
from observability import TradingMetrics

class TestLatencyMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_decision_latency_recording(self):
        """Test that decision latency is recorded during inference."""
        # Mocks
        mock_model = MagicMock()
        mock_model.predict_probs.return_value = {"up": torch.tensor(0.6)}
        mock_model.get_volatility_anomaly_score.return_value = torch.tensor(0.1)
        
        # Mock engine: MagicMock for sync methods, AsyncMock for async methods
        mock_engine = MagicMock()
        mock_engine.process_with_context = AsyncMock()
        mock_engine.process_with_context.return_value = []
        mock_engine.get_statistics.return_value = {}
        
        # Orchestrator doesn't take buffer in run_cycle, it takes snapshot
        snapshot = {
            "ticks": torch.zeros(60),
            "candles": torch.zeros((60, 6)),
            "tick_count": 100,
            "candle_count": 100 
        }
        
        mock_fb = MagicMock()
        mock_fb.build.return_value = {
            "ticks": torch.zeros(60),
            "candles": torch.zeros((60, 6)),
            "vol_metrics": torch.zeros(10)
        }
        
        mock_settings = MagicMock()
        mock_metrics = MagicMock(spec=TradingMetrics)
        
        # Simulate engine thinking time
        async def delayed_process(*args, **kwargs):
            await asyncio.sleep(0.01)
            return []
        mock_engine.process_with_context.side_effect = delayed_process
        
        # Initialize Orchestrator
        orchestrator = InferenceOrchestrator(
            model=mock_model,
            engine=mock_engine,
            executor=AsyncMock(),
            feature_builder=mock_fb,
            device="cpu",
            settings=mock_settings,
            metrics=mock_metrics
        )
        
        # Run
        await orchestrator.run_cycle(
            market_snapshot=snapshot
        )
        
        # Verify
        mock_metrics.record_decision_latency.assert_called_once()
        # latency should be > 0
        args, _ = mock_metrics.record_decision_latency.call_args
        latency = args[0]
        self.assertGreater(latency, 0.0)
        print(f"Recorded decision latency: {latency*1000:.2f}ms")

if __name__ == '__main__':
    unittest.main()
