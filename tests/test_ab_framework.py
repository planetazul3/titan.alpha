
import unittest
import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from execution.decision import DecisionEngine
from execution.sqlite_shadow_store import SQLiteShadowStore
from config.settings import Settings
from config.constants import CONTRACT_TYPES

class TestABFramework(unittest.TestCase):
    def setUp(self):
        self.db_path = Path("test_ab_framework.db")
        if self.db_path.exists():
            os.remove(self.db_path)
            
        # Shared Store
        self.store = SQLiteShadowStore(self.db_path)
        
        # Settings
        os.environ["ENVIRONMENT"] = "development"
        # Reload settings to pick up env var if needed or just init
        self.settings = Settings()
        
        # Override thresholds for test predictability
        self.settings.thresholds.confidence_threshold_high = 0.7
        self.settings.thresholds.learning_threshold_min = 0.5
        self.settings.shadow_trade.min_probability_track = 0.5
        # Ensure regime caution doesn't filter
        self.settings.hyperparams.regime_caution_threshold = 0.1
        self.settings.hyperparams.regime_veto_threshold = 0.2
        
    def tearDown(self):
        self.store.close()
        if self.db_path.exists():
            os.remove(self.db_path)

    async def async_test_champion_challenger_logging(self):
        try:
            # 1. Setup Champion Engine (REAL)
            champion_engine = DecisionEngine(
                settings=self.settings,
                shadow_store=self.store,
                model_version="champion-v1",
                execution_mode="REAL"
            )
            
            # 2. Setup Challenger Engine (SHADOW)
            challenger_engine = DecisionEngine(
                settings=self.settings,
                shadow_store=self.store,
                model_version="challenger-v1",
                execution_mode="SHADOW"
            )
            
            # 3. Simulate Inputs
            probs = {"rise_fall_prob": 0.8}
            recon_error = 0.05
            
            # 4. Run Process (Concurrent simulation)
            # Champion (0.8 probability)
            await champion_engine.process_with_context(
                probs=probs,
                reconstruction_error=recon_error,
                tick_window=[100.0, 101.0],
                candle_window=[[100.0, 101.0, 99.0, 100.5]],
                entry_price=100.5,
                timestamp=None
            )
            
            # Challenger (0.55 probability)
            await challenger_engine.process_with_context(
                probs={"rise_fall_prob": 0.55}, # Different probs
                reconstruction_error=recon_error,
                tick_window=[100.0, 101.0],
                candle_window=[[100.0, 101.0, 99.0, 100.5]],
                entry_price=100.5,
                timestamp=None
            )
            
            # 5. Flush tasks
            await champion_engine.shutdown()
            await challenger_engine.shutdown()
            
            # 6. Verify Store
            records = self.store.query()
            self.assertEqual(len(records), 2)
            
            champion_record = next(r for r in records if r.model_version == "champion-v1")
            challenger_record = next(r for r in records if r.model_version == "challenger-v1")
            
            # Verify Metadata
            self.assertEqual(champion_record.metadata.get("execution_mode"), "REAL")
            self.assertEqual(challenger_record.metadata.get("execution_mode"), "SHADOW")
            
            # Verify Probs
            self.assertAlmostEqual(champion_record.probability, 0.8)
            self.assertAlmostEqual(challenger_record.probability, 0.55)
            
        finally:
            # Engines might hold references?
            pass

    def test_run_async(self):
        asyncio.run(self.async_test_champion_challenger_logging())

if __name__ == '__main__':
    unittest.main()
