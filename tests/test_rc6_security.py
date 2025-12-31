
import pytest
import os
from unittest.mock import patch, MagicMock
from config.settings import Settings

class TestRC6ConfigurationSecurity:
    """Verification for RC-6 Configuration Security Remediation."""

    def test_is_test_mode_detection(self):
        """Verify is_test_mode property detects test context."""
        # We are running inside pytest, so this should effectively be True
        # But we create a fresh settings instance
        # Note: Pydantic BaseSettings loads from env, need to be careful with existing env
        
        # Test Case 1: Detect pytest via env var (simulated)
        with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "True", "ENVIRONMENT": "production"}):
            # This combination logic:
            # is_test_mode = True (due to PYTEST_CURRENT_TEST)
            # is_production = True
            # But the validator SHOULD raise RuntimeError because running test in production env
            
            with pytest.raises(RuntimeError, match="strictly forbidden"):
                Settings()

    def test_safe_environment_allows_init(self):
        """Verify initialization works in safe environment."""
        with patch.dict(os.environ, {
            "PYTEST_CURRENT_TEST": "True", 
            "ENVIRONMENT": "test",
            "DERIV_API_TOKEN": "" # Safe
        }):
            settings = Settings()
            assert settings.is_test_mode
            assert not settings.is_production()

    def test_production_token_in_test_mode_raises(self):
        """RC-6 Core Requirement: Prod token in test mode must fail."""
        with patch.dict(os.environ, {
            "PYTEST_CURRENT_TEST": "True",
            "ENVIRONMENT": "test",
            "DERIV_API_TOKEN": "prod_12345_dangerous_token"
        }):
            with pytest.raises(RuntimeError, match="Production token detected"):
                Settings()

    def test_production_token_in_production_mode_allowed(self):
        """RC-6: Prod token allowed in prod mode (but not running under pytest)."""
        # We must unset PYTEST_CURRENT_TEST to simulate real run
        env_patch = {
            "ENVIRONMENT": "production",
            "DERIV_API_TOKEN": "prod_allowed_token",
            # Ensure other required fields are present if not loaded from .env
            "TRADING__SYMBOL": "R_100",
            "TRADING__STAKE_AMOUNT": "10",
        }
        
        # We can't really "unset" PYTEST_CURRENT_TEST easily inside pytest without 
        # potentially confusing things, but patch.dict allows popping.
        with patch.dict(os.environ, env_patch, clear=True):
            # Also need to make sure PYTEST_CURRENT_TEST is gone
            if "PYTEST_CURRENT_TEST" in os.environ:
                 del os.environ["PYTEST_CURRENT_TEST"]
                 
            # Re-inject critical ones that might be missing if we cleared too much?
            # BaseSettings will read from .env if we are not careful.
            # But duplicate envs override .env.
            
            # Note: mocking os.environ affects Settings() load
            settings = Settings()
            assert settings.is_production()
            # Should not raise
            
    def test_production_token_case_insensitivity(self):
        """Verify PROD token detection is case insensitive."""
        with patch.dict(os.environ, {
            "PYTEST_CURRENT_TEST": "True",
            "ENVIRONMENT": "test",
            "DERIV_API_TOKEN": "PROD_TOKEN_UPPERCASE"
        }):
             with pytest.raises(RuntimeError, match="Production token detected"):
                Settings()
