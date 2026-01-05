"""
Unit tests for utils/seed.py.

Tests global seeding and random state management.
"""

import random
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from utils.seed import get_random_state, restore_random_state, set_global_seed


class TestSeedUtils(unittest.TestCase):
    """Test seed utilities."""

    def test_set_global_seed_basic(self):
        """Test setting a basic integer seed."""
        with patch("random.seed") as mock_random, \
             patch("numpy.random.seed") as mock_np_random, \
             patch("torch.manual_seed") as mock_torch_seed, \
             patch("torch.cuda.manual_seed_all") as mock_cuda_seed, \
             patch("os.environ", {}) as mock_env:
            
            # Mock CUDA availability
            with patch("torch.cuda.is_available", return_value=True):
                 set_global_seed(42)

            mock_random.assert_called_with(42)
            mock_np_random.assert_called_with(42)
            mock_torch_seed.assert_called_with(42)
            mock_cuda_seed.assert_called_with(42)
            self.assertEqual(mock_env["PYTHONHASHSEED"], "42")

    def test_set_global_seed_default(self):
        """Test setting default seed when None provided."""
        from config.constants import DEFAULT_SEED
        
        with patch("random.seed") as mock_random:
            set_global_seed(None)
            mock_random.assert_called_with(DEFAULT_SEED)

    def test_set_global_seed_invalid_type(self):
        """Test invalid seed types raise TypeError."""
        with self.assertRaises(TypeError):
            set_global_seed("not_an_int")
            
        with self.assertRaises(TypeError):
            set_global_seed(12.34)

    def test_set_global_seed_negative(self):
        """Test negative seed raises ValueError."""
        with self.assertRaises(ValueError):
            set_global_seed(-1)

    def test_cudnn_determinism(self):
        """Test cuDNN flags are set if available."""
        with patch("torch.backends.cudnn.is_available", return_value=True), \
             patch("torch.backends.cudnn") as mock_cudnn:
            
            set_global_seed(123)
            
            self.assertTrue(mock_cudnn.deterministic)
            self.assertFalse(mock_cudnn.benchmark)

    def test_get_and_restore_state(self):
        """Test capturing and restoring random state."""
        # 1. Set initial state and capture
        set_global_seed(42)
        state_42 = get_random_state()
        
        # 2. Change state by drawing numbers
        _ = random.random()
        _ = np.random.rand(1)
        _ = torch.rand(1)
        
        state_changed = get_random_state()
        self.assertNotEqual(state_42["python"], state_changed["python"])
        
        # 3. Restore state
        restore_random_state(state_42)
        state_restored = get_random_state()
        
        # 4. Verify match (Python tuples vs lists might differ in DeepDiff but direct comparison works for random state)
        self.assertEqual(state_42["python"], state_restored["python"])
        
        # Numpy state is a tuple (str, ndarray, int, int, float)
        # We need to compare carefully or assume np.random.get_state() returns comparable objects
        np.testing.assert_equal(state_42["numpy"][1], state_restored["numpy"][1]) # Check state array

        # Torch state
        self.assertTrue(torch.equal(state_42["torch"], state_restored["torch"]))

    @patch("torch.cuda.is_available", return_value=True)
    def test_gpu_state_save_restore(self, mock_cuda_avail):
        """Test GPU state logic is called."""
        with patch("torch.cuda.get_rng_state_all") as mock_get, \
             patch("torch.cuda.set_rng_state_all") as mock_set:
            
            state = get_random_state()
            mock_get.assert_called_once()
            
            restore_random_state(state)
            mock_set.assert_called_once()
