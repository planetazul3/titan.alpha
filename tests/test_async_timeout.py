
import unittest
import asyncio
import time
from execution.sqlite_mixin import SQLiteTransactionMixin

class TestAsyncTimeout(unittest.IsolatedAsyncioTestCase):
    
    async def test_run_with_timeout_success(self):
        """Test that fast operations succeed."""
        loop = asyncio.get_running_loop()
        
        def fast_func():
            return "ok"
            
        result = await SQLiteTransactionMixin.run_with_timeout(loop, fast_func, 1.0)
        self.assertEqual(result, "ok")

    async def test_run_with_timeout_fail(self):
        """Test that slow operations raise TimeoutError."""
        loop = asyncio.get_running_loop()
        
        def slow_func():
            time.sleep(0.5)
            return "too slow"
            
        with self.assertRaises(asyncio.TimeoutError):
            await SQLiteTransactionMixin.run_with_timeout(loop, slow_func, timeout=0.1)

    async def test_exception_propagation(self):
        """Test that regular exceptions are propagated."""
        loop = asyncio.get_running_loop()
        
        def boom():
            raise ValueError("boom")
            
        with self.assertRaises(ValueError):
            await SQLiteTransactionMixin.run_with_timeout(loop, boom, timeout=1.0)

if __name__ == '__main__':
    unittest.main()
