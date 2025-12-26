"""
Performance Tracking for High-Frequency Inference.

This module provides real-time monitoring of system performance, including:
- Inference latency percentiles (p50, p95, p99)
- Memory usage (RAM + GPU if available)
- Device utilization context

Metrics are essential for identifying bottlenecks in the critical path.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque

import numpy as np
import psutil
import torch

from utils.device import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """Snapshot of system performance."""
    
    timestamp: float
    latency_ms: float
    memory_mb: float
    cpu_percent: float
    gpu_memory_mb: float = 0.0


class PerformanceTracker:
    """
    Tracks critical system performance metrics with sliding window.
    
    Attributes:
        window_size: Number of samples to keep for percentile calculation
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latencies: Deque[float] = deque(maxlen=window_size)
        self._stats_history: Deque[PerformanceStats] = deque(maxlen=window_size)
        self.device = resolve_device("auto")
        self._start_time = time.time()
        self._total_inferences = 0
        
        # Determine if GPU monitoring is possible
        self._gpu_available = torch.cuda.is_available()
        
        logger.info(f"PerformanceTracker initialized (window={window_size}, device={self.device})")

    def record_inference(self, latency_sec: float) -> None:
        """Record a single inference execution."""
        latency_ms = latency_sec * 1000.0
        self._latencies.append(latency_ms)
        self._total_inferences += 1
        
        # Optional: Record full system stats periodically (e.g. every 10th call) to save overhead
        if self._total_inferences % 10 == 0:
            self._capture_system_stats(latency_ms)

    def _capture_system_stats(self, latency_ms: float) -> None:
        """Capture low-level system stats."""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        gpu_mem = 0.0
        if self._gpu_available:
            try:
                gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass
        
        stats = PerformanceStats(
            timestamp=time.time(),
            latency_ms=latency_ms,
            memory_mb=mem_info.rss / (1024 * 1024),
            cpu_percent=process.cpu_percent(),
            gpu_memory_mb=gpu_mem
        )
        self._stats_history.append(stats)

    def get_latency_percentiles(self) -> dict[str, float]:
        """Calculate p50, p95, p99 latency."""
        if not self._latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            
        data = list(self._latencies)
        return {
            "p50": np.percentile(data, 50),
            "p95": np.percentile(data, 95),
            "p99": np.percentile(data, 99),
            "max": np.max(data)
        }
        
    def get_summary(self) -> dict:
        """Get summary of current performance state."""
        metrics = self.get_latency_percentiles()
        
        current_mem = 0.0
        if self._stats_history:
            current_mem = self._stats_history[-1].memory_mb
            
        return {
            "latency": metrics,
            "memory_mb": round(current_mem, 1),
            "total_inferences": self._total_inferences,
            "uptime_sec": round(time.time() - self._start_time, 1)
        }
