import time
import torch
import numpy as np
import os
import sys

# Add current dir to sys.path
sys.path.append(os.getcwd())

from models.core import DerivOmniModel
from config.settings import Settings

def main():
    settings = Settings()
    device = torch.device('cpu')
    
    print("--- Performance Benchmarking ---")
    
    # 1. Model Inference Latency
    print("1. Benchmarking Model Inference...")
    model = DerivOmniModel(settings).to(device)
    model.eval()
    
    # Generate dummy input
    batch_size = 1
    seq_len_ticks = settings.data_shapes.sequence_length_ticks
    seq_len_candles = settings.data_shapes.sequence_length_candles
    
    ticks = torch.randn(batch_size, seq_len_ticks)
    candles = torch.randn(batch_size, seq_len_candles, 10)
    vol_metrics = torch.randn(batch_size, 4)
    
    latencies = []
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(ticks, candles, vol_metrics)
            
    # Measure
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(ticks, candles, vol_metrics)
        latencies.append(time.perf_counter() - start)
        
    avg_latency = np.mean(latencies) * 1000
    p95_latency = np.percentile(latencies, 95) * 1000
    
    print(f"   Avg Latency: {avg_latency:.2f} ms")
    print(f"   P95 Latency: {p95_latency:.2f} ms")
    
    # 2. Database Performance (Simple write test)
    print("2. Benchmarking Database I/O (trading_state.db)...")
    import sqlite3
    db_path = 'data_cache/trading_state.db'
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        write_times = []
        for i in range(50):
            start = time.perf_counter()
            conn.execute("INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)", 
                         (f'bench_{i}', 'value', time.time()))
            conn.commit()
            write_times.append(time.perf_counter() - start)
        
        avg_write = np.mean(write_times) * 1000
        print(f"   Avg Commit latency: {avg_write:.2f} ms")
        conn.execute("DELETE FROM kv_store WHERE key LIKE 'bench_%'")
        conn.commit()
        conn.close()
    else:
        print("   Database not found, skipping I/O benchmark.")

    # Generate baseline report
    with open('/home/planetazul3/.gemini/antigravity/brain/15733ef8-b8cd-47b0-9756-f5f4f6a58dc9/PERFORMANCE_BASELINE.md', 'w') as f:
        f.write("# PERFORMANCE_BASELINE.md\n\n")
        f.write("## System Metrics\n")
        f.write(f"- **Device**: CPU\n")
        f.write(f"- **Model Parameters**: {sum(p.numel() for p in model.parameters()):,}\n\n")
        
        f.write("## Latency Benchmarks\n")
        f.write("| Operation | Metric | Value |\n")
        f.write("|-----------|--------|-------|\n")
        f.write(f"| Inference | Avg | {avg_latency:.2f} ms |\n")
        f.write(f"| Inference | P95 | {p95_latency:.2f} ms |\n")
        if 'avg_write' in locals():
            f.write(f"| DB Commit | Avg | {avg_write:.2f} ms |\n")
            
        f.write("\n## Resource Consumption (Estimated)\n")
        f.write("- **Memory Footprint**: ~150MB (Base)\n")
        f.write("- **CPU Utilization**: Low-Medium during inference\n")

if __name__ == "__main__":
    main()
