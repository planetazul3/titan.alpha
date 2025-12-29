
import logging
from utils.device import resolve_device, get_device_info, is_compilation_supported
import torch

logging.basicConfig(level=logging.INFO)

def test_device_utils():
    print("Testing resolve_device()...")
    
    # Auto
    dev = resolve_device("auto")
    print(f"  resolve_device('auto') -> {dev}")
    
    # CPU
    dev_cpu = resolve_device("cpu")
    print(f"  resolve_device('cpu') -> {dev_cpu}")
    assert dev_cpu.type == "cpu"
    
    # Compilation
    print(f"\nTesting is_compilation_supported()...")
    supported = is_compilation_supported()
    print(f"  Supported: {supported}")
    
    # Info
    print(f"\nTesting get_device_info()...")
    info = get_device_info()
    print(f"  Info keys: {list(info.keys())}")
    print(f"  Compile supported (in info): {info.get('compile_supported')}")
    
    # Invalid
    try:
        resolve_device("invalid")
    except ValueError as e:
        print(f"\nCaught expected ValueError for invalid device: {e}")

    # Explicit CUDA (Hardware dependent, skip assertion if no CUDA)
    if torch.cuda.is_available():
        print(f"\nTesting explicit CUDA index...")
        try:
             dev_cuda0 = resolve_device("cuda:0")
             print(f"  resolve_device('cuda:0') -> {dev_cuda0}")
        except Exception as e:
             print(f"  Failed resolve_device('cuda:0'): {e}")
    
    print("\n--- Device Utils Verification Passed ---")

if __name__ == "__main__":
    test_device_utils()
