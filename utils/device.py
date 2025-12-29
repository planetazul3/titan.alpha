"""
Device selection and management utilities for PyTorch.

This module provides utilities to select and query compute devices
(CPU, CUDA, MPS) based on availability and user preference.

Example:
    >>> from utils.device import resolve_device, get_device_info
    >>> device = resolve_device('auto')
    >>> info = get_device_info()
    >>> print(f"Using {device}, CUDA available: {info['cuda_available']}")
"""

import logging
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> torch.device:
    """
    Resolve compute device based on availability and preference.

    The device selection priority for 'auto' mode:
    1. CUDA (NVIDIA GPUs) if available
    2. MPS (Apple Silicon) if available
    3. CPU as fallback

    Args:
        preference: Device preference. Options:
            - 'cpu': Force CPU usage
            - 'cuda': Use CUDA, fail if unavailable
            - 'mps': Use MPS, fail if unavailable
            - 'auto': Automatically select best available device

    Returns:
        torch.device object representing the selected device

    Raises:
        RuntimeError: If specified device preference is unavailable
        ValueError: If preference is not a valid option

    Example:
        >>> device = resolve_device('auto')
        >>> model = model.to(device)
        >>> batch = batch.to(device)
    """
    valid_preferences = ("cpu", "cuda", "mps", "auto")
    # Allow "cuda:X" format
    if not (preference in valid_preferences or preference.startswith("cuda:")):
        raise ValueError(
            f"Invalid device preference '{preference}'. Must be one of {valid_preferences} or 'cuda:X'"
        )

    # CPU requested - always available
    if preference == "cpu":
        logger.info("Using CPU device (user preference)")
        return torch.device("cpu")

    # Handle explicit CUDA definition (e.g. "cuda:1")
    if preference.startswith("cuda:"):
        if not torch.cuda.is_available():
             raise RuntimeError(f"CUDA device '{preference}' requested but CUDA is not available.")
        
        # Verify index
        try:
            parts = preference.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx >= torch.cuda.device_count():
                     raise RuntimeError(f"CUDA device index {idx} out of range (count={torch.cuda.device_count()})")
        except ValueError:
            pass # Let torch.device handle validation if we miss something specific

        logger.info(f"Using pinned CUDA device: {preference}")
        return torch.device(preference)

    # CUDA requested or auto mode
    if preference in ["cuda", "auto"]:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            logger.info(f"Using CUDA device: {device_name} ({device_count} GPU(s) available)")
            return torch.device("cuda")
        elif preference == "cuda":
            raise RuntimeError(
                "CUDA device requested but CUDA is not available. "
                "Install CUDA-enabled PyTorch or use 'auto'/'cpu' preference."
            )

    # MPS requested or auto mode (after CUDA check)
    if preference in ["mps", "auto"]:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS device (Apple Silicon)")
            return torch.device("mps")
        elif preference == "mps":
            raise RuntimeError(
                "MPS device requested but MPS is not available. "
                "Ensure you're on Apple Silicon with PyTorch >= 1.12"
            )

    # Fallback to CPU
    logger.info("Using CPU device (fallback - no accelerator available)")
    return torch.device("cpu")


def is_compilation_supported() -> bool:
    """
    Check if the current environment supports `torch.compile()`.

    Returns:
        bool: True if torch.compile is available and supported on this platform.
    """
    # 1. Check PyTorch version (>= 2.0.0)
    try:
        version_major = int(torch.__version__.split('.')[0])
        if version_major < 2:
            return False
    except (ValueError, IndexError):
        return False
        
    # 2. Check Platform
    # Windows support for torch.compile is experimental/limited in 2.0
    import platform
    if platform.system() == "Windows":
        return False
        
    return hasattr(torch, "compile")


def get_device_info() -> dict[str, Any]:
    """
    Return comprehensive information about available compute devices.

    Returns:
        Dictionary containing:
            - cuda_available (bool): CUDA availability
            - mps_available (bool): MPS availability
            - cpu_count (int): Number of CPU cores
            - cuda_device_count (int): Number of CUDA devices (if available)
            - cuda_device_name (str): Primary CUDA device name (if available)
            - cuda_memory_allocated (int): Memory allocated on GPU 0 (if available)
            - cuda_memory_reserved (int): Memory reserved on GPU 0 (if available)
            - torch_version (str): PyTorch version

    Example:
        >>> info = get_device_info()
        >>> print(f"CUDA: {info['cuda_available']}, CPUs: {info['cpu_count']}")
    """
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "compile_supported": is_compilation_supported(),
    }

    # Add CUDA-specific information
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)

        # Get properties for device 0
        props = torch.cuda.get_device_properties(0)
        info["cuda_compute_capability"] = f"{props.major}.{props.minor}"
        info["cuda_total_memory"] = props.total_memory

    return info


def print_device_info() -> None:
    """
    Print formatted device information to console.

    Useful for debugging and logging device configuration.

    Example:
        >>> from utils.device import print_device_info
        >>> print_device_info()
        Device Information:
        ==================
        PyTorch Version: 2.0.0
        CUDA Available: True
        ...
    """
    info = get_device_info()

    print("\nDevice Information:")
    print("=" * 50)
    print(f"PyTorch Version: {info['torch_version']}")
    print(f"CPU Threads: {info['cpu_count']}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"MPS Available: {info['mps_available']}")

    if info["cuda_available"]:
        print("\nCUDA Details:")
        print(f"  Device Count: {info['cuda_device_count']}")
        print(f"  Device Name: {info['cuda_device_name']}")
        print(f"  Compute Capability: {info.get('cuda_compute_capability', 'N/A')}")
        print(f"  Total Memory: {info.get('cuda_total_memory', 0) / 1e9:.2f} GB")
        print(f"  Memory Allocated: {info['cuda_memory_allocated'] / 1e6:.2f} MB")
        print(f"  Memory Reserved: {info['cuda_memory_reserved'] / 1e6:.2f} MB")

    print("=" * 50)
