
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

def setup_logging(script_name: str = "app", log_level: int = logging.INFO) -> tuple[logging.Logger, Path, Path]:
    """
    Configure logging to both console and file.
    
    Args:
        script_name: Name of the script/component (used for log filename)
        log_level: Logging level (default: INFO)
        
    Returns:
        Tuple[logging.Logger, Path, Path]: (logger, log_dir, log_file)
    """
    # Find project root by looking for marker files
    # Start from current file's directory and walk up
    current_dir = Path(__file__).resolve().parent
    project_root = None
    
    # Walk up directory tree looking for project markers
    for parent in [current_dir] + list(current_dir.parents):
        if any((parent / marker).exists() for marker in [".git", ".env", "pyproject.toml"]):
            project_root = parent
            break
    
    # Fallback to current working directory if no markers found
    if project_root is None:
        project_root = Path.cwd()
    
    # Create logs directory under project root
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    # Configure handlers
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
    
    # Configure root logger
    # Force reconfiguration to override any previous basicConfig
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger()
    logger.info(f"Logging initialized. Writing to console and {log_file}")
    
    return logger, log_dir, log_file
