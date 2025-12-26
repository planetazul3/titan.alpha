
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


def cleanup_logs(log_dir: Path, retention_days: int = 7) -> int:
    """
    Remove log files older than retention period.
    
    Args:
        log_dir: Directory containing logs
        retention_days: Retention period in days
        
    Returns:
        Number of files deleted
    """
    import time
    
    if not log_dir.exists():
        return 0
        
    cutoff_time = time.time() - (retention_days * 86400)
    deleted_count = 0
    
    try:
        for log_file in log_dir.glob("*.log"):
            try:
                if log_file.is_file() and log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    deleted_count += 1
            except Exception as e:
                # Log to stderr since logger might not be setup or is restricted
                print(f"Failed to delete old log {log_file}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error during log cleanup: {e}", file=sys.stderr)
        
    return deleted_count
