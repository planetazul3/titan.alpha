"""
CLI argument parsing for live trading.

Separates argument parsing from business logic for:
- Testability and reusability
- Fail-fast validation
- Clean module boundaries

Reference: docs/plans/live_script_refactoring.md Section 4.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def parse_live_trading_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse and validate CLI arguments for live trading.
    
    Args:
        argv: Command line arguments (defaults to sys.argv)
        
    Returns:
        Parsed and validated arguments
        
    Exit codes:
        2: Invalid arguments (argparse default)
    """
    parser = argparse.ArgumentParser(
        description="x.titan Live Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/live.py --test                     # Test connection only
  python scripts/live.py --shadow-only              # Shadow mode (no real trades)
  python scripts/live.py --strategy compound        # Compounding strategy
  python scripts/live.py --strategy kelly           # Kelly criterion sizing

Exit codes:
  0   - Clean exit
  1   - Fatal error
  130 - SIGINT (Ctrl+C)
  143 - SIGTERM
        """
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Checkpoint Arguments
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint name to load (e.g., 'best_model'). Auto-selects 'best_model.pt' if not specified."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--skip-checkpoint-verify",
        action="store_true",
        help="Skip checkpoint verification (not recommended for production)"
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Mode Arguments
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connection only, don't trade"
    )
    parser.add_argument(
        "--shadow-only",
        action="store_true",
        help="Run in shadow mode (no real trades, log outcomes only)"
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Position Sizing Strategy Arguments
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--compound",
        action="store_true",
        help="Enable compounding strategy (alias for --strategy compound)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["fixed", "compound", "martingale", "kelly"],
        default="fixed",
        help="Position sizing strategy (default: fixed)"
    )
    parser.add_argument(
        "--x-amount",
        type=str,
        default=None,
        help="Multiplier for compound/martingale (e.g., '2x', 'reinvest')"
    )
    parser.add_argument(
        "--winstrikes",
        type=int,
        default=5,
        help="Max consecutive wins/losses for compounding/martingale (default: 5)"
    )
    
    args = parser.parse_args(argv)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Validation (Fail-Fast)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Handle --compound as alias for --strategy compound
    if args.compound and args.strategy == "fixed":
        args.strategy = "compound"
    
    # Validate checkpoint directory exists if specified
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        parser.error(f"Checkpoint directory not found: {ckpt_dir}")
    
    # Warn about unsafe combinations
    if args.strategy == "martingale" and not args.shadow_only:
        print(
            "⚠️ WARNING: Martingale strategy can be risky. "
            "Consider using --shadow-only first.",
            file=sys.stderr
        )
    
    return args


def get_strategy_from_args(args: argparse.Namespace) -> str:
    """
    Determine position sizing strategy from args.
    
    Handles --compound alias and defaults.
    """
    if args.compound:
        return "compound"
    return args.strategy or "fixed"


def validate_checkpoint(args: argparse.Namespace) -> Path | None:
    """
    Determine checkpoint path from arguments.
    
    Returns:
        Path to checkpoint file, or None if no checkpoint
        
    Raises:
        SystemExit: If checkpoint specified but not found (unless --test or --skip-checkpoint-verify)
    """
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if args.checkpoint:
        # Explicit checkpoint specified
        ckpt_name = args.checkpoint
        if not ckpt_name.endswith('.pt'):
            ckpt_name = f"{ckpt_name}.pt"
        checkpoint_path = checkpoint_dir / ckpt_name
    else:
        # Default to best_model.pt
        checkpoint_path = checkpoint_dir / "best_model.pt"
    
    if checkpoint_path.exists():
        return checkpoint_path
    
    # Checkpoint not found
    if args.test or args.skip_checkpoint_verify:
        return None
    
    print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
    sys.exit(1)
