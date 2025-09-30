import subprocess
import random
import torch
import numpy as np
import os
import sys


def check_assert_enabled():
    if (not __debug__) or sys.flags.optimize > 0 or os.environ.get("PYTHONOPTIMIZE"):
        raise SystemExit("Asserts required. Do not run with -O/-OO or PYTHONOPTIMIZE.")


def get_git_info() -> str:
    """
    Get git hash and dirty status for the current repository.

    Returns:
        Tuple of (short_hash, is_dirty) where:
        - short_hash: 7-character git hash, or None if not a git repo
        - is_dirty: True if there are uncommitted changes
    """
    try:
        # Get short git hash (7 characters)
        result = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True, check=True)
        short_hash = result.stdout.strip()

        # Check if working tree is dirty
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        is_dirty = len(result.stdout.strip()) > 0

        info_str = short_hash
        if is_dirty:
            info_str += "_unstaged"

        return info_str

    except subprocess.CalledProcessError as e:
        # Git command failed - might be corrupted repo
        raise RuntimeError(f"Git command failed: {e}") from e
    except FileNotFoundError as e:
        # Git not available
        raise RuntimeError("Git executable not found - please install git") from e


def set_global_seed(seed: int):
    """Set global random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_script_name() -> str:
    """Get the name of the calling script without .py extension."""
    import inspect
    import os

    # Get the filename of the script that called this function
    frame = inspect.currentframe()
    try:
        # Go up the call stack to find the main script
        caller_frame = frame.f_back if frame else None
        while caller_frame:
            filename = caller_frame.f_code.co_filename
            if filename.endswith(".py") and os.path.basename(filename) != "training_utils.py":
                return os.path.splitext(os.path.basename(filename))[0]
            caller_frame = caller_frame.f_back
    finally:
        if frame:
            del frame

    # Fallback
    return "unknown_script"
