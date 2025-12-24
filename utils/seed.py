"""
Utility functions for setting random seeds for reproducibility.
"""

import random
import torch
import numpy as np
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - PyTorch (CPU and CUDA)
    - NumPy (if available)
    - CUDA deterministic operations
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set NumPy seed if available
    try:
        np.random.seed(seed)
    except NameError:
        pass  # NumPy not imported
    
    # Enable deterministic CUDA operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

