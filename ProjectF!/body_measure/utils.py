"""
Utility functions shared across modules.
- One Euro filter wrapper
- Pixel-to-centimeter conversion using ArUco calibration
- Helper to draw text with Thai fonts via Pillow
"""

import numpy as np
from OneEuroFilter import OneEuroFilter
from typing import Tuple

def create_filter(rate: float, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> OneEuroFilter:
    """Create a OneEuroFilter instance.
    Args:
        rate: Expected sampling rate (Hz).
        min_cutoff: Minimum cutoff frequency.
        beta: Speed coefficient.
        d_cutoff: Cutoff frequency for derivative.
    """
    return OneEuroFilter(rate, min_cutoff, beta, d_cutoff)

def smooth_point(filter_obj: OneEuroFilter, point: np.ndarray) -> np.ndarray:
    """Smooth a 2‑D point using the provided filter.
    Returns a new NumPy array with smoothed coordinates.
    """
    return np.array([filter_obj(point[0]), filter_obj(point[1])])

def pixel_to_cm(px: float, scale: float) -> float:
    """Convert pixel length to centimeters using the calibrated scale (px/cm)."""
    return px / scale if scale != 0 else 0.0
