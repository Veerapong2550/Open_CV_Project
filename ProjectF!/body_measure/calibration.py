"""
ArUco based calibration utilities.
Uses the existing `get_reference_scale` from `test.py` for the pixel‑to‑cm ratio.
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional

# Configuration constants are in config.py
from .config import REFERENCE_MARKER_CM, ARUCO_DICT

def _load_aruco():
    aruco = cv2.aruco
    # Use getattr to get dictionary constant safely
    dict_attr = getattr(aruco, ARUCO_DICT, None)
    if dict_attr is None:
        raise ValueError(f"Unsupported ArUco dictionary: {ARUCO_DICT}")
    dictionary = aruco.getPredefinedDictionary(dict_attr)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    return detector

def detect_marker(frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Detect ArUco markers in a frame.
    Returns the corners and ids (or (None, None) if not found).
    """
    detector = _load_aruco()
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids

# Helper to load get_reference_scale from sibling test.py without importing the whole package name.
def _load_reference_scale_function():
    import importlib.util, sys
    test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test.py"))
    spec = importlib.util.spec_from_file_location("test_module", test_path)
    test_mod = importlib.util.module_from_spec(spec)
    sys.modules["test_module"] = test_mod
    spec.loader.exec_module(test_mod)
    return getattr(test_mod, "get_reference_scale")

# Wrapper that uses the function from test.py
def get_pixel_scale(frame: np.ndarray) -> Optional[float]:
    """Return the calibrated pixel‑to‑cm scale (px/cm) using the ArUco marker.
    If calibration fails, returns None.
    """
    get_ref = _load_reference_scale_function()
    scale_px, _ = get_ref(frame)
    return scale_px
