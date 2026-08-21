"""ArUco calibration, independent from the legacy ``test.py`` script."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import ARUCO_DICT, ARUCO_MARKER_ID


def get_pixel_scale(frame: np.ndarray, marker_size_cm: float,
                    marker_id: int = ARUCO_MARKER_ID) -> tuple[Optional[float], Optional[np.ndarray]]:
    """Return (pixels_per_cm, marker_corners) for the requested marker."""
    if marker_size_cm <= 0 or not hasattr(cv2, "aruco"):
        return None, None
    aruco = cv2.aruco
    dictionary_id = getattr(aruco, ARUCO_DICT, None)
    if dictionary_id is None:
        raise ValueError(f"Unsupported ArUco dictionary: {ARUCO_DICT}")
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    try:
        corners, ids, _ = aruco.ArucoDetector(dictionary, aruco.DetectorParameters()).detectMarkers(frame)
    except AttributeError:
        corners, ids, _ = aruco.detectMarkers(frame, dictionary)
    if ids is None:
        return None, None
    for candidate, candidate_id in zip(corners, ids.flatten()):
        if int(candidate_id) == marker_id:
            points = candidate.reshape(4, 2)
            sides = [np.linalg.norm(points[index] - points[(index + 1) % 4]) for index in range(4)]
            return float(np.mean(sides) / marker_size_cm), points.astype(int)
    return None, None
