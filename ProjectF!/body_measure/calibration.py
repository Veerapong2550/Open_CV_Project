"""ArUco calibration, independent from the legacy ``test.py`` script."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import ARUCO_DICT, ARUCO_MARKER_ID


_detector_cache: dict[str, object] = {}


def _get_aruco_detector(aruco, dict_name: str):
    if dict_name in _detector_cache:
        return _detector_cache[dict_name]
    dictionary_id = getattr(aruco, dict_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unsupported ArUco dictionary: {dict_name}")
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    detector = None
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    _detector_cache[dict_name] = (detector, dictionary)
    return detector, dictionary


def get_pixel_scale(frame: np.ndarray, marker_size_cm: float,
                    marker_id: int = ARUCO_MARKER_ID) -> tuple[Optional[float], Optional[np.ndarray]]:
    """Return (pixels_per_cm, marker_corners) for the requested marker."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None, None
    if marker_size_cm <= 0 or not hasattr(cv2, "aruco"):
        return None, None
    aruco = cv2.aruco
    detector, dictionary = _get_aruco_detector(aruco, ARUCO_DICT)
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = aruco.detectMarkers(frame, dictionary)
    if ids is None:
        return None, None
    for candidate, candidate_id in zip(corners, ids.flatten()):
        if int(candidate_id) == marker_id:
            points = candidate.reshape(4, 2)
            sides = [np.linalg.norm(points[index] - points[(index + 1) % 4]) for index in range(4)]
            return float(np.mean(sides) / marker_size_cm), points.astype(np.int32)
    return None, None


def estimate_pixel_scale_from_height(nose: tuple[float, float],
                                     ankle: tuple[float, float],
                                     user_height_cm: float,
                                     head_compensation_factor: float = 1.06) -> float | None:
    """Calculate fallback pixel-to-cm scale from user's known height.

    Dist(nose, ankle) * head_compensation_factor / user_height_cm
    """
    if not nose or not ankle or len(nose) < 2 or len(ankle) < 2 or user_height_cm <= 0:
        return None
    body_height_px = float(np.hypot(nose[0] - ankle[0], nose[1] - ankle[1])) * head_compensation_factor
    if body_height_px < 20.0:
        return None
    return float(body_height_px / user_height_cm)
