"""Saving full-body camera captures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import CAPTURE_DIR


def save_full_body_capture(frame: np.ndarray) -> Path | None:
    """Save a camera frame and return its path, or ``None`` if saving fails."""
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"full_body_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        path = CAPTURE_DIR / filename
        return path if cv2.imwrite(str(path), frame) else None
    except (OSError, cv2.error):
        return None
