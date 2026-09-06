"""Saving full-body camera captures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import cv2
import numpy as np

from .config import CAPTURE_DIR


def save_full_body_capture(frame: np.ndarray, capture_type: str = "full_body") -> Path | None:
    """Save a verified full-body frame and return its path, or ``None`` on failure.

    ``capture_type`` is included in the filename so a completed two-view
    session can distinguish its front shoulder image from its side posture
    image.  The application calls this only after its landmark/size/stability
    gates have passed.
    """
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        safe_type = re.sub(r"[^a-z0-9_-]+", "_", capture_type.lower()).strip("_") or "full_body"
        filename = f"{safe_type}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        path = CAPTURE_DIR / filename
        return path if cv2.imwrite(str(path), frame) else None
    except (OSError, cv2.error):
        return None
