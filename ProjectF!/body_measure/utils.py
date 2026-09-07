"""Shared measurement, smoothing, drawing, and gesture helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HAND_CONNECTIONS = ((0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
                    (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
                    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
                    (17, 18), (18, 19), (19, 20))

# MediaPipe Pose's 33 landmark topology.  Keeping this locally avoids relying
# on private drawing helpers and makes the camera feedback visible at all
# times, not only once a measurement becomes valid.
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


def is_peace_sign(landmarks: Iterable) -> bool:
    """Recognise a V (peace) sign with more tolerant distance checks.

    This version relaxes the extension and folding thresholds to improve
    detection range when the hand is farther from the camera.
    """
    points = list(landmarks)
    if len(points) < 21:
        return False

    def distance(first: int, second: int) -> float:
        return math.hypot(points[first].x - points[second].x,
                          points[first].y - points[second].y)

    # Use wrist‑to‑middle‑finger distance as hand size reference.
    hand_size = max(distance(0, 9), 0.01)

    # Relaxed multipliers (original 0.22/0.12/0.18).
    EXT_FACTOR = 0.18
    FOLD_FACTOR = 0.15
    VERT_FACTOR = 0.20

    def extended(tip: int, pip: int) -> bool:
        return distance(tip, 0) > distance(pip, 0) + hand_size * EXT_FACTOR

    def folded(tip: int, pip: int) -> bool:
        return (
            distance(tip, 0) <= distance(pip, 0) + hand_size * FOLD_FACTOR
            or points[tip].y >= points[pip].y - hand_size * VERT_FACTOR
        )

    return (
        extended(8, 6) and extended(12, 10) and
        folded(16, 14) and folded(20, 18)
    )


def draw_hand_landmarks(frame: np.ndarray, landmarks: Iterable) -> None:
    height, width = frame.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in landmarks]
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (0, 200, 0), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1, cv2.LINE_AA)


def draw_pose_landmarks(frame: np.ndarray, landmarks: Iterable, min_visibility: float = 0.25) -> None:
    """Overlay the detected pose, including when the app is waiting to start."""
    height, width = frame.shape[:2]
    points = list(landmarks)
    visible = [getattr(point, "visibility", 1.0) >= min_visibility for point in points]
    pixels = [(int(point.x * width), int(point.y * height)) for point in points]
    for start, end in POSE_CONNECTIONS:
        if start < len(visible) and end < len(visible) and visible[start] and visible[end]:
            cv2.line(frame, pixels[start], pixels[end], (255, 180, 0), 2, cv2.LINE_AA)
    for point, is_visible in zip(pixels, visible):
        if is_visible:
            cv2.circle(frame, point, 3, (0, 220, 255), -1, cv2.LINE_AA)


def draw_posture_guides(frame: np.ndarray, points: dict[str, tuple[float, float]]) -> None:
    """Overlay the side-view alignment chain used by the posture screen."""
    required = ("ear", "shoulder", "hip", "ankle")
    if not all(name in points for name in required):
        return
    ear, shoulder, hip, ankle = (tuple(map(int, points[name])) for name in required)
    # The yellow chain is the measurement path; the grey line through the hip
    # is a visual vertical reference, not an anatomical spine estimate.
    cv2.line(frame, ear, shoulder, (0, 220, 255), 3, cv2.LINE_AA)
    cv2.line(frame, shoulder, hip, (0, 220, 255), 3, cv2.LINE_AA)
    cv2.line(frame, hip, ankle, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (hip[0], 0), (hip[0], frame.shape[0] - 1), (105, 105, 105), 1, cv2.LINE_AA)
    for point, color in ((ear, (0, 80, 255)), (shoulder, (0, 255, 0)),
                         (hip, (255, 170, 0)), (ankle, (255, 255, 255))):
        cv2.circle(frame, point, 6, color, -1, cv2.LINE_AA)


def draw_shoulder_measurement_guides(frame: np.ndarray,
                                     points: dict[str, tuple[float, float]]) -> None:
    """Overlay front-view torso-midline to left/right shoulder measurements."""
    required = ("left_shoulder", "torso_midline_proxy", "right_shoulder")
    if not all(name in points for name in required):
        return
    left, midline, right = (tuple(map(int, points[name])) for name in required)
    cv2.line(frame, left, midline, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(frame, midline, right, (0, 200, 255), 3, cv2.LINE_AA)
    for point, color in ((left, (0, 255, 0)), (midline, (255, 255, 255)), (right, (0, 200, 255))):
        cv2.circle(frame, point, 7, color, -1, cv2.LINE_AA)


_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def draw_thai_text(frame: np.ndarray, text: str, pos: tuple[int, int], size: int = 24,
                   color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Draw Thai text using Tahoma when available, otherwise Pillow's default."""
    font = _font_cache.get(size)
    if font is None:
        try:
            font = ImageFont.truetype(str(Path("C:/Windows/Fonts/tahoma.ttf")), size)
        except OSError:
            font = ImageFont.load_default()
        _font_cache[size] = font
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    np.copyto(frame, cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
