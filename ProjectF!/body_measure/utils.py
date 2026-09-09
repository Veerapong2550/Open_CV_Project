"""Shared measurement, smoothing, drawing, and gesture helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class OneEuroFilter:
    """Small, dependency-free One Euro filter for a scalar signal.

    Formula from spec:
        alpha = dt / (dt + tau), tau = 1 / (2 * pi * fc), fc = min_cutoff + beta * |dx|
    """

    def __init__(self, timestamp: float, value: float, min_cutoff: float = 0.3,
                 beta: float = 0.01, derivative_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.value = float(value)
        self.derivative = 0.0
        self.timestamp = float(timestamp)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return dt / (dt + tau)

    def update(self, timestamp: float, value: float) -> float:
        dt = timestamp - self.timestamp
        if dt <= 0:
            return self.value
        raw_derivative = (value - self.value) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        self.derivative = derivative_alpha * raw_derivative + (1 - derivative_alpha) * self.derivative
        value_alpha = self._alpha(self.min_cutoff + self.beta * abs(self.derivative), dt)
        self.value = value_alpha * value + (1 - value_alpha) * self.value
        self.timestamp = timestamp
        return self.value


class PointSmoother:
    """Filters 2D coordinate streams using OneEuroFilter for X and Y."""

    def __init__(self, min_cutoff: float = 0.3, beta: float = 0.01):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[str, tuple[OneEuroFilter, OneEuroFilter]] = {}

    def clear(self) -> None:
        self._filters.clear()

    def update(self, name: str, timestamp: float, x: float, y: float) -> tuple[float, float]:
        if name not in self._filters:
            self._filters[name] = (
                OneEuroFilter(timestamp, x, min_cutoff=self.min_cutoff, beta=self.beta),
                OneEuroFilter(timestamp, y, min_cutoff=self.min_cutoff, beta=self.beta),
            )
            return x, y
        x_filter, y_filter = self._filters[name]
        return x_filter.update(timestamp, x), y_filter.update(timestamp, y)


HAND_CONNECTIONS = ((0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
                    (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
                    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
                    (17, 18), (18, 19), (19, 20))

# Facial landmarks not needed for body posture & measurement (0..10: nose, eyes, ears, mouth)
# Complete exclusion leaves the user's face 100% clean with zero points and zero lines.
UNUSED_FACE_LANDMARKS = frozenset(range(11))

# MediaPipe Pose landmark topology excluding all facial mesh lines and finger/toe clutter.
# Focused strictly on the spine, torso, and main limb segments for posture analysis.
POSE_CONNECTIONS = (
    # Shoulders and Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Arms
    (11, 13), (13, 15), (12, 14), (14, 16),
    # Legs
    (23, 25), (24, 26), (25, 27), (26, 28),
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
    """Overlay the detected pose, excluding unnecessary facial landmarks (eyes, mouth)."""
    height, width = frame.shape[:2]
    points = list(landmarks)
    visible = [getattr(point, "visibility", 1.0) >= min_visibility for point in points]
    pixels = [(int(point.x * width), int(point.y * height)) for point in points]
    for start, end in POSE_CONNECTIONS:
        if (start not in UNUSED_FACE_LANDMARKS and end not in UNUSED_FACE_LANDMARKS
                and start < len(visible) and end < len(visible)
                and visible[start] and visible[end]):
            cv2.line(frame, pixels[start], pixels[end], (255, 180, 0), 2, cv2.LINE_AA)
    for idx, (point, is_visible) in enumerate(zip(pixels, visible)):
        if is_visible and idx not in UNUSED_FACE_LANDMARKS:
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
_dummy_img = Image.new("RGB", (1, 1))
_dummy_draw = ImageDraw.Draw(_dummy_img)


def draw_thai_text(frame: np.ndarray, text: str, pos: tuple[int, int], size: int = 24,
                   color: tuple[int, int, int] = (255, 255, 255),
                   background: tuple[int, int, int] | None = (18, 24, 30)) -> None:
    """Draw readable Thai camera-overlay text using ROI-only processing for high FPS."""
    if not text:
        return
    font = _font_cache.get(size)
    if font is None:
        try:
            font = ImageFont.truetype(str(Path("C:/Windows/Fonts/tahoma.ttf")), size)
        except OSError:
            font = ImageFont.load_default()
        _font_cache[size] = font

    box = _dummy_draw.textbbox(pos, text, font=font)
    padding = max(5, size // 5) if background is not None else 0

    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, box[0] - padding)
    y1 = max(0, box[1] - padding)
    x2 = min(frame_w, box[2] + padding)
    y2 = min(frame_h, box[3] + padding)

    if x2 <= x1 or y2 <= y1:
        return

    # Process only the small sub-region containing the text
    roi = frame[y1:y2, x1:x2]
    image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    rel_pos = (pos[0] - x1, pos[1] - y1)
    if background is not None:
        rel_box = (box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1)
        draw.rounded_rectangle(
            (rel_box[0] - padding, rel_box[1] - padding, rel_box[2] + padding, rel_box[3] + padding),
            radius=padding,
            fill=(background[2], background[1], background[0]),
        )
    draw.text(rel_pos, text, font=font, fill=(color[2], color[1], color[0]))
    frame[y1:y2, x1:x2] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

