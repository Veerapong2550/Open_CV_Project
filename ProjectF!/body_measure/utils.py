"""Shared measurement, smoothing, drawing, and gesture helpers."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class OneEuroFilter:
    """Small, dependency-free One Euro filter for a scalar signal."""

    def __init__(self, timestamp: float, value: float, min_cutoff: float = 0.3,
                 beta: float = 0.01, derivative_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.value = value
        self.derivative = 0.0
        self.timestamp = timestamp

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
    def __init__(self):
        self._filters: dict[str, tuple[OneEuroFilter, OneEuroFilter]] = {}

    def clear(self) -> None:
        self._filters.clear()

    def update(self, name: str, timestamp: float, x: float, y: float) -> tuple[float, float]:
        if name not in self._filters:
            self._filters[name] = (OneEuroFilter(timestamp, x), OneEuroFilter(timestamp, y))
        x_filter, y_filter = self._filters[name]
        return x_filter.update(timestamp, x), y_filter.update(timestamp, y)


class StableMeasurement:
    """Records medians only after a run of consistently valid frames."""

    def __init__(self, required_samples: int):
        self.required_samples = required_samples
        self.shoulders: deque[float] = deque(maxlen=required_samples)
        self.lefts: deque[float] = deque(maxlen=required_samples)
        self.rights: deque[float] = deque(maxlen=required_samples)

    def clear(self) -> None:
        self.shoulders.clear()
        self.lefts.clear()
        self.rights.clear()

    def add(self, shoulder: float, left: float, right: float) -> None:
        self.shoulders.append(shoulder)
        self.lefts.append(left)
        self.rights.append(right)

    @property
    def progress(self) -> float:
        return len(self.shoulders) / self.required_samples

    @property
    def ready(self) -> bool:
        return len(self.shoulders) == self.required_samples

    def result(self) -> tuple[float, float, float] | None:
        if not self.ready:
            return None
        return tuple(float(np.median(values)) for values in (self.shoulders, self.lefts, self.rights))


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
    points = list(landmarks)
    return (points[8].y < points[6].y and points[12].y < points[10].y
            and points[16].y > points[14].y and points[20].y > points[18].y)


def draw_hand_landmarks(frame: np.ndarray, landmarks: Iterable) -> None:
    height, width = frame.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in landmarks]
    for start, end in HAND_CONNECTIONS:
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
        if visible[start] and visible[end]:
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


def shoulder_center(left: tuple[float, float], right: tuple[float, float],
                    nose: tuple[float, float], base: tuple[float, float]) -> tuple[float, float]:
    lx, ly = left
    rx, ry = right
    nx, ny = nose
    bx, by = base
    denominator = (rx - lx) * (by - ny) - (ry - ly) * (bx - nx)
    ratio = 0.5 if abs(denominator) < 1e-6 else ((nx - lx) * (by - ny) - (ny - ly) * (bx - nx)) / denominator
    ratio = max(0.0, min(1.0, ratio))
    return lx + ratio * (rx - lx), ly + ratio * (ry - ly)


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
