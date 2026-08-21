"""Application entry point integrating camera, vision, calibration, and UI."""

from __future__ import annotations

from datetime import datetime
import math
import time

import cv2

from .calibration import get_pixel_scale
from .config import (CAMERA_INDEX, DEFAULT_HEIGHT_CM, DEFAULT_MARKER_CM, FRAME_HEIGHT,
                     FRAME_WIDTH, GESTURE_HOLD_SECONDS, REQUIRED_STABLE_SAMPLES,
                     VISIBILITY_THRESHOLD, WINDOW_SIZE, WINDOW_TITLE)
from .state_machine import GestureStateMachine, State
from .ui import MeasurementUI
from .utils import (PointSmoother, StableMeasurement, draw_hand_landmarks, draw_thai_text,
                    is_peace_sign, shoulder_center)
from .vision import VisionEngine


def _quality_and_measurement(pose_result, frame, timestamp, user_height_cm, marker_size_cm,
                             marker_scale, smoother, samples):
    """Analyse one frame and return a status and any new stable measurement."""
    height, width = frame.shape[:2]
    if not pose_result.pose_landmarks:
        samples.clear()
        return "Keep your full body visible", None

    landmarks = pose_result.pose_landmarks[0]
    left_shoulder, right_shoulder, nose = landmarks[11], landmarks[12], landmarks[0]
    left_ankle, right_ankle = landmarks[27], landmarks[28]
    required = (left_shoulder, right_shoulder, left_ankle, right_ankle)
    if not all(getattr(point, "visibility", 1.0) > VISIBILITY_THRESHOLD for point in required):
        samples.clear()
        return "Keep your full body visible", None

    left = smoother.update("left_shoulder", timestamp, left_shoulder.x * width, left_shoulder.y * height)
    right = smoother.update("right_shoulder", timestamp, right_shoulder.x * width, right_shoulder.y * height)
    nose_xy = smoother.update("nose", timestamp, nose.x * width, nose.y * height)
    base = smoother.update("ankle_midpoint", timestamp,
                           (left_ankle.x + right_ankle.x) * width / 2,
                           (left_ankle.y + right_ankle.y) * height / 2)
    neck = shoulder_center(left, right, nose_xy, base)
    shoulder_angle = abs(math.degrees(math.atan2(right[1] - left[1], right[0] - left[0])))
    if abs(neck[0] - width / 2) > width * 0.10:
        samples.clear()
        return "Move to the centre line", None
    if shoulder_angle > 5.0:
        samples.clear()
        return "Keep shoulders level", None

    scale = marker_scale
    if scale is None and marker_size_cm > 0:
        samples.clear()
        return "Show ArUco ID 0 beside your shoulders", None
    if scale is None:
        # This fallback estimates scale from a known user height, so it remains
        # sensitive to perspective and should not be presented as calibration.
        scale = math.dist(nose_xy, base) * 1.06 / user_height_cm if user_height_cm > 0 else None
    if scale is None or scale <= 0:
        samples.clear()
        return "Unable to determine scale", None

    shoulder_cm = math.dist(left, right) / scale
    left_cm = math.dist(left, neck) / scale
    right_cm = math.dist(right, neck) / scale
    samples.add(shoulder_cm, left_cm, right_cm)
    for point, color in ((left, (0, 255, 0)), (right, (0, 255, 0)), (neck, (0, 255, 255))):
        cv2.circle(frame, (int(point[0]), int(point[1])), 7, color, -1, cv2.LINE_AA)
    cv2.line(frame, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0), 3, cv2.LINE_AA)
    result = samples.result()
    if result is None:
        return f"Hold still — collecting samples ({len(samples.shoulders)}/{samples.required_samples})", None
    return "Measurement complete", {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "input_height_cm": user_height_cm,
        "shoulder_cm": result[0],
        "left_shoulder_cm": result[1],
        "right_shoulder_cm": result[2],
        "samples_used": len(samples.shoulders),
        "calibration": "ArUco marker" if marker_size_cm > 0 else "Height-based estimate",
    }


def run(user_height_cm: float = DEFAULT_HEIGHT_CM, marker_size_cm: float = DEFAULT_MARKER_CM,
        video_source: int = CAMERA_INDEX):
    """Run the integrated application and return its last stable measurement.

    A peace sign held for 1.5 seconds starts the measurement; repeat it to exit.
    Passing a positive ``marker_size_cm`` enables true ArUco calibration (ID 0).
    """
    if user_height_cm <= 0 or marker_size_cm < 0:
        raise ValueError("user_height_cm must be positive and marker_size_cm cannot be negative")
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera source {video_source}")

    ui = None
    vision = None
    measurement = None
    try:
        ui = MeasurementUI(WINDOW_TITLE, WINDOW_SIZE)
        vision = VisionEngine()
        state_machine = GestureStateMachine(GESTURE_HOLD_SECONDS)
        smoother = PointSmoother()
        samples = StableMeasurement(REQUIRED_STABLE_SAMPLES)
        while ui.is_open:
            ok, camera_frame = cap.read()
            if not ok:
                break
            # Detect before mirroring: an ArUco pattern can no longer decode as
            # the same ID after a horizontal flip.
            marker_scale, marker_corners = get_pixel_scale(camera_frame, marker_size_cm)
            frame = cv2.flip(camera_frame, 1)
            if marker_corners is not None:
                marker_corners = marker_corners.copy()
                marker_corners[:, 0] = frame.shape[1] - 1 - marker_corners[:, 0]
                cv2.polylines(frame, [marker_corners], True, (0, 255, 255), 2, cv2.LINE_AA)
            timestamp = time.monotonic()
            pose_result, hand_result = vision.process(frame, int(timestamp * 1000))
            peace = bool(hand_result.hand_landmarks and is_peace_sign(hand_result.hand_landmarks[0]))
            if hand_result.hand_landmarks:
                draw_hand_landmarks(frame, hand_result.hand_landmarks[0])

            state, progress, transitioned = state_machine.update(peace, timestamp)
            if transitioned and state is State.MEASURING:
                smoother.clear()
                samples.clear()
                measurement = None
            if state is State.EXIT:
                break
            if state is State.WAITING:
                status = "Hold a peace sign to begin"
                if peace:
                    status += f" ({progress * GESTURE_HOLD_SECONDS:.1f}/{GESTURE_HOLD_SECONDS:.1f}s)"
            else:
                status, new_measurement = _quality_and_measurement(
                    pose_result, frame, timestamp, user_height_cm, marker_size_cm, marker_scale,
                    smoother, samples)
                if new_measurement:
                    measurement = new_measurement
                if measurement:
                    draw_thai_text(frame, f"Shoulder width: {measurement['shoulder_cm']:.1f} cm", (30, 70), 28, (0, 255, 0))
            draw_thai_text(frame, status, (30, 30), 24, (255, 255, 255))
            ui.show_frame(frame)
    finally:
        cap.release()
        if vision is not None:
            vision.close()
        if ui is not None:
            ui.destroy()
    return measurement


if __name__ == "__main__":
    print(run())
