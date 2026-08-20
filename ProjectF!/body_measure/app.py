"""
Entry point that wires together vision, calibration, filtering, state machine and UI.
"""

import time
import os
import sys
import numpy as np
import cv2

# Ensure the parent directory (ProjectF!) is in sys.path for importing test.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from test import get_reference_scale  # reuse existing calibration helper from test.py

from .vision import VisionEngine
from .calibration import get_pixel_scale, detect_marker
from .state_machine import GestureStateMachine, State
from .ui import MeasurementUI
from .utils import create_filter, smooth_point, pixel_to_cm

def run():
    """Run the real‑time precision body measurement application.
    This function creates the UI, processes video frames, performs calibration,
    runs MediaPipe detection, applies smoothing, and manages the measurement state.
    """
    ui = MeasurementUI()
    engine = VisionEngine()
    sm = GestureStateMachine()
    filters = {}
    pixel_scale = None  # px/cm ratio after calibration

    try:
        while True:
            ret, frame = ui.cap.read()
            if not ret:
                continue
            timestamp = int(time.time() * 1000)

            # Calibration: if we don't have a scale yet, try to detect ArUco marker
            if pixel_scale is None:
                corners, ids = detect_marker(frame)
                if ids is not None and len(ids) > 0:
                    scale = get_pixel_scale(frame)
                    if scale is not None:
                        pixel_scale = scale
                        print(f"Calibration complete: {pixel_scale:.2f} px/cm")

            # Vision processing
            pose_res, hand_res = engine.process(frame, timestamp)

            # Detect peace sign gesture (simple heuristic)
            peace = False
            if hand_res and hand_res.hand_landmarks:
                landmarks = hand_res.hand_landmarks[0]
                # Indices for hand landmarks (MediaPipe)
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_tip = landmarks[20]
                if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and \
                   ring_tip.y > thumb_tip.y and pinky_tip.y > thumb_tip.y:
                    peace = True

            # Update state machine
            state = sm.update(peace)
            if state == State.EXIT:
                break

            # Apply One Euro filter to pose landmarks (if present)
            if pose_res and pose_res.pose_landmarks:
                for i, lm in enumerate(pose_res.pose_landmarks[0]):
                    key = f"lm_{i}"
                    if key not in filters:
                        # Assuming ~30 FPS video
                        filters[key] = create_filter(rate=30)
                    pt = np.array([lm.x * frame.shape[1], lm.y * frame.shape[0]])
                    filtered_pt = smooth_point(filters[key], pt)
                    # Here you could compute body ratios using filtered_pt and pixel_scale

            # UI overlay – currently only draws static Thai text; you can extend to draw landmarks
            # The UI class already draws overlay each frame, so we just let it display the raw frame.

            # Slight pause to keep loop around 30 FPS
            time.sleep(0.01)
    finally:
        ui.stop()
        print("Application terminated.")
