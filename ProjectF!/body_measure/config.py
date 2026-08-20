"""
Configuration constants used across the project.
Adjust values here to tune the system.
"""

# MediaPipe models (update paths if needed)
POSE_MODEL = "pose_landmarker_lite.task"
HAND_MODEL = "hand_landmarker_lite.task"

# Calibration
REFERENCE_MARKER_CM = 5.0  # real‑world size of the ArUco marker in cm
ARUCO_DICT = "DICT_4X4_50"

# One Euro Filter parameters (tune as needed)
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.0
ONE_EURO_DERIV_CUTOFF = 1.0

# UI settings
WINDOW_TITLE = "Precision Body Measurement"
WINDOW_SIZE = (960, 540)
