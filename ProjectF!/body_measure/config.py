"""Central configuration for the body-measurement application."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSE_MODEL = PROJECT_ROOT / "pose_landmarker_full.task"
HAND_MODEL = PROJECT_ROOT / "hand_landmarker.task"

DEFAULT_HEIGHT_CM = 170.0
DEFAULT_MARKER_CM = 0.0  # 0 means use the height-based estimate.
ARUCO_MARKER_ID = 0
ARUCO_DICT = "DICT_4X4_50"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_TITLE = "Precision Body Measurement"
WINDOW_SIZE = (960, 540)

VISIBILITY_THRESHOLD = 0.6
# Complete immediately on the first valid pose.  This prioritizes instant
# feedback; the displayed value can vary more than a multi-frame median.
REQUIRED_STABLE_SAMPLES = 1
GESTURE_HOLD_SECONDS = 1.5
