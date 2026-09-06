"""Central configuration for the body-measurement application."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The Heavy model is more accurate for small/distant landmarks when it is
# available locally.  Keeping the Full model as a fallback means the project
# still runs with the model that is already bundled in the repository.
POSE_HEAVY_MODEL = PROJECT_ROOT / "pose_landmarker_heavy.task"
POSE_FULL_MODEL = PROJECT_ROOT / "pose_landmarker_full.task"
POSE_MODEL = POSE_HEAVY_MODEL if POSE_HEAVY_MODEL.is_file() else POSE_FULL_MODEL
HAND_MODEL = PROJECT_ROOT / "hand_landmarker.task"
CAPTURE_DIR = PROJECT_ROOT / "captures"

DEFAULT_HEIGHT_CM = 170.0
DEFAULT_MARKER_CM = 0.0  # 0 means report distance-independent ratios/angles only.
ARUCO_MARKER_ID = 0
ARUCO_DICT = "DICT_4X4_50"

CAMERA_INDEX = 0
# Preserve substantially more landmark detail for people standing farther
# from the camera.  OpenCV will negotiate down to the closest mode supported
# by a camera, so this remains safe for 720p webcams.
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
WINDOW_TITLE = "Precision Body Measurement"
WINDOW_SIZE = (960, 540)

# Lower landmark acceptance is paired with multi-frame stability checks in
# posture.py.  It permits a distant but coherent pose without accepting a
# one-frame result as a measurement.
VISIBILITY_THRESHOLD = 0.40
POSTURE_VISIBILITY_THRESHOLD = 0.35
MIN_ANALYSIS_BODY_HEIGHT_PX = 150
MIN_ANALYSIS_BODY_HEIGHT_RATIO = 0.14
POSTURE_STABLE_SAMPLES = 24

# A profile is required to screen for a rounded upper back.  This is the
# maximum projected shoulder/hip width relative to torso length that is
# treated as a side view.
SIDE_VIEW_MAX_WIDTH_TO_TORSO = 0.55

# Distant-pose refinement/search settings.  The second, image-mode detector
# is used only for a small person or after the full-frame detector loses it.
FAR_POSE_REFINEMENT_INTERVAL_MS = 100
FAR_POSE_SEARCH_INTERVAL_MS = 350
FAR_POSE_ACTIVE_ROI_TTL_MS = 1500
GESTURE_HOLD_SECONDS = 1.5
