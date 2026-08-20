"""
MediaPipe based visual perception module.
Provides pose and hand landmark detection.
"""

import mediapipe as mp
import cv2
import numpy as np
from .config import POSE_MODEL, HAND_MODEL

class VisionEngine:
    def __init__(self):
        # Pose Landmarker
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                result_callback=self._pose_callback,
            )
        )
        # Hand Landmarker
        self.hand = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                result_callback=self._hand_callback,
            )
        )
        self.latest_pose = None
        self.latest_hand = None

    def _pose_callback(self, result, timestamp_ms):
        self.latest_pose = result

    def _hand_callback(self, result, timestamp_ms):
        self.latest_hand = result

    def process(self, frame: np.ndarray, timestamp_ms: int):
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.pose.detect_async(image, timestamp_ms)
        self.hand.detect_async(image, timestamp_ms)
        return self.latest_pose, self.latest_hand
