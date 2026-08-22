"""Synchronous MediaPipe pose and hand detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import HAND_MODEL, POSE_MODEL


class VisionEngine:
    def __init__(self, pose_model: Path = POSE_MODEL, hand_model: Path = HAND_MODEL):
        for model in (pose_model, hand_model):
            if not Path(model).is_file():
                raise FileNotFoundError(f"MediaPipe model not found: {model}")
        base_options = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.VIDEO,
            # A person occupying the full camera frame has smaller features
            # than a close-up.  These values still reject weak detections, but
            # avoid hiding a valid full-body pose before tracking can start.
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        ))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        ))

    def process(self, frame: np.ndarray, timestamp_ms: int):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.pose.detect_for_video(image, timestamp_ms), self.hand.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        self.pose.close()
        self.hand.close()
