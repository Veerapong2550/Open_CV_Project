"""MediaPipe pose and hand detection, including a distant-person refinement pass."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import (FAR_POSE_ACTIVE_ROI_TTL_MS, FAR_POSE_REFINEMENT_INTERVAL_MS,
                     FAR_POSE_SEARCH_INTERVAL_MS, HAND_MODEL, POSE_MODEL)


@dataclass(frozen=True)
class PoseLandmark:
    """A small, MediaPipe-version-independent copy of a normalized landmark."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    presence: float = 0.0


@dataclass
class PoseFrameResult:
    """The subset of MediaPipe's result API consumed by this application."""

    pose_landmarks: list[list[PoseLandmark]]


class VisionEngine:
    """Run a normal video tracker plus crop-based detection for small people.

    The video tracker supplies smooth, inexpensive full-frame poses.  When a
    person is small, a second image-mode detector receives an enlarged crop
    around the body.  That crop gives the model a larger subject in its input
    (unlike merely upscaling the whole camera image) and lets it recover after
    a full-frame detector briefly loses a distant person.
    """

    def __init__(self, pose_model: Path = POSE_MODEL, hand_model: Path = HAND_MODEL):
        for model in (pose_model, hand_model):
            if not Path(model).is_file():
                raise FileNotFoundError(f"MediaPipe model not found: {model}")
        base_options = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        # Permissive detector thresholds help a small/distant body.  The app
        # later requires landmark quality, enough pixels, and 24 stable frames
        # before presenting a posture result.
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.28,
            min_pose_presence_confidence=0.30,
            min_tracking_confidence=0.30,
        ))
        self.roi_pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.24,
            min_pose_presence_confidence=0.26,
            min_tracking_confidence=0.26,
        ))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.65,
        ))
        self._active_roi: tuple[int, int, int, int] | None = None
        self._active_roi_until_ms = 0
        self._last_refine_ms = -FAR_POSE_REFINEMENT_INTERVAL_MS
        self._last_search_ms = -FAR_POSE_SEARCH_INTERVAL_MS
        self._last_refined_result: PoseFrameResult | None = None
        self._search_index = 0

    @staticmethod
    def _image(frame: np.ndarray) -> mp.Image:
        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

    @staticmethod
    def _empty_result() -> PoseFrameResult:
        return PoseFrameResult(pose_landmarks=[])

    @staticmethod
    def _map_result(raw_result, source_roi: tuple[int, int, int, int], frame_width: int,
                    frame_height: int) -> PoseFrameResult:
        """Map crop-normalized landmarks back into full-frame coordinates."""
        if not getattr(raw_result, "pose_landmarks", None):
            return VisionEngine._empty_result()
        left, top, right, bottom = source_roi
        roi_width = max(right - left, 1)
        roi_height = max(bottom - top, 1)
        mapped_poses: list[list[PoseLandmark]] = []
        for pose in raw_result.pose_landmarks:
            mapped_poses.append([
                PoseLandmark(
                    x=(left + float(point.x) * roi_width) / frame_width,
                    y=(top + float(point.y) * roi_height) / frame_height,
                    # z is not used by the 2D screen; retain a normalized
                    # value for callers that want it later.
                    z=float(getattr(point, "z", 0.0)) * roi_width / frame_width,
                    visibility=float(getattr(point, "visibility", 0.0) or 0.0),
                    presence=float(getattr(point, "presence", 0.0) or 0.0),
                )
                for point in pose
            ])
        return PoseFrameResult(pose_landmarks=mapped_poses)

    @staticmethod
    def _quality(result: PoseFrameResult) -> float:
        """Score a candidate by landmarks needed for full-body posture work."""
        if not result.pose_landmarks:
            return 0.0
        pose = result.pose_landmarks[0]
        required = (0, 7, 8, 11, 12, 23, 24, 27, 28)
        if len(pose) <= max(required):
            return 0.0
        visibility = [pose[index].visibility for index in required]
        return float(np.mean(visibility) + 0.20 * sum(value >= 0.35 for value in visibility) / len(visibility))

    @staticmethod
    def _person_roi(result: PoseFrameResult, frame_width: int,
                    frame_height: int) -> tuple[int, int, int, int] | None:
        if not result.pose_landmarks:
            return None
        points = [
            landmark for landmark in result.pose_landmarks[0]
            if landmark.visibility >= 0.22 and -0.2 <= landmark.x <= 1.2 and -0.2 <= landmark.y <= 1.2
        ]
        if len(points) < 7:
            return None
        xs = [point.x * frame_width for point in points]
        ys = [point.y * frame_height for point in points]
        raw_width = max(xs) - min(xs)
        raw_height = max(ys) - min(ys)
        if raw_width < 8 or raw_height < 20:
            return None
        # Retain enough border for a moving wrist/ankle yet magnify a distant
        # person much more than a full-frame pass can.
        horizontal_margin = max(raw_width * 0.45, raw_height * 0.20, 24.0)
        vertical_margin = max(raw_height * 0.18, 20.0)
        left = max(0, math.floor(min(xs) - horizontal_margin))
        top = max(0, math.floor(min(ys) - vertical_margin))
        right = min(frame_width, math.ceil(max(xs) + horizontal_margin))
        bottom = min(frame_height, math.ceil(max(ys) + vertical_margin))
        return (left, top, right, bottom) if right - left >= 48 and bottom - top >= 64 else None

    @staticmethod
    def _enlarged_roi_image(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray | None:
        left, top, right, bottom = roi
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        # Avoid a needless huge allocation for close subjects, but preserve
        # all source pixels before MediaPipe normalizes a small crop.
        largest_side = max(crop.shape[:2])
        if largest_side < 720:
            scale = min(3.0, 720.0 / largest_side)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def _needs_refinement(result: PoseFrameResult, frame_height: int) -> bool:
        if not result.pose_landmarks:
            return True
        pose = result.pose_landmarks[0]
        visible_y = [point.y * frame_height for point in pose if point.visibility >= 0.30]
        if len(visible_y) < 6:
            return True
        return max(visible_y) - min(visible_y) < frame_height * 0.62

    @staticmethod
    def _search_rois(frame_width: int, frame_height: int) -> tuple[tuple[int, int, int, int], ...]:
        """Prioritized overlapping focus crops for a body too small in-frame.

        A narrow, mid-height crop can enlarge a distant full body about twice
        as much as a full-frame pass.  The centre is searched first because
        the on-screen guidance asks the user to stand there; adjacent crops
        recover a person who is slightly left/right or high/low in the view.
        """
        crop_width = min(frame_width, max(64, int(frame_width * 0.42)))
        crop_height = min(frame_height, max(96, int(frame_height * 0.48)))
        center_left = (frame_width - crop_width) // 2
        center_top = (frame_height - crop_height) // 2

        def crop(left: int, top: int) -> tuple[int, int, int, int]:
            bounded_left = max(0, min(frame_width - crop_width, left))
            bounded_top = max(0, min(frame_height - crop_height, top))
            return (bounded_left, bounded_top, bounded_left + crop_width, bounded_top + crop_height)

        horizontal_step = int(crop_width * 0.75)
        vertical_step = int(crop_height * 0.70)
        return (
            crop(center_left, center_top),
            crop(center_left - horizontal_step, center_top),
            crop(center_left + horizontal_step, center_top),
            crop(center_left, center_top - vertical_step),
            crop(center_left, center_top + vertical_step),
            crop(center_left - horizontal_step, center_top - vertical_step),
            crop(center_left + horizontal_step, center_top - vertical_step),
            crop(center_left - horizontal_step, center_top + vertical_step),
            crop(center_left + horizontal_step, center_top + vertical_step),
        )

    def _detect_roi(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> PoseFrameResult:
        crop = self._enlarged_roi_image(frame, roi)
        if crop is None:
            return self._empty_result()
        raw_result = self.roi_pose.detect(self._image(crop))
        height, width = frame.shape[:2]
        return self._map_result(raw_result, roi, width, height)

    def _remember_roi(self, result: PoseFrameResult, frame_width: int, frame_height: int,
                      timestamp_ms: int) -> None:
        roi = self._person_roi(result, frame_width, frame_height)
        if roi is not None:
            self._active_roi = roi
            self._active_roi_until_ms = timestamp_ms + FAR_POSE_ACTIVE_ROI_TTL_MS

    def process(self, frame: np.ndarray, timestamp_ms: int):
        """Return full-frame pose coordinates and the regular hand result."""
        height, width = frame.shape[:2]
        full_raw = self.pose.detect_for_video(self._image(frame), timestamp_ms)
        full_result = self._map_result(full_raw, (0, 0, width, height), width, height)
        hand_result = self.hand.detect_for_video(self._image(frame), timestamp_ms)

        if full_result.pose_landmarks:
            self._remember_roi(full_result, width, height, timestamp_ms)
            if self._needs_refinement(full_result, height) and self._active_roi is not None:
                if timestamp_ms - self._last_refine_ms >= FAR_POSE_REFINEMENT_INTERVAL_MS:
                    self._last_refine_ms = timestamp_ms
                    refined = self._detect_roi(frame, self._active_roi)
                    if self._quality(refined) >= self._quality(full_result):
                        self._last_refined_result = refined
                        self._remember_roi(refined, width, height, timestamp_ms)
                        return refined, hand_result
            return full_result, hand_result

        # A full-frame detector can briefly lose a small person.  First reuse
        # the last known person crop, then rotate across center/left/right
        # search crops at a limited rate to avoid excessive camera latency.
        if self._active_roi is not None and timestamp_ms <= self._active_roi_until_ms:
            if timestamp_ms - self._last_refine_ms >= FAR_POSE_REFINEMENT_INTERVAL_MS:
                self._last_refine_ms = timestamp_ms
                refined = self._detect_roi(frame, self._active_roi)
                if refined.pose_landmarks:
                    self._last_refined_result = refined
                    self._remember_roi(refined, width, height, timestamp_ms)
                    return refined, hand_result
            if self._last_refined_result is not None:
                return self._last_refined_result, hand_result

        if timestamp_ms - self._last_search_ms >= FAR_POSE_SEARCH_INTERVAL_MS:
            self._last_search_ms = timestamp_ms
            rois = self._search_rois(width, height)
            roi = rois[self._search_index % len(rois)]
            self._search_index += 1
            recovered = self._detect_roi(frame, roi)
            if recovered.pose_landmarks:
                self._last_refined_result = recovered
                self._remember_roi(recovered, width, height, timestamp_ms)
                return recovered, hand_result
        return self._empty_result(), hand_result

    def close(self) -> None:
        self.pose.close()
        self.roi_pose.close()
        self.hand.close()
