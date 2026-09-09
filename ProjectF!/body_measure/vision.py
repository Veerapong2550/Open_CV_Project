"""MediaPipe pose and hand detection, including a distant-person refinement pass."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import (FAR_HAND_DETECTION_CONFIDENCE, FAR_HAND_PRESENCE_CONFIDENCE,
                     FAR_HAND_REFINEMENT_INTERVAL_MS, FAR_POSE_ACTIVE_ROI_TTL_MS,
                     FAR_POSE_REFINEMENT_INTERVAL_MS, FAR_POSE_SEARCH_INTERVAL_MS,
                     HAND_DETECTION_CONFIDENCE, HAND_MODEL, HAND_PRESENCE_CONFIDENCE,
                     HAND_TRACKING_CONFIDENCE, POSE_MODEL)


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


@dataclass(frozen=True)
class HandLandmark:
    """A normalized hand landmark mapped into the original camera frame."""

    x: float
    y: float
    z: float = 0.0


@dataclass
class HandFrameResult:
    """The hand-result shape used by the gesture and drawing helpers."""

    hand_landmarks: list[list[HandLandmark]]


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
        # Full-frame tracking remains fast for nearby hands.  It looks for two
        # hands so a non-gesture hand cannot hide the hand making the command.
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
        ))
        # A separate image-mode detector works on an enlarged crop around a
        # pose wrist.  It is intentionally independent from VIDEO tracking:
        # when a distant hand first appears it may be only a few pixels in the
        # full camera image and therefore has no tracker history to follow.
        self.roi_hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=FAR_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=FAR_HAND_PRESENCE_CONFIDENCE,
        ))
        self._active_roi: tuple[int, int, int, int] | None = None
        self._active_roi_until_ms = 0
        self._last_refine_ms = -FAR_POSE_REFINEMENT_INTERVAL_MS
        self._last_search_ms = -FAR_POSE_SEARCH_INTERVAL_MS
        self._last_refined_result: PoseFrameResult | None = None
        self._search_index = 0
        self._last_hand_refine_ms = -FAR_HAND_REFINEMENT_INTERVAL_MS
        # MediaPipe VIDEO mode rejects equal or decreasing timestamps.  A fast
        # camera loop can produce two frames in the same millisecond.
        self._last_timestamp_ms = -1

    def _next_timestamp(self, timestamp_ms: int) -> int:
        """Return a non-negative timestamp strictly newer than the last one."""
        timestamp_ms = max(int(timestamp_ms), 0)
        self._last_timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    @staticmethod
    def _image(frame: np.ndarray) -> mp.Image:
        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

    @staticmethod
    def _mp_image(rgb_frame: np.ndarray) -> mp.Image:
        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

    @staticmethod
    def _empty_result() -> PoseFrameResult:
        return PoseFrameResult(pose_landmarks=[])

    @staticmethod
    def _empty_hand_result() -> HandFrameResult:
        return HandFrameResult(hand_landmarks=[])

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
    def _map_hand_result(raw_result, source_roi: tuple[int, int, int, int], frame_width: int,
                         frame_height: int) -> HandFrameResult:
        """Map crop-normalized hand points back into full-frame coordinates."""
        if not getattr(raw_result, "hand_landmarks", None):
            return VisionEngine._empty_hand_result()
        left, top, right, bottom = source_roi
        roi_width = max(right - left, 1)
        roi_height = max(bottom - top, 1)
        mapped_hands: list[list[HandLandmark]] = []
        for hand in raw_result.hand_landmarks:
            mapped_hands.append([
                HandLandmark(
                    x=(left + float(point.x) * roi_width) / frame_width,
                    y=(top + float(point.y) * roi_height) / frame_height,
                    z=float(getattr(point, "z", 0.0)) * roi_width / frame_width,
                )
                for point in hand
            ])
        return HandFrameResult(hand_landmarks=mapped_hands)

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

    @staticmethod
    def _square_roi(center_x: float, center_y: float, half_size: float, frame_width: int,
                    frame_height: int) -> tuple[int, int, int, int] | None:
        """Return a bounded square focus region, retaining a useful minimum size."""
        side = int(round(max(96.0, half_size * 2.0)))
        side = min(side, frame_width, frame_height)
        if side < 48:
            return None
        left = int(round(center_x - side / 2))
        top = int(round(center_y - side / 2))
        left = max(0, min(frame_width - side, left))
        top = max(0, min(frame_height - side, top))
        return (left, top, left + side, top + side)

    @staticmethod
    def _hand_rois_from_pose(result: PoseFrameResult, frame_width: int,
                             frame_height: int) -> tuple[tuple[int, int, int, int], ...]:
        """Focus on the space just beyond each visible wrist.

        The pose wrist and elbow are much easier to find at distance than the
        21 detailed hand points.  Extending the crop a little past the wrist
        covers a raised V sign while its square shape magnifies the hand before
        the second hand model sees it.
        """
        if not result.pose_landmarks:
            return ()
        pose = result.pose_landmarks[0]
        rois: list[tuple[int, int, int, int]] = []
        visible_y = [point.y * frame_height for point in pose if point.visibility >= 0.25]
        body_height = max(visible_y) - min(visible_y) if len(visible_y) >= 2 else 0.0
        for wrist_index, elbow_index in ((15, 13), (16, 14)):
            if len(pose) <= wrist_index:
                continue
            wrist = pose[wrist_index]
            if wrist.visibility < 0.15:
                continue
            wrist_x = wrist.x * frame_width
            wrist_y = wrist.y * frame_height
            elbow = pose[elbow_index] if len(pose) > elbow_index else None
            if elbow is not None and elbow.visibility >= 0.12:
                elbow_x = elbow.x * frame_width
                elbow_y = elbow.y * frame_height
                forearm = math.hypot(wrist_x - elbow_x, wrist_y - elbow_y)
                # Move toward the fingertips rather than centring only on the
                # wrist, whose point sits at the lower edge of a raised hand.
                center_x = wrist_x + (wrist_x - elbow_x) * 0.30
                center_y = wrist_y + (wrist_y - elbow_y) * 0.30
            else:
                forearm = 0.0
                center_x, center_y = wrist_x, wrist_y
            half_size = max(48.0, forearm * 1.50, body_height * 0.12)
            roi = VisionEngine._square_roi(center_x, center_y, half_size, frame_width, frame_height)
            if roi is not None and roi not in rois:
                rois.append(roi)
        return tuple(rois)

    def _detect_hand_roi(self, frame: np.ndarray,
                         roi: tuple[int, int, int, int]) -> HandFrameResult:
        crop = self._enlarged_roi_image(frame, roi)
        if crop is None:
            return self._empty_hand_result()
        raw_result = self.roi_hand.detect(self._image(crop))
        height, width = frame.shape[:2]
        return self._map_hand_result(raw_result, roi, width, height)

    @staticmethod
    def _merge_hands(*results: HandFrameResult) -> HandFrameResult:
        """Combine full and crop detections while dropping the same hand twice."""
        merged: list[list[HandLandmark]] = []
        for result in results:
            for hand in result.hand_landmarks:
                if not hand:
                    continue
                wrist = hand[0]
                is_duplicate = any(
                    existing and math.hypot(existing[0].x - wrist.x, existing[0].y - wrist.y) < 0.06
                    for existing in merged
                )
                if not is_duplicate:
                    merged.append(hand)
        return HandFrameResult(hand_landmarks=merged[:2])

    def _detect_hands(self, frame: np.ndarray, pose_result: PoseFrameResult,
                      timestamp_ms: int, rgb_frame: np.ndarray | None = None) -> HandFrameResult:
        """Detect nearby hands normally and distant hands in enlarged wrist crops."""
        height, width = frame.shape[:2]
        if rgb_frame is None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        full_raw = self.hand.detect_for_video(self._mp_image(rgb_frame), timestamp_ms)
        full_result = self._map_hand_result(full_raw, (0, 0, width, height), width, height)

        # Crop passes are rate-limited so the camera remains responsive.  The
        # regular hand tracker still runs every frame, so a vanished gesture is
        # never held over from a stale crop result.
        if timestamp_ms - self._last_hand_refine_ms < FAR_HAND_REFINEMENT_INTERVAL_MS:
            return full_result
        self._last_hand_refine_ms = timestamp_ms
        roi_results = [self._detect_hand_roi(frame, roi)
                       for roi in self._hand_rois_from_pose(pose_result, width, height)]
        return self._merge_hands(full_result, *roi_results)

    def _remember_roi(self, result: PoseFrameResult, frame_width: int, frame_height: int,
                      timestamp_ms: int) -> None:
        roi = self._person_roi(result, frame_width, frame_height)
        if roi is not None:
            self._active_roi = roi
            self._active_roi_until_ms = timestamp_ms + FAR_POSE_ACTIVE_ROI_TTL_MS

    def process(self, frame: np.ndarray, timestamp_ms: int) -> tuple[PoseFrameResult, HandFrameResult]:
        """Return pose coordinates plus near/far hand landmarks in camera space."""
        timestamp_ms = self._next_timestamp(timestamp_ms)
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        full_raw = self.pose.detect_for_video(self._mp_image(rgb_frame), timestamp_ms)
        full_result = self._map_result(full_raw, (0, 0, width, height), width, height)
        selected_result = self._empty_result()

        if full_result.pose_landmarks:
            self._remember_roi(full_result, width, height, timestamp_ms)
            selected_result = full_result
            if self._needs_refinement(full_result, height) and self._active_roi is not None:
                if timestamp_ms - self._last_refine_ms >= FAR_POSE_REFINEMENT_INTERVAL_MS:
                    self._last_refine_ms = timestamp_ms
                    refined = self._detect_roi(frame, self._active_roi)
                    if self._quality(refined) >= self._quality(full_result):
                        self._last_refined_result = refined
                        self._remember_roi(refined, width, height, timestamp_ms)
                        selected_result = refined
        else:
            # A full-frame detector can briefly lose a small person.  First
            # reuse the last known person crop, then rotate across search crops
            # at a limited rate to avoid excessive camera latency.
            if self._active_roi is not None and timestamp_ms <= self._active_roi_until_ms:
                if timestamp_ms - self._last_refine_ms >= FAR_POSE_REFINEMENT_INTERVAL_MS:
                    self._last_refine_ms = timestamp_ms
                    refined = self._detect_roi(frame, self._active_roi)
                    if refined.pose_landmarks:
                        self._last_refined_result = refined
                        self._remember_roi(refined, width, height, timestamp_ms)
                        selected_result = refined
                if not selected_result.pose_landmarks and self._last_refined_result is not None:
                    selected_result = self._last_refined_result

            if not selected_result.pose_landmarks and timestamp_ms - self._last_search_ms >= FAR_POSE_SEARCH_INTERVAL_MS:
                self._last_search_ms = timestamp_ms
                rois = self._search_rois(width, height)
                roi = rois[self._search_index % len(rois)]
                self._search_index += 1
                recovered = self._detect_roi(frame, roi)
                if recovered.pose_landmarks:
                    self._last_refined_result = recovered
                    self._remember_roi(recovered, width, height, timestamp_ms)
                    selected_result = recovered

        return selected_result, self._detect_hands(frame, selected_result, timestamp_ms, rgb_frame=rgb_frame)

    def close(self) -> None:
        self.pose.close()
        self.roi_pose.close()
        self.hand.close()
        self.roi_hand.close()
