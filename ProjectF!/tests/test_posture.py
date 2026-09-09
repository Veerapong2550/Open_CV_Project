"""Geometry tests for the distance-normalised posture screening metrics."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from body_measure.posture import (PostureSampleBuffer, ShoulderSampleBuffer,
                                  analyse_front_shoulder_frame, analyse_posture_frame)


@dataclass
class Landmark:
    x: float = 0.5
    y: float = 0.5
    visibility: float = 0.99


def _pose(kind: str) -> list[Landmark]:
    landmarks = [Landmark() for _ in range(33)]

    if kind == "front":
        # Broad projected shoulder/hip widths identify a frontal image.
        values = {
            0: (0.50, 0.16), 7: (0.47, 0.19), 8: (0.53, 0.19),
            11: (0.35, 0.36), 12: (0.65, 0.36),
            23: (0.38, 0.62), 24: (0.62, 0.62),
            27: (0.40, 0.92), 28: (0.60, 0.92),
        }
    elif kind == "rounded":
        # Ear and shoulder move to the same side of the hip.  Values are
        # intentionally normalized so changing image resolution has no effect.
        values = {
            0: (0.63, 0.15), 7: (0.59, 0.18), 8: (0.60, 0.18),
            11: (0.56, 0.36), 12: (0.57, 0.36),
            23: (0.50, 0.62), 24: (0.51, 0.62),
            27: (0.50, 0.92), 28: (0.51, 0.92),
        }
    else:  # neutral profile
        values = {
            0: (0.54, 0.15), 7: (0.50, 0.18), 8: (0.51, 0.18),
            11: (0.50, 0.36), 12: (0.51, 0.36),
            23: (0.50, 0.62), 24: (0.51, 0.62),
            27: (0.50, 0.92), 28: (0.51, 0.92),
        }
    for index, (x, y) in values.items():
        landmarks[index] = Landmark(x, y)
    return landmarks


class PostureGeometryTests(unittest.TestCase):
    def test_neutral_profile_is_side_view_with_neutral_screen(self):
        metrics, reason = analyse_posture_frame(_pose("neutral"), 1920, 1080)
        self.assertEqual(reason, "พร้อมวิเคราะห์")
        self.assertEqual(metrics["view"], "side")
        self.assertEqual(metrics["screening_level"], "neutral")
        self.assertLess(metrics["head_shoulder_offset_ratio"], 0.01)

    def test_forward_chain_is_flagged_without_depends_on_resolution(self):
        low_resolution, _ = analyse_posture_frame(_pose("rounded"), 960, 540)
        high_resolution, _ = analyse_posture_frame(_pose("rounded"), 3840, 2160)
        self.assertEqual(low_resolution["view"], "side")
        self.assertEqual(low_resolution["screening_level"], "elevated")
        self.assertAlmostEqual(low_resolution["head_shoulder_offset_ratio"],
                               high_resolution["head_shoulder_offset_ratio"])
        self.assertAlmostEqual(low_resolution["shoulder_hip_offset_ratio"],
                               high_resolution["shoulder_hip_offset_ratio"])

    def test_front_view_is_not_mislabelled_as_rounded_back_analysis(self):
        metrics, reason = analyse_posture_frame(_pose("front"), 1920, 1080)
        self.assertEqual(metrics["view"], "front")
        self.assertIn("ด้านข้าง", reason)
        self.assertNotIn("screening_score", metrics)

    def test_front_view_measures_equal_projected_left_and_right_shoulders(self):
        metrics, reason = analyse_front_shoulder_frame(_pose("front"), 1920, 1080)
        self.assertEqual(reason, "พร้อมวัดความยาวไหล่")
        self.assertEqual(metrics["view"], "front")
        self.assertAlmostEqual(metrics["left_shoulder_length_ratio"], 0.5)
        self.assertAlmostEqual(metrics["right_shoulder_length_ratio"], 0.5)
        self.assertAlmostEqual(metrics["shoulder_length_difference_ratio"], 0.0)
        self.assertEqual(metrics["shoulder_balance_level"], "within_tolerance")

    def test_front_view_flags_large_image_projection_difference_for_recheck(self):
        landmarks = _pose("front")
        landmarks[11].x = 0.25
        metrics, reason = analyse_front_shoulder_frame(landmarks, 1920, 1080)
        self.assertEqual(reason, "พร้อมวัดความยาวไหล่")
        self.assertGreater(metrics["shoulder_length_difference_ratio"], 0.07)
        self.assertEqual(metrics["shoulder_balance_level"], "recheck")

    def test_front_shoulder_measurement_rejects_an_oblique_view(self):
        landmarks = _pose("front")
        for index, x in ((11, 0.45), (12, 0.55), (23, 0.47), (24, 0.53)):
            landmarks[index].x = x
        metrics, reason = analyse_front_shoulder_frame(landmarks, 1920, 1080)
        self.assertIsNone(metrics)
        self.assertIn("หน้าตรง", reason)

    def test_front_shoulder_measurement_rejects_midline_outside_shoulder_span(self):
        landmarks = _pose("front")
        for index in (7, 8, 23, 24):
            landmarks[index].x += 0.30
        metrics, reason = analyse_front_shoulder_frame(landmarks, 1920, 1080)
        self.assertIsNone(metrics)
        self.assertIn("แนวกึ่งกลาง", reason)

    def test_buffer_returns_median_and_stability_information(self):
        metrics, _ = analyse_posture_frame(_pose("rounded"), 1920, 1080)
        metrics["pixels_per_cm"] = 10.0
        buffer = PostureSampleBuffer(required_samples=3)
        for _ in range(3):
            buffer.add(metrics)
        result = buffer.result()
        self.assertTrue(result["stable"])
        self.assertEqual(result["samples_used"], 3)
        self.assertEqual(result["screening_level"], "elevated")
        self.assertEqual(result["pixels_per_cm"], 10.0)

    def test_shoulder_buffer_reports_stable_front_comparison(self):
        metrics, _ = analyse_front_shoulder_frame(_pose("front"), 1920, 1080)
        metrics["pixels_per_cm"] = 10.0
        buffer = ShoulderSampleBuffer(required_samples=3)
        for _ in range(3):
            buffer.add(metrics)
        result = buffer.result()
        self.assertTrue(result["stable"])
        self.assertEqual(result["samples_used"], 3)
        self.assertEqual(result["shoulder_balance_level"], "within_tolerance")
        self.assertEqual(result["pixels_per_cm"], 10.0)


    def test_front_shoulder_measurement_rejects_off_center(self):
        landmarks = _pose("front")
        # Shift entire body by +0.15 screen width so center is 0.65 (offset = 0.15 > 0.10)
        for index in range(len(landmarks)):
            landmarks[index].x += 0.15
        metrics, reason = analyse_front_shoulder_frame(landmarks, 1920, 1080)
        self.assertIsNone(metrics)
        self.assertIn("กึ่งกลางภาพ", reason)

    def test_front_shoulder_measurement_rejects_shoulder_tilt_above_limit(self):
        landmarks = _pose("front")
        # Tilt right shoulder down significantly (tilt ~ 11.3 degrees > 5.0 deg)
        landmarks[12].y = 0.42  # dy = 0.06, dx = 0.30 -> atan2(0.06, 0.30) ~ 11.3 deg
        metrics, reason = analyse_front_shoulder_frame(landmarks, 1920, 1080)
        self.assertIsNone(metrics)
        self.assertIn("ระดับไหล่เอียงเกินกำหนด", reason)

    def test_front_shoulder_measurement_falls_back_to_nose_when_ears_obscured(self):
        landmarks = _pose("front")
        # Set ear visibility below threshold (e.g. hair covering ears)
        landmarks[7].visibility = 0.10
        landmarks[8].visibility = 0.10
        landmarks[0].visibility = 0.95  # nose is visible
        metrics, reason = analyse_front_shoulder_frame(landmarks, 1920, 1080)
        self.assertIsNotNone(metrics)
        self.assertEqual(reason, "พร้อมวัดความยาวไหล่")
        self.assertAlmostEqual(metrics["left_shoulder_length_ratio"], 0.5, places=2)


class SideBodyOcclusionTests(unittest.TestCase):
    def test_has_side_body_accepts_single_sided_profile(self):
        from body_measure.app import _has_side_body
        from body_measure.vision import PoseFrameResult, PoseLandmark

        # Side profile where far shoulder (12), far hip (24), and far ankle (28) are occluded
        profile_landmarks = [PoseLandmark(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
        profile_landmarks[0] = PoseLandmark(x=0.5, y=0.15, visibility=0.85)   # nose
        profile_landmarks[11] = PoseLandmark(x=0.5, y=0.35, visibility=0.80)  # near shoulder
        profile_landmarks[12] = PoseLandmark(x=0.5, y=0.35, visibility=0.10)  # occluded far shoulder
        profile_landmarks[23] = PoseLandmark(x=0.5, y=0.60, visibility=0.75)  # near hip
        profile_landmarks[24] = PoseLandmark(x=0.5, y=0.60, visibility=0.15)  # occluded far hip
        profile_landmarks[27] = PoseLandmark(x=0.5, y=0.90, visibility=0.80)  # near ankle
        profile_landmarks[28] = PoseLandmark(x=0.5, y=0.90, visibility=0.10)  # occluded far ankle

        pose_result = PoseFrameResult(pose_landmarks=[profile_landmarks])
        self.assertTrue(_has_side_body(pose_result))

    def test_has_side_body_rejects_when_near_ankle_missing(self):
        from body_measure.app import _has_side_body
        from body_measure.vision import PoseFrameResult, PoseLandmark

        profile_landmarks = [PoseLandmark(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
        profile_landmarks[0] = PoseLandmark(x=0.5, y=0.15, visibility=0.85)
        profile_landmarks[11] = PoseLandmark(x=0.5, y=0.35, visibility=0.80)
        profile_landmarks[23] = PoseLandmark(x=0.5, y=0.60, visibility=0.75)
        # neither ankle is visible
        profile_landmarks[27] = PoseLandmark(x=0.5, y=0.90, visibility=0.10)
        profile_landmarks[28] = PoseLandmark(x=0.5, y=0.90, visibility=0.10)

        pose_result = PoseFrameResult(pose_landmarks=[profile_landmarks])
        self.assertFalse(_has_side_body(pose_result))

    def test_side_posture_succeeds_without_face_nose(self):
        # Verify side posture analysis does not require nose/face landmarks
        landmarks = _pose("rounded")
        landmarks[0].visibility = 0.0  # Nose completely invisible/covered
        metrics, reason = analyse_posture_frame(landmarks, 1920, 1080)
        self.assertIsNotNone(metrics)
        self.assertEqual(reason, "พร้อมวิเคราะห์")
        self.assertEqual(metrics["view"], "side")
        self.assertEqual(metrics["screening_level"], "elevated")

    def test_standalone_posture_measurement_without_shoulders(self):
        import numpy as np
        from body_measure.app import _quality_and_posture_measurement
        from body_measure.vision import PoseFrameResult, PoseLandmark

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        landmarks = _pose("rounded")
        pose_result = PoseFrameResult(pose_landmarks=[[
            PoseLandmark(lm.x, lm.y, visibility=lm.visibility) for lm in landmarks
        ]])
        buffer = PostureSampleBuffer(required_samples=3)

        # Collect 3 samples
        result = None
        for _ in range(3):
            status, result, live = _quality_and_posture_measurement(
                pose_result, frame, user_height_cm=175.0, marker_size_cm=0.0,
                marker_scale=None, samples=buffer, shoulder_summary=None
            )
        self.assertIsNotNone(result)
        self.assertEqual(result["samples_used"], 3)
        self.assertIsNone(result["shoulders"])
        self.assertEqual(result["posture"]["screening_level"], "elevated")
        self.assertIn("วิเคราะห์อาการหลังค่อมหลังงอเสร็จสิ้น", status)


if __name__ == "__main__":
    unittest.main()
