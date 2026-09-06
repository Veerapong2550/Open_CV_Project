"""Geometry tests for the distance-normalised posture screening metrics."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from body_measure.posture import PostureSampleBuffer, analyse_posture_frame


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


if __name__ == "__main__":
    unittest.main()
