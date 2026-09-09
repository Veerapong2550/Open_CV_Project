"""Unit tests for fallback calibration, One Euro filter, and Excel export."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from body_measure.calibration import estimate_pixel_scale_from_height, get_pixel_scale
from body_measure.utils import OneEuroFilter, PointSmoother
from body_measure.export import save_measurement_to_excel
from body_measure.ui import MeasurementUI


class CalibrationAndFilterTests(unittest.TestCase):
    def test_estimate_pixel_scale_from_height_calculates_correct_scale(self):
        nose = (500.0, 100.0)
        ankle = (500.0, 900.0)
        # distance is 800.0 px * 1.06 = 848.0 px
        # height = 170.0 cm -> scale = 848.0 / 170.0 = 4.988235... px/cm
        scale = estimate_pixel_scale_from_height(nose, ankle, 170.0)
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(scale, 848.0 / 170.0, places=4)

    def test_estimate_pixel_scale_rejects_non_positive_or_tiny_height(self):
        nose = (500.0, 100.0)
        ankle = (500.0, 105.0)  # distance is 5px (< 20px min)
        self.assertIsNone(estimate_pixel_scale_from_height(nose, ankle, 170.0))
        self.assertIsNone(estimate_pixel_scale_from_height(nose, (500.0, 900.0), 0.0))
        self.assertIsNone(estimate_pixel_scale_from_height(nose, (500.0, 900.0), -10.0))

    def test_one_euro_filter_reduces_noise(self):
        filt = OneEuroFilter(timestamp=0.0, value=100.0, min_cutoff=0.3, beta=0.01)
        # Add slight jitter around 100
        val1 = filt.update(0.033, 102.0)
        val2 = filt.update(0.066, 99.0)
        val3 = filt.update(0.099, 101.0)
        # Filtered values should be smoothed closer to 100 than the noisy input
        self.assertLess(abs(val1 - 100.0), 2.0)
        self.assertLess(abs(val2 - 100.0), 2.0)
        self.assertLess(abs(val3 - 100.0), 2.0)

    def test_point_smoother_tracks_2d_points(self):
        smoother = PointSmoother()
        x1, y1 = smoother.update("shoulder", 0.0, 10.0, 20.0)
        self.assertEqual((x1, y1), (10.0, 20.0))
        x2, y2 = smoother.update("shoulder", 0.05, 12.0, 22.0)
        self.assertLess(x2, 12.0)
        self.assertLess(y2, 22.0)
        smoother.clear()
        self.assertEqual(len(smoother._filters), 0)

    def test_excel_export_creates_valid_workbook(self):
        sample_measurement = {
            "measured_at": "2026-09-07 17:00:00",
            "input_height_cm": 170.0,
            "calibration": "ประเมินจากส่วนสูง (170 ซม.)",
            "samples_used": 24,
            "shoulder_cm": 42.5,
            "left_shoulder_cm": 21.0,
            "right_shoulder_cm": 21.5,
            "shoulder_difference_cm": 0.5,
            "posture": {
                "screening_level": "neutral",
                "screening_score": 0,
                "head_shoulder_offset_ratio": 0.04,
                "shoulder_hip_offset_ratio": 0.03,
                "trunk_inclination_deg": 2.1,
                "stable": True,
                "landmark_confidence": 0.95,
            },
            "shoulders": {
                "left_shoulder_length_ratio": 0.49,
                "right_shoulder_length_ratio": 0.51,
                "shoulder_length_difference_ratio": 0.02,
                "shoulder_balance_label": "ความต่างซ้าย–ขวาอยู่ในช่วงความคลาดเคลื่อนของภาพ",
                "samples_used": 18,
                "stable": True,
                "landmark_confidence": 0.92,
                "shoulder_tilt_deg": 1.2,
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_out.xlsx"
            saved_path = save_measurement_to_excel(sample_measurement, out_file)
            self.assertTrue(saved_path.is_file())
            self.assertGreater(saved_path.stat().st_size, 1000)

    def test_excel_export_handles_none_posture_and_shoulders(self):
        sparse_measurement = {
            "measured_at": "2026-09-07 17:00:00",
            "posture": None,
            "shoulders": None,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_sparse.xlsx"
            saved_path = save_measurement_to_excel(sparse_measurement, out_file)
            self.assertTrue(saved_path.is_file())

    def test_excel_export_creates_parent_directory_if_missing(self):
        sample_measurement = {
            "measured_at": "2026-09-07 17:00:00",
            "posture": {"screening_level": "neutral"},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "nested" / "sub" / "reports.xlsx"
            saved_path = save_measurement_to_excel(sample_measurement, out_file)
            self.assertTrue(saved_path.is_file())

    def test_get_pixel_scale_rejects_empty_or_invalid_frame(self):
        scale, corners = get_pixel_scale(None, 5.0)
        self.assertIsNone(scale)
        self.assertIsNone(corners)
        scale, corners = get_pixel_scale(np.zeros((0, 0, 3), dtype=np.uint8), 5.0)
        self.assertIsNone(scale)
        self.assertIsNone(corners)
        scale, corners = get_pixel_scale(np.zeros((100, 100, 3), dtype=np.uint8), 0.0)
        self.assertIsNone(scale)
        self.assertIsNone(corners)

    def test_ui_is_open_reports_false_when_root_destroyed(self):
        ui = MeasurementUI("Test", (320, 240))
        self.assertTrue(ui.is_open)
        ui.root.destroy()
        self.assertFalse(ui.is_open)


if __name__ == "__main__":
    unittest.main()
