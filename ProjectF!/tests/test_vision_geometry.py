"""Pure geometry checks for the enlarged distant-hand pass."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from body_measure.vision import PoseFrameResult, PoseLandmark, VisionEngine


@dataclass
class RawPoint:
    x: float
    y: float
    z: float = 0.0


@dataclass
class RawHandResult:
    hand_landmarks: list[list[RawPoint]]


class VisionGeometryTests(unittest.TestCase):
    def test_video_timestamp_is_always_strictly_increasing(self):
        engine = object.__new__(VisionEngine)
        engine._last_timestamp_ms = -1
        self.assertEqual(engine._next_timestamp(100), 100)
        self.assertEqual(engine._next_timestamp(100), 101)
        self.assertEqual(engine._next_timestamp(99), 102)

    def test_crop_hand_points_map_back_to_camera_coordinates(self):
        result = VisionEngine._map_hand_result(
            RawHandResult([[RawPoint(0.25, 0.50)]]), (100, 200, 300, 400), 800, 600,
        )
        point = result.hand_landmarks[0][0]
        self.assertAlmostEqual(point.x, 0.1875)
        self.assertAlmostEqual(point.y, 0.50)

    def test_visible_wrist_produces_an_enlarged_focus_crop(self):
        pose = [PoseLandmark(0.5, 0.5, visibility=0.0) for _ in range(33)]
        # A raised right wrist and its elbow, plus a vertical body extent for
        # sizing the crop proportionally when the person is far away.
        pose[0] = PoseLandmark(0.50, 0.12, visibility=0.9)
        pose[27] = PoseLandmark(0.45, 0.90, visibility=0.9)
        pose[13] = PoseLandmark(0.35, 0.48, visibility=0.8)
        pose[15] = PoseLandmark(0.28, 0.32, visibility=0.8)
        rois = VisionEngine._hand_rois_from_pose(PoseFrameResult([pose]), 1280, 720)
        self.assertEqual(len(rois), 1)
        left, top, right, bottom = rois[0]
        self.assertEqual(right - left, bottom - top)
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, 1280)
        self.assertLessEqual(bottom, 720)


    def test_face_detail_landmarks_excluded_from_connections_and_drawing(self):
        import numpy as np
        from body_measure.utils import POSE_CONNECTIONS, UNUSED_FACE_LANDMARKS, draw_pose_landmarks

        # Verify all facial landmarks (0..10: nose, eyes, ears, mouth) are in UNUSED_FACE_LANDMARKS
        self.assertEqual(UNUSED_FACE_LANDMARKS, frozenset(range(11)))

        # Verify NO facial landmarks are in any POSE_CONNECTIONS
        for start, end in POSE_CONNECTIONS:
            self.assertNotIn(start, UNUSED_FACE_LANDMARKS, f"Landmark {start} should not be in POSE_CONNECTIONS")
            self.assertNotIn(end, UNUSED_FACE_LANDMARKS, f"Landmark {end} should not be in POSE_CONNECTIONS")

        # Verify drawing executes cleanly without error and ignores face points
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = [PoseLandmark(0.5, 0.5, visibility=0.99) for _ in range(33)]
        draw_pose_landmarks(frame, landmarks)
        # Check that body canvas was drawn on
        self.assertGreater(frame.sum(), 0)

        # Check that if ONLY face landmarks are visible, nothing is drawn on canvas
        face_only_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_only_landmarks = [PoseLandmark(0.5, 0.5, visibility=0.99 if i < 11 else 0.0) for i in range(33)]
        draw_pose_landmarks(face_only_frame, face_only_landmarks)
        self.assertEqual(face_only_frame.sum(), 0, "No drawing should occur for face-only landmarks")


if __name__ == "__main__":
    unittest.main()
