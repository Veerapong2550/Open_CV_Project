"""Unit tests for the tolerant V-sign recogniser."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from body_measure.utils import is_peace_sign
from body_measure.state_machine import GestureStateMachine, State


@dataclass
class Landmark:
    x: float = 0.50
    y: float = 0.60


def _hand(*, ring_extended: bool = False) -> list[Landmark]:
    points = [Landmark() for _ in range(21)]
    points[0] = Landmark(0.50, 0.80)   # wrist
    points[6] = Landmark(0.42, 0.55)   # index PIP
    points[8] = Landmark(0.37, 0.28)   # index tip
    points[9] = Landmark(0.50, 0.60)   # hand-size reference
    points[10] = Landmark(0.53, 0.53)  # middle PIP
    points[12] = Landmark(0.55, 0.25)  # middle tip
    points[14] = Landmark(0.59, 0.56)  # ring PIP
    points[16] = Landmark(0.60, 0.66)  # folded ring tip
    points[18] = Landmark(0.66, 0.59)  # pinky PIP
    points[20] = Landmark(0.68, 0.69)  # folded pinky tip
    if ring_extended:
        points[16] = Landmark(0.60, 0.24)
    return points


class GestureTests(unittest.TestCase):
    def test_v_sign_is_accepted_without_exact_joint_alignment(self):
        self.assertTrue(is_peace_sign(_hand()))

    def test_open_ring_finger_is_not_accepted_as_v_sign(self):
        self.assertFalse(is_peace_sign(_hand(ring_extended=True)))

    def test_incomplete_landmarks_are_not_a_gesture(self):
        self.assertFalse(is_peace_sign(_hand()[:10]))

    def test_peace_sign_starts_but_cannot_exit_a_measurement(self):
        machine = GestureStateMachine(hold_seconds=1.5)
        state, _, transitioned = machine.update(True, 10.0)
        self.assertEqual(state, State.WAITING)
        self.assertFalse(transitioned)
        state, _, transitioned = machine.update(True, 11.5)
        self.assertEqual(state, State.MEASURING)
        self.assertTrue(transitioned)
        machine.update(False, 11.6)
        machine.update(True, 12.0)
        state, _, transitioned = machine.update(True, 13.5)
        self.assertEqual(state, State.MEASURING)
        self.assertFalse(transitioned)


if __name__ == "__main__":
    unittest.main()
