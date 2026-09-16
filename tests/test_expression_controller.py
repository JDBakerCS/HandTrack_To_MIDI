"""Tests for expression-hand position mapping and MIDI-value smoothing."""

import unittest
from dataclasses import dataclass

from expression_controller import (
    SmoothedMidiControl,
    palm_vertical_position,
    vertical_position_to_midi,
)


@dataclass
class Point:
    y: float


class ExpressionMappingTests(unittest.TestCase):
    def test_palm_position_averages_stable_hand_landmarks(self) -> None:
        landmarks = [Point(0.5) for _ in range(21)]
        for index, value in zip((0, 5, 9, 13, 17), (0.2, 0.3, 0.4, 0.5, 0.6)):
            landmarks[index].y = value

        self.assertAlmostEqual(0.4, palm_vertical_position(landmarks))

    def test_raised_hand_opens_control_and_lowered_hand_closes_it(self) -> None:
        self.assertEqual(127, vertical_position_to_midi(0.15))
        self.assertEqual(64, vertical_position_to_midi(0.50))
        self.assertEqual(0, vertical_position_to_midi(0.85))

    def test_positions_outside_playable_area_are_clamped(self) -> None:
        self.assertEqual(127, vertical_position_to_midi(-0.5))
        self.assertEqual(0, vertical_position_to_midi(1.5))

    def test_invalid_landmark_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            palm_vertical_position([Point(0.5) for _ in range(20)])

    def test_invalid_expression_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            vertical_position_to_midi(0.5, top=0.9, bottom=0.1)


class SmoothedMidiControlTests(unittest.TestCase):
    def test_first_value_is_emitted_immediately(self) -> None:
        control = SmoothedMidiControl()
        self.assertEqual(100, control.update(100, 1.0))

    def test_exponential_smoothing_reduces_a_large_jump(self) -> None:
        control = SmoothedMidiControl(smoothing=0.5, dead_zone=0, max_rate_hz=10)

        self.assertEqual(100, control.update(100, 0.0))
        self.assertIsNone(control.update(80, 0.05))
        self.assertEqual(85, control.update(80, 0.10))

    def test_dead_zone_suppresses_small_changes(self) -> None:
        control = SmoothedMidiControl(smoothing=1.0, dead_zone=3, max_rate_hz=100)

        self.assertEqual(64, control.update(64, 0.0))
        self.assertIsNone(control.update(66, 0.02))
        self.assertEqual(68, control.update(68, 0.04))

    def test_reset_makes_next_value_immediate(self) -> None:
        control = SmoothedMidiControl()
        control.update(20, 1.0)
        control.reset()

        self.assertEqual(110, control.update(110, 1.01))
        self.assertEqual(110, control.current_value)

    def test_invalid_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SmoothedMidiControl(smoothing=0.0)
        with self.assertRaises(ValueError):
            SmoothedMidiControl(dead_zone=128)
        with self.assertRaises(ValueError):
            SmoothedMidiControl(max_rate_hz=0.0)


if __name__ == "__main__":
    unittest.main()
