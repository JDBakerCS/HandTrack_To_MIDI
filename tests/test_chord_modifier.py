"""Camera-free tests for expression-hand seventh pinch recognition."""

import unittest
from dataclasses import dataclass

from chord_modifier import PinchModifierStabilizer, analyze_seventh_pinch
from gesture_music import DOMINANT_SEVENTH_MODIFIER, QUALITY_SEVENTH_MODIFIER


@dataclass
class Point:
    x: float
    y: float


def make_expression_hand(pinch: str = "none", scale: float = 1.0):
    """Build landmarks with configurable palm size and fingertip pinch."""

    def point(x: float, y: float) -> Point:
        return Point(0.5 + (x - 0.5) * scale, 0.5 + (y - 0.5) * scale)

    landmarks = [point(0.5, 0.5) for _ in range(21)]
    landmarks[5] = point(0.30, 0.60)
    landmarks[17] = point(0.70, 0.60)
    landmarks[4] = point(0.40, 0.40)
    landmarks[8] = point(0.25, 0.20)
    landmarks[12] = point(0.55, 0.18)
    landmarks[16] = point(0.70, 0.25)

    if pinch == "index":
        landmarks[8] = point(0.41, 0.40)
    elif pinch == "middle":
        landmarks[4] = point(0.50, 0.40)
        landmarks[12] = point(0.51, 0.40)
    elif pinch == "ring":
        landmarks[4] = point(0.50, 0.40)
        landmarks[16] = point(0.49, 0.40)
    elif pinch == "tmr":
        landmarks[4] = point(0.50, 0.40)
        landmarks[12] = point(0.51, 0.40)
        landmarks[16] = point(0.49, 0.41)
    elif pinch != "none":
        raise ValueError("Unknown test pinch")
    return landmarks


class PinchRecognitionTests(unittest.TestCase):
    def test_thumb_index_pinch_selects_quality_seventh(self) -> None:
        analysis = analyze_seventh_pinch(make_expression_hand("index"))
        self.assertEqual(QUALITY_SEVENTH_MODIFIER, analysis.modifier)

    def test_thumb_middle_ring_pinch_selects_dominant_seventh(self) -> None:
        analysis = analyze_seventh_pinch(make_expression_hand("tmr"))
        self.assertEqual(DOMINANT_SEVENTH_MODIFIER, analysis.modifier)

    def test_thumb_middle_alone_does_not_select_dominant_seventh(self) -> None:
        analysis = analyze_seventh_pinch(make_expression_hand("middle"))
        self.assertIsNone(analysis.modifier)

    def test_thumb_ring_alone_does_not_select_dominant_seventh(self) -> None:
        analysis = analyze_seventh_pinch(make_expression_hand("ring"))
        self.assertIsNone(analysis.modifier)

    def test_open_hand_selects_no_modifier(self) -> None:
        analysis = analyze_seventh_pinch(make_expression_hand())
        self.assertIsNone(analysis.modifier)

    def test_normalization_is_independent_of_hand_size(self) -> None:
        small = analyze_seventh_pinch(make_expression_hand("index", scale=0.5))
        large = analyze_seventh_pinch(make_expression_hand("index", scale=1.5))
        self.assertAlmostEqual(small.index_ratio, large.index_ratio)
        self.assertEqual(small.modifier, large.modifier)

    def test_release_hysteresis_retains_active_pinch(self) -> None:
        landmarks = make_expression_hand()
        # Palm width is 0.4, so this 0.16 gap has ratio 0.4: above engage
        # threshold 0.3 but below release threshold 0.45.
        landmarks[8] = Point(landmarks[4].x + 0.16, landmarks[4].y)

        inactive = analyze_seventh_pinch(landmarks)
        active = analyze_seventh_pinch(
            landmarks, active_modifier=QUALITY_SEVENTH_MODIFIER
        )
        self.assertIsNone(inactive.modifier)
        self.assertEqual(QUALITY_SEVENTH_MODIFIER, active.modifier)

    def test_invalid_thresholds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze_seventh_pinch(
                make_expression_hand(), engage_threshold=0.5, release_threshold=0.4
            )


class PinchModifierStabilizerTests(unittest.TestCase):
    def test_modifier_activates_and_releases_after_hold_time(self) -> None:
        stabilizer = PinchModifierStabilizer(hold_seconds=0.12)

        self.assertIsNone(stabilizer.update(QUALITY_SEVENTH_MODIFIER, 1.00))
        self.assertIsNone(stabilizer.update(QUALITY_SEVENTH_MODIFIER, 1.10))
        self.assertEqual(
            QUALITY_SEVENTH_MODIFIER,
            stabilizer.update(QUALITY_SEVENTH_MODIFIER, 1.12),
        )
        self.assertEqual(QUALITY_SEVENTH_MODIFIER, stabilizer.update(None, 1.20))
        self.assertIsNone(stabilizer.update(None, 1.32))

    def test_changed_modifier_restarts_hold_time(self) -> None:
        stabilizer = PinchModifierStabilizer(hold_seconds=0.10)
        stabilizer.update(QUALITY_SEVENTH_MODIFIER, 1.00)
        stabilizer.update(DOMINANT_SEVENTH_MODIFIER, 1.05)

        self.assertIsNone(stabilizer.update(DOMINANT_SEVENTH_MODIFIER, 1.10))
        self.assertEqual(
            DOMINANT_SEVENTH_MODIFIER,
            stabilizer.update(DOMINANT_SEVENTH_MODIFIER, 1.16),
        )


if __name__ == "__main__":
    unittest.main()
