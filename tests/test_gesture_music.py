"""Tests for translating stable gestures into musical intent."""

import unittest

from gesture_classifier import ChordGesture
from gesture_music import chord_intent_from_gesture


def make_gesture(degree: int, quality: str) -> ChordGesture:
    """Create the minimum realistic gesture needed by mapping tests."""

    return ChordGesture(
        degree=degree,
        roman_numeral=str(degree),
        quality=quality,
        pose_name="Test Pose",
        confidence=0.9,
    )


class GestureMusicTests(unittest.TestCase):
    def test_major_and_minor_quality_are_normalized(self) -> None:
        major = chord_intent_from_gesture(make_gesture(1, "Major"))
        minor = chord_intent_from_gesture(make_gesture(6, "Minor"))

        self.assertEqual((1, "major"), (major.degree, major.quality))
        self.assertEqual((6, "minor"), (minor.degree, minor.quality))

    def test_performance_configuration_is_forwarded(self) -> None:
        intent = chord_intent_from_gesture(
            make_gesture(4, "Major"),
            tonic="D",
            scale="natural_minor",
            octave=3,
            voicing_style="open",
            velocity=72,
        )

        self.assertEqual("D", intent.tonic)
        self.assertEqual("natural_minor", intent.scale)
        self.assertEqual(3, intent.octave)
        self.assertEqual("open", intent.voicing_style)
        self.assertEqual(72, intent.velocity)
        self.assertIsNone(intent.inversion)

    def test_dominant_seventh_option_applies_only_to_v_major(self) -> None:
        dominant = chord_intent_from_gesture(
            make_gesture(5, "Major"), dominant_seventh=True
        )
        minor_five = chord_intent_from_gesture(
            make_gesture(5, "Minor"), dominant_seventh=True
        )
        major_four = chord_intent_from_gesture(
            make_gesture(4, "Major"), dominant_seventh=True
        )

        self.assertEqual("dominant7", dominant.extension)
        self.assertIsNone(minor_five.extension)
        self.assertIsNone(major_four.extension)


if __name__ == "__main__":
    unittest.main()
