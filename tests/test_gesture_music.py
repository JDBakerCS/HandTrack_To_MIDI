"""Tests for translating stable gestures into musical intent."""

import unittest

from gesture_classifier import ChordGesture
from gesture_music import (
    DOMINANT_SEVENTH_MODIFIER,
    QUALITY_SEVENTH_MODIFIER,
    chord_intent_from_gesture,
)


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

    def test_quality_seventh_modifier_follows_the_selector_quality(self) -> None:
        major = chord_intent_from_gesture(
            make_gesture(1, "Major"),
            seventh_modifier=QUALITY_SEVENTH_MODIFIER,
        )
        minor = chord_intent_from_gesture(
            make_gesture(2, "Minor"),
            seventh_modifier=QUALITY_SEVENTH_MODIFIER,
        )

        self.assertEqual("major7", major.extension)
        self.assertEqual("minor7", minor.extension)

    def test_dominant_modifier_works_on_any_major_degree(self) -> None:
        intent = chord_intent_from_gesture(
            make_gesture(2, "Major"),
            seventh_modifier=DOMINANT_SEVENTH_MODIFIER,
        )
        self.assertEqual("dominant7", intent.extension)

    def test_dominant_modifier_rejects_a_minor_pose(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a major pose"):
            chord_intent_from_gesture(
                make_gesture(2, "Minor"),
                seventh_modifier=DOMINANT_SEVENTH_MODIFIER,
            )

    def test_explicit_modifier_overrides_automatic_v7(self) -> None:
        intent = chord_intent_from_gesture(
            make_gesture(5, "Major"),
            dominant_seventh=True,
            seventh_modifier=QUALITY_SEVENTH_MODIFIER,
        )
        self.assertEqual("major7", intent.extension)

    def test_unknown_modifier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chord_intent_from_gesture(
                make_gesture(1, "Major"), seventh_modifier="mystery"
            )


if __name__ == "__main__":
    unittest.main()
