"""Camera-free tests for the custom scale-degree gesture classifier."""

import unittest
from dataclasses import dataclass
from typing import Iterable, List

from gesture_classifier import ChordGesture, GestureStabilizer, analyze_hand


# --- Synthetic hand builder ------------------------------------------------


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0


LONG_FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_X = {
    "index": 0.32,
    "middle": 0.44,
    "ring": 0.56,
    "pinky": 0.68,
}


def make_hand(
    extended_fingers: Iterable[str], direction: str = "up", vulcan: bool = False
) -> List[Point]:
    """Build a simple landmark hand with known straight and folded fingers."""

    extended = set(extended_fingers)
    landmarks = [Point(0.5, 0.5) for _ in range(21)]
    is_up = direction == "up"
    landmarks[0] = Point(0.5, 0.82 if is_up else 0.18)

    x_positions = dict(FINGER_X)
    if vulcan:
        x_positions.update(index=0.35, middle=0.42, ring=0.58, pinky=0.65)

    for finger_name, indices in LONG_FINGER_LANDMARKS.items():
        mcp, pip, dip, tip = indices
        x_position = x_positions[finger_name]
        if finger_name in extended:
            y_positions = (0.62, 0.52, 0.42, 0.30) if is_up else (
                0.38, 0.48, 0.58, 0.70
            )
        else:
            y_positions = (0.62, 0.54, 0.60, 0.66) if is_up else (
                0.38, 0.46, 0.40, 0.34
            )
        for landmark_index, y_position in zip(indices, y_positions):
            landmarks[landmark_index] = Point(x_position, y_position)

    # The thumb is horizontal in both directions. Its distance from the index
    # knuckle distinguishes an open thumb from one folded over the palm.
    landmarks[1] = Point(0.44, 0.60 if is_up else 0.40)
    if "thumb" in extended:
        thumb_points = (0.34, 0.20, 0.05)
    else:
        thumb_points = (0.44, 0.40, 0.36)
    for landmark_index, x_position in zip((2, 3, 4), thumb_points):
        landmarks[landmark_index] = Point(x_position, 0.60 if is_up else 0.40)

    return landmarks


# --- Pose vocabulary tests -------------------------------------------------


class GestureClassifierTests(unittest.TestCase):
    def test_all_seven_degrees_in_both_qualities(self) -> None:
        pose_cases = (
            (("index",), 1, False, "Major", "Minor"),
            (("index", "middle"), 2, False, "Major", "Minor"),
            (("index", "middle", "ring"), 3, False, "Major", "Minor"),
            (("index", "middle", "ring", "pinky"), 4, False, "Major", "Minor"),
            (
                ("thumb", "index", "middle", "ring", "pinky"),
                5,
                False,
                "Major",
                "Minor",
            ),
            (("index", "pinky"), 6, False, "Minor", "Major"),
            (
                ("thumb", "index", "middle", "ring", "pinky"),
                7,
                True,
                "Major",
                "Minor",
            ),
        )

        for fingers, degree, vulcan, up_quality, down_quality in pose_cases:
            for direction, quality in (("up", up_quality), ("down", down_quality)):
                with self.subTest(degree=degree, quality=quality):
                    analysis = analyze_hand(make_hand(fingers, direction, vulcan))
                    self.assertIsNotNone(analysis.gesture)
                    self.assertEqual(degree, analysis.gesture.degree)
                    self.assertEqual(quality, analysis.gesture.quality)

    def test_closed_hand_is_not_a_chord_gesture(self) -> None:
        analysis = analyze_hand(make_hand(()))
        self.assertIsNone(analysis.gesture)

    def test_wrong_landmark_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze_hand(make_hand(("index",))[:-1])


# --- Debounce behavior tests ----------------------------------------------


class GestureStabilizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gesture = ChordGesture(1, "I", "Major", "Index", 0.9)

    def test_gesture_becomes_stable_after_hold_time(self) -> None:
        stabilizer = GestureStabilizer(hold_seconds=0.25)
        self.assertIsNone(stabilizer.update(self.gesture, 1.0))
        self.assertIsNone(stabilizer.update(self.gesture, 1.20))
        self.assertEqual(self.gesture, stabilizer.update(self.gesture, 1.25))

    def test_changed_gesture_restarts_hold_time(self) -> None:
        stabilizer = GestureStabilizer(hold_seconds=0.25)
        minor = ChordGesture(1, "I", "Minor", "Index", 0.9)
        stabilizer.update(self.gesture, 1.0)
        stabilizer.update(minor, 1.20)
        self.assertIsNone(stabilizer.update(minor, 1.40))
        self.assertEqual(minor, stabilizer.update(minor, 1.45))


if __name__ == "__main__":
    unittest.main()
