"""End-to-end tests for pinch modifiers becoming seventh-chord MIDI notes."""

import unittest
from dataclasses import dataclass

from chord_modifier import PinchModifierStabilizer, analyze_seventh_pinch
from gesture_classifier import ChordGesture
from gesture_music import InvalidChordModifierError, chord_intent_from_gesture
from harmony_engine import VoicingEngine
from midi_chord_player import ChordMidiPlayer


@dataclass
class Point:
    x: float
    y: float


class FakeOutput:
    def __init__(self) -> None:
        self.messages = []

    def send(self, message) -> None:
        self.messages.append(message)


def expression_hand(pinch: str):
    landmarks = [Point(0.5, 0.5) for _ in range(21)]
    landmarks[5] = Point(0.30, 0.60)
    landmarks[17] = Point(0.70, 0.60)
    landmarks[4] = Point(0.40, 0.40)
    landmarks[8] = Point(0.25, 0.20)
    landmarks[12] = Point(0.55, 0.18)
    if pinch == "index":
        landmarks[8] = Point(0.41, 0.40)
    elif pinch == "middle":
        landmarks[4] = Point(0.50, 0.40)
        landmarks[12] = Point(0.51, 0.40)
    return landmarks


def selector_gesture(quality: str) -> ChordGesture:
    return ChordGesture(1, "I", quality, "Index", 0.9)


class SeventhModifierPipelineTests(unittest.TestCase):
    def test_pinch_paths_produce_expected_seventh_chords(self) -> None:
        cases = (
            ("index", "Major", "major7", (60, 64, 67, 71)),
            ("index", "Minor", "minor7", (60, 63, 67, 70)),
            ("middle", "Major", "dominant7", (60, 64, 67, 70)),
        )

        for pinch, quality, extension, expected_notes in cases:
            with self.subTest(pinch=pinch, quality=quality):
                analysis = analyze_seventh_pinch(expression_hand(pinch))
                stabilizer = PinchModifierStabilizer(hold_seconds=0.0)
                modifier = stabilizer.update(analysis.modifier, timestamp=1.0)
                intent = chord_intent_from_gesture(
                    selector_gesture(quality), seventh_modifier=modifier
                )
                notes = VoicingEngine().voice(intent)
                output = FakeOutput()
                ChordMidiPlayer(output).play_chord(notes)

                self.assertEqual(extension, intent.extension)
                self.assertEqual(expected_notes, notes)
                self.assertEqual(
                    list(expected_notes), [message.note for message in output.messages]
                )

    def test_invalid_dominant_minor_keeps_existing_player_state(self) -> None:
        output = FakeOutput()
        player = ChordMidiPlayer(output)
        player.play_chord((60, 63, 67))
        output.messages.clear()
        modifier = analyze_seventh_pinch(expression_hand("middle")).modifier

        with self.assertRaises(InvalidChordModifierError):
            chord_intent_from_gesture(
                selector_gesture("Minor"), seventh_modifier=modifier
            )

        self.assertEqual((60, 63, 67), player.active_notes)
        self.assertEqual([], output.messages)


if __name__ == "__main__":
    unittest.main()
