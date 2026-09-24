"""End-to-end tests for the camera-free gesture-to-MIDI pipeline."""

import unittest

from gesture_classifier import ChordGesture
from gesture_music import chord_intent_from_gesture
from harmony_engine import VoicingEngine
from midi_chord_player import ChordMidiPlayer


class FakeOutput:
    def __init__(self) -> None:
        self.messages = []

    def send(self, message) -> None:
        self.messages.append(message)


def gesture(degree: int, numeral: str, quality: str) -> ChordGesture:
    return ChordGesture(degree, numeral, quality, "Test Pose", 0.9)


class GestureMidiPipelineTests(unittest.TestCase):
    def test_gesture_change_becomes_voice_led_midi_chords(self) -> None:
        output = FakeOutput()
        player = ChordMidiPlayer(output)
        voicing_engine = VoicingEngine()

        first_intent = chord_intent_from_gesture(gesture(1, "I", "Major"))
        first_notes = voicing_engine.voice(first_intent)
        player.play_chord(first_notes, velocity=first_intent.velocity)

        second_intent = chord_intent_from_gesture(gesture(4, "IV", "Major"))
        second_notes = voicing_engine.voice(second_intent)
        player.play_chord(second_notes, velocity=second_intent.velocity)

        self.assertEqual((60, 64, 67), first_notes)
        self.assertEqual((60, 65, 69), second_notes)
        self.assertEqual(
            ["note_on"] * 5 + ["note_off"] * 2,
            [message.type for message in output.messages],
        )
        self.assertEqual(
            [60, 64, 67, 65, 69, 64, 67],
            [message.note for message in output.messages],
        )

    def test_v_major_can_become_a_four_note_dominant_seventh(self) -> None:
        output = FakeOutput()
        player = ChordMidiPlayer(output)
        voicing_engine = VoicingEngine()
        intent = chord_intent_from_gesture(
            gesture(5, "V", "Major"), dominant_seventh=True
        )

        notes = voicing_engine.voice(intent)
        player.play_chord(notes, velocity=intent.velocity)

        self.assertEqual((67, 71, 74, 77), notes)
        self.assertEqual(4, len(output.messages))
        self.assertTrue(all(message.type == "note_on" for message in output.messages))


if __name__ == "__main__":
    unittest.main()
