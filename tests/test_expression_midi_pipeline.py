"""End-to-end test for expression landmarks becoming MIDI CC messages."""

import unittest
from dataclasses import dataclass

from expression_controller import (
    SmoothedMidiControl,
    palm_vertical_position,
    vertical_position_to_midi,
)
from midi_chord_player import ChordMidiPlayer


@dataclass
class Point:
    y: float


class FakeOutput:
    def __init__(self) -> None:
        self.messages = []

    def send(self, message) -> None:
        self.messages.append(message)


class ExpressionMidiPipelineTests(unittest.TestCase):
    def test_palm_height_becomes_filter_control_change(self) -> None:
        landmarks = [Point(0.20) for _ in range(21)]
        output = FakeOutput()
        player = ChordMidiPlayer(output, channel=1)
        control = SmoothedMidiControl(
            smoothing=1.0,
            dead_zone=0,
            max_rate_hz=30.0,
        )

        palm_y = palm_vertical_position(landmarks)
        raw_value = vertical_position_to_midi(palm_y)
        expression_value = control.update(raw_value, timestamp=1.0)
        player.send_control_change(control=74, value=expression_value)

        message = output.messages[-1]
        self.assertEqual(118, raw_value)
        self.assertEqual("control_change", message.type)
        self.assertEqual(74, message.control)
        self.assertEqual(118, message.value)
        self.assertEqual(0, message.channel)


if __name__ == "__main__":
    unittest.main()
