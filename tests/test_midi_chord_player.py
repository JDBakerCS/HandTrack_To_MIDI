"""Camera-free tests for chord MIDI playback and port-name handling."""

import unittest

from midi_chord_player import ChordMidiPlayer, resolve_output_port


class FakeOutput:
    """Record Mido messages without opening a real Windows MIDI device."""

    def __init__(self) -> None:
        self.messages = []

    def send(self, message) -> None:
        self.messages.append(message)


class PortResolutionTests(unittest.TestCase):
    def test_exact_port_name_wins(self) -> None:
        available = ("Port1", "Port1 1")
        self.assertEqual("Port1", resolve_output_port("Port1", available))

    def test_loopmidi_name_can_omit_mido_numeric_suffix(self) -> None:
        available = ("Microsoft GS Wavetable Synth 0", "Port1 1")
        self.assertEqual("Port1 1", resolve_output_port("Port1", available))

    def test_ambiguous_alias_lists_matches(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            resolve_output_port("Port1", ("Port1 1", "Port1 2"))

    def test_missing_port_lists_available_ports(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Microsoft GS"):
            resolve_output_port("Missing", ("Microsoft GS Wavetable Synth 0",))


class ChordMidiPlayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = FakeOutput()
        self.player = ChordMidiPlayer(self.output, channel=2)

    def test_play_chord_sends_note_ons_on_selected_channel(self) -> None:
        changed = self.player.play_chord((60, 64, 67), velocity=90)

        self.assertTrue(changed)
        self.assertEqual((60, 64, 67), self.player.active_notes)
        self.assertEqual(["note_on"] * 3, [m.type for m in self.output.messages])
        self.assertEqual([1] * 3, [m.channel for m in self.output.messages])
        self.assertEqual([90] * 3, [m.velocity for m in self.output.messages])

    def test_held_chord_does_not_repeat_note_ons(self) -> None:
        self.player.play_chord((60, 64, 67))
        changed = self.player.play_chord((60, 64, 67))

        self.assertFalse(changed)
        self.assertEqual(3, len(self.output.messages))

    def test_chord_change_releases_old_notes_before_new_notes(self) -> None:
        self.player.play_chord((60, 64, 67))
        self.output.messages.clear()

        self.player.play_chord((60, 65, 69))

        self.assertEqual(
            ["note_off", "note_off", "note_off", "note_on", "note_on", "note_on"],
            [message.type for message in self.output.messages],
        )
        self.assertEqual((60, 65, 69), self.player.active_notes)

    def test_stop_releases_active_notes_once(self) -> None:
        self.player.play_chord((60, 64, 67))
        self.output.messages.clear()

        self.assertTrue(self.player.stop())
        self.assertFalse(self.player.stop())
        self.assertEqual(["note_off"] * 3, [m.type for m in self.output.messages])

    def test_control_change_uses_the_players_channel(self) -> None:
        self.player.send_control_change(control=74, value=91)

        message = self.output.messages[-1]
        self.assertEqual("control_change", message.type)
        self.assertEqual(1, message.channel)
        self.assertEqual(74, message.control)
        self.assertEqual(91, message.value)

    def test_invalid_control_change_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.player.send_control_change(control=128, value=64)
        with self.assertRaises(ValueError):
            self.player.send_control_change(control=74, value=-1)

    def test_panic_sends_all_notes_off_control_change(self) -> None:
        self.player.play_chord((60, 64, 67))
        self.output.messages.clear()

        self.player.panic()

        panic_message = self.output.messages[-1]
        self.assertEqual("control_change", panic_message.type)
        self.assertEqual(123, panic_message.control)
        self.assertEqual(0, panic_message.value)

    def test_invalid_channel_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ChordMidiPlayer(self.output, channel=17)


if __name__ == "__main__":
    unittest.main()
