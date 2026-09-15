"""Camera-free tests for chord intent and deterministic chord construction."""

import unittest

from harmony_engine import (
    ChordIntent,
    build_chord,
    chord_root,
    format_chord,
    midi_note_name,
    note_to_midi,
)


# --- Pitch conversion tests ------------------------------------------------


class PitchConversionTests(unittest.TestCase):
    def test_middle_c_is_midi_note_60(self) -> None:
        self.assertEqual(60, note_to_midi("C", 4))
        self.assertEqual("C4", midi_note_name(60))

    def test_flat_and_sharp_tonics_share_pitch_classes(self) -> None:
        self.assertEqual(note_to_midi("C#", 4), note_to_midi("Db", 4))
        self.assertEqual(note_to_midi("F#", 3), note_to_midi("G♭", 3))


# --- Chord construction tests ---------------------------------------------


class ChordConstructionTests(unittest.TestCase):
    def test_c_major_root_position(self) -> None:
        notes = build_chord(ChordIntent())
        self.assertEqual((60, 64, 67), notes)
        self.assertEqual("C4 - E4 - G4", format_chord(notes))

    def test_all_c_major_scale_degree_roots(self) -> None:
        roots = tuple(
            chord_root(ChordIntent(degree=degree)) for degree in range(1, 8)
        )
        self.assertEqual((60, 62, 64, 65, 67, 69, 71), roots)

    def test_scale_degree_can_be_major_or_minor(self) -> None:
        d_major = build_chord(ChordIntent(degree=2, quality="major"))
        d_minor = build_chord(ChordIntent(degree=2, quality="minor"))
        self.assertEqual((62, 66, 69), d_major)
        self.assertEqual((62, 65, 69), d_minor)

    def test_vi_minor_matches_a_minor_in_c(self) -> None:
        notes = build_chord(ChordIntent(degree=6, quality="minor"))
        self.assertEqual((69, 72, 76), notes)
        self.assertEqual("A4 - C5 - E5", format_chord(notes))

    def test_v_dominant_seventh_in_c(self) -> None:
        notes = build_chord(
            ChordIntent(degree=5, quality="major", extension="dominant7")
        )
        self.assertEqual((67, 71, 74, 77), notes)
        self.assertEqual("G4 - B4 - D5 - F5", format_chord(notes))

    def test_triads_support_both_inversions(self) -> None:
        first = build_chord(ChordIntent(inversion=1))
        second = build_chord(ChordIntent(inversion=2))
        self.assertEqual((64, 67, 72), first)
        self.assertEqual((67, 72, 76), second)

    def test_open_voicing_moves_the_middle_voice_up(self) -> None:
        notes = build_chord(ChordIntent(voicing_style="open"))
        self.assertEqual((60, 67, 76), notes)


# --- Intent validation tests ----------------------------------------------


class ChordIntentValidationTests(unittest.TestCase):
    def test_degree_must_be_one_through_seven(self) -> None:
        with self.assertRaises(ValueError):
            ChordIntent(degree=8)

    def test_dominant_seventh_requires_major_quality(self) -> None:
        with self.assertRaises(ValueError):
            ChordIntent(quality="minor", extension="dominant7")

    def test_inversion_must_exist_for_the_chord(self) -> None:
        with self.assertRaises(ValueError):
            ChordIntent(inversion=3)


if __name__ == "__main__":
    unittest.main()
