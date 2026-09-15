"""Pure musical logic for turning chord intent into MIDI note numbers.

The camera layer should communicate *what* the player intends to perform. This
module decides *which notes* express that intent. It deliberately has no camera,
MediaPipe, Mido, or synthesizer dependencies, which keeps the musical behavior
fast and straightforward to test.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


# --- Musical vocabulary ----------------------------------------------------


NOTE_PITCH_CLASSES = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}
PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
}
TRIAD_INTERVALS = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
}
SUPPORTED_EXTENSIONS = (None, "dominant7")
SUPPORTED_VOICINGS = ("close", "open")
SUPPORTED_ARTICULATIONS = ("sustain", "legato", "staccato")


# --- Public musical intent -------------------------------------------------


@dataclass(frozen=True)
class ChordIntent:
    """A high-level chord request produced by gestures or another controller."""

    tonic: str = "C"
    scale: str = "major"
    degree: int = 1
    quality: str = "major"
    extension: Optional[str] = None
    inversion: Optional[int] = 0
    octave: int = 4
    voicing_style: str = "close"
    velocity: int = 96
    articulation: str = "sustain"

    def __post_init__(self) -> None:
        normalized_tonic = normalize_note_name(self.tonic)
        object.__setattr__(self, "tonic", normalized_tonic)

        if self.scale not in SCALE_INTERVALS:
            raise ValueError(f"Unsupported scale: {self.scale}")
        if not 1 <= self.degree <= 7:
            raise ValueError("degree must be between 1 and 7")
        if self.quality not in TRIAD_INTERVALS:
            raise ValueError(f"Unsupported chord quality: {self.quality}")
        if self.extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported chord extension: {self.extension}")
        if self.extension == "dominant7" and self.quality != "major":
            raise ValueError("dominant7 requires a major triad quality")
        if self.voicing_style not in SUPPORTED_VOICINGS:
            raise ValueError(f"Unsupported voicing style: {self.voicing_style}")
        if self.articulation not in SUPPORTED_ARTICULATIONS:
            raise ValueError(f"Unsupported articulation: {self.articulation}")
        if not 1 <= self.velocity <= 127:
            raise ValueError("velocity must be between 1 and 127")

        voice_count = 4 if self.extension == "dominant7" else 3
        if self.inversion is not None and not 0 <= self.inversion < voice_count:
            raise ValueError(
                f"inversion must be between 0 and {voice_count - 1}, or None"
            )


# --- Pitch and chord construction -----------------------------------------


def normalize_note_name(note_name: str) -> str:
    """Normalize ASCII or Unicode accidental names and validate the tonic."""

    normalized = note_name.strip().replace("♯", "#").replace("♭", "b").upper()
    if normalized not in NOTE_PITCH_CLASSES:
        raise ValueError(f"Unsupported tonic: {note_name}")
    return normalized


def note_to_midi(note_name: str, octave: int) -> int:
    """Convert a pitch name and scientific-pitch octave to a MIDI note number."""

    pitch_class = NOTE_PITCH_CLASSES[normalize_note_name(note_name)]
    midi_note = (octave + 1) * 12 + pitch_class
    if not 0 <= midi_note <= 127:
        raise ValueError("tonic and octave produce a note outside MIDI range")
    return midi_note


def midi_note_name(midi_note: int) -> str:
    """Return a readable sharp-based name such as C4 for MIDI note 60."""

    if not 0 <= midi_note <= 127:
        raise ValueError("MIDI note must be between 0 and 127")
    pitch_name = PITCH_CLASS_NAMES[midi_note % 12]
    octave = midi_note // 12 - 1
    return f"{pitch_name}{octave}"


def chord_root(intent: ChordIntent) -> int:
    """Find the MIDI root for the requested scale degree."""

    tonic_midi = note_to_midi(intent.tonic, intent.octave)
    degree_offset = SCALE_INTERVALS[intent.scale][intent.degree - 1]
    root = tonic_midi + degree_offset
    if root > 127:
        raise ValueError("scale degree root falls outside MIDI range")
    return root


def _apply_inversion(notes: Sequence[int], inversion: int) -> Tuple[int, ...]:
    inverted = list(notes)
    for _ in range(inversion):
        inverted.append(inverted.pop(0) + 12)
    return tuple(inverted)


def _apply_voicing_style(
    notes: Sequence[int], voicing_style: str
) -> Tuple[int, ...]:
    if voicing_style == "close":
        return tuple(notes)

    # Open voicing moves the second voice up an octave. This widens a triad and
    # produces a consistent, transparent spread for a dominant seventh chord.
    opened = list(notes)
    opened[1] += 12
    return tuple(sorted(opened))


def build_chord(intent: ChordIntent) -> Tuple[int, ...]:
    """Convert a validated intent into ordered MIDI notes."""

    root = chord_root(intent)
    intervals = list(TRIAD_INTERVALS[intent.quality])
    if intent.extension == "dominant7":
        intervals.append(10)

    notes = tuple(root + interval for interval in intervals)
    inversion = intent.inversion if intent.inversion is not None else 0
    notes = _apply_inversion(notes, inversion)
    notes = _apply_voicing_style(notes, intent.voicing_style)

    if any(note > 127 for note in notes):
        raise ValueError("chord voicing falls outside MIDI range")
    return notes


def format_chord(notes: Sequence[int]) -> str:
    """Format MIDI notes for terminal output and human musical review."""

    return " - ".join(midi_note_name(note) for note in notes)
