"""Print chord voicings for musical review without opening a MIDI port."""

import argparse
import sys
from dataclasses import dataclass
from typing import Tuple

from harmony_engine import ChordIntent, VoicingEngine, format_chord


# --- Progression vocabulary ------------------------------------------------


ROMAN_NUMERALS = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
}
SUPPORTED_CHORD_TYPES = ("major", "minor", "dominant7")
DEFAULT_PROGRESSION = "1:major,4:major,5:dominant7,1:major"


@dataclass(frozen=True)
class ProgressionStep:
    degree: int
    chord_type: str


def parse_progression(value: str) -> Tuple[ProgressionStep, ...]:
    """Parse comma-separated steps such as 1:major,4:major,5:dominant7."""

    steps = []
    for raw_step in value.split(","):
        pieces = raw_step.strip().lower().split(":")
        if len(pieces) != 2:
            raise argparse.ArgumentTypeError(
                "progression steps must use DEGREE:TYPE format"
            )

        degree_text, chord_type = pieces
        try:
            degree = int(degree_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid scale degree: {degree_text}"
            ) from error

        if not 1 <= degree <= 7:
            raise argparse.ArgumentTypeError("scale degrees must be between 1 and 7")
        if chord_type not in SUPPORTED_CHORD_TYPES:
            choices = ", ".join(SUPPORTED_CHORD_TYPES)
            raise argparse.ArgumentTypeError(
                f"chord type must be one of: {choices}"
            )
        steps.append(ProgressionStep(degree, chord_type))

    if not steps:
        raise argparse.ArgumentTypeError("progression must contain at least one step")
    return tuple(steps)


# --- Command-line configuration -------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview chord construction and automatic voice leading."
    )
    parser.add_argument("--tonic", default="C", help="Key tonic, such as C or F#")
    parser.add_argument(
        "--scale",
        choices=("major", "natural_minor"),
        default="major",
        help="Scale used to place each scale-degree root",
    )
    parser.add_argument(
        "--progression",
        type=parse_progression,
        default=parse_progression(DEFAULT_PROGRESSION),
        metavar="STEPS",
        help="Comma-separated DEGREE:TYPE steps",
    )
    parser.add_argument("--octave", type=int, default=4, help="Tonic octave")
    parser.add_argument(
        "--voicing",
        choices=("close", "open"),
        default="close",
        help="Chord spacing style",
    )
    parser.add_argument(
        "--lowest-note", type=int, default=36, help="Lowest allowed MIDI note"
    )
    parser.add_argument(
        "--highest-note", type=int, default=96, help="Highest allowed MIDI note"
    )
    return parser.parse_args()


# --- Human-readable harmony preview ---------------------------------------


def run(args: argparse.Namespace) -> int:
    try:
        engine = VoicingEngine(args.lowest_note, args.highest_note)
        print(
            f"{args.tonic} {args.scale.replace('_', ' ')} | "
            f"{args.voicing} voicing | automatic inversions"
        )
        print("-" * 72)

        for step_number, step in enumerate(args.progression, start=1):
            is_dominant_seventh = step.chord_type == "dominant7"
            intent = ChordIntent(
                tonic=args.tonic,
                scale=args.scale,
                degree=step.degree,
                quality="major" if is_dominant_seventh else step.chord_type,
                extension="dominant7" if is_dominant_seventh else None,
                inversion=None,
                octave=args.octave,
                voicing_style=args.voicing,
            )
            notes = engine.voice(intent)
            label = f"{ROMAN_NUMERALS[step.degree]} {step.chord_type}"
            print(
                f"{step_number:>2}. {label:<15} "
                f"{format_chord(notes):<24} MIDI {notes}"
            )
    except ValueError as error:
        print(f"Harmony preview error: {error}", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
