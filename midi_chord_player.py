"""Safe MIDI-port selection and chord note lifecycle management."""

from typing import Iterable, List, Protocol, Sequence, Tuple

import mido
from mido import Message


# --- MIDI output boundary --------------------------------------------------


class MidiOutput(Protocol):
    """Small output interface implemented by Mido ports and test doubles."""

    def send(self, message: Message) -> None:
        """Send one MIDI message."""


def _numbered_port_alias(port_name: str) -> str:
    """Remove the numeric suffix Mido adds to some Windows device names."""

    base_name, separator, suffix = port_name.rpartition(" ")
    if separator and suffix.isdigit():
        return base_name
    return port_name


def resolve_output_port(requested_name: str, available_names: Sequence[str]) -> str:
    """Resolve an exact port or an unambiguous Windows-friendly alias.

    For example, loopMIDI may display ``Port1`` while Mido reports ``Port1 1``.
    Exact names always win; otherwise the trailing Mido number may be omitted.
    """

    requested = requested_name.strip()
    if not requested:
        raise ValueError("MIDI output port name cannot be empty")

    exact_matches = [
        name for name in available_names if name.casefold() == requested.casefold()
    ]
    if exact_matches:
        return exact_matches[0]

    alias_matches = [
        name
        for name in available_names
        if _numbered_port_alias(name).casefold() == requested.casefold()
    ]
    if len(alias_matches) == 1:
        return alias_matches[0]
    if len(alias_matches) > 1:
        matches = ", ".join(alias_matches)
        raise RuntimeError(
            f"MIDI output port '{requested}' is ambiguous. Matches: {matches}"
        )

    ports = ", ".join(available_names) if available_names else "none"
    raise RuntimeError(
        f"MIDI output port '{requested}' was not found. Available ports: {ports}"
    )


def open_midi_output(requested_name: str):
    """Open a requested Mido output and return both the port and resolved name."""

    try:
        available_names = mido.get_output_names()
        resolved_name = resolve_output_port(requested_name, available_names)
        return mido.open_output(resolved_name), resolved_name
    except (OSError, RuntimeError) as error:
        raise RuntimeError(str(error)) from error


# --- Chord note lifecycle --------------------------------------------------


class ChordMidiPlayer:
    """Send one active chord at a time without repeating held gestures."""

    def __init__(self, output: MidiOutput, channel: int = 1) -> None:
        if not 1 <= channel <= 16:
            raise ValueError("MIDI channel must be between 1 and 16")
        self.output = output
        # The command line uses musician-friendly channels 1-16; MIDI messages
        # and Mido encode those channels as zero-based values 0-15.
        self.channel = channel - 1
        self._active_notes: Tuple[int, ...] = ()

    @property
    def active_notes(self) -> Tuple[int, ...]:
        return self._active_notes

    @staticmethod
    def _validated_notes(notes: Iterable[int]) -> Tuple[int, ...]:
        # Preserve voice order while removing accidental duplicate note-ons.
        ordered_notes = tuple(dict.fromkeys(notes))
        if not ordered_notes:
            raise ValueError("A chord must contain at least one MIDI note")
        if any(not 0 <= note <= 127 for note in ordered_notes):
            raise ValueError("MIDI notes must be between 0 and 127")
        return ordered_notes

    def play_chord(self, notes: Iterable[int], velocity: int = 96) -> bool:
        """Replace the active chord, returning whether MIDI messages were sent."""

        validated_notes = self._validated_notes(notes)
        if not 1 <= velocity <= 127:
            raise ValueError("MIDI velocity must be between 1 and 127")
        if validated_notes == self._active_notes:
            return False

        previous_notes = self._active_notes
        previous_note_set = set(previous_notes)
        next_note_set = set(validated_notes)
        notes_to_start = tuple(
            note for note in validated_notes if note not in previous_note_set
        )
        notes_to_stop = tuple(
            note for note in previous_notes if note not in next_note_set
        )

        # Keep common tones sounding instead of reattacking the whole chord.
        # New voices start before obsolete voices stop, avoiding an all-notes-
        # off gap that can produce a click in synths with sharp envelopes.
        started_notes: List[int] = []
        try:
            for note in notes_to_start:
                self.output.send(
                    Message(
                        "note_on",
                        channel=self.channel,
                        note=note,
                        velocity=velocity,
                    )
                )
                started_notes.append(note)
            for note in notes_to_stop:
                self.output.send(
                    Message(
                        "note_off",
                        channel=self.channel,
                        note=note,
                        velocity=0,
                    )
                )
        except Exception:
            # Include old and newly started voices so stop() can safely release
            # everything after a partial transition.
            self._active_notes = tuple(
                dict.fromkeys((*previous_notes, *started_notes))
            )
            self.stop()
            raise

        self._active_notes = validated_notes
        return bool(notes_to_start or notes_to_stop)

    def stop(self) -> bool:
        """Release the active chord, returning whether notes needed releasing."""

        if not self._active_notes:
            return False

        notes_to_stop = self._active_notes
        self._active_notes = ()
        for note in notes_to_stop:
            self.output.send(
                Message(
                    "note_off",
                    channel=self.channel,
                    note=note,
                    velocity=0,
                )
            )
        return True

    def send_control_change(self, control: int, value: int) -> None:
        """Send one validated continuous-controller value on this MIDI channel."""

        if not 0 <= control <= 127:
            raise ValueError("MIDI control number must be between 0 and 127")
        if not 0 <= value <= 127:
            raise ValueError("MIDI control value must be between 0 and 127")
        self.output.send(
            Message(
                "control_change",
                channel=self.channel,
                control=control,
                value=value,
            )
        )

    def panic(self) -> None:
        """Release tracked notes and request that the synth silence the channel."""

        self.stop()
        self.output.send(
            Message(
                "control_change",
                channel=self.channel,
                control=123,
                value=0,
            )
        )
