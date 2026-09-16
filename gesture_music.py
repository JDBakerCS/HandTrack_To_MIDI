"""Translate recognized hand gestures and modifiers into harmony requests.

This small boundary keeps camera geometry out of the harmony engine and keeps
MIDI note construction out of the camera loop. It is deliberately camera-free
so every gesture-to-chord decision can be tested with ordinary unit tests.
"""

from typing import Optional

from gesture_classifier import ChordGesture
from harmony_engine import ChordIntent


QUALITY_SEVENTH_MODIFIER = "quality7"
DOMINANT_SEVENTH_MODIFIER = "dominant7"
SUPPORTED_SEVENTH_MODIFIERS = (
    None,
    QUALITY_SEVENTH_MODIFIER,
    DOMINANT_SEVENTH_MODIFIER,
)


def chord_intent_from_gesture(
    gesture: ChordGesture,
    *,
    tonic: str = "C",
    scale: str = "major",
    octave: int = 4,
    voicing_style: str = "close",
    velocity: int = 96,
    dominant_seventh: bool = False,
    seventh_modifier: Optional[str] = None,
) -> ChordIntent:
    """Build a musical intent from one stabilized selector-hand gesture.

    The pose classifier uses display-friendly title case (``Major`` and
    ``Minor``), while the harmony engine uses lowercase configuration values.
    An explicit expression-hand modifier overrides the legacy automatic-V7
    option. ``quality7`` follows the selector pose, producing major7 or minor7.
    """

    quality = gesture.quality.lower()
    if seventh_modifier not in SUPPORTED_SEVENTH_MODIFIERS:
        choices = ", ".join(
            modifier
            for modifier in SUPPORTED_SEVENTH_MODIFIERS
            if modifier is not None
        )
        raise ValueError(f"seventh_modifier must be one of: none, {choices}")

    if seventh_modifier == QUALITY_SEVENTH_MODIFIER:
        extension = "major7" if quality == "major" else "minor7"
    elif seventh_modifier == DOMINANT_SEVENTH_MODIFIER:
        if quality != "major":
            raise ValueError("Dominant seventh modifier requires a major pose")
        extension = "dominant7"
    else:
        extension = (
            "dominant7"
            if dominant_seventh and gesture.degree == 5 and quality == "major"
            else None
        )

    return ChordIntent(
        tonic=tonic,
        scale=scale,
        degree=gesture.degree,
        quality=quality,
        extension=extension,
        inversion=None,
        octave=octave,
        voicing_style=voicing_style,
        velocity=velocity,
    )
