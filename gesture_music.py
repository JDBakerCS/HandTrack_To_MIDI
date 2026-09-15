"""Translate recognized hand gestures into harmony-engine requests.

This small boundary keeps camera geometry out of the harmony engine and keeps
MIDI note construction out of the camera loop. It is deliberately camera-free
so every gesture-to-chord decision can be tested with ordinary unit tests.
"""

from gesture_classifier import ChordGesture
from harmony_engine import ChordIntent


def chord_intent_from_gesture(
    gesture: ChordGesture,
    *,
    tonic: str = "C",
    scale: str = "major",
    octave: int = 4,
    voicing_style: str = "close",
    velocity: int = 96,
    dominant_seventh: bool = False,
) -> ChordIntent:
    """Build a musical intent from one stabilized selector-hand gesture.

    The pose classifier uses display-friendly title case (``Major`` and
    ``Minor``), while the harmony engine uses lowercase configuration values.
    A dominant seventh is optional because there is not yet a separate gesture
    that distinguishes V major from V7.
    """

    quality = gesture.quality.lower()
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
