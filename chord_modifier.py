"""Recognize and stabilize expression-hand seventh-chord pinch modifiers."""

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Protocol, Sequence

from gesture_music import (
    DOMINANT_SEVENTH_MODIFIER,
    QUALITY_SEVENTH_MODIFIER,
    SUPPORTED_SEVENTH_MODIFIERS,
)


# --- Pinch geometry --------------------------------------------------------


class Landmark(Protocol):
    x: float
    y: float


@dataclass(frozen=True)
class PinchAnalysis:
    """Normalized pinch distances and the currently observed modifier."""

    index_ratio: float
    middle_ratio: float
    ring_ratio: float
    modifier: Optional[str]


MODIFIER_LABELS = {
    None: "Triad",
    QUALITY_SEVENTH_MODIFIER: "Quality 7",
    DOMINANT_SEVENTH_MODIFIER: "Dominant 7",
}
DEFAULT_PINCH_ENGAGE_THRESHOLD = 0.30
DEFAULT_PINCH_RELEASE_THRESHOLD = 0.45


def _distance(first: Landmark, second: Landmark) -> float:
    return sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2)


def analyze_seventh_pinch(
    landmarks: Sequence[Landmark],
    *,
    active_modifier: Optional[str] = None,
    engage_threshold: float = DEFAULT_PINCH_ENGAGE_THRESHOLD,
    release_threshold: float = DEFAULT_PINCH_RELEASE_THRESHOLD,
) -> PinchAnalysis:
    """Recognize thumb-index or thumb-middle-ring pinch using palm-width ratios.

    An active modifier uses the larger release threshold. This hysteresis keeps
    a held pinch stable even when fingertip landmarks jitter near the boundary.
    """

    if len(landmarks) != 21:
        raise ValueError("Expected exactly 21 hand landmarks")
    if active_modifier not in SUPPORTED_SEVENTH_MODIFIERS:
        raise ValueError("Unsupported active seventh modifier")
    if not 0.0 < engage_threshold < release_threshold:
        raise ValueError("Pinch thresholds must satisfy 0 < engage < release")

    palm_width = max(_distance(landmarks[5], landmarks[17]), 0.001)
    thumb_tip = landmarks[4]
    index_ratio = _distance(thumb_tip, landmarks[8]) / palm_width
    middle_ratio = _distance(thumb_tip, landmarks[12]) / palm_width
    ring_ratio = _distance(thumb_tip, landmarks[16]) / palm_width

    # Dominant 7 deliberately requires both the middle and ring fingertips to
    # meet the thumb. Using the larger distance means one stray fingertip cannot
    # make an ordinary thumb-middle movement look like the TMR gesture.
    tmr_ratio = max(middle_ratio, ring_ratio)
    ratios = {
        QUALITY_SEVENTH_MODIFIER: index_ratio,
        DOMINANT_SEVENTH_MODIFIER: tmr_ratio,
    }

    if (
        active_modifier is not None
        and ratios[active_modifier] <= release_threshold
    ):
        modifier = active_modifier
    else:
        engaged = [
            (ratio, modifier_name)
            for modifier_name, ratio in ratios.items()
            if ratio <= engage_threshold
        ]
        modifier = min(engaged)[1] if engaged else None

    return PinchAnalysis(
        index_ratio=index_ratio,
        middle_ratio=middle_ratio,
        ring_ratio=ring_ratio,
        modifier=modifier,
    )


# --- Temporal stabilization ------------------------------------------------


class PinchModifierStabilizer:
    """Accept a modifier only after the observation remains stable briefly."""

    def __init__(self, hold_seconds: float = 0.12) -> None:
        if hold_seconds < 0.0:
            raise ValueError("hold_seconds must be non-negative")
        self.hold_seconds = hold_seconds
        self._candidate: Optional[str] = None
        self._candidate_since = 0.0
        self._stable: Optional[str] = None

    @property
    def stable_modifier(self) -> Optional[str]:
        return self._stable

    def update(self, observation: Optional[str], timestamp: float) -> Optional[str]:
        """Return the stable modifier after considering one detected-hand frame."""

        if observation not in SUPPORTED_SEVENTH_MODIFIERS:
            raise ValueError("Unsupported seventh modifier observation")
        if observation != self._candidate:
            self._candidate = observation
            self._candidate_since = timestamp

        if timestamp - self._candidate_since >= self.hold_seconds:
            self._stable = self._candidate
        return self._stable
