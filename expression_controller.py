"""Camera-free mapping and smoothing for continuous expression controls."""

from typing import Optional, Protocol, Sequence


# --- Landmark-to-controller mapping ---------------------------------------


class VerticalLandmark(Protocol):
    """Smallest landmark interface needed for expression-hand height."""

    y: float


PALM_LANDMARK_INDICES = (0, 5, 9, 13, 17)


def palm_vertical_position(landmarks: Sequence[VerticalLandmark]) -> float:
    """Return the average normalized height of the wrist and palm knuckles."""

    if len(landmarks) != 21:
        raise ValueError("Expected exactly 21 hand landmarks")
    return sum(landmarks[index].y for index in PALM_LANDMARK_INDICES) / len(
        PALM_LANDMARK_INDICES
    )


def vertical_position_to_midi(
    position: float,
    *,
    top: float = 0.15,
    bottom: float = 0.85,
) -> int:
    """Map hand height to MIDI 0-127, with a raised hand producing 127."""

    if not 0.0 <= top < bottom <= 1.0:
        raise ValueError("Expression range must satisfy 0 <= top < bottom <= 1")

    clipped_position = max(top, min(bottom, position))
    normalized_height = (bottom - clipped_position) / (bottom - top)
    return round(normalized_height * 127)


# --- Temporal smoothing and output throttling -----------------------------


class SmoothedMidiControl:
    """Smooth MIDI values and emit only meaningful, rate-limited changes."""

    def __init__(
        self,
        *,
        smoothing: float = 0.25,
        dead_zone: int = 2,
        max_rate_hz: float = 30.0,
    ) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be greater than 0 and at most 1")
        if not 0 <= dead_zone <= 127:
            raise ValueError("dead_zone must be between 0 and 127")
        if max_rate_hz <= 0.0:
            raise ValueError("max_rate_hz must be positive")

        self.smoothing = smoothing
        self.dead_zone = dead_zone
        self.minimum_interval = 1.0 / max_rate_hz
        self._smoothed_value: Optional[float] = None
        self._last_sent_value: Optional[int] = None
        self._last_sent_at: Optional[float] = None

    @property
    def current_value(self) -> Optional[int]:
        """Return the current smoothed value, whether or not it was emitted."""

        if self._smoothed_value is None:
            return None
        return round(self._smoothed_value)

    @property
    def last_sent_value(self) -> Optional[int]:
        return self._last_sent_value

    def reset(self) -> None:
        """Forget smoothing and output history for a new performance session."""

        self._smoothed_value = None
        self._last_sent_value = None
        self._last_sent_at = None

    def update(self, raw_value: int, timestamp: float) -> Optional[int]:
        """Return a MIDI value when one should be sent, otherwise return None."""

        if not 0 <= raw_value <= 127:
            raise ValueError("raw_value must be between 0 and 127")

        if self._smoothed_value is None:
            self._smoothed_value = float(raw_value)
        else:
            self._smoothed_value += self.smoothing * (
                raw_value - self._smoothed_value
            )
        candidate = round(self._smoothed_value)

        if (
            self._last_sent_value is not None
            and abs(candidate - self._last_sent_value) < self.dead_zone
        ):
            return None
        if (
            self._last_sent_at is not None
            and timestamp - self._last_sent_at < self.minimum_interval
        ):
            return None

        self._last_sent_value = candidate
        self._last_sent_at = timestamp
        return candidate
