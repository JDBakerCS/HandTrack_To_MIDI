"""Rule-based hand-pose classification for scale-degree gestures.

This module deliberately has no OpenCV, MediaPipe, or MIDI imports. Keeping the
geometry separate makes it possible to test and tune gesture rules without a
camera or music software running.
"""

from dataclasses import dataclass
from math import acos, degrees, sqrt
from typing import Dict, Optional, Protocol, Sequence, Tuple


# --- Public data shapes ----------------------------------------------------


class Landmark(Protocol):
    """Smallest landmark interface required from MediaPipe or a test fixture."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ChordGesture:
    """A recognized scale-degree gesture before it is converted into MIDI."""

    degree: int
    roman_numeral: str
    quality: str
    pose_name: str
    confidence: float

    @property
    def display_name(self) -> str:
        return f"{self.roman_numeral} {self.quality}"


@dataclass(frozen=True)
class PoseAnalysis:
    """Debug information used by the camera overlay and threshold tuning."""

    extended_fingers: Tuple[str, ...]
    direction: str
    vulcan_gap_ratio: float
    gesture: Optional[ChordGesture]


# --- MediaPipe landmark layout and gesture vocabulary ---------------------


FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
LONG_FINGER_ORDER = ("index", "middle", "ring", "pinky")

# Each long finger stores MCP, PIP, DIP, and fingertip landmark indices.
LONG_FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

ROMAN_NUMERALS = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
}

# Exact extended-finger patterns for I-IV and VI. Degree III also accepts a
# relaxed thumb because live testing showed the intended IMR pose as TIMR. V
# and VII both use all five fingers, so their fingertip gap is checked below.
POSE_PATTERNS = {
    ("index",): (1, "Index"),
    ("index", "middle"): (2, "Index + Middle"),
    ("index", "middle", "ring"): (3, "Three Fingers"),
    ("thumb", "index", "middle", "ring"): (3, "Three Fingers"),
    ("index", "middle", "ring", "pinky"): (4, "Four Fingers"),
    ("index", "pinky"): (6, "Index + Pinky"),
}

# These are intentionally named constants because camera testing will likely
# reveal that one or more thresholds should be adjusted for the user's hands.
FINGER_EXTENDED_THRESHOLD = 0.55
VERTICAL_STRENGTH_THRESHOLD = 0.45
VERTICAL_DOMINANCE_THRESHOLD = 0.60
VULCAN_GAP_RATIO_THRESHOLD = 1.65


# --- Geometry helpers ------------------------------------------------------


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _distance(first: Landmark, second: Landmark, include_z: bool = True) -> float:
    z_distance = (first.z - second.z) if include_z else 0.0
    return sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + z_distance**2
    )


def _joint_angle(first: Landmark, middle: Landmark, last: Landmark) -> float:
    """Return the angle at ``middle`` in degrees, including landmark depth."""

    first_vector = (
        first.x - middle.x,
        first.y - middle.y,
        first.z - middle.z,
    )
    last_vector = (
        last.x - middle.x,
        last.y - middle.y,
        last.z - middle.z,
    )
    first_length = sqrt(sum(component**2 for component in first_vector))
    last_length = sqrt(sum(component**2 for component in last_vector))
    if first_length == 0.0 or last_length == 0.0:
        return 0.0

    cosine = sum(
        first_component * last_component
        for first_component, last_component in zip(first_vector, last_vector)
    ) / (first_length * last_length)
    return degrees(acos(_clamp(cosine, -1.0, 1.0)))


def _angle_to_extension_score(angle: float) -> float:
    """Map a bent-to-straight joint angle onto a zero-to-one score."""

    return _clamp((angle - 120.0) / 45.0)


# --- Finger state and direction analysis ----------------------------------


def _finger_extension_scores(landmarks: Sequence[Landmark]) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    for finger_name, (mcp, pip, dip, tip) in LONG_FINGER_LANDMARKS.items():
        pip_angle = _joint_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
        dip_angle = _joint_angle(landmarks[pip], landmarks[dip], landmarks[tip])
        scores[finger_name] = min(
            _angle_to_extension_score(pip_angle),
            _angle_to_extension_score(dip_angle),
        )

    # A straight thumb can still lie across the palm. Requiring separation from
    # the index knuckle prevents that tucked position from counting as extended.
    thumb_angle = _joint_angle(landmarks[2], landmarks[3], landmarks[4])
    palm_width = max(_distance(landmarks[5], landmarks[17], include_z=False), 0.001)
    thumb_separation = _distance(
        landmarks[4], landmarks[5], include_z=False
    ) / palm_width
    thumb_separation_score = _clamp((thumb_separation - 0.35) / 0.45)
    scores["thumb"] = min(
        _angle_to_extension_score(thumb_angle), thumb_separation_score
    )
    return scores


def _finger_direction(
    landmarks: Sequence[Landmark], extended_fingers: Tuple[str, ...]
) -> Tuple[str, float]:
    """Classify the long fingers as pointing up, down, or sideways."""

    active_fingers = [
        finger_name
        for finger_name in extended_fingers
        if finger_name in LONG_FINGER_LANDMARKS
    ]
    if not active_fingers:
        return "Sideways", 0.0

    x_distance = 0.0
    y_distance = 0.0
    for finger_name in active_fingers:
        mcp, _, _, tip = LONG_FINGER_LANDMARKS[finger_name]
        x_distance += landmarks[tip].x - landmarks[mcp].x
        y_distance += landmarks[tip].y - landmarks[mcp].y

    x_distance /= len(active_fingers)
    y_distance /= len(active_fingers)
    palm_length = max(
        _distance(landmarks[0], landmarks[9], include_z=False), 0.001
    )
    vertical_strength = abs(y_distance) / palm_length
    vertical_dominance = abs(y_distance) / max(
        abs(x_distance) + abs(y_distance), 0.001
    )

    if (
        vertical_strength < VERTICAL_STRENGTH_THRESHOLD
        or vertical_dominance < VERTICAL_DOMINANCE_THRESHOLD
    ):
        return "Sideways", _clamp(vertical_dominance)

    direction = "Up" if y_distance < 0.0 else "Down"
    confidence = (
        _clamp(vertical_strength / 1.5) + _clamp(vertical_dominance)
    ) / 2.0
    return direction, confidence


def _vulcan_gap_ratio(landmarks: Sequence[Landmark]) -> float:
    middle_ring_gap = _distance(landmarks[12], landmarks[16], include_z=False)
    index_middle_gap = _distance(landmarks[8], landmarks[12], include_z=False)
    ring_pinky_gap = _distance(landmarks[16], landmarks[20], include_z=False)
    neighboring_gap = max((index_middle_gap + ring_pinky_gap) / 2.0, 0.001)
    return middle_ring_gap / neighboring_gap


# --- Public gesture classifier --------------------------------------------


def analyze_hand(landmarks: Sequence[Landmark]) -> PoseAnalysis:
    """Analyze 21 MediaPipe landmarks and classify the project's custom pose."""

    if len(landmarks) != 21:
        raise ValueError("Expected exactly 21 hand landmarks")

    extension_scores = _finger_extension_scores(landmarks)
    extended_fingers = tuple(
        finger_name
        for finger_name in FINGER_ORDER
        if extension_scores[finger_name] >= FINGER_EXTENDED_THRESHOLD
    )
    direction, direction_confidence = _finger_direction(
        landmarks, extended_fingers
    )
    vulcan_ratio = _vulcan_gap_ratio(landmarks)

    degree_and_name = POSE_PATTERNS.get(extended_fingers)
    gap_confidence = 1.0

    if extended_fingers == FINGER_ORDER:
        if vulcan_ratio >= VULCAN_GAP_RATIO_THRESHOLD:
            degree_and_name = (7, "Vulcan")
        else:
            degree_and_name = (5, "Open Hand")

        # Confidence rises as the observed gap moves farther from the V/VII
        # decision boundary in either direction.
        gap_confidence = _clamp(
            0.5 + abs(vulcan_ratio - VULCAN_GAP_RATIO_THRESHOLD) / 2.0
        )

    gesture: Optional[ChordGesture] = None
    if degree_and_name is not None and direction in ("Up", "Down"):
        degree, pose_name = degree_and_name
        expected_fingers = set(extended_fingers)
        pattern_confidence = sum(
            extension_scores[finger_name]
            if finger_name in expected_fingers
            else 1.0 - extension_scores[finger_name]
            for finger_name in FINGER_ORDER
        ) / len(FINGER_ORDER)
        overall_confidence = (
            pattern_confidence + direction_confidence + gap_confidence
        ) / 3.0
        # VI is intentionally inverted because vi is naturally minor in a
        # major scale and the upward pose is more comfortable to hold.
        if degree == 6:
            quality = "Minor" if direction == "Up" else "Major"
        else:
            quality = "Major" if direction == "Up" else "Minor"

        gesture = ChordGesture(
            degree=degree,
            roman_numeral=ROMAN_NUMERALS[degree],
            quality=quality,
            pose_name=pose_name,
            confidence=round(overall_confidence, 2),
        )

    return PoseAnalysis(
        extended_fingers=extended_fingers,
        direction=direction,
        vulcan_gap_ratio=round(vulcan_ratio, 2),
        gesture=gesture,
    )


# --- Temporal stabilization ------------------------------------------------


class GestureStabilizer:
    """Accept a gesture only after it remains unchanged for a short hold time."""

    def __init__(self, hold_seconds: float = 0.25) -> None:
        if hold_seconds < 0.0:
            raise ValueError("hold_seconds must be non-negative")
        self.hold_seconds = hold_seconds
        self._candidate: Optional[ChordGesture] = None
        self._candidate_since = 0.0
        self._stable: Optional[ChordGesture] = None

    @staticmethod
    def _gesture_key(gesture: Optional[ChordGesture]) -> Optional[Tuple[int, str]]:
        if gesture is None:
            return None
        return gesture.degree, gesture.quality

    def update(
        self, observation: Optional[ChordGesture], timestamp: float
    ) -> Optional[ChordGesture]:
        """Return the current stable gesture after considering one observation."""

        if self._gesture_key(observation) != self._gesture_key(self._candidate):
            self._candidate = observation
            self._candidate_since = timestamp
        else:
            # Keep the latest confidence while preserving the original timer.
            self._candidate = observation

        if timestamp - self._candidate_since >= self.hold_seconds:
            self._stable = self._candidate
        return self._stable
