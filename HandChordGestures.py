"""Preview two-hand chord gestures without sending MIDI notes.

This is Milestone 1 from the project README. It is intentionally visual-only so
gesture thresholds can be tested safely before note-on and note-off behavior is
introduced.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Set

import cv2
import mediapipe as mp

from gesture_classifier import GestureStabilizer, PoseAnalysis, analyze_hand


# --- Application constants -------------------------------------------------


DEFAULT_MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)
HAND_COLORS = {
    "Left": (80, 220, 80),
    "Right": (255, 180, 60),
    "Unknown": (220, 220, 220),
}


# --- Command-line configuration -------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview two-hand scale-degree gestures without MIDI output."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe gesture recognizer model",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.25,
        help="Seconds a pose must remain unchanged before it becomes stable",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum custom gesture confidence from 0.0 to 1.0",
    )
    args = parser.parse_args()

    if args.hold_seconds < 0.0:
        parser.error("--hold-seconds must be non-negative")
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0.0 and 1.0")
    return args


# --- Camera overlay helpers ------------------------------------------------


def draw_hand(frame, landmarks: Sequence, hand_name: str) -> None:
    height, width = frame.shape[:2]
    points = [
        (round(landmark.x * width), round(landmark.y * height))
        for landmark in landmarks
    ]
    color = HAND_COLORS.get(hand_name, HAND_COLORS["Unknown"])

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 2)
    # Draw every one of MediaPipe's 21 landmarks. Knuckles and finger joints
    # use small nodes, while the wrist and fingertips remain easy to spot.
    for index, point in enumerate(points):
        if index in FINGERTIP_INDICES:
            radius = 7
        elif index == 0:
            radius = 6
        else:
            radius = 4
        cv2.circle(frame, point, radius, color, -1)


def _finger_summary(analysis: PoseAnalysis) -> str:
    abbreviations = {
        "thumb": "T",
        "index": "I",
        "middle": "M",
        "ring": "R",
        "pinky": "P",
    }
    return "".join(
        abbreviations[finger_name] for finger_name in analysis.extended_fingers
    ) or "none"


def draw_status_panel(frame, status_lines: Sequence[str]) -> None:
    # A dark backing rectangle keeps labels readable over a bright camera feed.
    panel_height = 45 + 28 * len(status_lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)

    cv2.putText(
        frame,
        "Gesture preview - no MIDI notes are being sent",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    for row, status_line in enumerate(status_lines, start=1):
        cv2.putText(
            frame,
            status_line,
            (12, 28 + row * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            2,
        )


def _hand_name(result, hand_index: int) -> str:
    if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
        return f"Unknown-{hand_index + 1}"
    return result.handedness[hand_index][0].category_name


def _gesture_text(analysis: PoseAnalysis) -> str:
    if analysis.gesture is None:
        return "unrecognized"
    return f"{analysis.gesture.display_name} ({analysis.gesture.confidence:.0%})"


def _stable_text(stable_gesture) -> str:
    return stable_gesture.display_name if stable_gesture is not None else "waiting"


# --- Main two-hand tracking loop ------------------------------------------


def run(args: argparse.Namespace) -> int:
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Camera {args.camera} could not be opened.", file=sys.stderr)
        return 1

    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    stabilizers: Dict[str, GestureStabilizer] = {}
    start_time = time.perf_counter()
    last_timestamp_ms = -1

    try:
        with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
            while camera.isOpened():
                success, frame = camera.read()
                if not success:
                    print("Camera stopped returning frames.", file=sys.stderr)
                    return 1

                # Mirror the frame so moving left/right feels like looking into
                # a mirror and MediaPipe handedness labels match the user.
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    int((time.perf_counter() - start_time) * 1000),
                )
                last_timestamp_ms = timestamp_ms
                result = recognizer.recognize_for_video(mp_image, timestamp_ms)

                now = time.perf_counter()
                status_lines = []
                seen_hands: Set[str] = set()

                for hand_index, landmarks in enumerate(result.hand_landmarks):
                    hand_name = _hand_name(result, hand_index)
                    seen_hands.add(hand_name)
                    stabilizer = stabilizers.setdefault(
                        hand_name, GestureStabilizer(args.hold_seconds)
                    )
                    analysis = analyze_hand(landmarks)
                    observation = analysis.gesture
                    if (
                        observation is not None
                        and observation.confidence < args.min_confidence
                    ):
                        observation = None
                    stable_gesture = stabilizer.update(observation, now)

                    draw_hand(frame, landmarks, hand_name)
                    status_lines.append(
                        f"{hand_name}: raw {_gesture_text(analysis)} | "
                        f"stable {_stable_text(stable_gesture)} | "
                        f"fingers {_finger_summary(analysis)} {analysis.direction}"
                    )

                # Clear a hand's stable pose after it has been absent for the
                # configured hold period, avoiding stale labels on re-entry.
                for hand_name, stabilizer in stabilizers.items():
                    if hand_name not in seen_hands:
                        stabilizer.update(None, now)

                if not status_lines:
                    status_lines.append("No hands detected")
                draw_status_panel(frame, status_lines)
                cv2.imshow("Two-Hand Chord Gesture Preview", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
