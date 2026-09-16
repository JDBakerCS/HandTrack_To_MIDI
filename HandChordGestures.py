"""Preview or perform two-hand chord gestures.

Without ``--port`` this remains the safe Milestone 1 visual preview. Supplying a
MIDI output port enables Milestone 2 chord playback through the harmony engine.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Set

import cv2
import mediapipe as mp

from chord_modifier import (
    DEFAULT_PINCH_ENGAGE_THRESHOLD,
    DEFAULT_PINCH_RELEASE_THRESHOLD,
    MODIFIER_LABELS,
    PinchModifierStabilizer,
    analyze_seventh_pinch,
)
from expression_controller import (
    SmoothedMidiControl,
    palm_vertical_position,
    vertical_position_to_midi,
)
from gesture_classifier import (
    ChordGesture,
    GestureStabilizer,
    PoseAnalysis,
    analyze_hand,
)
from gesture_music import InvalidChordModifierError, chord_intent_from_gesture
from harmony_engine import VoicingEngine, format_chord, normalize_note_name
from midi_chord_player import ChordMidiPlayer, open_midi_output


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

# OpenCV colors use blue-green-red order. Each anatomical group receives one
# color so finger geometry remains readable even when two hands overlap.
LANDMARK_GROUPS = (
    ("Wrist", (0,), (255, 255, 255)),
    ("Thumb", (1, 2, 3, 4), (255, 80, 255)),
    ("Index", (5, 6, 7, 8), (80, 255, 80)),
    ("Middle", (9, 10, 11, 12), (255, 255, 80)),
    ("Ring", (13, 14, 15, 16), (80, 170, 255)),
    ("Pinky", (17, 18, 19, 20), (255, 100, 120)),
)
LANDMARK_COLORS = {
    landmark_index: color
    for _, landmark_indices, color in LANDMARK_GROUPS
    for landmark_index in landmark_indices
}


# --- Command-line configuration -------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview gestures or perform their chords through MIDI."
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
        default=0.15,
        help="Seconds a pose must remain unchanged (default: 0.15)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum custom gesture confidence from 0.0 to 1.0 (default: 0.55)",
    )
    parser.add_argument(
        "--port",
        help="MIDI output port; omit this option for visual-only preview",
    )
    parser.add_argument(
        "--channel",
        type=int,
        choices=range(1, 17),
        default=1,
        metavar="1-16",
        help="MIDI channel (default: 1)",
    )
    parser.add_argument(
        "--selector-hand",
        choices=("Left", "Right"),
        default="Right",
        help="Hand that selects chords; the other hand is reserved for expression",
    )
    parser.add_argument(
        "--minor-direction",
        choices=("down", "sideways"),
        default="sideways",
        help="Direction used for minor gestures I-V and VII (default: sideways)",
    )
    parser.add_argument(
        "--expression-cc",
        type=int,
        choices=range(128),
        default=74,
        metavar="0-127",
        help="MIDI CC controlled by expression-hand height (default: 74)",
    )
    parser.add_argument(
        "--expression-smoothing",
        type=float,
        default=0.25,
        help="Expression smoothing from 0 exclusive to 1 inclusive (default: 0.25)",
    )
    parser.add_argument(
        "--expression-dead-zone",
        type=int,
        default=2,
        help="Minimum MIDI-value change before sending expression (default: 2)",
    )
    parser.add_argument(
        "--expression-rate",
        type=float,
        default=30.0,
        help="Maximum expression CC messages per second (default: 30)",
    )
    parser.add_argument(
        "--expression-top",
        type=float,
        default=0.15,
        help="Top of the expression hand's active camera range (default: 0.15)",
    )
    parser.add_argument(
        "--expression-bottom",
        type=float,
        default=0.85,
        help="Bottom of the expression hand's active camera range (default: 0.85)",
    )
    parser.add_argument(
        "--modifier-hold-seconds",
        type=float,
        default=0.12,
        help="Seconds a seventh pinch must remain stable (default: 0.12)",
    )
    parser.add_argument(
        "--pinch-engage",
        type=float,
        default=DEFAULT_PINCH_ENGAGE_THRESHOLD,
        help="Normalized distance that engages a seventh pinch (default: 0.30)",
    )
    parser.add_argument(
        "--pinch-release",
        type=float,
        default=DEFAULT_PINCH_RELEASE_THRESHOLD,
        help="Normalized distance that releases a seventh pinch (default: 0.45)",
    )
    parser.add_argument("--tonic", default="C", help="Tonic note, such as C or F#")
    parser.add_argument(
        "--scale",
        choices=("major", "natural_minor"),
        default="major",
        help="Scale used to place degrees I-VII",
    )
    parser.add_argument(
        "--octave",
        type=int,
        choices=range(-1, 8),
        default=4,
        metavar="-1-7",
        help="Root octave before automatic inversion (default: 4)",
    )
    parser.add_argument(
        "--voicing",
        choices=("close", "open"),
        default="close",
        help="Chord spacing style (default: close)",
    )
    parser.add_argument(
        "--velocity",
        type=int,
        choices=range(1, 128),
        default=96,
        metavar="1-127",
        help="Chord note velocity (default: 96)",
    )
    parser.add_argument(
        "--lowest-note",
        type=int,
        default=36,
        help="Lowest MIDI note allowed in automatic voicings (default: 36)",
    )
    parser.add_argument(
        "--highest-note",
        type=int,
        default=96,
        help="Highest MIDI note allowed in automatic voicings (default: 96)",
    )
    parser.add_argument(
        "--dominant-seven",
        action="store_true",
        help="Play the major V gesture as a dominant seventh chord",
    )
    args = parser.parse_args()

    if args.hold_seconds < 0.0:
        parser.error("--hold-seconds must be non-negative")
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0.0 and 1.0")
    if not 0 <= args.lowest_note < args.highest_note <= 127:
        parser.error("MIDI range must satisfy 0 <= lowest-note < highest-note <= 127")
    if not 0.0 < args.expression_smoothing <= 1.0:
        parser.error("--expression-smoothing must be greater than 0 and at most 1")
    if not 0 <= args.expression_dead_zone <= 127:
        parser.error("--expression-dead-zone must be between 0 and 127")
    if args.expression_rate <= 0.0:
        parser.error("--expression-rate must be positive")
    if not 0.0 <= args.expression_top < args.expression_bottom <= 1.0:
        parser.error(
            "Expression range must satisfy 0 <= expression-top "
            "< expression-bottom <= 1"
        )
    if args.modifier_hold_seconds < 0.0:
        parser.error("--modifier-hold-seconds must be non-negative")
    if not 0.0 < args.pinch_engage < args.pinch_release:
        parser.error("Pinch thresholds must satisfy 0 < pinch-engage < pinch-release")
    try:
        args.tonic = normalize_note_name(args.tonic)
    except ValueError as error:
        parser.error(str(error))
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
    # Draw every one of MediaPipe's 21 landmarks. Anatomical colors identify
    # each finger; node size distinguishes joints from the wrist and fingertips.
    for index, point in enumerate(points):
        if index in FINGERTIP_INDICES:
            radius = 7
        elif index == 0:
            radius = 6
        else:
            radius = 4
        cv2.circle(frame, point, radius, LANDMARK_COLORS[index], -1)
        cv2.circle(frame, point, radius, (25, 25, 25), 1)


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


def draw_status_panel(
    frame, status_lines: Sequence[str], heading: str
) -> None:
    # A dark backing rectangle keeps labels readable over a bright camera feed.
    panel_height = 72 + 28 * len(status_lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)

    cv2.putText(
        frame,
        heading,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    # The compact legend uses the same colors as the landmark nodes.
    legend_x = 16
    legend_y = 52
    for label, _, color in LANDMARK_GROUPS:
        cv2.circle(frame, (legend_x, legend_y - 5), 5, color, -1)
        cv2.putText(
            frame,
            label,
            (legend_x + 10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
        )
        label_width = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )[0][0]
        legend_x += label_width + 27

    for row, status_line in enumerate(status_lines, start=1):
        cv2.putText(
            frame,
            status_line,
            (12, 52 + row * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            2,
        )


def draw_expression_meter(
    frame,
    *,
    top: float,
    bottom: float,
    value: Optional[int],
    control: int,
) -> None:
    """Draw the active vertical range and current expression value."""

    frame_height, frame_width = frame.shape[:2]
    top_y = round(top * frame_height)
    bottom_y = round(bottom * frame_height)
    meter_x = frame_width - 28

    cv2.rectangle(
        frame,
        (meter_x - 8, top_y),
        (meter_x + 8, bottom_y),
        (25, 25, 25),
        -1,
    )
    cv2.line(frame, (meter_x, top_y), (meter_x, bottom_y), (210, 210, 210), 2)
    cv2.putText(
        frame,
        f"CC{control}",
        (meter_x - 48, max(18, top_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "127",
        (meter_x - 44, top_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (210, 210, 210),
        1,
    )
    cv2.putText(
        frame,
        "0",
        (meter_x - 22, bottom_y + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (210, 210, 210),
        1,
    )

    if value is not None:
        marker_y = round(bottom_y - (value / 127.0) * (bottom_y - top_y))
        cv2.circle(frame, (meter_x, marker_y), 8, (80, 255, 255), -1)
        cv2.circle(frame, (meter_x, marker_y), 8, (25, 25, 25), 1)


def _hand_name(result, hand_index: int) -> str:
    if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
        return f"Unknown-{hand_index + 1}"
    return result.handedness[hand_index][0].category_name


def _gesture_text(analysis: PoseAnalysis) -> str:
    if analysis.gesture is None:
        return "unrecognized"
    return f"{analysis.gesture.display_name} ({analysis.gesture.confidence:.0%})"


def _stable_text(stable_gesture: Optional[ChordGesture]) -> str:
    return stable_gesture.display_name if stable_gesture is not None else "waiting"


# --- Main two-hand tracking loop ------------------------------------------


def run(args: argparse.Namespace) -> int:
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    camera = None
    midi_output = None
    player: Optional[ChordMidiPlayer] = None
    resolved_port = ""

    try:
        if args.port:
            midi_output, resolved_port = open_midi_output(args.port)
            player = ChordMidiPlayer(midi_output, channel=args.channel)
            print(f"Opened MIDI output port: {resolved_port}")

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
        voicing_engine = VoicingEngine(args.lowest_note, args.highest_note)
        expression_control = SmoothedMidiControl(
            smoothing=args.expression_smoothing,
            dead_zone=args.expression_dead_zone,
            max_rate_hz=args.expression_rate,
        )
        modifier_stabilizer = PinchModifierStabilizer(
            args.modifier_hold_seconds
        )
        expression_hand = "Left" if args.selector_hand == "Right" else "Right"
        last_selector_key = None
        active_chord_text = "none"
        start_time = time.perf_counter()
        last_timestamp_ms = -1

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
                stable_gestures: Dict[str, Optional[ChordGesture]] = {}
                stable_modifier = modifier_stabilizer.stable_modifier
                modifier_warning = ""

                for hand_index, landmarks in enumerate(result.hand_landmarks):
                    hand_name = _hand_name(result, hand_index)
                    seen_hands.add(hand_name)
                    draw_hand(frame, landmarks, hand_name)

                    if hand_name == args.selector_hand:
                        stabilizer = stabilizers.setdefault(
                            hand_name, GestureStabilizer(args.hold_seconds)
                        )
                        analysis = analyze_hand(
                            landmarks, minor_direction=args.minor_direction
                        )
                        observation = analysis.gesture
                        if (
                            observation is not None
                            and observation.confidence < args.min_confidence
                        ):
                            observation = None
                        stable_gesture = stabilizer.update(observation, now)
                        stable_gestures[hand_name] = stable_gesture
                        status_lines.append(
                            f"{hand_name} [selector]: raw {_gesture_text(analysis)} | "
                            f"stable {_stable_text(stable_gesture)} | "
                            f"fingers {_finger_summary(analysis)} {analysis.direction}"
                        )
                    elif hand_name == expression_hand:
                        palm_y = palm_vertical_position(landmarks)
                        raw_value = vertical_position_to_midi(
                            palm_y,
                            top=args.expression_top,
                            bottom=args.expression_bottom,
                        )
                        expression_value = expression_control.update(raw_value, now)
                        if player is not None and expression_value is not None:
                            player.send_control_change(
                                args.expression_cc, expression_value
                            )
                        pinch_analysis = analyze_seventh_pinch(
                            landmarks,
                            active_modifier=stable_modifier,
                            engage_threshold=args.pinch_engage,
                            release_threshold=args.pinch_release,
                        )
                        stable_modifier = modifier_stabilizer.update(
                            pinch_analysis.modifier, now
                        )
                        status_lines.append(
                            f"{hand_name} [expression]: "
                            f"CC{args.expression_cc} {expression_control.current_value} | "
                            f"7th {MODIFIER_LABELS[stable_modifier]} | "
                            f"pinch I {pinch_analysis.index_ratio:.2f} "
                            f"M {pinch_analysis.middle_ratio:.2f} "
                            f"R {pinch_analysis.ring_ratio:.2f}"
                        )
                    else:
                        status_lines.append(f"{hand_name} [unassigned]")

                # Clear a hand's stable pose after it has been absent for the
                # configured hold period, avoiding stale labels on re-entry.
                for hand_name, stabilizer in stabilizers.items():
                    if hand_name not in seen_hands:
                        stable_gestures[hand_name] = stabilizer.update(None, now)

                # Only the configured selector hand owns harmony. A stable pose
                # is converted to intent once, voiced once, and sent once.
                selector_gesture = stable_gestures.get(args.selector_hand)
                selector_key = (
                    (
                        selector_gesture.degree,
                        selector_gesture.quality,
                        stable_modifier,
                    )
                    if selector_gesture is not None
                    else None
                )
                if player is not None:
                    if selector_key is None:
                        if player.stop():
                            print("Chord off")
                        last_selector_key = None
                        active_chord_text = "none"
                    elif selector_key != last_selector_key:
                        try:
                            intent = chord_intent_from_gesture(
                                selector_gesture,
                                tonic=args.tonic,
                                scale=args.scale,
                                octave=args.octave,
                                voicing_style=args.voicing,
                                velocity=args.velocity,
                                dominant_seventh=args.dominant_seven,
                                seventh_modifier=stable_modifier,
                            )
                        except InvalidChordModifierError as error:
                            # Keep the previous chord sounding until the player
                            # releases the pinch or selects a compatible pose.
                            modifier_warning = str(error)
                        else:
                            notes = voicing_engine.voice(intent)
                            player.play_chord(notes, velocity=intent.velocity)
                            last_selector_key = selector_key
                            chord_type = {
                                "major7": "Major 7",
                                "minor7": "Minor 7",
                                "dominant7": "Dominant 7",
                            }.get(intent.extension, selector_gesture.quality)
                            active_chord_text = (
                                f"{selector_gesture.roman_numeral} {chord_type}: "
                                f"{format_chord(notes)}"
                            )
                            print(f"Chord on: {active_chord_text} | MIDI {notes}")

                if not status_lines:
                    status_lines.append("No hands detected")
                if player is not None:
                    status_lines.insert(0, f"Active chord: {active_chord_text}")
                    status_lines.insert(
                        1,
                        f"Seventh modifier: {MODIFIER_LABELS[stable_modifier]}",
                    )
                    if modifier_warning:
                        status_lines.insert(2, f"Modifier warning: {modifier_warning}")
                    heading = (
                        f"MIDI: {resolved_port} | chord: {args.selector_hand} | "
                        f"CC{args.expression_cc}: {expression_hand}"
                    )
                else:
                    heading = "Gesture preview - no MIDI notes are being sent"
                draw_status_panel(frame, status_lines, heading)
                draw_expression_meter(
                    frame,
                    top=args.expression_top,
                    bottom=args.expression_bottom,
                    value=expression_control.current_value,
                    control=args.expression_cc,
                )
                cv2.imshow("Two-Hand Chord Gestures", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Performance stopped: {error}", file=sys.stderr)
        return 1
    finally:
        # Panic runs before closing the port so both tracked note-offs and MIDI
        # all-notes-off reach LMMS even after an exception or camera failure.
        if player is not None:
            try:
                player.panic()
            except (OSError, RuntimeError) as error:
                print(f"MIDI cleanup warning: {error}", file=sys.stderr)
        if midi_output is not None:
            try:
                midi_output.close()
            except (OSError, RuntimeError) as error:
                print(f"MIDI close warning: {error}", file=sys.stderr)
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
