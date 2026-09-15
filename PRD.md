# Hand Gesture Music Controller

Status: Draft

## 1. Product idea

Use two hands in front of a webcam to control musical chords and sound effects.
One hand selects a scale degree and chord quality. The other hand provides
continuous expressive control for effects such as filters, reverb, delay, and
pan.

The project starts from the HandTrack_To_MIDI Python application and its
MediaPipe, OpenCV, Mido, and virtual-MIDI workflow.

## 2. Goals

- Select seven scale-degree chords with recognizable hand poses.
- Select major or minor using the direction of the gesture.
- Send reliable MIDI note messages for the selected chord.
- Use the second hand for smooth, expressive MIDI CC messages.
- Provide clear on-screen feedback about the detected gesture and active chord.
- Keep key, scale, octave, MIDI channel, port, and CC mappings configurable.

## 3. Initial non-goals

- Training a custom machine-learning model in the first version.
- Supporting rapid percussive note triggering.
- Building a complete digital audio workstation or synthesizer.
- Supporting more than two hands.

## 4. Gesture vocabulary

The selector hand uses extended-finger patterns. Fingers are assumed to be
index, middle, ring, and pinky unless otherwise stated. The thumb should remain
tucked for poses where it is not part of the pattern.

| Scale degree | Major gesture | Minor gesture |
| --- | --- | --- |
| I | Index pointing up | Index pointing down |
| II | Index and middle pointing up | Index and middle pointing down |
| III | Index, middle, and ring pointing up | Same fingers pointing down |
| IV | Four fingers pointing up | Four fingers pointing down |
| V | Open hand pointing up | Open hand pointing down |
| VI | Index and pinky pointing up | Index and pinky pointing down |
| VII | Vulcan hand signal pointing up | Vulcan hand signal pointing down |

For this document, “up” and “down” mean the direction of the fingertips in the
camera image. This can be revised if palm orientation proves more natural or
reliable.

## 5. Musical behavior

- Begin with a configurable key, using C major as the first test case.
- Convert the selected scale degree into a chord using a configurable chord
  formula and octave.
- Initially support major and minor triads.
- Add diminished, seventh, suspended, and extended chords later.
- When the gesture changes, turn off the previous chord before turning on the
  new chord.
- Send an all-notes-off or equivalent cleanup message when the application
  exits.

An open music decision is whether “major/minor” should freely override the
quality of every scale degree, or whether the application should enforce normal
diatonic harmony for the selected key and scale.

## 6. Two-hand control model

### Selector hand

- Detect the hand’s identity and landmark positions.
- Classify the finger pattern.
- Classify fingertip direction as up or down.
- Apply confidence, debounce, and hold-time rules before changing chords.

### Expression hand

Start with a small set of mappings:

- Vertical position -> filter cutoff
- Horizontal position -> pan
- Finger spread or hand openness -> reverb/delay amount
- Pinch distance -> effect intensity
- Wrist rotation -> modulation rate

All continuous values should be smoothed and rate-limited before MIDI CC
messages are sent.

## 7. Technical approach

1. Extend hand tracking from one hand to two hands.
2. Use MediaPipe raw landmarks to recognize custom poses rather than relying
   only on its built-in gesture names.
3. Separate the application into vision, gesture classification, music mapping,
   MIDI output, and configuration responsibilities.
4. Keep the existing continuous MIDI script as a reference while building the
   new controller.
5. Use the loopMIDI port name supplied by the user, currently `Port1 1`.

## 8. Milestones

### Milestone 1: Gesture display

- Detect two hands.
- Recognize the I through VII poses.
- Display the detected degree, quality, and confidence on the camera window.
- Do not send MIDI notes yet.

### Milestone 2: Chord output

- Start with C major and I through IV.
- Send note-on and note-off messages to the virtual MIDI port.
- Confirm the chords in a DAW or MIDI monitor.
- Add V through VII after the basic mapping is stable.

### Milestone 3: Expression control

- Assign one movement to one MIDI CC.
- Add smoothing and a dead zone to reduce jitter.
- Add the remaining effect mappings one at a time.

### Milestone 4: Usability and configuration

- Add calibration for camera position and hand size.
- Add an on-screen status panel.
- Move musical and MIDI settings into a configuration file.
- Add a panic/stop control and robust cleanup.

## 9. Acceptance criteria for the first usable prototype

- The app detects two hands without confusing the selector and expression hand.
- Each gesture remains stable when held naturally for a short period.
- I through VII and major/minor are displayed correctly under normal lighting.
- Changing gestures does not leave stuck MIDI notes.
- A DAW or MIDI monitor receives the expected chord notes on `Port1 1`.
- Moving the expression hand changes at least one effect smoothly.
- Pressing `q` or Escape closes the camera and MIDI connection safely.

## 10. Risks and open questions

- Camera angle and hand rotation may make up/down classification unreliable.
- The Vulcan signal must be distinguished from an open hand by its finger gaps.
- Finger-count poses may be confused during transitions between chords.
- The best representation for key changes and scales is still undecided.
- Chord voicings, inversions, velocity, sustain, and quantization need musical
  testing.
- The project should eventually include a small test suite for pose-to-chord
  mapping that does not require a webcam.
