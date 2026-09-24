# Hand Gesture Music Controller

Status: Draft

## Development preview and performance mode

Milestones 1 and 2 plus the first Milestone 3 expression control are implemented
in `HandChordGestures.py`. It detects two hands, sends voiced MIDI chords from
the selector hand, maps the other hand's height to a smoothed MIDI CC, and uses
expression-hand pinches to add seventh-chord extensions.

Run the preview from the activated virtual environment:

```powershell
python HandChordGestures.py
```

Press `q` or Escape to close it. Live testing established responsive defaults
of `0.15` seconds and `0.55` confidence. If recognition becomes too sensitive,
try the earlier conservative values:

```powershell
python HandChordGestures.py --hold-seconds 0.25 --min-confidence 0.65
```

Omitting `--port` always keeps this visual-only mode, making it useful for
gesture tuning without producing notes.

The preview color-codes all 21 hand landmarks so joint movement is easier to
follow:

- White: wrist
- Magenta: thumb
- Green: index finger
- Cyan: middle finger
- Orange: ring finger
- Pink: pinky finger

### MIDI chord performance

Before starting the camera application:

1. Open loopMIDI, create `Port1`, and leave loopMIDI running.
2. Start LMMS after the port exists and leave its MIDI interface on `WinMM MIDI`.
3. Add an instrument such as TripleOscillator to the Song Editor.
4. On that instrument track, select `MIDI -> Input -> Port1`.
5. Click the instrument's on-screen piano once to confirm LMMS audio works.

Start C-major chord performance with the right hand as the selector:

```powershell
python HandChordGestures.py --port Port1
```

Sideways minor gestures are the default for degrees I-V and VII. Either
horizontal direction is accepted. To compare them with the original downward
minor gestures, use:

```powershell
python HandChordGestures.py --port Port1 --minor-direction down
```

The friendly `Port1` name also matches the numbered name, such as `Port1 1`,
that Mido reports on Windows. Use the left hand as the selector when preferred:

```powershell
python HandChordGestures.py --port Port1 --selector-hand Left
```

The other detected hand is labeled `expression` and does not send notes yet.
That separation prevents the future effects hand from accidentally selecting a
second chord.

Add a dominant seventh to the major V gesture or try open voicings:

```powershell
python HandChordGestures.py --port Port1 --dominant-seven
python HandChordGestures.py --port Port1 --voicing open
```

A held stable pose sends one chord onset. Changing the stable pose releases the
old notes before starting the new chord. Removing the selector hand releases
the chord after the configured hold period. Pressing `q` or Escape releases the
active notes and sends MIDI all-notes-off before closing the port.

If LMMS receives no notes, confirm that `Port1` is visible in loopMIDI, restart
LMMS after creating the port, and reselect `MIDI -> Input -> Port1` on the
instrument track.

### Expression-hand filter control

By default, the non-selector hand controls MIDI CC74, the conventional
brightness/filter-cutoff controller. With the default right selector hand, the
left hand is the expression hand. Raise it for `127` and lower it for `0`. The
camera overlay shows the current value on a vertical CC74 meter.

LMMS needs a one-time connection between incoming CC74 and the instrument's
filter cutoff:

1. Open the TripleOscillator instrument and choose a saw or square waveform so
   filtering is easy to hear.
2. Open `ENV/LFO`, enable the filter by clicking its title bar/light, and choose
   a low-pass filter.
3. Right-click the filter `CUTOFF` knob and select `Connect to controller`.
4. Select `MIDI controller`, enable `Auto Detect`, then move only the expression
   hand vertically while the Python application is running.
5. Hold a chord with the selector hand and raise/lower the expression hand.

The normal performance command enables both chord and expression output:

```powershell
python HandChordGestures.py --port Port1
```

Use another CC number when a synth expects a different control:

```powershell
python HandChordGestures.py --port Port1 --expression-cc 1
```

Expression output uses exponential smoothing, a two-value dead zone, and a
30-message-per-second limit. A larger smoothing value responds faster; a
smaller value moves more gently. For example:

```powershell
python HandChordGestures.py --port Port1 --expression-smoothing 0.4
```

### Seventh-chord pinch modifiers

The expression hand can change the held selector chord without giving up its
vertical CC74 control:

| Expression-hand shape | Major selector pose | Minor selector pose |
| --- | --- | --- |
| No pinch | Major triad | Minor triad |
| Thumb to index | Major 7 | Minor 7 |
| Thumb, middle, and ring together | Dominant 7 | Invalid; previous chord is retained |

Pinch distance is normalized by palm width, stabilized for 0.12 seconds, and
uses different engage/release thresholds to prevent flicker. A tracking dropout
shorter than 0.30 seconds retains the modifier. Removing the expression hand for
longer returns the chord to a triad, and changing the selector while that hand
is absent drops the old modifier immediately.

The normal performance command enables the pinch modifiers automatically:

```powershell
python HandChordGestures.py --port Port1
```

The overlay reports `Triad`, `Quality 7`, or `Dominant 7`, along with normalized
index (`I`), middle (`M`), and ring (`R`) pinch distances. Dominant 7 engages
only when both `M` and `R` are close enough to the thumb. If a firm pinch does
not engage, try slightly larger thresholds:

```powershell
python HandChordGestures.py --port Port1 --pinch-engage 0.38 --pinch-release 0.55
```

If modifiers engage accidentally, use smaller values such as `0.22` and `0.35`.
The hand-loss grace can also be tuned; for example, use
`--modifier-loss-seconds 0.5` if brief expression-hand dropouts are common.
The existing `--dominant-seven` option remains available for automatically
turning an unmodified major V gesture into V7; an active pinch takes precedence.

### Harmony preview

`HarmonyPreview.py` prints chord notes and automatic inversions without opening
the camera or a MIDI port. Its default progression is I-IV-V7-I in C major:

```powershell
python HarmonyPreview.py
```

Use `DEGREE:TYPE` steps to review another progression. Supported chord types are
`major`, `minor`, `dominant7`, `major7`, and `minor7`:

```powershell
python HarmonyPreview.py --tonic D --progression 1:major,6:minor,4:major,5:dominant7
```

Compare close and open spacing with `--voicing close` or `--voicing open`. The
preview shows both note names and raw MIDI note numbers for human review.

<img width="797" height="885" alt="image" src="https://github.com/user-attachments/assets/23175ef7-82cc-4e64-89c7-fa370c70248d" />


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
| I | Index pointing up | Index pointing sideways |
| II | Index and middle pointing up | Index and middle pointing sideways |
| III | Index, middle, and ring pointing up | Same fingers pointing sideways |
| IV | Four fingers pointing up | Four fingers pointing sideways |
| V | Open hand pointing up | Open hand pointing sideways |
| VI | Index and pinky pointing down | Index and pinky pointing up |
| VII | Vulcan hand signal pointing up | Vulcan hand signal pointing sideways |

“Up,” “down,” and “sideways” mean fingertip direction in the camera image. Both
screen-left and screen-right count as sideways. Degree III accepts either `IMR`
or the relaxed-thumb `TIMR` detected during live testing. The
`--minor-direction down` option restores the original mapping for comparison.
VI keeps its intentional exception: pointing up is minor and pointing down is
major because vi is naturally minor in a major scale and the upward gesture is
easier to perform.

## 5. Musical behavior

- Begin with a configurable key, using C major as the first test case.
- Convert the selected scale degree into a chord using a configurable chord
  formula and octave.
- Initially support major and minor triads.
- Major, minor, dominant-seven, major-seven, and minor-seven chords are
  supported; diminished, suspended, and larger extensions come later.
- When the gesture changes, turn off the previous chord before turning on the
  new chord.
- Send an all-notes-off or equivalent cleanup message when the application
  exits.

An open music decision is whether “major/minor” should freely override the
quality of every scale degree, or whether the application should enforce normal
diatonic harmony for the selected key and scale.

## 6. Musical intent and voicing model

Hand gestures should describe musical intent instead of directly encoding every
MIDI note. The first internal representation will be a `ChordIntent` containing:

- Key and scale
- Scale degree I through VII
- Chord quality, initially major or minor
- Optional extension, such as a dominant seventh
- Inversion preference
- Register and voicing style
- Performance values such as intensity and articulation

The voicing engine will convert that intent into concrete MIDI note numbers. It
will be responsible for:

- Finding the chord root from the selected key and scale degree
- Building major, minor, and later extended chord formulas
- Applying root position, first inversion, or second inversion
- Keeping notes inside a configured playable range
- Producing close, open, and later instrument-specific voicings
- Choosing a nearby inversion when automatic voice leading is enabled

The harmony layer is tested independently of the camera and MIDI port, then the
live application connects stabilized gestures to its `ChordIntent` interface.

## 7. Two-hand control model

### Selector hand

- Detect the hand’s identity and landmark positions.
- Classify the finger pattern.
- Classify fingertip direction as up, down, sideways, or diagonal.
- Apply confidence, debounce, and hold-time rules before changing chords.

### Expression hand

The implemented mappings are:

- Vertical position -> filter cutoff
- Thumb-index pinch -> major 7 or minor 7, following the selector quality
- Thumb-middle-ring pinch -> dominant 7 with a major selector pose

Planned mappings include:

- Horizontal position -> pan
- Finger spread or hand openness -> reverb/delay amount
- Wrist rotation -> modulation rate

All continuous values should be smoothed and rate-limited before MIDI CC
messages are sent.

## 8. Technical approach

1. Extend hand tracking from one hand to two hands.
2. Use MediaPipe raw landmarks to recognize custom poses rather than relying
   only on its built-in gesture names.
3. Separate the application into vision, gesture classification, music mapping,
   MIDI output, and configuration responsibilities.
4. Keep the existing continuous MIDI script as a reference while building the
   new controller.
5. Use the loopMIDI port name supplied by the user, accepting both `Port1` and
   Mido's numbered form such as `Port1 1`.
6. Pass recognized gestures into `ChordIntent` rather than constructing MIDI
   messages inside the camera loop.

## 9. Milestones

### Milestone 1: Gesture display

- Detect two hands.
- Recognize the I through VII poses.
- Display the detected degree, quality, and confidence on the camera window.
- Do not send MIDI notes yet.

### Milestone 2: Chord output

Implementation complete and undergoing camera-and-LMMS musical tuning.

- Build and test `ChordIntent` and the voicing engine first.
- Start with C major and all seven scale degrees.
- Support major/minor triads, inversions, register limits, and close voicing.
- Add automatic nearest-inversion voice leading.
- Send note-on and note-off messages to the virtual MIDI port.
- Confirm the chords in a DAW or MIDI monitor.

### Milestone 3: Expression control

Vertical position to smoothed CC74 and stable seventh-chord pinch modifiers are
implemented and awaiting camera-and-LMMS tuning. Remaining controls will be
added one at a time.

- Assign one movement to one MIDI CC.
- Add smoothing and a dead zone to reduce jitter.
- Add the remaining effect mappings one at a time.

### Milestone 4: Usability and configuration

- Add calibration for camera position and hand size.
- Add an on-screen status panel.
- Move musical and MIDI settings into a configuration file.
- Add a panic/stop control and robust cleanup.

## 10. Acceptance criteria for the first usable prototype

- The app detects two hands without confusing the selector and expression hand.
- Each gesture remains stable when held naturally for a short period.
- I through VII and major/minor are displayed correctly under normal lighting.
- Changing gestures does not leave stuck MIDI notes.
- A DAW or MIDI monitor receives the expected chord notes on `Port1 1`.
- Moving the expression hand changes at least one effect smoothly.
- Expression-hand pinches select major7, minor7, and dominant7 without stuck
  notes or rapid chord flicker.
- Pressing `q` or Escape closes the camera and MIDI connection safely.

## 11. Risks and open questions

- Camera angle and hand rotation may make directional classification unreliable.
- The Vulcan signal must be distinguished from an open hand by its finger gaps.
- Finger-count poses may be confused during transitions between chords.
- The best representation for key changes and scales is still undecided.
- Chord voicings, inversions, velocity, sustain, and quantization need musical
  testing.
- The project should eventually include a small test suite for pose-to-chord
  mapping that does not require a webcam.
