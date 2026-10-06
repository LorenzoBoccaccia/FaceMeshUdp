# Calibration — geometry and math

## Goal

Turn per-frame face-mesh measurements into the point on screen the user looks at, and into
gaze angles for games. One 9-point session provides everything except the viewing
distance, which the user measures once (`--viewing-distance`, default 100 cm).

## Measurements and frames

- **Camera frame**: MediaPipe's metric camera space. `+x` image right, `+y` up, `+z` toward
  the viewer (the camera looks along `−z`), millimetres.
- **Head rotation `R`**: top-left 3×3 of MediaPipe's facial transformation matrix. It maps
  the face's own axes (x to the face's left, y up, z out of the face) into the camera frame.
- **Person axes**: right / up / back as seen by the person, `PERSON_AXES = diag(−1, 1, −1)`
  applied to face or camera axes. All yaw/pitch angles in the code are Fick angles in these
  axes: yaw to the right, then pitch up, `direction(yaw, pitch) = (sin y·cos p, sin p, −cos y·cos p)`.
- **Eye position `e`**: midpoint of the four eye corners, unprojected with MediaPipe's camera
  model (63° vertical field of view) at the head's reported depth. Lateral position is metric
  whatever the real camera's field of view is; absolute depth is not, and nothing relies on it.
- **Raw eye reading `r`**: iris offset from the eye corners and nose bridge, measured on the
  face's own axes (landmarks de-rotated by `Rᵀ`), averaged over both eyes. Head rotation does
  not leak into it.

## Nine-point procedure

Each target shows a green dot near a screen edge or corner (where the eyes look) and a red
dot half-way from the centre toward the opposite side (where the nose aims). Head and eyes
therefore turn in opposite directions while the eyes stay within about 10–20° of straight
ahead, where MediaPipe's iris landmarks still behave. At `C` both dots coincide at the
centre and the user faces it with head and eyes aligned, held for 5 s so the eye noise can
be measured over seconds, not just across a few frames. Click, hold still through the blink,
and the capture window records the frames. A target whose capture window saw no
usable face is asked again.

Each `CalibrationPoint` drops its blink frames (eye opening below 85% of the capture's
median) and keeps the per-axis median raw eye reading and its robust spread, the median eye
position, the median eye opening, and the chordal mean head rotation of the rest.

## Model

Everything is expressed relative to the reference pose at `C`:

- `S0 = R_C · PERSON_AXES` is the reference person frame; the screen centre lies straight
  ahead of `e_C` at the viewing distance `D`.
- The screen's axes are the reference axes rotated by a **roll** about the line of sight and
  a **tilt** about the screen's horizontal axis (eyes above or below the centre, monitor tilt).
  Pixel offsets map to millimetres with the OS-reported pixel density.
- Eye rotation within the head: `eye = W · (r − r_C)`, a full 2×2 matrix (gains and
  cross-talk), zero at `C` by the protocol.
- Gaze direction: `R · PERSON_AXES · direction(eye)`, a rotation composition, so head roll
  and large head/eye combinations are handled exactly.
- Head aim: the head turns only part of the way to the red dot,
  `head angles = G · (angles to the red dot)` per axis.

Because head rotations are taken relative to `R_C`, a constant bias in MediaPipe's head
orientation cancels.

## Fit

`fit_gaze_model` minimises, over the 8 non-centre targets:

- eye residual: `W · (r_i − r_C)` minus the eye-in-head angles that point from `e_i` at the
  green dot, given head rotation `R_C⁻¹ R_i`;
- head residual: the head's forward angles minus `G ×` the angles from `e_i` to the red dot,

for roll, tilt, `W` (4) and `G` (2), with a robust loss. The anti-correlated head and eye
targets make head and eye rotations vary independently, and head rotation is measured in
true degrees, which ties the eye gains to the head.

The viewing distance is an input, not a fitted parameter: with only these targets a larger
distance can be traded for a smaller head-aim gain and eye gain, so the data cannot pin it.
A wrong distance scales the eye contribution relative to the head (10% off gives about 5%).

The profile file stores the measured points, the screen and the distance; the model is
refitted at every start, so `--viewing-distance` can be corrected without recalibrating.

The fit also derives two per-user levels from the points:

- **eye noise**: the spread of the raw eye reading over the 5 s centre hold, carried through
  `W`, in degrees per axis; the smoother measures jumps against it. Short holds understate it,
  because the reading wanders slowly;
- **blink level**: 85% of the narrowest eye opening seen at any target (looking down narrows
  the eyes), below which the eyes count as closing.

## Runtime

`BlinkRejectionStep` withholds gaze from the moment the eye opening drops below the blink
level until the lid is back to 95% of its pre-blink opening, or has stopped reopening above
the blink level (a blink that ends in a downward look settles narrower). Iris readings in
between are lid artefacts, worth up to 30° of false downward gaze.

`GazeSmoothingStep` forwards the mean of the samples in a trailing window of observed gaze:
frames without a gaze reading, such as blinks, do not age it, so the view resumes where it
stopped. A sample further than `--smooth-threshold` times the eye noise from the current
fixation's median, confirmed by the next one, starts a new fixation. Everything in the window
before it is replaced by its average, as it stood when the gaze left, with the weight of the
last `--smooth-reset` milliseconds; the window then refills with new frames. Older fixations
therefore cannot pull the view back as they age out, and the view moves from where it is
straight toward the new fixation without a snap. A held gaze settles exactly on its calibrated
direction within one window, so the view centre cannot drift.

For each frame, `GazeModel.project` builds the gaze ray from the eye position, intersects
it with the screen plane, and reports:

- `screen_px`: the gaze point in pixels, plus where the head alone (`head_screen_px`) and the
  eyes alone from the reference pose (`eye_screen_px`) point, for the overlays;
- `yaw`, `pitch`: the gaze point's offset from the screen centre as angles at the viewing
  distance. They are zero at the centre and do not change when the head moves while the
  eyes stay on the same point; this is what opentrack and FreeTrack receive.

## Failure modes to watch

- **Head not aligned with the eyes at `C`**: the reference pose defines the screen axes; a
  wrong `C` biases everything.
- **Wrong viewing distance**: eye and head contributions stop matching; measure it.
- **Clicking before settling**: the blink phase and the median per target reduce the effect.
- **Display without a physical size**: calibration refuses to run, since targets cannot be
  placed in millimetres.
