# Calibration — geometry and math

## Goal

Turn a sequence of per-frame face-mesh events into pixel-space gaze estimates. To do that we need:

1. A 3D model of the monitor (pose, orientation, physical size) expressed in the camera reference frame.
2. A per-user mapping from raw eye-socket angles to true eye-in-head angles.

Both fall out of a single 9-point calibration session.

## Reference frames

- **Camera frame** is the frame the face mesh reports in: origin at the camera optical centre, `+x` right, `+y` down, `+z` forward (away from the camera along its optical axis). Head position `h = (hx, hy, hz)` and head pose `(yaw, pitch)` are expressed in this frame.
- **Screen frame** is an affine 2D frame embedded in a plane in camera space. Origin is at the centre of the screen; axes `ax`, `ay` lie in the screen plane; a pixel `(p_x, p_y)` maps to the 3D point `P = c + (p_x - o_x) · A + (p_y - o_y) · B`, where `A`, `B` have units of **mm per pixel** along the two in-plane directions. `(o_x, o_y)` is the pixel location of the origin (the centre dot).

`A = (1/scale_x) · ax` and `B = (1/scale_y) · ay` — the stored `screen_scale_*` fields are in pixels per mm.

## What we know and what we must recover

| quantity | source | type |
| --- | --- | --- |
| head position `h_i` per sample | face mesh | reliable |
| head pose `(yaw_i, pitch_i)` → forward ray `d_i` | face mesh | reliable |
| raw eye angles `(e_yaw_i, e_pitch_i)` | face mesh | biased / per-user scaled |
| target pixel `(p_x_i, p_y_i)` on screen | UX — we chose it | exact |
| monitor position / orientation / scale | unknown | solve for it |
| eye-socket → true-eye-angle mapping | unknown | solve for it |

Previously the code pulled physical monitor size from the OS (`GetDeviceCaps HORZSIZE/VERTSIZE`) and assumed the screen plane was `z = 0` with world-aligned axes. Both assumptions are now dropped.

## Nine-point procedure

Nine target pairs are shown in sequence: `C, T, TL, L, BL, B, BR, R, TR`. Each target has two on-screen dots:

- **Red dot** (`nose_target`) — where the user should *aim the nose* (head forward ray).
- **Green dot** (`eye_target`) — where the user should *look* with the eyes. Always placed at the point opposite the red dot through the screen centre: `eye_target = 2·C − nose_target`.

`C` is the degenerate case where both dots coincide at the screen centre: head forward and gaze aligned on the centre.

### UX flow per target

1. Render red + green dots at the pre-computed pixel positions.
2. Wait for a mouse click. The click is interpreted as *"my nose is on the red dot now"*. No pose-based alignment check — we trust the user. This is the critical inversion: UX no longer guides the head to an assumed position, the head tells us where the dots actually are.
3. **Blink phase** (≈500 ms): dots blink white as acknowledgement. Lets the user settle.
4. **Capture phase** (≈500 ms): record head pose, head position, and raw eye angles each frame, then average.
5. Advance to the next target.

The result of the session is nine `CalibrationPoint` records, each carrying `head_x/y/z`, `head_yaw/pitch`, `raw_eye_yaw/pitch`, and the pixel targets.

## Screen geometry fit

### Per-point constraint

For point `i`, the head-forward ray from `h_i` along direction `d_i` must hit the 3D point that corresponds to the red-dot pixel:

    h_i + t_i · d_i = c + u_i · A + v_i · B

with `u_i = p_x_i − o_x`, `v_i = p_y_i − o_y`, and `t_i > 0` the per-point ray length. This is three scalar equations in the unknowns `(c, A, B, t_i)`.

### Joint linear fit

The system is **linear** in `(c, A, B, t_1, …, t_9)` — 18 unknowns. Nine points give 27 equations. Stack them into `M x = b` and solve with `numpy.linalg.lstsq`. Monitor pose and scale come out directly:

- `c` — screen centre in camera mm.
- `axis_x = A / ‖A‖`, `scale_x = 1 / ‖A‖` (pixels per mm along column direction).
- `axis_y_ortho = B − (B·axis_x)·axis_x`, `axis_y = axis_y_ortho / ‖axis_y_ortho‖`, `scale_y = 1 / ‖axis_y_ortho‖`.
- `normal = axis_x × axis_y`.

Orthogonalisation pushes residual into one axis, matching the convention that `apply_calibration_model` already enforces at runtime.

### Conditioning

Eight effective degrees of freedom (`c` has 3, orientation has 3, scale has 2) are observed by 18 constraints. The plane normal is identifiable only if the nine head-ray origins span a nontrivial volume — i.e. the user's head must shift in `x`/`y` across targets. Pointing the nose at the four corner dots naturally induces 5–10 cm of head translation at a 60 cm viewing distance, which is enough parallax.

If the user keeps the head unnaturally still, the fit degenerates: the normal is unobserved. We accept that and report `screen_fit_rmse` so downstream code can see when the fit is weak.

## Eye calibration

Once screen geometry is known, each point's *target eye angle* is computable:

1. Green-dot pixel `(ex_i, ey_i) = 2·C − nose_target_i` lies at `Q_i = c + (ex_i − o_x)·A + (ey_i − o_y)·B` in camera frame.
2. True gaze direction from head `Q_i − h_i` → yaw/pitch via `screen_xy_to_head_angles`.
3. Target eye angle = true gaze angle − head angle at that frame.

Compare to the measured raw eye delta `raw_eye − raw_eye_C`:

- Ratio `target_eye_yaw_i / raw_eye_delta_yaw_i` bucketed by sign of the delta → `yaw_coefficient_positive` / `yaw_coefficient_negative` (same for pitch). Sign-bucketed because the eye-in-socket signal is asymmetric.
- Cross-axis coupling (`yaw_from_pitch_coupling`, `pitch_from_yaw_coupling`) fit linearly on the residual after the diagonal coefficients are applied.
- The coefficient applied at runtime interpolates between `negative` and `positive` as a function of the eye delta's position in `[eye_*_min, eye_*_max]`.

Because the head-forward ray and the eye target are on opposite sides of the screen centre, the eye delta for each outer point has a reliably large magnitude — good signal-to-noise for the ratio.

## Application at runtime

`apply_calibration_model` takes a runtime face-mesh event and returns pixel-space gaze:

1. Subtract calibration centre from raw eye angles to get `eye_delta`.
2. Apply interpolated coefficients + coupling → `corrected_eye_yaw/pitch`.
3. Add to head angles → `absolute_gaze_yaw/pitch`.
4. Trace the ray from `h` along that direction and intersect the calibrated screen plane via `project_head_angles_to_screen_xy`.
5. Decompose the intersection in `(axis_x, axis_y)` and scale by `(scale_x, scale_y)` → pixel offset from origin.

All of the intersection maths lives in `gaze_primitives.py`; calibration only has to supply `(c, axis_x, axis_y, scale_x, scale_y)`.

## Failure modes to watch

- **User clicks before settling** — head pose still moving, captured angles noisy. Mitigated by the 500 ms blink before capture.
- **User moves the mouse to click** — clicking requires head/eye drift. The blink phase lets things settle again before sampling.
- **User keeps head too still** — plane normal unobserved. Report `screen_fit_rmse` and let the caller decide.
- **User clicks with nose *not* on the red dot** — garbage in, garbage out. There is no automated check for this; the whole approach trusts the click.
