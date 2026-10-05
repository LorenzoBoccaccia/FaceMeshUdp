# FaceMeshUdp

Python face-tracking app built on MediaPipe FaceLandmarker. Produces gaze/head-pose output with an optional overlay, capture tooling, a 9-point calibration workflow, UDP forwarding to OpenTrack, and direct FreeTrack 2.0 Enhanced output to games. See [eyes.ini](eyes.ini) for an example opentrack profile that consumes the UDP stream.

[![Demo reel](demo.gif)](https://youtu.be/I_M037X3Fb8)

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Calibrate and run

`calibrate.bat` records the `default` profile into `calibration-default.json`, `start.bat`
loads it and publishes to games over FreeTrack (TrackIR interface, 500 ms gaze smoothing, 200 ms tail kept on gaze jumps beyond 5 degrees). Both run from the repository root, use `.venv`, and pass any
extra arguments through to the app (`start.bat --camera-index 1`).

Measure the distance from your eyes to the screen and pass it to calibration, e.g.
`.\calibrate.bat --viewing-distance 65`; it is stored in the profile (default 100 cm). The
gaze model is refitted from the recorded points at every start, so passing
`--viewing-distance` to `start.bat` corrects it without recalibrating. See
[docs/calibration.md](docs/calibration.md) for the model.

```powershell
.\calibrate.bat
.\start.bat
```

Equivalent direct invocation:

```powershell
python -m facemesh_app.main --calibrate --viewing-distance 65
python -m facemesh_app.main --opentrack
```

## Options

At least one mode flag (`--overlay`, `--capture`, `--opentrack`, `--freetrack`, `--calibrate`) must be set.

Modes:

- `--overlay` — transparent overlay window with live landmarks
- `--capture` — save frames/mesh data on click (implies overlay)
- `--capture-live` / `--live` — show live camera feed in the capture window
- `--opentrack` — forward calibrated gaze output to opentrack's UDP tracker
- `--freetrack` — publish calibrated gaze straight to FreeTrack/TrackIR games (Windows, no opentrack process needed)
- `--calibrate` / `--calibration` — run the 9-point calibration workflow
- `--force-recalibrate` — ignore any stored profile and recalibrate
- `--calibration-profile NAME` — named calibration profile (defaults to `default`)
- `--viewing-distance CM` — eye-to-screen distance; default the profile's, or 100 for a new calibration

Camera:

- `--camera-index N` (`CAMERA_INDEX`, default 0) — which device to use

Resolution, fps, fourcc, and backend are no longer user-configurable: the
mediapipe FaceLandmarker downsamples internally to fixed sizes (128x128
detector, 256x256 landmarks), so high-resolution capture only inflates
per-frame buffer copies without improving accuracy. The app probes a fixed
ladder of (backend, format, size) candidates from cheapest to most expensive
on startup and accepts the first one that the camera actually delivers
without hitting the driver's CPU-scaling slow path.

OpenTrack:

- `--opentrack-host HOST` (`OPENTRACK_HOST`, default `127.0.0.1`)
- `--opentrack-port PORT` (`OPENTRACK_PORT`, default 4242)
- The packet carries the eye position in cm (to your right, up, and away from the camera on MediaPipe's depth scale), the gaze yaw/pitch and head roll

FreeTrack:

- `--freetrack-interface both|freetrack|npclient` — interface exposed to games (default `both`)
- `--freetrack-multiplier FACTOR` — scale the gaze angles sent to games, so a small gaze shift turns the game view further (default 1); the result is limited to ±180° yaw and ±90° pitch
- `--opentrack-dir DIR` (`OPENTRACK_DIR`, default auto-detected) — opentrack install whose client DLLs and `TrackIR.exe` games use; opentrack must be installed but not running

Misc:

- `--smooth MS` — average the forwarded gaze over the last MS milliseconds (default 0, raw)
- `--smooth-reset MS` — when the gaze jumps to a new fixation, keep only the last MS milliseconds of the previous one in the average (default off)
- `--smooth-threshold DEG` — distance from the current fixation, just above fixation jitter, that counts as a jump (default 1)
- `--overlay-fps FPS` — overlay redraw rate (default 60)
- `--log-interval SECONDS` — periodic stats interval (default 2.0)
- `--quiet` — suppress console output

## Capture output

Left-click the overlay in capture mode to write:

- `captures/mesh_capture_*.png` — frame with landmarks drawn
- `captures/mesh_capture_*.json` — 478 landmarks, 52 blendshapes, 4x4 transform matrix

## Build a standalone executable

```powershell
task build-exe
```

Outputs `dist/facemesh.exe` via PyInstaller.

## Profiling

Set `FACEMESH_PROFILE=1` to launch under yappi; stats are written on exit.
