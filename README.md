# FaceMesh

Python face-tracking app built on MediaPipe FaceLandmarker. Produces gaze/head-pose output with an optional overlay, capture tooling, a 9-point calibration workflow, and UDP forwarding to OpenTrack. See [eyes.ini](eyes.ini) for an example opentrack profile that consumes the UDP stream.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Calibrate and run

```powershell
python -m facemesh_app.main --calibrate
python -m facemesh_app.main --udp 
```

## Options

At least one mode flag (`--overlay`, `--capture`, `--udp`, `--calibrate`) must be set.

Modes:

- `--overlay` — transparent overlay window with live landmarks
- `--capture` — save frames/mesh data on click (implies overlay)
- `--capture-live` / `--live` — show live camera feed in the capture window
- `--udp` — forward calibrated gaze output over UDP
- `--calibrate` / `--calibration` — run the 9-point calibration workflow
- `--force-recalibrate` — ignore any stored profile and recalibrate
- `--calibration-profile NAME` — named calibration profile (defaults to `default`)
- `--calibration-samples N` — minimum samples per point (default: 5)

Camera (env vars in parentheses override defaults):

- `--camera-index N` (`CAMERA_INDEX`, default 0)
- `--camera-backend {auto,msmf,dshow,any}` (`CAMERA_BACKEND`, default `dshow`)
- `--camera-width W` (`CAMERA_WIDTH`, default 1920)
- `--camera-height H` (`CAMERA_HEIGHT`, default 1080)
- `--camera-fps FPS` (`CAMERA_FPS`, default 60)
- `--camera-fourcc CODEC` (`CAMERA_FOURCC`, default `MJPG`)

UDP:

- `--udp-host HOST` (`UDP_HOST`, default `127.0.0.1`)
- `--udp-port PORT` (`UDP_PORT`, default 4242)

Misc:

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
