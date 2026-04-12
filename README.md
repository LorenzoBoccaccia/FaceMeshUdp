# FaceMesh (Python)

A simplified face mesh data capture application using MediaPipe FaceLandmarker.

## Usage

Run the application:

```powershell
python -m facemesh_app.main
```

## Features

- **Face Mesh Data Capture**: Captures raw face mesh data including landmarks, blendshapes, and transformation matrix
- **Optional Overlay Display**: Shows live face mesh visualization on screen
- **Capture Mode**: Save mesh data snapshots for analysis

## Command Line Options

```powershell
python -m facemesh_app.main --overlay --capture
python -m facemesh_app.main --no-overlay
python -m facemesh_app.main --duration 10.0
```

### Available Options

- `--overlay` / `--no-overlay`: Enable/disable overlay window (default: enabled)
- `--capture` / `--no-capture`: Enable capture mode (requires overlay)
- `--duration SECONDS`: Run time in seconds (0 = continuous)
- `--camera-index N`: Camera device index (default: 0)
- `--camera-backend`: Camera backend (auto/msmf/dshow/any, default: auto)
- `--camera-width WIDTH`: Camera resolution width (default: auto)
- `--camera-height HEIGHT`: Camera resolution height (default: auto)
- `--camera-fps FPS`: Camera FPS (default: auto)
- `--camera-fourcc CODEC`: Camera codec (default: MJPG)
- `--quiet`: Suppress console output
- `--log-interval SECONDS`: Log interval in seconds (default: 2.0)
- `--overlay-fps FPS`: Overlay refresh rate (default: 60)

## Capture Mode

When `--capture` is enabled, left-clicking on the overlay window saves:
- `captures/mesh_capture_*.png`: Camera frame with face mesh landmarks
- `captures/mesh_capture_*.json`: Raw mesh data (landmarks, blendshapes, transform matrix)

## Output Data

The application captures raw face mesh data:

- **Landmarks**: 478 face mesh landmarks as (x, y, z) coordinates
- **Blendshapes**: 52 facial expression blendshape scores
- **Transform Matrix**: 4x4 facial transformation matrix

## Build Executable

To build a standalone executable (inside project venv):

```powershell
task build-exe
```
