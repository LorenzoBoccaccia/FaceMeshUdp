# FaceMesh (Python) - faceudp

Run:

```powershell
python -m facemesh_app.main
```

Feature toggles:

```powershell
python -m facemesh_app.main --udp --overlay --calibrate
python -m facemesh_app.main --no-udp --overlay --no-calibrate
python -m facemesh_app.main --udp --no-overlay
python -m facemesh_app.main --overlay --capture
```

Capture mode (`--capture`) uses black debug view and saves `captures/gaze_debug_*.png/.json` on left click.
Saved PNG now uses the camera frame at click time with facemesh points plus face/eye gaze normals drawn.

Build executable (inside project venv):

```powershell
task build-exe
```
