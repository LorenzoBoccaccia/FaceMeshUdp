#!/usr/bin/env python3
"""
FaceMesh Data Capture Application
Main entry point for raw face mesh data capture.
"""

import argparse
import os
import time
import threading
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .facemesh_dao import FaceMeshEvent
from .capture import save_test_capture, reset_capture_dir, build_camera_capture_marked_image

from .overlay import (
    OverlayManager, get_display_geo
)
# Constants
MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def ms_now():
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def open_camera(args):
    """Open camera with specified backend and settings."""
    backends = {
        "auto": [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW), ("any", None)],
        "msmf": [("msmf", cv2.CAP_MSMF), ("any", None)],
        "dshow": [("dshow", cv2.CAP_DSHOW), ("any", None)],
        "any": [("any", None)],
    }
    errors = []
    for name, backend in backends.get(args.camera_backend, backends["auto"]):
        cap = cv2.VideoCapture(args.camera_index, backend) if backend is not None else cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            cap.release()
            errors.append(f"{name}: open failed")
            continue

        fourcc = (args.camera_fourcc or "").strip().upper()
        if len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if args.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.camera_width))
        if args.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.camera_height))
        if args.camera_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(args.camera_fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, _ = cap.read()
        if not ok:
            cap.release()
            errors.append(f"{name}: read failed")
            continue

        info = {
            "backend": name,
            "index": args.camera_index,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }
        return cap, info

    raise RuntimeError(f"Unable to open webcam ({'; '.join(errors)})")


def ensure_model():
    """Download MediaPipe model if not present."""
    if MODEL_PATH.exists():
        return
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))


class FaceMeshWorker(threading.Thread):
    """Worker thread for MediaPipe face mesh data capture."""
    
    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.stop_evt = threading.Event()
        self.ready_evt = threading.Event()

        self.lock = threading.Lock()
        self.seq = 0
        self.latest: Optional[FaceMeshEvent] = None
        self.latest_frame = None
        self.error = None
        self.status = deque(maxlen=80)

    def _status(self, msg, force=False):
        """Add status message and optionally print."""
        with self.lock:
            self.status.append({"ts": ms_now(), "line": str(msg)})
        if not self.args.quiet or force:
            print(msg, flush=True)

    def snapshot(self) -> Tuple[int, Optional[FaceMeshEvent]]:
        """Get snapshot of latest event."""
        with self.lock:
            return self.seq, self.latest

    def capture_snapshot(self) -> Dict:
        """Capture full snapshot including frame and landmarks."""
        with self.lock:
            snap_evt = self.latest
            snap_frame = None if self.latest_frame is None else self.latest_frame.copy()
            if snap_evt is not None and snap_evt.landmarks is not None:
                snap_landmarks = list(snap_evt.landmarks)
            else:
                snap_landmarks = None
            snap_seq = self.seq
        return {"seq": snap_seq, "evt": snap_evt, "frame": snap_frame, "landmarks": snap_landmarks}

    def status_tail(self, n=20) -> List[Dict]:
        """Get last n status messages."""
        with self.lock:
            return list(self.status)[-n:]

    def stop(self):
        """Signal worker to stop."""
        self.stop_evt.set()

    def run(self):
        """Main worker loop for face mesh data capture."""
        cap = None
        landmarker = None
        try:
            ensure_model()
            base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
            opts = vision.FaceLandmarkerOptions(
                base_options=base,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            )
            landmarker = vision.FaceLandmarker.create_from_options(opts)
            cap, info = open_camera(self.args)

            self._status("FaceLandmarker initialized")
            self._status(
                f"Camera backend={info['backend']} index={info['index']} "
                f"{info['width']}x{info['height']} {info['fps']:.1f}fps",
                force=True,
            )
            self._status("Webcam capture started. Running until terminated.", force=True)
            self.ready_evt.set()

            next_log = time.time() + self.args.log_interval
            while not self.stop_evt.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))
                evt = FaceMeshEvent.from_landmarker_result(result, ts=ms_now())

                with self.lock:
                    self.seq += 1
                    self.latest = evt
                    self.latest_frame = frame_bgr

                if self.args.log_interval > 0 and time.time() >= next_log:
                    if evt.has_face:
                        self._status(f"Face detected - landmarks: {evt.landmark_count}", force=True)
                    else:
                        self._status("No face detected", force=True)
                    next_log += self.args.log_interval
        except Exception as e:
            self.error = f"FaceMesh worker failed: {e}"
            self._status(self.error, force=True)
            self.ready_evt.set()
        finally:
            if cap is not None:
                cap.release()
            if landmarker is not None:
                landmarker.close()


def enrich_runtime_evt(evt: FaceMeshEvent, screen_w: float, screen_h: float) -> Dict:
    """Extract raw face mesh data from event."""
    if not evt:
        return None
    return dict(evt.to_overlay_dict())



def start_capture_loop(args, worker: FaceMeshWorker,
                      display: Dict, run_duration: float):
    """Main capture loop for face mesh data capture with overlay."""
    overlay_enabled = bool(args.overlay)
    capture_enabled = bool(args.capture and overlay_enabled)
    capture_live_enabled = bool(args.capture and args.capture_live)
    live_window_name = "FaceMesh Capture Live"

    w = int(display["width"])
    h = int(display["height"])

    # Initialize overlay if enabled
    overlay_manager = None
    if overlay_enabled:
        overlay_manager = OverlayManager(display, capture_enabled, args.overlay_fps)
        overlay_manager.initialize()
    if capture_live_enabled:
        cv2.namedWindow(live_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(live_window_name, min(1400, w), min(900, h))

    latest_evt = None
    last_seq = -1
    last_live_seq = -1
    latest_live_frame = None

    started = time.time()
    running = True
    while running:
        if worker.error:
            raise RuntimeError(worker.error)
        
        # Handle overlay events
        if overlay_manager:
            overlay_manager.handle_events(
                on_click=lambda pos: save_test_capture(
                    display, w, h, pos,
                    worker,
                ) if capture_enabled else None
            )
            if not overlay_manager.is_running():
                running = False
                break

        # Get latest mesh data
        seq_id, evt = worker.snapshot()
        if seq_id != last_seq and evt and evt.type == "mesh":
            last_seq = seq_id
            latest_evt = enrich_runtime_evt(evt, w, h)

        # Live capture preview window with same render stack as saved capture.
        if capture_live_enabled:
            snap = worker.capture_snapshot()
            snap_seq = int(snap.get("seq", -1))
            if snap_seq != last_live_seq:
                last_live_seq = snap_seq
                snap_frame = snap.get("frame")
                snap_landmarks = snap.get("landmarks")
                eye_ellipses = None
                if snap_frame is not None and isinstance(snap_landmarks, list) and snap_landmarks:
                    mirrored_frame = cv2.flip(snap_frame, 1)
                    eye_ellipses = estimate_eye_ellipses(mirrored_frame, snap_landmarks, mirror_x=True)
                live_img, _ = build_camera_capture_marked_image(
                    snap,
                    overlay_w=float(w),
                    overlay_h=float(h),
                    click_pos=(0.0, 0.0),
                    eye_ellipses=eye_ellipses,
                    draw_click=False,
                )
                if live_img is not None:
                    latest_live_frame = live_img

            if latest_live_frame is not None:
                cv2.imshow(live_window_name, latest_live_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                running = False
                if overlay_manager:
                    overlay_manager.should_exit = True

        # Render overlay with mesh data
        if overlay_manager:
            overlay_manager.render_mesh(latest_evt)
        else:
            time.sleep(0.001)

        # Check duration
        if run_duration > 0 and (time.time() - started) >= run_duration:
            print(f"{run_duration:.0f}s elapsed, exiting.", flush=True)
            running = False

    if overlay_manager:
        overlay_manager.shutdown()
    if capture_live_enabled:
        cv2.destroyWindow(live_window_name)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FaceMesh data capture app")
    
    # Display options
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True,
                        help="Show overlay window")
    parser.add_argument("--capture", action=argparse.BooleanOptionalAction, default=False,
                        help="Capture mode (click to save mesh data)")
    parser.add_argument("--capture-live", "--live", dest="capture_live",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Show live camera preview with mesh + iris ovals + normals in capture mode")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Run time in seconds (0 = continuous)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")
    parser.add_argument("--log-interval", type=float, default=2.0,
                        help="Log interval in seconds")
    parser.add_argument("--overlay-fps", type=int, default=60,
                        help="Overlay refresh rate")
    
    # Camera options
    parser.add_argument("--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")),
                        help="Camera device index")
    parser.add_argument("--camera-backend", choices=["auto", "msmf", "dshow", "any"],
                        default=os.getenv("CAMERA_BACKEND", "auto").lower(),
                        help="Camera backend")
    parser.add_argument("--camera-width", type=int, default=int(os.getenv("CAMERA_WIDTH", "0")),
                        help="Camera resolution width (0 = default)")
    parser.add_argument("--camera-height", type=int, default=int(os.getenv("CAMERA_HEIGHT", "0")),
                        help="Camera resolution height (0 = default)")
    parser.add_argument("--camera-fps", type=float, default=float(os.getenv("CAMERA_FPS", "0")),
                        help="Camera FPS (0 = default)")
    parser.add_argument("--camera-fourcc", type=str, default=os.getenv("CAMERA_FOURCC", "MJPG"),
                        help="Camera codec")
    
    return parser.parse_args()


def normalize_runtime_args(args):
    """Validate and normalize runtime arguments."""
    if args.capture_live and not args.capture:
        print("Live capture preview requires --capture; ignoring --capture-live.", flush=True)
        args.capture_live = False
    if args.capture and not args.overlay:
        print("Capture mode requires overlay; ignoring --capture with --no-overlay.", flush=True)
        args.capture = False
        args.capture_live = False
    if args.capture:
        reset_capture_dir()
        print(f"Cleared previous capture session at {Path('captures').resolve()}", flush=True)


def main():
    """Main entry point."""
    args = parse_args()
    run_duration = max(0.0, args.duration)
    normalize_runtime_args(args)

    display = get_display_geo()

    worker = FaceMeshWorker(args)
    worker.start()
    if not worker.ready_evt.wait(timeout=20):
        raise RuntimeError("Timed out waiting for mesh worker init")
    if worker.error:
        raise RuntimeError(worker.error)

    print(f"App started on '{display['name']}' ({display['width']}x{display['height']}).", flush=True)
    print(f"Camera request: backend={args.camera_backend} index={args.camera_index}", flush=True)
    if args.capture:
        print(f"Capture mode enabled (--capture). Click to save mesh_capture_*.png/json into {Path('captures').resolve()}", flush=True)
    if args.capture and args.capture_live:
        print("Live preview enabled (--capture-live/--live): camera window shows mesh + iris ovals + normals.", flush=True)
    if run_duration <= 0:
        print("Runtime: continuous.", flush=True)
    else:
        print(f"Runtime: {int(run_duration)}s test mode.", flush=True)

    try:
        start_capture_loop(args, worker, display, run_duration)
    finally:
        worker.stop()
        worker.join(timeout=3.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
