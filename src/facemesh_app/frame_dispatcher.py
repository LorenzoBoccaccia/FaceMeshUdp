"""
FrameDispatcher module for FaceMesh application.
Handles frame processing pipeline and coordinates FaceMesh worker.
"""

import logging
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable, Any

import cv2
import numpy as np

from .facemesh_dao import (
    FaceMeshEvent,
    CalibrationMatrix,
    CalibrationPoint,
    compute_calibration_matrix,
    save_calibration,
    load_calibration,
    safe_float,
)
from .capture import save_test_capture, build_camera_capture_marked_image
from .overlay import OverlayManager, get_display_geo
from .state_machine import StateMachine, DispatcherState
from .pipeline_steps import (
    FaceMeshStep,
    CalibrationAdapterStep,
    CalibrationControllerStep,
    CaptureStep,
    OverlayStep,
    UDPForwardStep,
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def ms_now():
    """Current time in milliseconds."""
    return int(time.time() * 1000)


def ensure_model():
    """Download the FaceLandmarker model if not present."""
    if MODEL_PATH.exists():
        return
    logger.info(f"Downloading FaceMesh model from {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    logger.info("FaceMesh model downloaded successfully")


def enrich_runtime_evt(evt: Optional[FaceMeshEvent], screen_w: float, screen_h: float) -> Optional[Dict]:
    """Build runtime event payload for overlay and capture rendering.

    Coordinate convention is inherited from FaceMeshEvent:
    - yaw: right-positive
    - pitch: up-positive
    """
    if not evt:
        return None

    result = dict(evt.to_overlay_dict())

    result["head_yaw"] = evt.head_yaw
    result["head_pitch"] = evt.head_pitch

    result["raw_left_eye_gaze_yaw"] = evt.left_eye_gaze_yaw
    result["raw_left_eye_gaze_pitch"] = evt.left_eye_gaze_pitch
    result["raw_right_eye_gaze_yaw"] = evt.right_eye_gaze_yaw
    result["raw_right_eye_gaze_pitch"] = evt.right_eye_gaze_pitch

    if evt.left_eye_gaze_yaw is not None and evt.right_eye_gaze_yaw is not None:
        result["raw_combined_eye_gaze_yaw"] = (
            evt.left_eye_gaze_yaw + evt.right_eye_gaze_yaw
        ) / 2.0
    else:
        result["raw_combined_eye_gaze_yaw"] = None

    if evt.left_eye_gaze_pitch is not None and evt.right_eye_gaze_pitch is not None:
        result["raw_combined_eye_gaze_pitch"] = (
            evt.left_eye_gaze_pitch + evt.right_eye_gaze_pitch
        ) / 2.0
    else:
        result["raw_combined_eye_gaze_pitch"] = None

    result["calibrated_left_eye_gaze_yaw"] = evt.calibrated_left_eye_gaze_yaw
    result["calibrated_left_eye_gaze_pitch"] = evt.calibrated_left_eye_gaze_pitch
    result["calibrated_right_eye_gaze_yaw"] = evt.calibrated_right_eye_gaze_yaw
    result["calibrated_right_eye_gaze_pitch"] = evt.calibrated_right_eye_gaze_pitch
    result["calibrated_combined_eye_gaze_yaw"] = evt.calibrated_combined_eye_gaze_yaw
    result["calibrated_combined_eye_gaze_pitch"] = evt.calibrated_combined_eye_gaze_pitch

    return result


class FaceMeshWorker(threading.Thread):
    """Background thread that processes frames through FaceMeshStep."""

    def __init__(self, args, face_mesh_step, calibration=None):
        super().__init__(daemon=True)
        self.args = args
        self.face_mesh_step = face_mesh_step
        self.calibration = calibration
        self.stop_evt = threading.Event()
        self.ready_evt = threading.Event()

        self.lock = threading.Lock()
        self.seq = 0
        self.latest: Optional[FaceMeshEvent] = None
        self.latest_frame = None
        self.error = None
        self.status = deque(maxlen=80)
        self.frame_queue = queue.Queue(maxsize=2)

    def _status(self, msg: str, force: bool = False):
        with self.lock:
            self.status.append({"ts": ms_now(), "line": str(msg)})
        if not self.args.quiet or force:
            logger.info(msg)
            print(msg, flush=True)

    def snapshot(self) -> Tuple[int, Optional[FaceMeshEvent]]:
        with self.lock:
            return self.seq, self.latest

    def capture_snapshot(self) -> Dict:
        with self.lock:
            snap_evt = self.latest
            snap_frame = None if self.latest_frame is None else self.latest_frame.copy()
            if snap_evt is not None and snap_evt.landmarks is not None:
                snap_landmarks = list(snap_evt.landmarks)
            else:
                snap_landmarks = None
            snap_seq = self.seq
        return {"seq": snap_seq, "evt": snap_evt, "frame": snap_frame, "landmarks": snap_landmarks}

    def status_tail(self, n: int = 20) -> List[Dict]:
        with self.lock:
            return list(self.status)[-n:]

    def submit_frame(self, frame: Any, timestamp_ms: int) -> None:
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((frame, timestamp_ms), timeout=0.001)
        except (queue.Full, queue.Empty):
            logger.debug("Frame dropped - queue full or busy")
        except Exception as e:
            logger.warning(f"Error submitting frame to worker: {e}")

    def stop(self):
        self.stop_evt.set()

    def run(self):
        try:
            self._status("FaceLandmarker initialized")
            self._status("Worker ready to process frames from external source.", force=True)
            self.ready_evt.set()

            next_log = time.time() + self.args.log_interval
            while not self.stop_evt.is_set():
                try:
                    frame_bgr, timestamp_ms = self.frame_queue.get(timeout=0.01)
                except queue.Empty:
                    continue

                evt = self.face_mesh_step.receive_frame(frame_bgr, timestamp_ms)
                if evt is not None and self.calibration is not None:
                    evt.calibration = self.calibration

                with self.lock:
                    self.seq += 1
                    self.latest = evt
                    self.latest_frame = frame_bgr

                if self.args.log_interval > 0 and time.time() >= next_log:
                    if evt is not None and evt.has_face:
                        self._status(
                            f"Face detected - landmarks: {evt.landmark_count} "
                            f"head=({safe_float(evt.head_yaw):.1f}, {safe_float(evt.head_pitch):.1f}) "
                            f"gaze=({safe_float(evt.calibrated_combined_eye_gaze_yaw):.1f}, "
                            f"{safe_float(evt.calibrated_combined_eye_gaze_pitch):.1f})",
                            force=True,
                        )
                    elif evt is not None:
                        self._status("No face detected", force=True)
                    next_log += self.args.log_interval
        except Exception as e:
            self.error = f"FaceMesh worker failed: {e}"
            self._status(self.error, force=True)
            logger.exception("FaceMesh worker exception")
            self.ready_evt.set()


class FrameDispatcher:
    """Unified frame processing dispatcher coordinating FaceMesh worker and pipeline steps."""

    def __init__(
        self,
        args,
        calibration=None,
        overlay_manager=None,
        state_machine=None,
        face_mesh_step=None,
        calibration_adapter_step=None,
        calibration_controller_step=None,
        capture_step=None,
        overlay_step=None,
        udp_forward_step=None,
    ):
        self.args = args
        self.calibration = calibration
        self.overlay_manager = overlay_manager
        self.state_machine = state_machine
        self.face_mesh_step = face_mesh_step
        self.calibration_adapter_step = calibration_adapter_step
        self.calibration_controller_step = calibration_controller_step
        self.capture_step = capture_step
        self.overlay_step = overlay_step
        self.udp_forward_step = udp_forward_step

        self.worker: Optional[FaceMeshWorker] = None
        self.display: Optional[Dict] = None
        self.running = False

        self.frames_processed = 0
        self.faces_detected = 0
        self.frames_with_face = 0
        self.frames_without_face = 0

        self.display_width = 0
        self.display_height = 0
        self.origin_x = 0
        self.origin_y = 0

        self.capture_callback: Optional[Callable[[np.ndarray, Optional['FaceMeshEvent'], Optional['CalibratedFaceAndGazeEvent']], None]] = None

    def receiveFrame(self, frame, timestamp_ms):
        """Forward a frame to the processing worker."""
        if self.worker is None:
            return
        self.worker.submit_frame(frame, timestamp_ms)

    def start(self):
        """Start the FaceMesh worker thread."""
        if self.worker is not None:
            raise RuntimeError("FrameDispatcher already started")

        self.worker = FaceMeshWorker(self.args, self.face_mesh_step, calibration=self.calibration)
        self.worker.start()

        if not self.worker.ready_evt.wait(timeout=20):
            raise RuntimeError("Timed out waiting for mesh worker init")
        if self.worker.error:
            raise RuntimeError(self.worker.error)

        if self.overlay_manager is not None:
            self.display = get_display_geo()
        self.running = True

    def stop(self):
        """Stop the worker and release resources."""
        if self.worker is None:
            return

        self.running = False

        if self.overlay_manager:
            self.overlay_manager.shutdown()
            self.overlay_manager = None

        self.worker.stop()
        self.worker.join(timeout=3.0)
        self.worker = None

    def run_capture_loop(self, run_duration: float = 0.0, on_capture_click: Optional[Callable] = None):
        """Run the main capture and display loop until duration elapses or user exits."""
        if self.worker is None:
            raise RuntimeError("FrameDispatcher not started")

        overlay_enabled = bool(self.args.overlay)
        capture_enabled = bool(self.args.capture and overlay_enabled)
        capture_live_enabled = bool(self.args.capture and self.args.capture_live)
        live_window_name = "FaceMesh Capture Live"

        w = int(self.display["width"])
        h = int(self.display["height"])

        if overlay_enabled:
            if self.overlay_manager is None:
                self.overlay_manager = OverlayManager(self.display, capture_enabled, self.args.overlay_fps)
            self.overlay_manager.initialize()

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
            if self.worker.error:
                raise RuntimeError(self.worker.error)

            if self.overlay_manager:
                self.overlay_manager.handle_events(
                    on_click=lambda pos: save_test_capture(
                        self.display,
                        w,
                        h,
                        pos,
                        self.worker,
                    ) if capture_enabled else None
                )
                if not self.overlay_manager.is_running():
                    running = False
                    break

            seq_id, evt = self.worker.snapshot()
            if seq_id != last_seq and evt and evt.type == "mesh":
                last_seq = seq_id
                latest_evt = enrich_runtime_evt(evt, w, h)

            if capture_live_enabled:
                snap = self.worker.capture_snapshot()
                snap_seq = int(snap.get("seq", -1))
                if snap_seq != last_live_seq:
                    last_live_seq = snap_seq
                    snap_frame = snap.get("frame")
                    snap_landmarks = snap.get("landmarks")
                    live_img, _ = build_camera_capture_marked_image(
                        snap,
                        overlay_w=float(w),
                        overlay_h=float(h),
                        click_pos=(0.0, 0.0),
                        draw_click=False,
                    )
                    if live_img is not None:
                        latest_live_frame = live_img

                if latest_live_frame is not None:
                    cv2.imshow(live_window_name, latest_live_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    running = False
                    if self.overlay_manager:
                        self.overlay_manager.should_exit = True

            if self.overlay_manager:
                self.overlay_manager.render_mesh(latest_evt)
            else:
                time.sleep(0.001)

            if run_duration > 0 and (time.time() - started) >= run_duration:
                print(f"{run_duration:.0f}s elapsed, exiting.", flush=True)
                running = False

        if self.overlay_manager:
            self.overlay_manager.shutdown()
            self.overlay_manager = None
        if capture_live_enabled:
            cv2.destroyWindow(live_window_name)

    def run_calibration_workflow(self) -> Tuple[Optional[CalibrationMatrix], List[CalibrationPoint]]:
        """Execute the 9-point calibration workflow with on-screen guidance."""
        if self.worker is None:
            raise RuntimeError("FrameDispatcher not started")

        logger.info("Starting 9-point calibration workflow...")
        print("Starting 9-point calibration workflow...", flush=True)
        print("Please follow the on-screen instructions and look at each calibration point.", flush=True)

        try:
            if self.overlay_manager is None:
                self.overlay_manager = OverlayManager(
                    self.display,
                    capture_enabled=False,
                    overlay_fps=self.args.overlay_fps,
                    calibration_mode=True,
                )
            self.overlay_manager.initialize()
            self.overlay_manager.start_calibration_sequence(self.display["width"], self.display["height"])

            calib_points: List[CalibrationPoint] = []

            while True:
                if self.worker.error:
                    raise RuntimeError(f"Worker error during calibration: {self.worker.error}")

                self.overlay_manager.handle_events()
                if not self.overlay_manager.is_running():
                    print("Calibration cancelled by user.", flush=True)
                    break

                _, latest_evt = self.worker.snapshot()
                evt_dict = enrich_runtime_evt(latest_evt, self.display["width"], self.display["height"])

                completed, calib_point = self.overlay_manager.update_calibration_state(evt_dict)
                self.overlay_manager.render_mesh(evt_dict)

                if calib_point is not None:
                    fallback_zeta = max(float(self.display["width"]), float(self.display["height"])) * 0.75
                    calib_point.head_yaw = safe_float(evt_dict.get("head_yaw") if evt_dict else None, 0.0)
                    calib_point.head_pitch = safe_float(evt_dict.get("head_pitch") if evt_dict else None, 0.0)
                    calib_point.zeta = safe_float(evt_dict.get("zeta") if evt_dict else None, fallback_zeta)
                    calib_points.append(calib_point)

                    print(
                        f"Calibration point {len(calib_points)}/9 completed at '{calib_point.name}' "
                        f"head=({calib_point.head_yaw:.2f}, {calib_point.head_pitch:.2f}) "
                        f"eye=({calib_point.raw_eye_yaw:.2f}, {calib_point.raw_eye_pitch:.2f}) "
                        f"zeta={calib_point.zeta:.2f}",
                        flush=True,
                    )

                if completed:
                    if len(calib_points) == 9:
                        print("All 9 calibration points collected.", flush=True)
                    else:
                        print(f"Calibration sequence ended with {len(calib_points)} points.", flush=True)
                    break

                time.sleep(0.001)

            calib_matrix = None
            if len(calib_points) == 9:
                print("Computing calibration matrix...", flush=True)
                calib_matrix = compute_calibration_matrix(calib_points)

                profile_name = getattr(self.args, "calibration_profile", "") or "default"
                calib_path = save_calibration(calib_matrix, calib_points, profile_name)

                print(f"Calibration saved to: {calib_path}", flush=True)
                print(
                    "Calibration matrix: "
                    f"eye_zero=({calib_matrix.center_yaw:.4f}, {calib_matrix.center_pitch:.4f}) "
                    f"face_zero=({calib_matrix.face_center_yaw:.4f}, {calib_matrix.face_center_pitch:.4f}) "
                    f"zeta={calib_matrix.center_zeta:.4f} "
                    f"samples={calib_matrix.sample_count}",
                    flush=True,
                )
            else:
                print(f"Insufficient calibration points ({len(calib_points)}/9). Cannot compute calibration matrix.", flush=True)

            return calib_matrix, calib_points

        except Exception as e:
            print(f"Error during calibration: {e}", flush=True)
            logger.exception("Calibration workflow exception")
            raise
        finally:
            if self.overlay_manager:
                self.overlay_manager.shutdown()
                print("Calibration overlay shutdown complete.", flush=True)

    def set_calibration(self, calibration: CalibrationMatrix) -> None:
        """Apply a new calibration matrix to the dispatcher and worker."""
        self.calibration = calibration
        if self.worker is not None:
            self.worker.calibration = calibration

    def get_latest_event(self) -> Optional[FaceMeshEvent]:
        """Return the most recent FaceMeshEvent from the worker."""
        if self.worker is None:
            return None
        _, evt = self.worker.snapshot()
        return evt

    def get_status_tail(self, n: int = 20) -> List[Dict]:
        """Return the last n status messages from the worker."""
        if self.worker is None:
            return []
        return self.worker.status_tail(n=n)

    def is_running(self) -> bool:
        """Check if the dispatcher is actively processing."""
        return self.running and self.worker is not None

    def _handle_calibration_complete(self, calibration_result: dict) -> None:
        """Apply completed calibration result and transition to OPERATIONAL state."""
        pitch = calibration_result.get("pitch", 0.0)
        yaw = calibration_result.get("yaw", 0.0)
        roll = calibration_result.get("roll", 0.0)

        self.calibration_adapter_step.update_calibration(pitch=pitch, yaw=yaw, roll=roll)
        self.state_machine.transition_to(DispatcherState.OPERATIONAL)

    def start_calibration(self) -> None:
        """Transition to CALIBRATION state and begin calibration collection."""
        self.state_machine.transition_to(DispatcherState.CALIBRATION)
        self.calibration_controller_step.start_calibration()

    def set_capture_enabled(self, enabled: bool) -> None:
        """Enable or disable the capture pipeline step."""
        self.capture_step.set_enabled(enabled)

    def set_overlay_enabled(self, enabled: bool) -> None:
        """Enable or disable the overlay pipeline step."""
        self.overlay_step.set_enabled(enabled)

    def set_overlay_show_hud(self, show_hud: bool) -> None:
        """Toggle HUD display on the overlay step."""
        self.overlay_step.set_show_hud(show_hud)

    def set_udp_forwarding_enabled(self, enabled: bool) -> None:
        """Enable or disable UDP data forwarding."""
        self.udp_forward_step.set_enabled(enabled)

    def update_calibration(self, pitch: float, yaw: float, roll: float) -> None:
        """Update the calibration adapter's pitch/yaw/roll offsets."""
        self.calibration_adapter_step.update_calibration(pitch=pitch, yaw=yaw, roll=roll)

    def update_display_geometry(self, width: int, height: int, origin_x: float, origin_y: float) -> None:
        """Propagate new display dimensions to the calibration adapter."""
        self.display_width = width
        self.display_height = height
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.calibration_adapter_step.update_display_geometry(
            width=width,
            height=height,
            origin_x=origin_x,
            origin_y=origin_y,
        )

    def get_state(self) -> DispatcherState:
        """Return the current dispatcher state."""
        return self.state_machine.get_state()

    def set_capture_callback(self, callback: Optional[Callable[[np.ndarray, Optional['FaceMeshEvent'], Optional['CalibratedFaceAndGazeEvent']], None]]) -> None:
        """Register a callback invoked on each captured frame."""
        self.capture_callback = callback

    def get_calibration_progress(self) -> float:
        """Return calibration progress as a fraction 0.0 to 1.0."""
        return self.calibration_controller_step.get_progress()

    def get_calibration_current_point_index(self) -> int:
        """Return the 0-based index of the current calibration point."""
        return self.calibration_controller_step.get_current_point_index()

    def is_calibration_complete(self) -> bool:
        """Return True if the calibration workflow has finished."""
        return self.calibration_controller_step.is_calibration_complete()

    def get_calibration_status(self) -> Dict[str, Any]:
        """Return a dict summarising calibration state, progress and values."""
        status = {
            "state": self.state_machine.get_state(),
            "progress": self.get_calibration_progress(),
            "current_point_index": self.get_calibration_current_point_index(),
            "num_points": self.calibration_controller_step.num_points,
            "is_complete": self.is_calibration_complete(),
            "current_calibration": None,
        }

        if status["is_complete"] or status["state"] == DispatcherState.OPERATIONAL:
            status["current_calibration"] = {
                "pitch": self.calibration_adapter_step.pitch_calibration,
                "yaw": self.calibration_adapter_step.yaw_calibration,
                "roll": self.calibration_adapter_step.roll_calibration,
            }

        return status

    def get_statistics(self) -> dict:
        """Return operational statistics counters."""
        return {
            "frames_processed": self.frames_processed,
            "faces_detected": self.faces_detected,
            "frames_with_face": self.frames_with_face,
            "frames_without_face": self.frames_without_face,
            "capture_frame_count": self.capture_step.get_frame_count(),
            "state": self.state_machine.get_state(),
            "calibration_complete": self.is_calibration_complete(),
        }

    def reset_statistics(self) -> None:
        """Zero out all frame and face counters."""
        self.frames_processed = 0
        self.faces_detected = 0
        self.frames_with_face = 0
        self.frames_without_face = 0

    def set_state_transition_callback(self, callback: Callable[[DispatcherState, DispatcherState], None]) -> None:
        """Register a callback for state machine transitions."""
        self.state_machine.set_transition_callback(callback)

    def clear_state_transition_callback(self) -> None:
        """Remove any registered state transition callback."""
        self.state_machine.clear_transition_callback()
