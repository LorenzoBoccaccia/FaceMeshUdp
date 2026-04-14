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
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .facemesh_dao import (
    FaceMeshEvent,
    CalibrationMatrix,
    CalibrationPoint,
    compute_calibration_matrix,
    save_calibration,
    load_calibration,
    safe_float,
)
from .capture import save_test_capture, reset_capture_dir, build_camera_capture_marked_image
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
    """Get current time in milliseconds."""
    return int(time.time() * 1000)


def ensure_model():
    """Ensure FaceMesh model file exists, download if missing."""
    if MODEL_PATH.exists():
        return
    logger.info(f"Downloading FaceMesh model from {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    logger.info("FaceMesh model downloaded successfully")


def enrich_runtime_evt(evt: Optional[FaceMeshEvent], screen_w: float, screen_h: float) -> Optional[Dict]:
    """Extract runtime event payload for overlay/capture.

    Coordinate convention is inherited from FaceMeshEvent:
    - yaw: right-positive
    - pitch: up-positive
    
    Args:
        evt: FaceMeshEvent instance or None
        screen_w: Screen width in pixels
        screen_h: Screen height in pixels
    
    Returns:
        Dictionary with enriched event data or None if evt is None
    """
    if not evt:
        return None

    result = dict(evt.to_overlay_dict())

    # Head pose
    result["head_yaw"] = evt.head_yaw
    result["head_pitch"] = evt.head_pitch

    # Raw eye gaze (used by calibration UI)
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

    # Calibrated eye gaze (used for runtime display)
    result["calibrated_left_eye_gaze_yaw"] = evt.calibrated_left_eye_gaze_yaw
    result["calibrated_left_eye_gaze_pitch"] = evt.calibrated_left_eye_gaze_pitch
    result["calibrated_right_eye_gaze_yaw"] = evt.calibrated_right_eye_gaze_yaw
    result["calibrated_right_eye_gaze_pitch"] = evt.calibrated_right_eye_gaze_pitch
    result["calibrated_combined_eye_gaze_yaw"] = evt.calibrated_combined_eye_gaze_yaw
    result["calibrated_combined_eye_gaze_pitch"] = evt.calibrated_combined_eye_gaze_pitch

    return result


class FaceMeshWorker(threading.Thread):
    """Background thread that processes frames with FaceMesh landmark detection."""
    
    def __init__(self, args: Any, calibration: Optional[CalibrationMatrix] = None):
        """Initialize FaceMesh worker thread.
        
        Args:
            args: Configuration object with camera settings and options
            calibration: Optional calibration matrix for gaze correction
        """
        super().__init__(daemon=True)
        self.args = args
        self.calibration = calibration
        self.stop_evt = threading.Event()
        self.ready_evt = threading.Event()

        self.lock = threading.Lock()
        self.seq = 0
        self.latest: Optional[FaceMeshEvent] = None
        self.latest_frame = None
        self.error = None
        self.status = deque(maxlen=80)
        
        # Frame queue for external frame input (from CameraReader)
        self.frame_queue = queue.Queue(maxsize=2)  # Small buffer to minimize latency

    def _status(self, msg: str, force: bool = False):
        """Log status message.
        
        Args:
            msg: Status message to log
            force: If True, force output even in quiet mode
        """
        with self.lock:
            self.status.append({"ts": ms_now(), "line": str(msg)})
        if not self.args.quiet or force:
            logger.info(msg)
            print(msg, flush=True)

    def snapshot(self) -> Tuple[int, Optional[FaceMeshEvent]]:
        """Get latest event snapshot.
        
        Returns:
            Tuple of (sequence_id, latest_event)
        """
        with self.lock:
            return self.seq, self.latest

    def capture_snapshot(self) -> Dict:
        """Get full snapshot including frame data.
        
        Returns:
            Dictionary with seq, evt, frame, and landmarks
        """
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
        """Get last n status messages.
        
        Args:
            n: Number of recent messages to retrieve
        
        Returns:
            List of status message dictionaries
        """
        with self.lock:
            return list(self.status)[-n:]

    def submit_frame(self, frame: Any, timestamp_ms: int) -> None:
        """Submit a frame for processing from an external source (e.g., CameraReader).
        
        This method is thread-safe and can be called from any thread.
        
        Args:
            frame: Frame data (numpy array in BGR format)
            timestamp_ms: Frame timestamp in milliseconds
        """
        try:
            # Put frame in queue with non-blocking to avoid deadlock
            # If queue is full, drop the oldest frame and add new one
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((frame, timestamp_ms), timeout=0.001)
        except (queue.Full, queue.Empty):
            # Drop frame if queue operations fail - better to skip than block
            logger.debug("Frame dropped - queue full or busy")
        except Exception as e:
            logger.warning(f"Error submitting frame to worker: {e}")

    def stop(self):
        """Signal the worker thread to stop."""
        self.stop_evt.set()

    def run(self):
        """Main worker thread loop - processes frames from external queue."""
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

            self._status("FaceLandmarker initialized")
            self._status("Worker ready to process frames from external source.", force=True)
            self.ready_evt.set()

            next_log = time.time() + self.args.log_interval
            while not self.stop_evt.is_set():
                try:
                    # Get frame from queue with timeout to allow checking stop_evt
                    frame_bgr, timestamp_ms = self.frame_queue.get(timeout=0.01)
                except queue.Empty:
                    # No frame available, continue loop
                    continue
                
                # Process the frame
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))
                evt = FaceMeshEvent.from_landmarker_result(result, ts=timestamp_ms, calibration=self.calibration)

                with self.lock:
                    self.seq += 1
                    self.latest = evt
                    self.latest_frame = frame_bgr

                if self.args.log_interval > 0 and time.time() >= next_log:
                    if evt.has_face:
                        self._status(
                            f"Face detected - landmarks: {evt.landmark_count} "
                            f"head=({safe_float(evt.head_yaw):.1f}, {safe_float(evt.head_pitch):.1f}) "
                            f"gaze=({safe_float(evt.calibrated_combined_eye_gaze_yaw):.1f}, "
                            f"{safe_float(evt.calibrated_combined_eye_gaze_pitch):.1f})",
                            force=True,
                        )
                    else:
                        self._status("No face detected", force=True)
                    next_log += self.args.log_interval
        except Exception as e:
            self.error = f"FaceMesh worker failed: {e}"
            self._status(self.error, force=True)
            logger.exception("FaceMesh worker exception")
            self.ready_evt.set()
        finally:
            if landmarker is not None:
                landmarker.close()

class FrameDispatcher:
    """Frame processing dispatcher that coordinates FaceMesh worker and processing pipeline."""
    
    def __init__(
        self,
        args: Any,
        calibration: Optional[CalibrationMatrix] = None,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        overlay_manager: Optional[OverlayManager] = None,
        state_machine: Optional['StateMachine'] = None,
        face_mesh_step: Optional['FaceMeshStep'] = None,
        calibration_adapter_step: Optional['CalibrationAdapterStep'] = None,
        calibration_controller_step: Optional['CalibrationControllerStep'] = None,
        capture_step: Optional['CaptureStep'] = None,
        overlay_step: Optional['OverlayStep'] = None,
        udp_forward_step: Optional['UDPForwardStep'] = None,
    ):
        """Initialize FrameDispatcher.
        
        Args:
            args: Configuration object with camera and application settings
            calibration: Optional calibration matrix for gaze correction
            model_complexity: FaceMesh model complexity (0 or 1)
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            overlay_manager: Optional OverlayManager instance for rendering visual overlays.
                             If None, a default manager will be created when needed.
                             Allows constructor injection for testing and flexibility.
            state_machine: Optional StateMachine instance for pipeline state management
            face_mesh_step: Optional FaceMeshStep instance for face mesh processing
            calibration_adapter_step: Optional CalibrationAdapterStep instance for display geometry
            calibration_controller_step: Optional CalibrationControllerStep instance for calibration workflow
            capture_step: Optional CaptureStep instance for live preview and frame counting
            overlay_step: Optional OverlayStep instance for rendering gaze dot and HUD
            udp_forward_step: Optional UDPForwardStep instance for forwarding data via UDP
        """
        self.args = args
        self.calibration = calibration
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
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
        
        # Ensure model is available
        ensure_model()
        logger.info("FrameDispatcher initialized")

    def receiveFrame(self, frame: Any, timestamp_ms: int) -> None:
        """Accept frame from CameraReader and send to processing pipeline.
        
        This method forwards the frame to the FaceMeshWorker for landmark detection.
        
        Args:
            frame: Frame data (numpy array in BGR format)
            timestamp_ms: Frame timestamp in milliseconds
        """
        if self.worker is None:
            logger.warning("Cannot process frame - worker not initialized")
            return
        
        try:
            # Forward frame to worker for processing
            self.worker.submit_frame(frame, timestamp_ms)
            logger.debug(f"Frame forwarded to worker at {timestamp_ms}ms")
        except Exception as e:
            logger.error(f"Error forwarding frame to worker: {e}")

    def start(self) -> None:
        """Start the frame processing pipeline."""
        if self.worker is not None:
            raise RuntimeError("FrameDispatcher already started")
        
        logger.info("Starting FrameDispatcher worker")
        self.worker = FaceMeshWorker(self.args, calibration=self.calibration)
        self.worker.start()
        
        # Wait for worker to be ready
        if not self.worker.ready_evt.wait(timeout=20):
            raise RuntimeError("Timed out waiting for mesh worker init")
        if self.worker.error:
            raise RuntimeError(self.worker.error)
        
        self.display = get_display_geo()
        self.running = True
        logger.info(f"FrameDispatcher started - ready to receive frames from CameraReader on display '{self.display['name']}' ({self.display['width']}x{self.display['height']})")

    def stop(self) -> None:
        """Stop the frame processing pipeline."""
        if self.worker is None:
            return
        
        logger.info("Stopping FrameDispatcher")
        self.running = False
        
        if self.overlay_manager:
            self.overlay_manager.shutdown()
            self.overlay_manager = None
        
        self.worker.stop()
        self.worker.join(timeout=3.0)
        self.worker = None
        logger.info("FrameDispatcher stopped")

    def run_capture_loop(
        self,
        run_duration: float = 0.0,
        on_capture_click: Optional[Callable] = None,
    ) -> None:
        """Run the main capture and display loop.
        
        Args:
            run_duration: Duration in seconds (0 = continuous)
            on_capture_click: Optional callback for capture button clicks
        """
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
        """Run the 9-point calibration workflow.
        
        Returns:
            Tuple of (calibration_matrix, calibration_points)
        """
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
        """Update the calibration matrix used by the worker.
        
        Args:
            calibration: New calibration matrix to apply
        """
        self.calibration = calibration
        if self.worker is not None:
            self.worker.calibration = calibration
            logger.info("Calibration updated in worker")

    def get_latest_event(self) -> Optional[FaceMeshEvent]:
        """Get the latest FaceMesh event.
        
        Returns:
            Latest FaceMeshEvent or None if not available
        """
        if self.worker is None:
            return None
        _, evt = self.worker.snapshot()
        return evt

    def get_status_tail(self, n: int = 20) -> List[Dict]:
        """Get recent status messages from worker.
        
        Args:
            n: Number of recent messages to retrieve
        
        Returns:
            List of status message dictionaries
        """
        if self.worker is None:
            return []
        return self.worker.status_tail(n=n)

    def is_running(self) -> bool:
        """Check if the dispatcher is currently running.
        
        Returns:
            True if running, False otherwise
        """
        return self.running and self.worker is not None


class FrameDispatcherV2:
    """
    FrameDispatcherV2 implements the new pipeline architecture for frame processing.
    
    This class replaces the old FrameDispatcher with a modular, state-machine-based
    pipeline architecture while maintaining backward-compatible API.
    
    The pipeline consists of 6 steps:
    1. FaceMeshStep - Extract face mesh data using MediaPipe
    2. CalibrationAdapterStep - Convert to calibrated event with display geometry
    3. CalibrationControllerStep - Manage calibration workflow
    4. CaptureStep - Handle live preview and frame counting
    5. OverlayStep - Render gaze dot and HUD overlay
    6. UDPForwardStep - Forward calibrated data via UDP
    
    State machine manages transitions between:
    - IDLE: Initial state, not processing frames
    - CALIBRATION: Collecting calibration data
    - OPERATIONAL: Normal operation with full pipeline
    """
    
    def __init__(self,
                 face_landmarker: vision.FaceLandmarker,
                 num_calibration_points: int = 9,
                 calibration_threshold: float = 0.5,
                 min_calibration_samples: int = 5,
                 display_width: int = 1920,
                 display_height: int = 1080,
                 origin_x: float = 960.0,
                 origin_y: float = 540.0,
                 state_machine: Optional[StateMachine] = None,
                 face_mesh_step: Optional[FaceMeshStep] = None,
                 calibration_adapter_step: Optional[CalibrationAdapterStep] = None,
                 calibration_controller_step: Optional[CalibrationControllerStep] = None,
                 capture_step: Optional[CaptureStep] = None,
                 overlay_step: Optional[OverlayStep] = None,
                 udp_forward_step: Optional[UDPForwardStep] = None,
                 calibration: Optional[CalibrationMatrix] = None):
        """
        Initialize FrameDispatcherV2 with pipeline architecture.
        
        Args:
            face_landmarker: MediaPipe FaceLandmarker instance
            num_calibration_points: Number of calibration points (default: 9)
            calibration_threshold: Stability threshold for calibration in degrees (default: 0.5)
            min_calibration_samples: Minimum number of samples required per calibration point (default: 5)
            display_width: Display width in pixels (default: 1920)
            display_height: Display height in pixels (default: 1080)
            origin_x: X coordinate of origin (default: 960.0)
            origin_y: Y coordinate of origin (default: 540.0)
            state_machine: Optional StateMachine instance for managing dispatcher states
            face_mesh_step: Optional FaceMeshStep instance for face mesh processing
            calibration_adapter_step: Optional CalibrationAdapterStep instance for display geometry
            calibration_controller_step: Optional CalibrationControllerStep instance for calibration workflow
            capture_step: Optional CaptureStep instance for live preview and frame counting
            overlay_step: Optional OverlayStep instance for rendering gaze dot and HUD
            udp_forward_step: Optional UDPForwardStep instance for forwarding data via UDP
            calibration: Optional CalibrationMatrix instance for gaze correction (default: None)
        """
        logger.info("Initializing FrameDispatcherV2 with pipeline architecture")
        
        # Initialize capture callback
        self.capture_callback: Optional[Callable[[np.ndarray, Optional['FaceMeshEvent'], Optional['CalibratedFaceAndGazeEvent']], None]] = None
        logger.debug("Capture callback initialized to None")
        
        # Initialize state machine (use provided instance or create default)
        self.state_machine = state_machine or StateMachine(initial_state=DispatcherState.IDLE)
        logger.debug(f"State machine initialized in {self.state_machine.get_state().value} state")
        
        # Initialize all pipeline steps (use provided instances or create defaults)
        self.face_mesh_step = face_mesh_step or FaceMeshStep(face_landmarker)
        logger.debug("FaceMeshStep initialized")
        
        # Extract calibration values from calibration matrix if provided
        pitch_calibration = 0.0
        yaw_calibration = 0.0
        roll_calibration = 0.0
        if calibration is not None:
            pitch_calibration = calibration.center_pitch
            yaw_calibration = calibration.center_yaw
            roll_calibration = 0.0
        
        self.calibration_adapter_step = calibration_adapter_step or CalibrationAdapterStep(
            pitch_calibration=pitch_calibration,
            yaw_calibration=yaw_calibration,
            roll_calibration=roll_calibration,
            display_width=display_width,
            display_height=display_height,
            origin_x=origin_x,
            origin_y=origin_y
        )
        logger.debug(f"CalibrationAdapterStep initialized with display geometry: {display_width}x{display_height}, "
                    f"origin=({origin_x}, {origin_y}), calibration=({pitch_calibration:.4f}, {yaw_calibration:.4f}, {roll_calibration:.4f})")
        
        self.calibration_controller_step = calibration_controller_step or CalibrationControllerStep(
            face_landmarker=face_landmarker,
            num_points=num_calibration_points,
            threshold=calibration_threshold,
            min_samples=min_calibration_samples
        )
        logger.debug(f"CalibrationControllerStep initialized: num_points={num_calibration_points}, "
                    f"threshold={calibration_threshold}, min_samples={min_calibration_samples}")
        
        self.capture_step = capture_step or CaptureStep(enabled=True)
        logger.debug("CaptureStep initialized (enabled=True)")
        
        self.overlay_step = overlay_step or OverlayStep(enabled=True, show_hud=True)
        logger.debug("OverlayStep initialized (enabled=True, show_hud=True)")
        
        self.udp_forward_step = udp_forward_step or UDPForwardStep(
            host="127.0.0.1",
            port=5005,
            enabled=False
        )
        logger.debug("UDPForwardStep initialized (host=127.0.0.1, port=5005, enabled=False)")
        
        # Store display geometry for reference
        self.display_width = display_width
        self.display_height = display_height
        self.origin_x = origin_x
        self.origin_y = origin_y
        
        # Initialize statistics counters
        self.frames_processed: int = 0
        self.faces_detected: int = 0
        self.frames_with_face: int = 0
        self.frames_without_face: int = 0
        logger.debug("Statistics counters initialized")
        
        logger.info("FrameDispatcherV2 initialization complete")
    
    def receiveFrame(self, frame: np.ndarray) -> None:
        """
        Receive and process a frame through the pipeline.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        State machine logic:
        - If CALIBRATION state: route to calibration controller only
        - If OPERATIONAL state: route through full pipeline
        - If IDLE state: do nothing
        """
        if frame is None:
            logger.debug("Received None frame, skipping")
            return
        
        current_state = self.state_machine.get_state()
        logger.debug(f"receiveFrame: current state={current_state.value}")
        
        if current_state == DispatcherState.CALIBRATION:
            # Calibration mode: only process with calibration controller
            logger.debug("Processing frame in CALIBRATION state")
            
            calibration_result = self.calibration_controller_step.receive_frame(frame)
            
            if calibration_result is not None:
                # Calibration complete
                logger.info("Calibration result received, handling completion")
                self._handle_calibration_complete(calibration_result)
                return
        
        elif current_state == DispatcherState.OPERATIONAL:
            # Operational mode: full pipeline
            logger.debug("Processing frame in OPERATIONAL state")
            
            # Initialize variables
            face_mesh_event = None
            calibrated_event = None
            frame_with_overlay = frame.copy() if frame is not None else None
            
            # Step 1: Face mesh detection
            timestamp_ms = int(time.time() * 1000)
            face_mesh_event = self.face_mesh_step.receive_frame(frame, timestamp_ms)
            
            if face_mesh_event is not None:
                # Step 2: Calibration adaptation
                calibrated_event = self.calibration_adapter_step.receive_frame(frame, face_mesh_event)
                
                if calibrated_event is not None:
                    # Step 3: Capture (conditional)
                    self.capture_step.receive_frame(frame, face_mesh_event, calibrated_event)
                    
                    # Step 4: Overlay (conditional)
                    frame_with_overlay = self.overlay_step.receive_frame(frame, face_mesh_event, calibrated_event)
                    
                    # Step 5: UDP forwarding (always runs)
                    self.udp_forward_step.receive_frame(frame_with_overlay, face_mesh_event, calibrated_event)
                    
                    # Step 6: Update capture display with overlay (if capture is enabled)
                    if self.capture_step.enabled and frame_with_overlay is not None:
                        cv2.imshow("Preview", frame_with_overlay)
                else:
                    logger.warning("Failed to create calibrated event, skipping downstream steps")
            else:
                logger.debug("No face detected, skipping pipeline steps")
            
            # Call capture callback if set (called even if no face detected)
            if self.capture_callback is not None:
                logger.debug("Calling capture callback")
                self.capture_callback(frame_with_overlay, face_mesh_event, calibrated_event)
        
        else:  # IDLE state
            logger.debug("Frame received in IDLE state, doing nothing")
            # IDLE state: do nothing
        
        # Track statistics for all states
        self.frames_processed += 1
        
        # Track face detection statistics (only in OPERATIONAL state)
        if current_state == DispatcherState.OPERATIONAL:
            if face_mesh_event is not None and face_mesh_event.has_face:
                self.faces_detected += 1
                self.frames_with_face += 1
            else:
                self.frames_without_face += 1
    
    def _handle_calibration_complete(self, calibration_result: dict) -> None:
        """
        Handle calibration completion.
        
        Extracts pitch/yaw/roll calibration values from result, updates
        calibration_adapter_step, transitions to OPERATIONAL state, and logs completion.
        
        Args:
            calibration_result: Dictionary containing calibration data with keys:
                - pitch: Pitch calibration value in degrees
                - yaw: Yaw calibration value in degrees
                - roll: Roll calibration value in degrees
        """
        logger.info("Handling calibration completion")
        
        # Extract calibration values
        pitch = calibration_result.get("pitch", 0.0)
        yaw = calibration_result.get("yaw", 0.0)
        roll = calibration_result.get("roll", 0.0)
        
        logger.info(f"Calibration values: pitch={pitch:.4f}, yaw={yaw:.4f}, roll={roll:.4f}")
        
        # Update calibration adapter step
        self.calibration_adapter_step.update_calibration(pitch=pitch, yaw=yaw, roll=roll)
        
        # Transition state to OPERATIONAL
        logger.info("Transitioning to OPERATIONAL state after calibration")
        self.state_machine.transition_to(DispatcherState.OPERATIONAL)
        
        logger.info("Calibration completion handling finished")
    
    def start_calibration(self) -> None:
        """
        Start the calibration process.
        
        Transitions to CALIBRATION state and starts calibration controller.
        """
        logger.info("Starting calibration process")
        logger.debug(f"Current state: {self.state_machine.get_state().value}")
        
        # Transition to CALIBRATION state
        self.state_machine.transition_to(DispatcherState.CALIBRATION)
        logger.info(f"Transitioned to CALIBRATION state: {self.state_machine.get_state().value}")
        
        # Start calibration controller
        self.calibration_controller_step.start_calibration()
        logger.info("Calibration controller started")
    
    def set_capture_enabled(self, enabled: bool) -> None:
        """
        Enable or disable capture/preview.
        
        Args:
            enabled: Whether capture is enabled
        """
        logger.debug(f"Setting capture enabled to: {enabled}")
        self.capture_step.set_enabled(enabled)
        logger.info(f"Capture {'enabled' if enabled else 'disabled'}")
    
    def set_overlay_enabled(self, enabled: bool) -> None:
        """
        Enable or disable overlay rendering.
        
        Args:
            enabled: Whether overlay is enabled
        """
        logger.debug(f"Setting overlay enabled to: {enabled}")
        self.overlay_step.set_enabled(enabled)
        logger.info(f"Overlay {'enabled' if enabled else 'disabled'}")
    
    def set_overlay_show_hud(self, show_hud: bool) -> None:
        """
        Enable or disable HUD display.
        
        Args:
            show_hud: Whether to show HUD
        """
        logger.debug(f"Setting overlay show_hud to: {show_hud}")
        self.overlay_step.set_show_hud(show_hud)
        logger.info(f"Overlay HUD {'enabled' if show_hud else 'disabled'}")
    
    def set_udp_forwarding_enabled(self, enabled: bool) -> None:
        """
        Enable or disable UDP forwarding.
        
        Args:
            enabled: Whether UDP forwarding is enabled
        """
        logger.debug(f"Setting UDP forwarding enabled to: {enabled}")
        self.udp_forward_step.set_enabled(enabled)
        logger.info(f"UDP forwarding {'enabled' if enabled else 'disabled'}")
    
    def update_calibration(self,
                           pitch: float,
                           yaw: float,
                           roll: float) -> None:
        """
        Update calibration values manually.
        
        Args:
            pitch: Pitch calibration value in degrees
            yaw: Yaw calibration value in degrees
            roll: Roll calibration value in degrees
        """
        logger.debug(f"Updating calibration: pitch={pitch:.4f}, yaw={yaw:.4f}, roll={roll:.4f}")
        self.calibration_adapter_step.update_calibration(pitch=pitch, yaw=yaw, roll=roll)
        logger.info(f"Calibration updated: pitch={pitch:.4f}, yaw={yaw:.4f}, roll={roll:.4f}")
    
    def update_display_geometry(self,
                                 width: int,
                                 height: int,
                                 origin_x: float,
                                 origin_y: float) -> None:
        """
        Update display geometry.
        
        Args:
            width: Display width in pixels
            height: Display height in pixels
            origin_x: X coordinate of origin
            origin_y: Y coordinate of origin
        """
        logger.debug(f"Updating display geometry: {width}x{height}, origin=({origin_x}, {origin_y})")
        
        # Update instance variables
        self.display_width = width
        self.display_height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        
        # Update calibration adapter step
        self.calibration_adapter_step.update_display_geometry(
            width=width,
            height=height,
            origin_x=origin_x,
            origin_y=origin_y
        )
        
        logger.info(f"Display geometry updated: {width}x{height}, origin=({origin_x}, {origin_y})")
    
    def get_state(self) -> DispatcherState:
        """
        Get current state.
        
        Returns:
            Current dispatcher state
        """
        return self.state_machine.get_state()
    
    def set_capture_callback(self, callback: Optional[Callable[[np.ndarray, Optional['FaceMeshEvent'], Optional['CalibratedFaceAndGazeEvent']], None]]) -> None:
        """
        Set callback for saving captured frames.
        
        Args:
            callback: Function to call for each captured frame.
                     Signature: callback(frame, face_mesh_event, calibrated_event) -> None
        """
        self.capture_callback = callback
        logger.info(f"Capture callback {'set' if callback is not None else 'cleared'}")
    
    def get_calibration_progress(self) -> float:
        """
        Get calibration progress (0.0 to 1.0).
        
        Returns:
            Calibration progress as a fraction (0.0 = not started, 1.0 = complete)
        """
        progress = self.calibration_controller_step.get_progress()
        logger.debug(f"Calibration progress: {progress:.2f}")
        return progress
    
    def get_calibration_current_point_index(self) -> int:
        """
        Get current calibration point index (0-based).
        
        Returns:
            Current calibration point index (0 to num_points-1)
        """
        point_index = self.calibration_controller_step.get_current_point_index()
        logger.debug(f"Current calibration point index: {point_index}")
        return point_index
    
    def is_calibration_complete(self) -> bool:
        """
        Check if calibration is complete.
        
        Returns:
            True if calibration is complete, False otherwise
        """
        complete = self.calibration_controller_step.is_calibration_complete()
        logger.debug(f"Calibration complete: {complete}")
        return complete
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive calibration status.
        
        Returns:
            Dictionary with calibration status information:
            {
                'state': DispatcherState,
                'progress': float,
                'current_point_index': int,
                'num_points': int,
                'is_complete': bool,
                'current_calibration': dict or None  # pitch/yaw/roll values if complete
            }
        """
        status = {
            'state': self.state_machine.get_state(),
            'progress': self.get_calibration_progress(),
            'current_point_index': self.get_calibration_current_point_index(),
            'num_points': self.calibration_controller_step.num_points,
            'is_complete': self.is_calibration_complete(),
            'current_calibration': None
        }
        
        # Add current calibration values if available
        if status['is_complete'] or status['state'] == DispatcherState.OPERATIONAL:
            status['current_calibration'] = {
                'pitch': self.calibration_adapter_step.pitch_calibration,
                'yaw': self.calibration_adapter_step.yaw_calibration,
                'roll': self.calibration_adapter_step.roll_calibration
            }
        
        logger.debug(f"Calibration status: state={status['state'].value}, "
                    f"progress={status['progress']:.2f}, "
                    f"point_index={status['current_point_index']}/{status['num_points']}, "
                    f"is_complete={status['is_complete']}, "
                    f"has_calibration={status['current_calibration'] is not None}")
        
        return status
    
    def get_statistics(self) -> dict:
        """
        Get performance and operational statistics.
        
        Returns:
            Dictionary with statistics:
            {
                'frames_processed': int,
                'faces_detected': int,
                'frames_with_face': int,
                'frames_without_face': int,
                'capture_frame_count': int,
                'state': DispatcherState,
                'calibration_complete': bool
            }
        """
        stats = {
            'frames_processed': self.frames_processed,
            'faces_detected': self.faces_detected,
            'frames_with_face': self.frames_with_face,
            'frames_without_face': self.frames_without_face,
            'capture_frame_count': self.capture_step.get_frame_count(),
            'state': self.state_machine.get_state(),
            'calibration_complete': self.is_calibration_complete()
        }
        
        logger.debug(f"Statistics: frames={stats['frames_processed']}, "
                    f"faces_detected={stats['faces_detected']}, "
                    f"with_face={stats['frames_with_face']}, "
                    f"without_face={stats['frames_without_face']}, "
                    f"capture_count={stats['capture_frame_count']}, "
                    f"state={stats['state'].value}, "
                    f"calib_complete={stats['calibration_complete']}")
        
        return stats
    
    def reset_statistics(self) -> None:
        """
        Reset all statistics counters.
        
        Note: capture_step frame count is not reset (it tracks total lifetime).
        """
        self.frames_processed = 0
        self.faces_detected = 0
        self.frames_with_face = 0
        self.frames_without_face = 0
        
        logger.info("Statistics reset")
    
    def set_state_transition_callback(self, callback: Callable[[DispatcherState, DispatcherState], None]) -> None:
        """
        Set callback for state transitions.
        
        Allows external code (e.g., UI) to monitor and react to state changes.
        
        Args:
            callback: Function to call on state transition.
                     Signature: callback(old_state: DispatcherState, new_state: DispatcherState) -> None
        """
        logger.debug("Setting state transition callback")
        self.state_machine.set_transition_callback(callback)
        logger.info("State transition callback set")
    
    def clear_state_transition_callback(self) -> None:
        """
        Clear state transition callback.
        
        Removes the previously registered callback, if any.
        """
        logger.debug("Clearing state transition callback")
        self.state_machine.clear_transition_callback()
        logger.info("State transition callback cleared")
    
    def get_latest_event(self) -> Optional[FaceMeshEvent]:
        """
        Get the latest FaceMeshEvent.
        
        Returns:
            Latest FaceMeshEvent or None if no event available
        """
        # In the new pipeline architecture, events are processed synchronously
        # We don't store a "latest event" like the old worker-based architecture
        # This method is provided for backward compatibility but returns None
        logger.debug("get_latest_event() called - returning None (events are processed synchronously)")
        return None
    
    def get_status_tail(self, n: int = 20) -> List[Dict]:
        """
        Get recent status messages.
        
        In the new pipeline architecture, there's no worker thread with status messages.
        This method is provided for backward compatibility but returns empty list.
        
        Args:
            n: Number of recent messages to retrieve (ignored in new architecture)
        
        Returns:
            Empty list (no worker status in new architecture)
        """
        logger.debug(f"get_status_tail({n}) called - returning empty list (no worker in new architecture)")
        return []
    
    def is_running(self) -> bool:
        """
        Check if the dispatcher is currently running.
        
        In the new pipeline architecture, running is determined by the state machine.
        Returns True if state is not IDLE, False otherwise.
        
        Returns:
            True if running (not in IDLE state), False otherwise
        """
        current_state = self.state_machine.get_state()
        is_running = current_state != DispatcherState.IDLE
        logger.debug(f"is_running() = {is_running} (state={current_state.value})")
        return is_running
