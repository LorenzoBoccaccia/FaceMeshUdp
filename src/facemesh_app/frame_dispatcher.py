"""
FrameDispatcher module for FaceMesh application.
Orchestrates the synchronous frame processing pipeline.
"""

import logging
import time
import urllib.request
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
    CaptureStep,
    OverlayStep,
    UDPForwardStep,
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def ensure_model():
    """Download the FaceLandmarker model if not present."""
    if MODEL_PATH.exists():
        return
    logger.info(f"Downloading FaceMesh model from {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    logger.info("FaceMesh model downloaded successfully")


def enrich_runtime_evt(
    evt: Optional[FaceMeshEvent], screen_w: float, screen_h: float
) -> Optional[Dict]:
    """Build runtime event payload for overlay and capture rendering."""
    if not evt:
        return None

    raw_left_yaw = evt.left_eye_gaze_yaw
    raw_left_pitch = evt.left_eye_gaze_pitch
    raw_right_yaw = evt.right_eye_gaze_yaw
    raw_right_pitch = evt.right_eye_gaze_pitch

    raw_combined_yaw = None
    if raw_left_yaw is not None and raw_right_yaw is not None:
        raw_combined_yaw = (raw_left_yaw + raw_right_yaw) / 2.0

    raw_combined_pitch = None
    if raw_left_pitch is not None and raw_right_pitch is not None:
        raw_combined_pitch = (raw_left_pitch + raw_right_pitch) / 2.0

    return {
        "type": evt.type,
        "hasFace": evt.has_face,
        "landmarkCount": evt.landmark_count,
        "ts": evt.ts,
        "zeta": evt.zeta,
        "head_yaw": evt.head_yaw,
        "head_pitch": evt.head_pitch,
        "raw_left_eye_gaze_yaw": raw_left_yaw,
        "raw_left_eye_gaze_pitch": raw_left_pitch,
        "raw_right_eye_gaze_yaw": raw_right_yaw,
        "raw_right_eye_gaze_pitch": raw_right_pitch,
        "raw_combined_eye_gaze_yaw": raw_combined_yaw,
        "raw_combined_eye_gaze_pitch": raw_combined_pitch,
    }


class FrameDispatcher:
    """Synchronous frame processing dispatcher coordinating pipeline steps."""

    def __init__(
        self,
        args,
        calibration=None,
        overlay_manager=None,
        state_machine=None,
        face_mesh_step=None,
        calibration_adapter_step=None,
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
        self.capture_step = capture_step
        self.overlay_step = overlay_step
        self.udp_forward_step = udp_forward_step

        self.display: Optional[Dict] = None
        self.running = False

        self.display_width = 0
        self.display_height = 0
        self.origin_x = 0.0
        self.origin_y = 0.0

        self._latest_evt: Optional[FaceMeshEvent] = None

    def start(self):
        """Initialize display geometry."""
        if self.overlay_manager is not None:
            self.display = get_display_geo()
        self.running = True

    def stop(self):
        """Stop and release overlay resources."""
        self.running = False
        if self.overlay_manager:
            self.overlay_manager.shutdown()
            self.overlay_manager = None

    def _process_frame(
        self, frame: np.ndarray, timestamp_ms: int, pixel_format: str = "bgr"
    ) -> Optional[FaceMeshEvent]:
        """Run FaceMesh detection on a single frame."""
        evt = self.face_mesh_step.receive_frame(frame, timestamp_ms, pixel_format)
        self._latest_evt = evt
        return evt

    def run_capture_loop(
        self, camera_reader, on_capture_click: Optional[Callable] = None
    ):
        """Run the main capture and display loop until user exits."""
        if self.display is None:
            raise RuntimeError("FrameDispatcher not started")

        overlay_enabled = bool(self.args.overlay)
        capture_enabled = bool(self.args.capture and overlay_enabled)
        capture_live_enabled = bool(self.args.capture and self.args.capture_live)
        live_window_name = "FaceMesh Capture Live"
        quiet = bool(getattr(self.args, "quiet", False))
        log_interval = float(getattr(self.args, "log_interval", 2.0))

        w = int(self.display["width"])
        h = int(self.display["height"])

        if overlay_enabled:
            if self.overlay_manager is None:
                self.overlay_manager = OverlayManager(
                    self.display, capture_enabled, self.args.overlay_fps
                )
            self.overlay_manager.initialize()

        if capture_live_enabled:
            cv2.namedWindow(live_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(live_window_name, min(1400, w), min(900, h))

        latest_evt = None
        last_log_time = 0.0
        running = True
        pixel_format = camera_reader.pixel_format

        while running:
            frame, timestamp_ms = camera_reader.read_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            evt = self._process_frame(frame, timestamp_ms, pixel_format)

            if evt and evt.type == "mesh":
                latest_evt = enrich_runtime_evt(evt, w, h)

            if not quiet and log_interval > 0:
                now = time.time()
                if now - last_log_time >= log_interval:
                    last_log_time = now
                    if evt is not None and evt.has_face:
                        logger.info(
                            f"Face detected - landmarks: {evt.landmark_count} "
                            f"head=({safe_float(evt.head_yaw):.1f}, {safe_float(evt.head_pitch):.1f}) "
                            f"gaze=({safe_float(evt.combined_eye_gaze_yaw):.1f}, "
                            f"{safe_float(evt.combined_eye_gaze_pitch):.1f})"
                        )
                    elif evt is not None:
                        logger.info("No face detected")

            if self.overlay_manager:
                self.overlay_manager.handle_events(
                    on_click=lambda pos: save_test_capture(
                        self.display,
                        w,
                        h,
                        pos,
                        frame,
                        evt,
                    )
                    if capture_enabled
                    else None
                )
                if not self.overlay_manager.is_running():
                    running = False
                    break

            if capture_live_enabled:
                snap = {
                    "evt": evt,
                    "frame": frame,
                    "landmarks": list(evt.landmarks) if evt and evt.landmarks else None,
                }
                live_img, _ = build_camera_capture_marked_image(
                    snap,
                    overlay_w=float(w),
                    overlay_h=float(h),
                    click_pos=(0.0, 0.0),
                    draw_click=False,
                )
                if live_img is not None:
                    cv2.imshow(live_window_name, live_img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    running = False
                    if self.overlay_manager:
                        self.overlay_manager.request_exit()

            if self.overlay_manager:
                self.overlay_manager.render_mesh(latest_evt)

        if self.overlay_manager:
            self.overlay_manager.shutdown()
            self.overlay_manager = None
        if capture_live_enabled:
            cv2.destroyWindow(live_window_name)

    def run_calibration_workflow(
        self, camera_reader
    ) -> Tuple[Optional[CalibrationMatrix], List[CalibrationPoint]]:
        """Execute the 9-point calibration workflow with on-screen guidance."""
        if self.display is None:
            raise RuntimeError("FrameDispatcher not started")

        logger.info("Starting 9-point calibration workflow...")
        print("Starting 9-point calibration workflow...", flush=True)
        print(
            "Please follow the on-screen instructions and look at each calibration point.",
            flush=True,
        )

        try:
            if self.overlay_manager is None:
                self.overlay_manager = OverlayManager(
                    self.display,
                    capture_enabled=False,
                    overlay_fps=self.args.overlay_fps,
                    calibration_mode=True,
                )
            self.overlay_manager.initialize()
            self.overlay_manager.start_calibration_sequence(
                self.display["width"], self.display["height"]
            )

            calib_points: List[CalibrationPoint] = []
            pixel_format = camera_reader.pixel_format

            while True:
                frame, timestamp_ms = camera_reader.read_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                evt = self._process_frame(frame, timestamp_ms, pixel_format)

                self.overlay_manager.handle_events()
                if not self.overlay_manager.is_running():
                    print("Calibration cancelled by user.", flush=True)
                    break

                evt_dict = enrich_runtime_evt(
                    evt, self.display["width"], self.display["height"]
                )

                completed, calib_point = self.overlay_manager.update_calibration_state(
                    evt_dict
                )
                self.overlay_manager.render_mesh(evt_dict)

                if calib_point is not None:
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
                        print(
                            f"Calibration sequence ended with {len(calib_points)} points.",
                            flush=True,
                        )
                    break

                time.sleep(0.001)

            calib_matrix = None
            if len(calib_points) == 9:
                print("Computing calibration matrix...", flush=True)
                calib_matrix = compute_calibration_matrix(calib_points)

                profile_name = (
                    getattr(self.args, "calibration_profile", "") or "default"
                )
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
                print(
                    f"Insufficient calibration points ({len(calib_points)}/9). Cannot compute calibration matrix.",
                    flush=True,
                )

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
        """Apply a new calibration matrix to the dispatcher."""
        self.calibration = calibration

    def get_latest_event(self) -> Optional[FaceMeshEvent]:
        """Return the most recent FaceMeshEvent."""
        return self._latest_evt

    def is_running(self) -> bool:
        """Check if the dispatcher is actively processing."""
        return self.running

    def _handle_calibration_complete(self, calibration_result: dict) -> None:
        """Apply completed calibration result and transition to OPERATIONAL state."""
        pitch = calibration_result.get("pitch", 0.0)
        yaw = calibration_result.get("yaw", 0.0)
        roll = calibration_result.get("roll", 0.0)

        self.calibration_adapter_step.update_calibration(
            pitch=pitch, yaw=yaw, roll=roll
        )
        self.state_machine.transition_to(DispatcherState.OPERATIONAL)

    def start_calibration(self) -> None:
        """Transition to CALIBRATION state."""
        self.state_machine.transition_to(DispatcherState.CALIBRATION)

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
        self.calibration_adapter_step.update_calibration(
            pitch=pitch, yaw=yaw, roll=roll
        )

    def update_display_geometry(
        self, width: int, height: int, origin_x: float, origin_y: float
    ) -> None:
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

    def set_state_transition_callback(
        self, callback: Callable[[DispatcherState, DispatcherState], None]
    ) -> None:
        """Register a callback for state machine transitions."""
        self.state_machine.set_transition_callback(callback)

    def clear_state_transition_callback(self) -> None:
        """Remove any registered state transition callback."""
        self.state_machine.clear_transition_callback()
