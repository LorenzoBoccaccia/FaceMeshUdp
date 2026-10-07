"""
FrameDispatcher module for FaceMesh application.
Orchestrates the synchronous frame processing pipeline.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable, Any

import cv2
import numpy as np

from .facemesh_dao import FaceMeshEvent
from .calibration import (
    CALIBRATION_MODEL_VERSION,
    CalibratedFaceAndGazeEvent,
    CalibrationPoint,
    CalibrationProfile,
    GazeModel,
    Screen,
    save_calibration,
)
from .capture import save_capture, build_camera_capture_marked_image
from .capture_window import CaptureWindowManager
from .overlay_calibration import CalibrationOverlayManager
from .overlay_common import get_display_geo
from .overlay_runtime import RuntimeOverlayManager
from .state_machine import DispatcherState

logger = logging.getLogger(__name__)

CALIBRATION_DATA_DIR = Path("calibration_data")
CALIBRATION_DATAPOINT_DIR = Path("calibration_datapoint")


class FrameDispatcher:
    """Synchronous frame processing dispatcher coordinating pipeline steps."""

    def __init__(
        self,
        args,
        overlay_manager=None,
        state_machine=None,
        face_mesh_step=None,
        calibration_adapter_step=None,
        blink_rejection_step=None,
        gaze_smoothing_step=None,
        opentrack_forward_step=None,
        freetrack_forward_step=None,
    ):
        self.args = args
        self.overlay_manager = overlay_manager
        self.state_machine = state_machine
        self.face_mesh_step = face_mesh_step
        self.calibration_adapter_step = calibration_adapter_step
        self.blink_rejection_step = blink_rejection_step
        self.gaze_smoothing_step = gaze_smoothing_step
        self.opentrack_forward_step = opentrack_forward_step
        self.freetrack_forward_step = freetrack_forward_step

        self.display: Optional[Dict] = None
        self.running = False

        self._latest_evt: Optional[FaceMeshEvent] = None

    def start(self):
        """Initialize display geometry."""
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

    def _run_pipeline_steps(
        self,
        frame: np.ndarray,
        evt: Optional[FaceMeshEvent],
        run_downstream: bool = True,
    ) -> Tuple[Optional[CalibratedFaceAndGazeEvent], np.ndarray]:
        """Run calibration adapter and downstream pipeline steps for a frame."""
        calibrated_evt = None
        if self.calibration_adapter_step is not None:
            calibrated_evt = self.calibration_adapter_step.receive_frame(frame, evt)
        if self.blink_rejection_step is not None:
            calibrated_evt = self.blink_rejection_step.receive_frame(frame, evt, calibrated_evt)

        pipeline_frame = frame
        if not run_downstream:
            return calibrated_evt, pipeline_frame

        gaze = None
        if self.gaze_smoothing_step is not None:
            gaze = self.gaze_smoothing_step.receive_frame(
                pipeline_frame, evt, calibrated_evt
            )

        if self.opentrack_forward_step is not None:
            self.opentrack_forward_step.receive_frame(
                pipeline_frame, evt, calibrated_evt, gaze
            )

        if self.freetrack_forward_step is not None:
            self.freetrack_forward_step.receive_frame(
                pipeline_frame, evt, calibrated_evt, gaze
            )

        return calibrated_evt, pipeline_frame

    def _calibration_sample_payload(
        self,
        evt: Optional[FaceMeshEvent],
        calibrated_evt: Optional[CalibratedFaceAndGazeEvent],
        timestamp_ms: int,
        phase: str,
        current_point: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build one calibration diagnostic sample for offline analysis."""
        has_face = bool(evt is not None and evt.has_face)
        position = evt.eye_position if has_face else None
        return {
            "frameTimestampMs": int(timestamp_ms),
            "phase": str(phase),
            "target": current_point,
            "hasFace": has_face,
            "headYaw": evt.head_yaw if has_face else None,
            "headPitch": evt.head_pitch if has_face else None,
            "headRoll": evt.roll if has_face else None,
            "eyePositionMm": position.tolist() if position is not None else None,
            "rawEye": (
                [evt.combined_eye_gaze_yaw, evt.combined_eye_gaze_pitch]
                if has_face
                else None
            ),
            "geometryInputs": evt.geometry_inputs() if has_face else None,
            "gaze": calibrated_evt.gaze.to_dict() if calibrated_evt else None,
        }

    def _save_calibration_session_data(
        self,
        session_timestamp_ms: int,
        samples: List[Dict[str, Any]],
        points: List[CalibrationPoint],
        model: Optional[GazeModel],
    ) -> Path:
        """Persist one calibration session payload for diagnostics."""
        CALIBRATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "modelVersion": CALIBRATION_MODEL_VERSION,
            "sessionTimestampMs": int(session_timestamp_ms),
            "profile": getattr(self.args, "calibration_profile", "") or "default",
            "display": self.display,
            "sampleCount": len(samples),
            "samples": samples,
            "points": [point.to_dict() for point in points],
            "model": model.to_dict() if model is not None else None,
        }
        path = CALIBRATION_DATA_DIR / f"calibration_session_{session_timestamp_ms}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _clear_calibration_session_data(self) -> int:
        """Ensure calibration diagnostics start with one fresh session payload."""
        CALIBRATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
        removed = 0
        for path in CALIBRATION_DATA_DIR.glob("calibration_session_*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                logger.warning("Failed to remove calibration session file: %s", path)
        CALIBRATION_DATAPOINT_DIR.mkdir(parents=True, exist_ok=True)
        for path in CALIBRATION_DATAPOINT_DIR.glob("*"):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                logger.warning("Failed to remove calibration datapoint file: %s", path)
        return removed

    def _save_calibration_datapoint(
        self,
        calib_point: CalibrationPoint,
        evt: Optional[FaceMeshEvent],
        frame: Any,
        timestamp_ms: int,
    ) -> None:
        """Dump per-point diagnostics (overlayed PNG + JSON) named by point position."""
        if self.display is None:
            return
        CALIBRATION_DATAPOINT_DIR.mkdir(parents=True, exist_ok=True)
        name = str(calib_point.name)
        img, err = build_camera_capture_marked_image(
            {"evt": evt, "frame": frame},
            overlay_w=float(self.display["width"]),
            overlay_h=float(self.display["height"]),
            click_pos=calib_point.nose_target_px,
            draw_click=True,
            draw_info_panel=True,
        )
        if img is not None:
            cv2.imwrite(str(CALIBRATION_DATAPOINT_DIR / f"{name}.png"), img)
        elif err:
            logger.warning("Calibration datapoint image failed for %s: %s", name, err)

        payload = {"timestamp_ms": timestamp_ms, "point": calib_point.to_dict()}
        try:
            (CALIBRATION_DATAPOINT_DIR / f"{name}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Calibration datapoint JSON failed for %s: %s", name, exc)

    def _transition_state(self, new_state: DispatcherState) -> None:
        if self.state_machine is None:
            return
        current_state = self.state_machine.get_state()
        if current_state == new_state:
            return
        self.state_machine.transition_to(new_state)

    def run_capture_loop(
        self, camera_reader, on_capture_click: Optional[Callable] = None
    ):
        """Run the main capture and display loop until user exits."""
        if self.display is None:
            raise RuntimeError("FrameDispatcher not started")
        self.start_operational()

        overlay_enabled = bool(self.args.overlay)
        capture_enabled = bool(self.args.capture)
        capture_live_enabled = bool(capture_enabled and self.args.capture_live)
        quiet = bool(getattr(self.args, "quiet", False))
        log_interval = float(getattr(self.args, "log_interval", 2.0))

        w = int(self.display["width"])
        h = int(self.display["height"])
        overlay_manager: Optional[RuntimeOverlayManager] = None
        capture_window_manager: Optional[CaptureWindowManager] = None
        last_log_time = time.time()
        running = True
        pixel_format = camera_reader.pixel_format

        camera_fps = float(getattr(camera_reader, "fps", 0.0) or 0.0)
        camera_period_ms = (1000.0 / camera_fps) if camera_fps > 0 else 0.0
        ewma_read_ms: Optional[float] = None
        ewma_proc_ms: Optional[float] = None
        ewma_alpha = 0.1
        frames_since_log = 0
        last_jump_count = 0

        try:
            if overlay_enabled:
                overlay_manager = RuntimeOverlayManager(
                    self.display,
                    overlay_fps=self.args.overlay_fps,
                )
                overlay_manager.initialize()

            if capture_enabled:
                capture_window_manager = CaptureWindowManager(self.display)
                capture_window_manager.initialize()

            while running:
                t_read_start = time.perf_counter()
                frame, timestamp_ms = camera_reader.read_frame()
                t_read_ms = (time.perf_counter() - t_read_start) * 1000.0
                if frame is None:
                    time.sleep(0.001)
                    continue

                t_proc_start = time.perf_counter()
                evt = self._process_frame(frame, timestamp_ms, pixel_format)
                calibrated_evt, _ = self._run_pipeline_steps(
                    frame, evt, run_downstream=True
                )
                t_proc_ms = (time.perf_counter() - t_proc_start) * 1000.0

                ewma_read_ms = (
                    t_read_ms
                    if ewma_read_ms is None
                    else ewma_alpha * t_read_ms + (1.0 - ewma_alpha) * ewma_read_ms
                )
                ewma_proc_ms = (
                    t_proc_ms
                    if ewma_proc_ms is None
                    else ewma_alpha * t_proc_ms + (1.0 - ewma_alpha) * ewma_proc_ms
                )
                frames_since_log += 1

                if not quiet and log_interval > 0:
                    now = time.time()
                    if now - last_log_time >= log_interval:
                        elapsed = now - last_log_time
                        processed_fps = (
                            frames_since_log / elapsed if elapsed > 0 else 0.0
                        )
                        buffering = (
                            camera_period_ms > 0
                            and ewma_proc_ms is not None
                            and ewma_read_ms is not None
                            and ewma_proc_ms > camera_period_ms
                            and ewma_read_ms < 0.5 * camera_period_ms
                        )
                        logger.info(
                            f"Pipeline: fps={processed_fps:.1f} "
                            f"t_read={ewma_read_ms:.1f}ms "
                            f"t_proc={ewma_proc_ms:.1f}ms "
                            f"cam_period={camera_period_ms:.1f}ms"
                            + (" [BUFFERING]" if buffering else "")
                        )
                        last_log_time = now
                        frames_since_log = 0
                        if (
                            self.gaze_smoothing_step is not None
                            and self.gaze_smoothing_step.reset_window_ms is not None
                        ):
                            jump_count = self.gaze_smoothing_step.jump_count
                            logger.info(
                                f"Gaze jumps: {jump_count - last_jump_count} "
                                f"in {elapsed:.1f}s"
                            )
                            last_jump_count = jump_count
                        if calibrated_evt is not None:
                            gaze = calibrated_evt.gaze
                            logger.info(
                                f"Gaze ({gaze.yaw:.1f}, {gaze.pitch:.1f}) deg on screen, "
                                f"head ({evt.head_yaw:.1f}, {evt.head_pitch:.1f}), "
                                f"eye ({gaze.eye_yaw:.1f}, {gaze.eye_pitch:.1f})"
                            )
                        elif evt is not None and evt.has_face:
                            logger.info("Face detected, no calibrated gaze")
                        elif evt is not None:
                            logger.info("No face detected")

                if overlay_manager is not None:
                    overlay_manager.handle_events()
                    if not overlay_manager.is_running():
                        running = False
                        break

                capture_live_img = None
                if capture_live_enabled and capture_window_manager is not None:
                    mouse_x, mouse_y = capture_window_manager.get_mouse_position()
                    snap = {"evt": evt, "frame": frame}
                    capture_live_img, _ = build_camera_capture_marked_image(
                        snap,
                        overlay_w=float(w),
                        overlay_h=float(h),
                        click_pos=(mouse_x, mouse_y),
                        draw_click=False,
                        draw_info_panel=False,
                    )

                if capture_window_manager is not None:
                    capture_window_manager.render(evt, calibrated_evt, capture_live_img)
                    if not capture_window_manager.is_running():
                        running = False
                        break
                    clicked = capture_window_manager.consume_click()
                    if clicked is not None:
                        if on_capture_click is not None:
                            on_capture_click(clicked)
                        else:
                            save_capture(
                                self.display,
                                w,
                                h,
                                clicked,
                                frame,
                                evt,
                                calibrated_evt,
                            )

                if overlay_manager is not None:
                    overlay_manager.render(calibrated_evt)
        finally:
            if overlay_manager is not None:
                overlay_manager.shutdown()
            if capture_window_manager is not None:
                capture_window_manager.shutdown()

    def run_calibration_workflow(
        self, camera_reader, viewing_distance_mm: float
    ) -> Optional[GazeModel]:
        """Execute the 9-point calibration workflow with on-screen guidance."""
        if self.display is None:
            raise RuntimeError("FrameDispatcher not started")
        screen = Screen.from_display(self.display)
        if screen is None:
            raise RuntimeError(
                "The display does not report its physical size, which calibration needs "
                "to place the targets in millimetres."
            )
        self.start_calibration()

        logger.info("Starting 9-point calibration workflow...")
        print("Starting 9-point calibration workflow...", flush=True)
        print(
            "Look forward for center, then align nose and eye with the dual targets at each step.",
            flush=True,
        )
        cleared_sessions = self._clear_calibration_session_data()
        if cleared_sessions > 0:
            print(
                f"Removed {cleared_sessions} previous calibration session file(s).",
                flush=True,
            )

        try:
            if not isinstance(self.overlay_manager, CalibrationOverlayManager):
                if self.overlay_manager is not None:
                    self.overlay_manager.shutdown()
                self.overlay_manager = CalibrationOverlayManager(
                    self.display,
                    overlay_fps=self.args.overlay_fps,
                )
            self.overlay_manager.initialize()
            self.overlay_manager.start_calibration_sequence(
                self.display["width"], self.display["height"]
            )

            calib_points: List[CalibrationPoint] = []
            session_timestamp_ms = int(time.time() * 1000)
            calibration_samples: List[Dict[str, Any]] = []
            pixel_format = camera_reader.pixel_format

            while True:
                frame, timestamp_ms = camera_reader.read_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                evt = self._process_frame(frame, timestamp_ms, pixel_format)
                calibrated_evt, _ = self._run_pipeline_steps(
                    frame, evt, run_downstream=False
                )

                self.overlay_manager.handle_events()
                if not self.overlay_manager.is_running():
                    print("Calibration cancelled by user.", flush=True)
                    break

                calibration_samples.append(
                    self._calibration_sample_payload(
                        evt=evt,
                        calibrated_evt=calibrated_evt,
                        timestamp_ms=timestamp_ms,
                        phase=self.overlay_manager.get_calibration_phase(),
                        current_point=self.overlay_manager.get_current_calib_point(),
                    )
                )

                completed, calib_point = self.overlay_manager.update_calibration_state(evt)
                self.overlay_manager.render()

                if calib_point is not None:
                    calib_points.append(calib_point)
                    print(
                        f"Calibration point {len(calib_points)}/9 completed at "
                        f"'{calib_point.name}' from {calib_point.sample_count} frames",
                        flush=True,
                    )
                    self._save_calibration_datapoint(
                        calib_point=calib_point,
                        evt=evt,
                        frame=frame,
                        timestamp_ms=timestamp_ms,
                    )

                if completed:
                    break

                time.sleep(0.001)

            model = None
            if len(calib_points) == 9:
                profile = CalibrationProfile(
                    screen=screen,
                    viewing_distance_mm=viewing_distance_mm,
                    points=tuple(calib_points),
                )
                model = profile.fit()
                profile_name = getattr(self.args, "calibration_profile", "") or "default"
                print(f"Calibration saved to: {save_calibration(profile, profile_name)}", flush=True)
                print(f"Calibration model: {model.describe()}", flush=True)
                for point in calib_points:
                    error = model.point_errors_px(point)
                    print(
                        f"  {point.name:>3s}: "
                        + (
                            "gaze misses the screen plane"
                            if error is None
                            else f"gaze {error[0]:+7.1f}, {error[1]:+7.1f} px from the eye target"
                        ),
                        flush=True,
                    )
                self.set_model(model)
                self.start_operational()
            else:
                print(
                    f"Calibration ended with {len(calib_points)}/9 points; nothing was saved.",
                    flush=True,
                )

            calibration_data_path = self._save_calibration_session_data(
                session_timestamp_ms=session_timestamp_ms,
                samples=calibration_samples,
                points=calib_points,
                model=model,
            )
            print(f"Calibration diagnostics saved to: {calibration_data_path}", flush=True)
            return model

        except Exception as e:
            print(f"Error during calibration: {e}", flush=True)
            logger.exception("Calibration workflow exception")
            raise
        finally:
            if self.overlay_manager:
                self.overlay_manager.shutdown()
                print("Calibration overlay shutdown complete.", flush=True)

    def set_model(self, model: GazeModel) -> None:
        """Use a new calibration model for the gaze output."""
        if self.calibration_adapter_step is not None:
            self.calibration_adapter_step.set_model(model)

    def get_latest_event(self) -> Optional[FaceMeshEvent]:
        """Return the most recent FaceMeshEvent."""
        return self._latest_evt

    def is_running(self) -> bool:
        """Check if the dispatcher is actively processing."""
        return self.running

    def start_calibration(self) -> None:
        """Transition to CALIBRATION state."""
        self._transition_state(DispatcherState.CALIBRATION)

    def start_operational(self) -> None:
        """Transition to OPERATIONAL state."""
        self._transition_state(DispatcherState.OPERATIONAL)

    def set_opentrack_forwarding_enabled(self, enabled: bool) -> None:
        """Enable or disable OpenTrack data forwarding."""
        if self.opentrack_forward_step is not None:
            self.opentrack_forward_step.set_enabled(enabled)

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
