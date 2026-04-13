# main.py
#!/usr/bin/env python3
"""
FaceMesh Data Capture Application
Main entry point for raw face mesh data capture.
"""

import argparse
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .facemesh_dao import load_calibration
from .camera_reader import CameraReader
from .frame_dispatcher import FrameDispatcher
from .overlay import OverlayManager, get_display_geo
from .pipeline_steps import (
    FaceMeshStep,
    CalibrationAdapterStep,
    CalibrationControllerStep,
    CaptureStep,
    OverlayStep,
    UDPForwardStep,
)
from .state_machine import StateMachine


def _backend_string_to_int(backend_str: str) -> int:
    """Convert backend string to cv2 CAP_* constant.
    
    Args:
        backend_str: Backend string ('auto', 'msmf', 'dshow', 'any')
    
    Returns:
        Corresponding cv2 CAP_* constant (defaults to CAP_ANY)
    """
    backend_map = {
        'auto': cv2.CAP_ANY,
        'msmf': cv2.CAP_MSMF,
        'dshow': cv2.CAP_DSHOW,
        'any': cv2.CAP_ANY
    }
    return backend_map.get(backend_str.lower(), cv2.CAP_ANY)


MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def ensure_model():
    """Ensure FaceMesh model file exists, download if missing."""
    if MODEL_PATH.exists():
        return
    print(f"Downloading FaceMesh model from {MODEL_URL}", flush=True)
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    print("FaceMesh model downloaded successfully", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="FaceMesh data capture app")

    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True, help="Show overlay window")
    parser.add_argument("--capture", action=argparse.BooleanOptionalAction, default=False, help="Capture mode")
    parser.add_argument(
        "--capture-live",
        "--live",
        dest="capture_live",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show live camera preview in capture mode",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Run time in seconds (0 = continuous)")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--log-interval", type=float, default=2.0, help="Log interval in seconds")
    parser.add_argument("--overlay-fps", type=int, default=60, help="Overlay refresh rate")

    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run 9-point calibration workflow",
    )
    parser.add_argument(
        "--calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alias for --calibrate",
    )
    parser.add_argument(
        "--calibration-profile",
        type=str,
        default="",
        help="Calibration profile name",
    )
    parser.add_argument("--force-recalibrate", action="store_true", help="Ignore existing calibration and recalibrate")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=5,
        help="Minimum number of calibration samples to collect (default: 5)"
    )

    parser.add_argument("--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")), help="Camera device index")
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "msmf", "dshow", "any"],
        default=os.getenv("CAMERA_BACKEND", "auto").lower(),
        help="Camera backend",
    )
    parser.add_argument("--camera-width", type=int, default=int(os.getenv("CAMERA_WIDTH", "0")), help="Camera width")
    parser.add_argument("--camera-height", type=int, default=int(os.getenv("CAMERA_HEIGHT", "0")), help="Camera height")
    parser.add_argument("--camera-fps", type=float, default=float(os.getenv("CAMERA_FPS", "0")), help="Camera FPS")
    parser.add_argument("--camera-fourcc", type=str, default=os.getenv("CAMERA_FOURCC", "MJPG"), help="Camera codec")

    parser.add_argument("--udp-host", type=str, default=os.getenv("UDP_HOST", "127.0.0.1"), help="UDP forward target host")
    parser.add_argument("--udp-port", type=int, default=int(os.getenv("UDP_PORT", "5005")), help="UDP forward target port")

    return parser.parse_args()


def main():
    # Argument parsing section
    args = parse_args()

    # Load calibration if applicable
    calibration = None
    if not args.force_recalibrate and not args.calibrate and not args.calibration:
        calibration, _ = load_calibration(args.calibration_profile)
        if calibration.sample_count > 0:
            profile_name = args.calibration_profile or "default"
            print(
                f"Loaded calibration from profile '{profile_name}': "
                f"eye_zero=({calibration.center_yaw:.4f}, {calibration.center_pitch:.4f}) "
                f"face_zero=({calibration.face_center_yaw:.4f}, {calibration.face_center_pitch:.4f}) "
                f"zeta={calibration.center_zeta:.4f} "
                f"samples={calibration.sample_count}",
                flush=True,
            )
        else:
            print("No existing calibration found. Running in uncalibrated mode.", flush=True)

    # Create StateMachine - foundational component for pipeline state management
    # This will be injected into FrameDispatcherV2 in future refactoring
    state_machine = StateMachine()

    # Get display geometry - needed for CalibrationAdapterStep configuration
    # The display geometry provides screen dimensions for coordinate transformations
    display = get_display_geo()

    # FaceMeshStep creation - first pipeline step that processes frames to detect facial landmarks
    # This will be injected into FrameDispatcherV2 in future refactoring
    ensure_model()  # Download model if needed
    base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = vision.FaceLandmarkerOptions(
        base_options=base,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(opts)
    face_mesh_step = FaceMeshStep(face_landmarker)

    # CalibrationAdapterStep creation - second pipeline step that applies calibration transformations
    # This will be injected into FrameDispatcherV2 in future refactoring
    # Extract calibration values from calibration matrix if available, otherwise use defaults
    pitch_calib = calibration.center_pitch if calibration else 0.0
    yaw_calib = calibration.center_yaw if calibration else 0.0
    roll_calib = 0.0  # Roll calibration not currently stored in calibration matrix
    calibration_adapter_step = CalibrationAdapterStep(
        pitch_calibration=pitch_calib,
        yaw_calibration=yaw_calib,
        roll_calibration=roll_calib,
        display_width=display['width'],
        display_height=display['height'],
        origin_x=float(display['width']) / 2.0,
        origin_y=float(display['height']) / 2.0
    )

    # CalibrationControllerStep creation - manages calibration workflow and collects calibration samples
    # This will be injected into FrameDispatcherV2 in future refactoring
    # Requires face_landmarker for face detection and min_samples for minimum samples per calibration point
    calibration_samples = getattr(args, 'calibration_samples', 5)
    calibration_controller_step = CalibrationControllerStep(
        face_landmarker=face_landmarker,
        min_samples=calibration_samples
    )

    # CaptureStep creation - handles live preview display and frame counting
    # This will be injected into FrameDispatcherV2 in future refactoring
    # Uses args.capture flag to determine if capture/preview is enabled
    capture_step = CaptureStep(enabled=args.capture)

    # OverlayStep creation - renders visual overlays on the display
    # This will be injected into FrameDispatcherV2 in future refactoring
    # Uses args.overlay flag to determine if overlay rendering is enabled
    overlay_step = OverlayStep(enabled=args.overlay, show_hud=True)

    # UDPForwardStep creation - forwards processed data to UDP server
    # This will be injected into FrameDispatcherV2 in future refactoring
    # UDP forwarding is disabled by default and can be enabled when needed for real-time data streaming
    udp_forward_step = UDPForwardStep(host=args.udp_host, port=args.udp_port)

    # OverlayManager creation - manages pygame overlay rendering and interaction
    # This will be injected into FrameDispatcher in the constructor call
    overlay_manager = OverlayManager(
        display=display,
        capture_enabled=args.capture,
        overlay_fps=args.overlay_fps,
        calibration_mode=args.calibrate or args.calibration
    )

    # Component instantiation - inject all pipeline steps into FrameDispatcher
    # This completes the dependency injection pattern, giving main.py full control
    frame_dispatcher = FrameDispatcher(
        args,
        calibration=calibration,
        overlay_manager=overlay_manager,
        state_machine=state_machine,
        face_mesh_step=face_mesh_step,
        calibration_adapter_step=calibration_adapter_step,
        calibration_controller_step=calibration_controller_step,
        capture_step=capture_step,
        overlay_step=overlay_step,
        udp_forward_step=udp_forward_step,
    )
    camera_reader = CameraReader(
        frame_dispatcher,
        args.camera_index,
        _backend_string_to_int(args.camera_backend),
        args.camera_fourcc,
        args.camera_width,
        args.camera_height,
        args.camera_fps
    )
    
    try:
        # Start the dispatcher worker
        frame_dispatcher.start()
        
        # Start camera frame capture
        camera_reader.startReceiving()
        
        # Run appropriate workflow
        if args.calibrate or args.calibration:
            print("Running calibration workflow...", flush=True)
            frame_dispatcher.run_calibration_workflow()
        else:
            # Run capture loop for overlay/capture modes
            mode = "capture" if args.capture else "overlay"
            print(f"Running in {mode} mode...", flush=True)
            frame_dispatcher.run_capture_loop(duration=args.duration)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        raise
    finally:
        # Cleanup
        print("Shutting down...", flush=True)
        camera_reader.stopReceiving()
        frame_dispatcher.stop()
        print("Shutdown complete", flush=True)


if __name__ == "__main__":
    main()
