#!/usr/bin/env python3
"""
FaceMesh Data Capture Application
Main entry point for raw face mesh data capture.
"""

import argparse
import logging
import os
import sys

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .facemesh_dao import load_calibration
from .camera_reader import CameraReader
from .frame_dispatcher import FrameDispatcher, ensure_model, MODEL_PATH
from .overlay import OverlayManager, get_display_geo
from .pipeline_steps import (
    FaceMeshStep,
    CalibrationAdapterStep,
    CaptureStep,
    OverlayStep,
    UDPForwardStep,
)
from .state_machine import StateMachine

logger = logging.getLogger(__name__)


def _backend_string_to_int(backend_str: str) -> int:
    """Convert backend string to cv2 CAP_* constant.

    Args:
        backend_str: Backend string ('auto', 'msmf', 'dshow', 'any')

    Returns:
        Corresponding cv2 CAP_* constant (defaults to CAP_ANY)
    """
    backend_map = {
        "auto": cv2.CAP_ANY,
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
        "any": cv2.CAP_ANY,
    }
    return backend_map.get(backend_str.lower(), cv2.CAP_ANY)


def _env_int(key: str, default: str) -> int:
    raw = os.getenv(key, default)
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"Environment variable {key}='{raw}' is not a valid integer, using default {default}"
        )
        return int(default)


def _env_float(key: str, default: str) -> float:
    raw = os.getenv(key, default)
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            f"Environment variable {key}='{raw}' is not a valid float, using default {default}"
        )
        return float(default)


def parse_args():
    parser = argparse.ArgumentParser(description="FaceMesh data capture app")

    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show overlay window",
    )
    parser.add_argument(
        "--capture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture mode",
    )
    parser.add_argument(
        "--capture-live",
        "--live",
        dest="capture_live",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show live camera preview in capture mode",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument(
        "--log-interval", type=float, default=2.0, help="Log interval in seconds"
    )
    parser.add_argument(
        "--overlay-fps", type=int, default=60, help="Overlay refresh rate"
    )

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
    parser.add_argument(
        "--force-recalibrate",
        action="store_true",
        help="Ignore existing calibration and recalibrate",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=5,
        help="Minimum number of calibration samples to collect (default: 5)",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=_env_int("CAMERA_INDEX", "0"),
        help="Camera device index",
    )
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "msmf", "dshow", "any"],
        default=os.getenv("CAMERA_BACKEND", "dshow").lower(),
        help="Camera backend",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=_env_int("CAMERA_WIDTH", "1920"),
        help="Camera width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=_env_int("CAMERA_HEIGHT", "1080"),
        help="Camera height",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=_env_float("CAMERA_FPS", "60"),
        help="Camera FPS",
    )
    parser.add_argument(
        "--camera-fourcc",
        type=str,
        default=os.getenv("CAMERA_FOURCC", "MJPG"),
        help="Camera codec",
    )

    parser.add_argument(
        "--udp-host",
        type=str,
        default=os.getenv("UDP_HOST", "127.0.0.1"),
        help="UDP forward target host",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=_env_int("UDP_PORT", "5005"),
        help="UDP forward target port",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    args = parse_args()

    calibration = None
    if not args.force_recalibrate and not args.calibrate and not args.calibration:
        try:
            calibration, _ = load_calibration(args.calibration_profile)
        except Exception as e:
            logger.warning(
                f"Failed to load calibration profile '{args.calibration_profile}': {e}"
            )
            calibration = None

        if calibration is not None and calibration.sample_count > 0:
            profile_name = args.calibration_profile or "default"
            logger.info(
                f"Loaded calibration from profile '{profile_name}': "
                f"eye_zero=({calibration.center_yaw:.4f}, {calibration.center_pitch:.4f}) "
                f"face_zero=({calibration.face_center_yaw:.4f}, {calibration.face_center_pitch:.4f}) "
                f"zeta={calibration.center_zeta:.4f} "
                f"samples={calibration.sample_count}",
            )
        else:
            logger.info("No existing calibration found. Running in uncalibrated mode.")

    state_machine = StateMachine()

    display = get_display_geo()

    try:
        ensure_model()
    except Exception as e:
        logger.error(f"Failed to download FaceMesh model: {e}")
        raise

    try:
        base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(opts)
    except Exception as e:
        logger.error(f"Failed to initialize FaceLandmarker: {e}")
        raise

    face_mesh_step = FaceMeshStep(face_landmarker)

    pitch_calib = calibration.center_pitch if calibration else 0.0
    yaw_calib = calibration.center_yaw if calibration else 0.0
    roll_calib = 0.0
    calibration_adapter_step = CalibrationAdapterStep(
        pitch_calibration=pitch_calib,
        yaw_calibration=yaw_calib,
        roll_calibration=roll_calib,
        display_width=display["width"],
        display_height=display["height"],
        origin_x=float(display["width"]) / 2.0,
        origin_y=float(display["height"]) / 2.0,
    )

    capture_step = CaptureStep(enabled=args.capture)

    overlay_step = OverlayStep(enabled=args.overlay, show_hud=True)

    udp_forward_step = UDPForwardStep(host=args.udp_host, port=args.udp_port)

    overlay_manager = OverlayManager(
        display=display,
        capture_enabled=args.capture,
        overlay_fps=args.overlay_fps,
        calibration_mode=args.calibrate or args.calibration,
    )

    frame_dispatcher = FrameDispatcher(
        args,
        calibration=calibration,
        overlay_manager=overlay_manager,
        state_machine=state_machine,
        face_mesh_step=face_mesh_step,
        calibration_adapter_step=calibration_adapter_step,
        capture_step=capture_step,
        overlay_step=overlay_step,
        udp_forward_step=udp_forward_step,
    )
    camera_reader = CameraReader(
        args.camera_index,
        _backend_string_to_int(args.camera_backend),
        args.camera_fourcc,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
    )

    try:
        frame_dispatcher.start()
        camera_reader.open()

        if args.calibrate or args.calibration:
            logger.info("Running calibration workflow...")
            frame_dispatcher.run_calibration_workflow(camera_reader)
        else:
            mode = "capture" if args.capture else "overlay"
            logger.info(f"Running in {mode} mode...")
            frame_dispatcher.run_capture_loop(camera_reader)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise
    finally:
        logger.info("Shutting down...")
        camera_reader.release()
        frame_dispatcher.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
