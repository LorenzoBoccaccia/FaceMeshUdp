#!/usr/bin/env python3
"""
Command-line entry point: track the face, calibrate the gaze and forward it to OpenTrack or FreeTrack.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from facemesh_app.calibration import DEFAULT_VIEWING_DISTANCE_MM, load_calibration
from facemesh_app.camera_reader import CameraReader
from facemesh_app.face_landmarker import FaceLandmarker, ensure_bundle
from facemesh_app.facemesh_dao import MM_PER_CM
from facemesh_app.frame_dispatcher import FrameDispatcher
from facemesh_app.pipeline_steps import (
    FaceMeshStep,
    CalibrationAdapterStep,
    BlinkRejectionStep,
    GazeSmoothingStep,
    OpenTrackForwardStep,
)
from facemesh_app.state_machine import StateMachine

logger = logging.getLogger(__name__)


def _env_int(key: str, default: str) -> int:
    raw = os.getenv(key, default)
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"Environment variable {key}='{raw}' is not a valid integer, using default {default}"
        )
        return int(default)


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam eye and head tracking for OpenTrack and FreeTrack")

    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show transparent overlay window",
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
        help="Show live camera+mesh content in capture window",
    )
    parser.add_argument(
        "--opentrack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward calibrated output to opentrack's UDP tracker input",
    )
    parser.add_argument(
        "--freetrack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Publish calibrated output to games over FreeTrack 2.0 Enhanced (Windows)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="MS",
        help="Average forwarded gaze over a trailing window of MS milliseconds (0: raw)",
    )
    parser.add_argument(
        "--smooth-reset",
        type=int,
        default=None,
        metavar="MS",
        help="When the gaze jumps to a new fixation, keep only the last MS milliseconds "
        "of the previous one in the average (default: off)",
    )
    parser.add_argument(
        "--smooth-threshold",
        type=float,
        default=3.0,
        metavar="SIGMA",
        help="Distance from the current fixation, in multiples of the calibrated eye noise, "
        "that counts as a jump for --smooth-reset (default: 3)",
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
        "--viewing-distance",
        type=float,
        default=None,
        metavar="CM",
        help="Distance from the eyes to the screen in centimetres (default: the "
        f"calibrated profile's distance, or {DEFAULT_VIEWING_DISTANCE_MM / MM_PER_CM:g} "
        "for a new calibration)",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=_env_int("CAMERA_INDEX", "0"),
        help="Camera device index",
    )

    parser.add_argument(
        "--opentrack-host",
        type=str,
        default=os.getenv("OPENTRACK_HOST", "127.0.0.1"),
        help="opentrack UDP tracker host",
    )
    parser.add_argument(
        "--opentrack-port",
        type=int,
        default=_env_int("OPENTRACK_PORT", "4242"),
        help="opentrack UDP tracker port",
    )

    parser.add_argument(
        "--freetrack-interface",
        choices=("both", "freetrack", "npclient"),
        default="both",
        help="Client interface exposed to games: FreeTrack, TrackIR (NPClient) or both",
    )
    parser.add_argument(
        "--freetrack-multiplier",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Scale the gaze angles sent to games, so a small gaze shift turns the game "
        "view further (default: 1)",
    )
    parser.add_argument(
        "--opentrack-dir",
        type=Path,
        default=os.getenv("OPENTRACK_DIR"),
        help="opentrack installation folder (containing opentrack.exe) whose client "
        "libraries games load for --freetrack (default: auto-detected)",
    )

    args = parser.parse_args()
    if args.smooth < 0:
        parser.error("--smooth must be 0 or a positive number of milliseconds")
    if args.smooth_reset is not None and not 0 <= args.smooth_reset < args.smooth:
        parser.error("--smooth-reset must be at least 0 and shorter than --smooth")
    if args.smooth_threshold <= 0:
        parser.error("--smooth-threshold must be a positive multiple of the eye noise")
    if args.viewing_distance is not None and args.viewing_distance <= 0:
        parser.error("--viewing-distance must be a positive number of centimetres")
    if args.freetrack_multiplier <= 0:
        parser.error("--freetrack-multiplier must be a positive factor")
    if args.freetrack and sys.platform != "win32":
        parser.error("--freetrack requires Windows")
    return args


def _exit_with(error: Exception) -> None:
    for line in str(error).splitlines():
        logger.error(line)
    sys.exit(1)


def main():
    if os.getenv("FACEMESH_PROFILE") == "1" and not os.getenv("_FACEMESH_PROFILE_ACTIVE"):
        os.environ["_FACEMESH_PROFILE_ACTIVE"] = "1"
        _run_with_yappi()
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    args = parse_args()
    if args.capture_live:
        args.capture = True

    opentrack_dir = None
    if args.freetrack:
        from facemesh_app.freetrack import FreeTrackSetupError, resolve_opentrack_dir

        try:
            opentrack_dir = resolve_opentrack_dir(
                args.opentrack_dir, args.freetrack_interface
            )
        except FreeTrackSetupError as e:
            _exit_with(e)
        logger.info(f"FreeTrack: using opentrack installation at {opentrack_dir}")

    no_explicit_mode = not (
        args.overlay
        or args.capture
        or args.capture_live
        or args.opentrack
        or args.freetrack
        or args.calibrate
        or args.calibration
        or args.force_recalibrate
    )

    viewing_distance_mm = (
        args.viewing_distance * MM_PER_CM if args.viewing_distance is not None else None
    )
    profile_name = args.calibration_profile or "default"
    calibrating = args.calibrate or args.calibration or args.force_recalibrate
    model = None
    if not calibrating:
        profile = load_calibration(args.calibration_profile)
        if profile is not None:
            model = profile.fit(viewing_distance_mm)
            logger.info(f"Calibration '{profile_name}': {model.describe()}")
        else:
            logger.info(f"No usable calibration for profile '{profile_name}'.")

    auto_transition_to_opentrack = False
    if no_explicit_mode:
        if model is not None:
            args.opentrack = True
            logger.info(
                "No mode specified; existing calibration found. Starting OpenTrack forwarder."
            )
        else:
            args.calibrate = True
            calibrating = True
            auto_transition_to_opentrack = True
            logger.info(
                "No mode specified and no calibration on disk. "
                "Running calibration, then OpenTrack forwarder."
            )
    if model is None and not calibrating and (args.freetrack or args.opentrack):
        _exit_with(
            RuntimeError(
                f"Gaze output needs a calibration for profile '{profile_name}'. "
                "Run calibrate.bat (or --calibrate) first."
            )
        )

    state_machine = StateMachine()

    try:
        bundle = ensure_bundle()
    except Exception as e:
        logger.error(f"Failed to download the face landmarker model: {e}")
        raise

    try:
        face_landmarker = FaceLandmarker.from_bundle(bundle, with_blendshapes=bool(args.capture))
    except Exception as e:
        logger.error(f"Failed to load the face landmarker model: {e}")
        raise

    face_mesh_step = FaceMeshStep(face_landmarker)

    calibration_adapter_step = CalibrationAdapterStep(model)

    gaze_smoothing_step = GazeSmoothingStep(
        window_ms=args.smooth,
        reset_window_ms=args.smooth_reset,
        reset_threshold_sigma=args.smooth_threshold,
    )
    if args.smooth <= 0:
        logger.info("Gaze smoothing: off (raw)")
    elif args.smooth_reset is None:
        logger.info(f"Gaze smoothing window: {args.smooth} ms")
    else:
        logger.info(
            f"Gaze smoothing window: {args.smooth} ms, {args.smooth_reset} ms tail kept "
            f"on jumps beyond {args.smooth_threshold:g}x the eye noise"
        )

    opentrack_forward_step = OpenTrackForwardStep(
        host=args.opentrack_host,
        port=args.opentrack_port,
        enabled=args.opentrack,
    )

    freetrack_forward_step = None
    if opentrack_dir is not None:
        from facemesh_app.freetrack import FreeTrackForwardStep, FreeTrackSetupError

        try:
            freetrack_forward_step = FreeTrackForwardStep(
                opentrack_dir=opentrack_dir,
                interface=args.freetrack_interface,
                multiplier=args.freetrack_multiplier,
                enabled=True,
            )
        except FreeTrackSetupError as e:
            _exit_with(e)

    frame_dispatcher = FrameDispatcher(
        args,
        overlay_manager=None,
        state_machine=state_machine,
        face_mesh_step=face_mesh_step,
        calibration_adapter_step=calibration_adapter_step,
        blink_rejection_step=BlinkRejectionStep(),
        gaze_smoothing_step=gaze_smoothing_step,
        opentrack_forward_step=opentrack_forward_step,
        freetrack_forward_step=freetrack_forward_step,
    )
    camera_reader = CameraReader(camera_id=args.camera_index)

    try:
        frame_dispatcher.start()
        camera_reader.open()

        if calibrating:
            frame_dispatcher.start_calibration()
            logger.info("Running calibration workflow...")
            calibrated_model = frame_dispatcher.run_calibration_workflow(
                camera_reader,
                viewing_distance_mm or DEFAULT_VIEWING_DISTANCE_MM,
            )
            if auto_transition_to_opentrack:
                if calibrated_model is None:
                    logger.info(
                        "Calibration did not complete; OpenTrack forwarder will not start."
                    )
                else:
                    logger.info(
                        "Calibration complete. Starting OpenTrack forwarder on "
                        f"{args.opentrack_host}:{args.opentrack_port}."
                    )
                    frame_dispatcher.set_opentrack_forwarding_enabled(True)
                    frame_dispatcher.run_capture_loop(camera_reader)
        else:
            frame_dispatcher.start_operational()
            active_modes = []
            if args.capture:
                active_modes.append("capture")
            if args.overlay:
                active_modes.append("overlay")
            if args.opentrack:
                active_modes.append("opentrack")
            if args.freetrack:
                active_modes.append("freetrack")
            if not active_modes:
                active_modes.append("tracking")
            logger.info(f"Running in mode(s): {', '.join(active_modes)}")
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
        if freetrack_forward_step is not None:
            freetrack_forward_step.close()
        logger.info("Shutdown complete")


def _run_with_yappi():
    import yappi

    yappi.set_clock_type(os.getenv("FACEMESH_PROFILE_CLOCK", "cpu"))
    yappi.start(builtins=True)
    try:
        main()
    finally:
        yappi.stop()
        out_dir = os.getenv("FACEMESH_PROFILE_DIR", "profile_out")
        os.makedirs(out_dir, exist_ok=True)

        func_stats = yappi.get_func_stats()
        func_stats.save(os.path.join(out_dir, "yappi.pstat"), type="pstat")
        func_stats.save(os.path.join(out_dir, "yappi.callgrind"), type="callgrind")

        with open(os.path.join(out_dir, "yappi_top.txt"), "w") as f:
            func_stats.sort("tsub", "desc").print_all(
                out=f,
                columns={
                    0: ("name", 80),
                    1: ("ncall", 10),
                    2: ("tsub", 10),
                    3: ("ttot", 10),
                    4: ("tavg", 10),
                },
            )

        thread_stats = yappi.get_thread_stats()
        with open(os.path.join(out_dir, "yappi_threads.txt"), "w") as f:
            thread_stats.print_all(out=f)

        print(f"[yappi] profile written to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
