#!/usr/bin/env python3
"""
Calibration analysis script.
Produces per-point error summaries from calibration session capture files.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))
from facemesh_app.calibration import (
    CalibrationMatrix,
    CalibrationPoint,
    apply_calibration_model,
    compute_calibration_matrix,
    save_calibration,
)
from facemesh_app.gaze_primitives import (
    project_head_angles_to_screen_xy,
    screen_xy_to_head_angles,
)


def safe_float(value: Any, fallback: float = 0.0) -> float:
    """Normalize numeric inputs so analysis can proceed on imperfect data."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isfinite(out):
        return out
    return fallback


def positive_or(value: Any, fallback: float) -> float:
    """Preserve positive values for geometry and depth calculations."""
    out = safe_float(value, fallback)
    if out > 1e-9:
        return out
    return fallback


def find_session_files(
    data_dir: Path,
    explicit_file: str,
    latest_only: bool,
) -> List[Path]:
    """Resolve calibration session files selected for analysis."""
    if explicit_file:
        target = Path(explicit_file)
        if not target.exists():
            print(f"Error: File not found: {target}")
            sys.exit(1)
        return [target]

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    files = sorted(data_dir.glob("calibration_session_*.json"))
    if not files:
        print(f"Error: No files matched {data_dir / 'calibration_session_*.json'}")
        sys.exit(1)

    if latest_only:
        return [max(files, key=lambda p: p.stat().st_mtime)]
    return files


def load_session_file(path: Path) -> Dict[str, Any]:
    """Load one calibration session payload."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: Could not read {path}: {exc}")
        sys.exit(1)


def summarize_sampling_frames(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate sampling-phase frame quality metrics grouped by target point."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        if sample.get("phase") != "sampling":
            continue
        target = sample.get("target") or {}
        name = str(target.get("name") or "")
        if not name:
            continue
        entry = grouped.setdefault(
            name,
            {
                "frame_count": 0,
                "face_count": 0,
                "raw_yaw_values": [],
                "raw_pitch_values": [],
                "corrected_yaw_values": [],
                "corrected_pitch_values": [],
            },
        )
        entry["frame_count"] += 1
        if bool(sample.get("hasFace")):
            entry["face_count"] += 1
        raw_yaw = sample.get("rawCombinedEyeYaw")
        raw_pitch = sample.get("rawCombinedEyePitch")
        if raw_yaw is not None:
            entry["raw_yaw_values"].append(safe_float(raw_yaw))
        if raw_pitch is not None:
            entry["raw_pitch_values"].append(safe_float(raw_pitch))
        corrected_yaw = sample.get("correctedYaw")
        corrected_pitch = sample.get("correctedPitch")
        if corrected_yaw is not None:
            entry["corrected_yaw_values"].append(safe_float(corrected_yaw))
        if corrected_pitch is not None:
            entry["corrected_pitch_values"].append(safe_float(corrected_pitch))
    return grouped


def avg(values: List[float]) -> Optional[float]:
    """Return average when there is data."""
    if not values:
        return None
    return float(sum(values) / len(values))


def stddev(values: List[float]) -> Optional[float]:
    """Return population standard deviation when there is data."""
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.pstdev(values))


def format_num(value: Optional[float], width: int = 7, precision: int = 2) -> str:
    """Render fixed-width numbers and blanks consistently."""
    if value is None or not math.isfinite(value):
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{precision}f}"


def session_points_to_calibration_points(
    session: Dict[str, Any],
) -> Tuple[List[CalibrationPoint], str]:
    """Build calibration points from one captured session payload."""
    session_points = session.get("points") or []
    if not session_points:
        return [], "Session has no calibration points."

    calibration_points: List[CalibrationPoint] = []
    for point in session_points:
        name = str(point.get("name") or "")
        if not name:
            continue
        raw_eye_yaw = safe_float(point.get("rawEyeYaw"))
        raw_eye_pitch = safe_float(point.get("rawEyePitch"))
        raw_left_eye_yaw = safe_float(point.get("rawLeftEyeYaw"), raw_eye_yaw)
        raw_left_eye_pitch = safe_float(point.get("rawLeftEyePitch"), raw_eye_pitch)
        raw_right_eye_yaw = safe_float(point.get("rawRightEyeYaw"), raw_eye_yaw)
        raw_right_eye_pitch = safe_float(point.get("rawRightEyePitch"), raw_eye_pitch)
        calibration_points.append(
            CalibrationPoint(
                name=name,
                screen_x=safe_float(point.get("screenX")),
                screen_y=safe_float(point.get("screenY")),
                raw_eye_yaw=raw_eye_yaw,
                raw_eye_pitch=raw_eye_pitch,
                raw_left_eye_yaw=raw_left_eye_yaw,
                raw_left_eye_pitch=raw_left_eye_pitch,
                raw_right_eye_yaw=raw_right_eye_yaw,
                raw_right_eye_pitch=raw_right_eye_pitch,
                sample_count=int(safe_float(point.get("sampleCount"), 0)),
                head_yaw=safe_float(point.get("headYaw")),
                head_pitch=safe_float(point.get("headPitch")),
                zeta=positive_or(point.get("zeta"), 1200.0),
                head_x=safe_float(point.get("headX"), 0.0),
                head_y=safe_float(point.get("headY"), 0.0),
                head_z=positive_or(point.get("headZ"), 1200.0),
            )
        )

    if not calibration_points:
        return [], "Session points are empty or malformed."
    return calibration_points, ""


def calibration_matrix_to_dict(matrix: CalibrationMatrix) -> Dict[str, Any]:
    """Normalize calibration matrix into analysis key space."""
    return {
        "centerYaw": float(matrix.center_yaw),
        "centerPitch": float(matrix.center_pitch),
        "faceCenterYaw": float(matrix.face_center_yaw),
        "faceCenterPitch": float(matrix.face_center_pitch),
        "centerZeta": float(matrix.center_zeta),
        "matrixYawYaw": float(matrix.matrix_yaw_yaw),
        "matrixYawPitch": float(matrix.matrix_yaw_pitch),
        "matrixPitchYaw": float(matrix.matrix_pitch_yaw),
        "matrixPitchPitch": float(matrix.matrix_pitch_pitch),
        "faceCenterX": float(matrix.face_center_x),
        "faceCenterY": float(matrix.face_center_y),
        "faceCenterZ": float(matrix.face_center_z),
        "screenCenterCamX": float(matrix.screen_center_cam_x),
        "screenCenterCamY": float(matrix.screen_center_cam_y),
        "screenCenterCamZ": float(matrix.screen_center_cam_z),
        "screenAxisXX": float(matrix.screen_axis_x_x),
        "screenAxisXY": float(matrix.screen_axis_x_y),
        "screenAxisXZ": float(matrix.screen_axis_x_z),
        "screenAxisYX": float(matrix.screen_axis_y_x),
        "screenAxisYY": float(matrix.screen_axis_y_y),
        "screenAxisYZ": float(matrix.screen_axis_y_z),
        "screenFitRmse": float(matrix.screen_fit_rmse),
        "sampleCount": int(matrix.sample_count),
    }


def rebuild_calibration_from_session(
    session: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[CalibrationPoint], Path, Path, str]:
    """Recompute and persist calibration from captured session points."""
    calibration_points, err = session_points_to_calibration_points(session)
    if err:
        return {}, [], Path(""), Path(""), err
    try:
        matrix = compute_calibration_matrix(calibration_points)
    except ValueError as exc:
        return {}, [], Path(""), Path(""), f"Calibration solve failed: {exc}"

    profile = str(session.get("profile") or "default")
    profile_path = save_calibration(matrix, calibration_points, profile=profile)
    legacy_path = Path("calibration.json")
    legacy_path.write_text(profile_path.read_text(encoding="utf-8"), encoding="utf-8")
    return calibration_matrix_to_dict(matrix), calibration_points, profile_path, legacy_path, ""


def per_point_error_rows(
    session: Dict[str, Any], matrix: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], str]:
    """Compute target-vs-prediction errors for each calibration point."""
    points = session.get("points") or []
    if not points:
        return [], "Session has no calibration points."
    if not matrix:
        return [], "Session has no analysis calibration matrix."

    point_by_name: Dict[str, Dict[str, Any]] = {
        str(point.get("name")): point for point in points if point.get("name")
    }
    center = point_by_name.get("C")
    if center is None:
        return [], "Session has no center point 'C'."

    center_eye_yaw = safe_float(matrix.get("centerYaw"))
    center_eye_pitch = safe_float(matrix.get("centerPitch"))
    face_center_yaw = safe_float(matrix.get("faceCenterYaw"))
    face_center_pitch = safe_float(matrix.get("faceCenterPitch"))
    center_zeta = positive_or(matrix.get("centerZeta"), 1200.0)
    face_center_x = safe_float(matrix.get("faceCenterX"), 0.0)
    face_center_y = safe_float(matrix.get("faceCenterY"), 0.0)
    face_center_z = positive_or(matrix.get("faceCenterZ"), center_zeta)
    screen_center_cam_x = safe_float(matrix.get("screenCenterCamX"), 0.0)
    screen_center_cam_y = safe_float(matrix.get("screenCenterCamY"), 0.0)
    screen_center_cam_z = safe_float(matrix.get("screenCenterCamZ"), center_zeta)
    screen_axis_x_x = safe_float(matrix.get("screenAxisXX"), 1.0)
    screen_axis_x_y = safe_float(matrix.get("screenAxisXY"), 0.0)
    screen_axis_x_z = safe_float(matrix.get("screenAxisXZ"), 0.0)
    screen_axis_y_x = safe_float(matrix.get("screenAxisYX"), 0.0)
    screen_axis_y_y = safe_float(matrix.get("screenAxisYY"), 1.0)
    screen_axis_y_z = safe_float(matrix.get("screenAxisYZ"), 0.0)
    has_screen_model = all(
        key in matrix
        for key in (
            "screenCenterCamX",
            "screenCenterCamY",
            "screenCenterCamZ",
            "screenAxisXX",
            "screenAxisXY",
            "screenAxisXZ",
            "screenAxisYX",
            "screenAxisYY",
            "screenAxisYZ",
            "screenFitRmse",
        )
    )
    screen_fit_rmse = safe_float(matrix.get("screenFitRmse"), -1.0)
    if not has_screen_model:
        return [], "Session has no geometric screen model."
    if screen_fit_rmse < 0.0:
        return [], "Session geometric screen model is invalid."

    m_yy = safe_float(matrix.get("matrixYawYaw"), 1.0)
    m_yp = safe_float(matrix.get("matrixYawPitch"), 0.0)
    m_py = safe_float(matrix.get("matrixPitchYaw"), 0.0)
    m_pp = safe_float(matrix.get("matrixPitchPitch"), 1.0)

    sampling_stats = summarize_sampling_frames(session.get("samples") or [])
    rows: List[Dict[str, Any]] = []

    center_x = safe_float(center.get("screenX"))
    center_y = safe_float(center.get("screenY"))

    for point in points:
        name = str(point.get("name") or "")
        head_x = safe_float(point.get("headX"), 0.0)
        head_y = safe_float(point.get("headY"), 0.0)
        head_z = positive_or(point.get("headZ"), center_zeta)
        target_total = screen_xy_to_head_angles(
            screen_x=safe_float(point.get("screenX")),
            screen_y=safe_float(point.get("screenY")),
            head_x=head_x,
            head_y=head_y,
            head_z=head_z,
            center_zeta=center_zeta,
            screen_center_cam_x=screen_center_cam_x,
            screen_center_cam_y=screen_center_cam_y,
            screen_center_cam_z=screen_center_cam_z,
            screen_axis_x_x=screen_axis_x_x,
            screen_axis_x_y=screen_axis_x_y,
            screen_axis_x_z=screen_axis_x_z,
            screen_axis_y_x=screen_axis_y_x,
            screen_axis_y_y=screen_axis_y_y,
            screen_axis_y_z=screen_axis_y_z,
            screen_fit_rmse=screen_fit_rmse,
            origin_x=center_x,
            origin_y=center_y,
        )
        if target_total is None:
            return [], f"Inverse projection failed at point '{name}'"
        target_total_yaw, target_total_pitch = target_total

        try:
            corrected = apply_calibration_model(
                raw_eye_yaw=safe_float(point.get("rawEyeYaw")),
                raw_eye_pitch=safe_float(point.get("rawEyePitch")),
                head_yaw=safe_float(point.get("headYaw")),
                head_pitch=safe_float(point.get("headPitch")),
                head_x=head_x,
                head_y=head_y,
                head_z=head_z,
                center_eye_yaw=center_eye_yaw,
                center_eye_pitch=center_eye_pitch,
                face_center_yaw=face_center_yaw,
                face_center_pitch=face_center_pitch,
                matrix_yaw_yaw=m_yy,
                matrix_yaw_pitch=m_yp,
                matrix_pitch_yaw=m_py,
                matrix_pitch_pitch=m_pp,
                center_zeta=center_zeta,
                face_center_x=face_center_x,
                face_center_y=face_center_y,
                face_center_z=face_center_z,
                screen_center_cam_x=screen_center_cam_x,
                screen_center_cam_y=screen_center_cam_y,
                screen_center_cam_z=screen_center_cam_z,
                screen_axis_x_x=screen_axis_x_x,
                screen_axis_x_y=screen_axis_x_y,
                screen_axis_x_z=screen_axis_x_z,
                screen_axis_y_x=screen_axis_y_x,
                screen_axis_y_y=screen_axis_y_y,
                screen_axis_y_z=screen_axis_y_z,
                screen_fit_rmse=screen_fit_rmse,
                origin_x=center_x,
                origin_y=center_y,
            )
        except ValueError as exc:
            return [], f"Geometric projection failed at point '{name}': {exc}"

        predicted_eye_yaw = corrected["corrected_eye_yaw"]
        predicted_eye_pitch = corrected["corrected_eye_pitch"]
        face_delta_yaw = corrected["face_delta_yaw"]
        face_delta_pitch = corrected["face_delta_pitch"]
        predicted_total_yaw = corrected["corrected_yaw"]
        predicted_total_pitch = corrected["corrected_pitch"]
        predicted_screen_x = corrected.get("corrected_screen_x")
        predicted_screen_y = corrected.get("corrected_screen_y")

        error_yaw = predicted_total_yaw - target_total_yaw
        error_pitch = predicted_total_pitch - target_total_pitch
        error_mag = math.hypot(error_yaw, error_pitch)
        screen_error_x = (
            safe_float(predicted_screen_x) - safe_float(point.get("screenX"))
            if predicted_screen_x is not None
            else None
        )
        screen_error_y = (
            safe_float(predicted_screen_y) - safe_float(point.get("screenY"))
            if predicted_screen_y is not None
            else None
        )
        screen_error_mag = (
            math.hypot(screen_error_x, screen_error_y)
            if screen_error_x is not None and screen_error_y is not None
            else None
        )

        stat = sampling_stats.get(name) or {}
        frame_count = int(stat.get("frame_count", 0))
        face_count = int(stat.get("face_count", 0))
        face_rate = (face_count / frame_count * 100.0) if frame_count > 0 else None
        raw_yaw_mean = avg(stat.get("raw_yaw_values", []))
        raw_pitch_mean = avg(stat.get("raw_pitch_values", []))
        raw_yaw_std = stddev(stat.get("raw_yaw_values", []))
        raw_pitch_std = stddev(stat.get("raw_pitch_values", []))
        corrected_yaw_mean = avg(stat.get("corrected_yaw_values", []))
        corrected_pitch_mean = avg(stat.get("corrected_pitch_values", []))
        corrected_yaw_std = stddev(stat.get("corrected_yaw_values", []))
        corrected_pitch_std = stddev(stat.get("corrected_pitch_values", []))

        rows.append(
            {
                "name": name,
                "point_samples": int(safe_float(point.get("sampleCount"), 0)),
                "sampling_frames": frame_count,
                "face_rate": face_rate,
                "target_yaw": target_total_yaw,
                "target_pitch": target_total_pitch,
                "predicted_yaw": predicted_total_yaw,
                "predicted_pitch": predicted_total_pitch,
                "predicted_screen_x": predicted_screen_x,
                "predicted_screen_y": predicted_screen_y,
                "face_delta_yaw": face_delta_yaw,
                "face_delta_pitch": face_delta_pitch,
                "predicted_eye_yaw": predicted_eye_yaw,
                "predicted_eye_pitch": predicted_eye_pitch,
                "predicted_yaw_linear": corrected["corrected_yaw_linear"],
                "predicted_pitch_linear": corrected["corrected_pitch_linear"],
                "error_yaw": error_yaw,
                "error_pitch": error_pitch,
                "error_mag": error_mag,
                "screen_error_x": screen_error_x,
                "screen_error_y": screen_error_y,
                "screen_error_mag": screen_error_mag,
                "head_x": head_x,
                "head_y": head_y,
                "head_z": head_z,
                "raw_yaw_mean": raw_yaw_mean,
                "raw_pitch_mean": raw_pitch_mean,
                "raw_yaw_std": raw_yaw_std,
                "raw_pitch_std": raw_pitch_std,
                "corrected_yaw_mean": corrected_yaw_mean,
                "corrected_pitch_mean": corrected_pitch_mean,
                "corrected_yaw_std": corrected_yaw_std,
                "corrected_pitch_std": corrected_pitch_std,
            }
        )

    return rows, ""


def print_session_summary(path: Path, session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Print one session report and return top-line aggregate metrics."""
    matrix, _, profile_path, legacy_path, rebuild_err = rebuild_calibration_from_session(
        session
    )
    rows, err = per_point_error_rows(session, matrix)
    points = session.get("points") or []
    center_point = next((p for p in points if str(p.get("name") or "") == "C"), {}) or {}
    center_x = safe_float(center_point.get("screenX"), 0.0)
    center_y = safe_float(center_point.get("screenY"), 0.0)
    line_width = 332
    print("\n" + "=" * line_width)
    print(f"SESSION: {path.name}")
    print(
        f"profile={session.get('profile', 'default')} "
        f"sessionTs={session.get('sessionTimestampMs')} "
        f"capturedFrames={session.get('sampleCount', 0)}"
    )
    if rebuild_err:
        print(f"Error: {rebuild_err}")
        return None
    print(
        f"calibration_rebuilt={profile_path.name} alias={legacy_path.name}",
    )
    if matrix:
        screen_fit_display = (
            safe_float(matrix.get("screenFitRmse"), -1.0)
            if "screenFitRmse" in matrix
            else float("nan")
        )
        screen_fit_text = (
            f"{screen_fit_display:.3f}"
            if math.isfinite(screen_fit_display)
            else "n/a"
        )
        print(
            "model: "
            f"eyeZero=({safe_float(matrix.get('centerYaw')):.3f}, {safe_float(matrix.get('centerPitch')):.3f}) "
            f"faceZero=({safe_float(matrix.get('faceCenterYaw')):.3f}, {safe_float(matrix.get('faceCenterPitch')):.3f}) "
            f"M=[[{safe_float(matrix.get('matrixYawYaw')):.3f}, {safe_float(matrix.get('matrixYawPitch')):.3f}], "
            f"[{safe_float(matrix.get('matrixPitchYaw')):.3f}, {safe_float(matrix.get('matrixPitchPitch')):.3f}]] "
            f"screenFitRmse={screen_fit_text}"
        )
    if err:
        print(f"Error: {err}")
        return None

    print("-" * line_width)
    print(
        f"{'Point':<6} {'PtSamp':>7} {'FrmSamp':>7} {'Face%':>7} "
        f"{'TargetYaw':>10} {'TargetPitch':>11} {'FaceDYaw':>9} {'FaceDPitch':>11} "
        f"{'CorrEyeYaw':>10} {'CorrEyePitch':>12} {'PredYaw':>9} {'PredPitch':>10} {'PredX':>8} {'PredY':>8} "
        f"{'ErrYaw':>8} {'ErrPitch':>9} {'ErrMag':>8} {'ErrXpx':>8} {'ErrYpx':>8} {'ErrPx':>8} "
        f"{'RawYawMean':>10} {'RawYawStd':>9} {'RawPitchMean':>12} {'RawPitchStd':>11} "
        f"{'CorrYawMean':>11} {'CorrYawStd':>10} {'CorrPitchMean':>13} {'CorrPitchStd':>12}"
    )
    print("-" * line_width)

    for row in rows:
        print(
            f"{row['name']:<6} "
            f"{row['point_samples']:>7d} "
            f"{row['sampling_frames']:>7d} "
            f"{format_num(row['face_rate'], width=7, precision=1)} "
            f"{format_num(row['target_yaw'], width=10)} "
            f"{format_num(row['target_pitch'], width=11)} "
            f"{format_num(row['face_delta_yaw'], width=9)} "
            f"{format_num(row['face_delta_pitch'], width=11)} "
            f"{format_num(row['predicted_eye_yaw'], width=10)} "
            f"{format_num(row['predicted_eye_pitch'], width=12)} "
            f"{format_num(row['predicted_yaw'], width=9)} "
            f"{format_num(row['predicted_pitch'], width=10)} "
            f"{format_num(row['predicted_screen_x'], width=8)} "
            f"{format_num(row['predicted_screen_y'], width=8)} "
            f"{format_num(row['error_yaw'], width=8)} "
            f"{format_num(row['error_pitch'], width=9)} "
            f"{format_num(row['error_mag'], width=8)} "
            f"{format_num(row['screen_error_x'], width=8)} "
            f"{format_num(row['screen_error_y'], width=8)} "
            f"{format_num(row['screen_error_mag'], width=8)} "
            f"{format_num(row['raw_yaw_mean'], width=10)} "
            f"{format_num(row['raw_yaw_std'], width=9)} "
            f"{format_num(row['raw_pitch_mean'], width=12)} "
            f"{format_num(row['raw_pitch_std'], width=11)} "
            f"{format_num(row['corrected_yaw_mean'], width=11)} "
            f"{format_num(row['corrected_yaw_std'], width=10)} "
            f"{format_num(row['corrected_pitch_mean'], width=13)} "
            f"{format_num(row['corrected_pitch_std'], width=12)}"
        )

    abs_err_yaw = [abs(row["error_yaw"]) for row in rows]
    abs_err_pitch = [abs(row["error_pitch"]) for row in rows]
    err_mag = [row["error_mag"] for row in rows]
    screen_err_mag = [
        row["screen_error_mag"]
        for row in rows
        if row.get("screen_error_mag") is not None
    ]

    mean_abs_yaw = sum(abs_err_yaw) / len(abs_err_yaw) if abs_err_yaw else float("nan")
    mean_abs_pitch = (
        sum(abs_err_pitch) / len(abs_err_pitch) if abs_err_pitch else float("nan")
    )
    rms_mag = (
        math.sqrt(sum(v * v for v in err_mag) / len(err_mag)) if err_mag else float("nan")
    )
    mean_screen_err = (
        sum(screen_err_mag) / len(screen_err_mag)
        if screen_err_mag
        else float("nan")
    )
    rms_screen_err = (
        math.sqrt(sum(v * v for v in screen_err_mag) / len(screen_err_mag))
        if screen_err_mag
        else float("nan")
    )
    worst = max(rows, key=lambda row: row["error_mag"]) if rows else None

    print("-" * line_width)
    print(
        "Aggregate: "
        f"mean|yaw_err|={mean_abs_yaw:.3f}deg "
        f"mean|pitch_err|={mean_abs_pitch:.3f}deg "
        f"rms_err_mag={rms_mag:.3f}deg "
        f"mean_screen_err={mean_screen_err:.3f}px "
        f"rms_screen_err={rms_screen_err:.3f}px"
    )
    if worst is not None:
        print(
            "Worst point: "
            f"{worst['name']} "
            f"errYaw={worst['error_yaw']:+.3f}deg "
            f"errPitch={worst['error_pitch']:+.3f}deg "
            f"errMag={worst['error_mag']:.3f}deg"
        )

    print_projection_inverse_identity(rows, matrix, center_x, center_y)

    return {
        "file": path.name,
        "profile": session.get("profile", "default"),
        "mean_abs_yaw": mean_abs_yaw,
        "mean_abs_pitch": mean_abs_pitch,
        "rms_mag": rms_mag,
        "mean_screen_err": mean_screen_err,
        "rms_screen_err": rms_screen_err,
        "worst_name": worst["name"] if worst else "",
        "worst_mag": worst["error_mag"] if worst else float("nan"),
    }


def print_projection_inverse_identity(
    rows: List[Dict[str, Any]],
    matrix: Dict[str, Any],
    origin_x: float,
    origin_y: float,
) -> None:
    """Report projection-inverse identity consistency for calibrated angles."""
    if not rows:
        return

    center_zeta = positive_or(matrix.get("centerZeta"), 1200.0)
    screen_center_cam_x = safe_float(matrix.get("screenCenterCamX"), 0.0)
    screen_center_cam_y = safe_float(matrix.get("screenCenterCamY"), 0.0)
    screen_center_cam_z = safe_float(matrix.get("screenCenterCamZ"), center_zeta)
    screen_axis_x_x = safe_float(matrix.get("screenAxisXX"), 1.0)
    screen_axis_x_y = safe_float(matrix.get("screenAxisXY"), 0.0)
    screen_axis_x_z = safe_float(matrix.get("screenAxisXZ"), 0.0)
    screen_axis_y_x = safe_float(matrix.get("screenAxisYX"), 0.0)
    screen_axis_y_y = safe_float(matrix.get("screenAxisYY"), 1.0)
    screen_axis_y_z = safe_float(matrix.get("screenAxisYZ"), 0.0)
    screen_fit_rmse = safe_float(matrix.get("screenFitRmse"), -1.0)

    identity_rows: List[Dict[str, Any]] = []
    for row in rows:
        projected = project_head_angles_to_screen_xy(
            yaw_deg=row.get("predicted_yaw_linear"),
            pitch_deg=row.get("predicted_pitch_linear"),
            head_x=row.get("head_x"),
            head_y=row.get("head_y"),
            head_z=row.get("head_z"),
            center_zeta=center_zeta,
            screen_center_cam_x=screen_center_cam_x,
            screen_center_cam_y=screen_center_cam_y,
            screen_center_cam_z=screen_center_cam_z,
            screen_axis_x_x=screen_axis_x_x,
            screen_axis_x_y=screen_axis_x_y,
            screen_axis_x_z=screen_axis_x_z,
            screen_axis_y_x=screen_axis_y_x,
            screen_axis_y_y=screen_axis_y_y,
            screen_axis_y_z=screen_axis_y_z,
            screen_fit_rmse=screen_fit_rmse,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        if projected is None:
            continue
        recovered = screen_xy_to_head_angles(
            screen_x=projected["screen_x"],
            screen_y=projected["screen_y"],
            head_x=row.get("head_x"),
            head_y=row.get("head_y"),
            head_z=row.get("head_z"),
            center_zeta=center_zeta,
            screen_center_cam_x=screen_center_cam_x,
            screen_center_cam_y=screen_center_cam_y,
            screen_center_cam_z=screen_center_cam_z,
            screen_axis_x_x=screen_axis_x_x,
            screen_axis_x_y=screen_axis_x_y,
            screen_axis_x_z=screen_axis_x_z,
            screen_axis_y_x=screen_axis_y_x,
            screen_axis_y_y=screen_axis_y_y,
            screen_axis_y_z=screen_axis_y_z,
            screen_fit_rmse=screen_fit_rmse,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        if recovered is None:
            continue
        recovered_yaw, recovered_pitch = recovered
        yaw_error = recovered_yaw - safe_float(row.get("predicted_yaw_linear"))
        pitch_error = recovered_pitch - safe_float(row.get("predicted_pitch_linear"))
        identity_rows.append(
            {
                "name": row.get("name", ""),
                "yaw_error": yaw_error,
                "pitch_error": pitch_error,
                "error_mag": math.hypot(yaw_error, pitch_error),
            }
        )

    print("-" * 332)
    print("PROJECTION/INVERSE IDENTITY")
    if not identity_rows:
        print("No valid rows for identity check")
        return
    print(
        f"{'Point':<6} {'YawErr':>10} {'PitchErr':>10} {'Mag':>10}"
    )
    for row in identity_rows:
        print(
            f"{str(row['name']):<6} "
            f"{format_num(row['yaw_error'], width=10, precision=4)} "
            f"{format_num(row['pitch_error'], width=10, precision=4)} "
            f"{format_num(row['error_mag'], width=10, precision=4)}"
        )
    mean_abs_yaw = sum(abs(r["yaw_error"]) for r in identity_rows) / len(identity_rows)
    mean_abs_pitch = sum(abs(r["pitch_error"]) for r in identity_rows) / len(identity_rows)
    rms = math.sqrt(sum(r["error_mag"] * r["error_mag"] for r in identity_rows) / len(identity_rows))
    print(
        "Identity aggregate: "
        f"mean|yaw|={mean_abs_yaw:.6f}deg "
        f"mean|pitch|={mean_abs_pitch:.6f}deg "
        f"rms_mag={rms:.6f}deg"
    )


def print_cross_session_summary(results: List[Dict[str, Any]]) -> None:
    """Print top-line metrics across multiple analyzed sessions."""
    if len(results) <= 1:
        return
    print("\n" + "=" * 140)
    print("CROSS-SESSION SUMMARY")
    print("-" * 140)
    print(
        f"{'File':<36} {'Profile':<12} {'Mean|Yaw|':>11} {'Mean|Pitch|':>13} {'RMS Mag':>10} {'MeanPx':>10} {'RmsPx':>10} "
        f"{'Worst Pt':>9} {'Worst Mag':>10}"
    )
    print("-" * 140)
    for item in results:
        print(
            f"{item['file']:<36} "
            f"{str(item['profile']):<12} "
            f"{item['mean_abs_yaw']:>11.3f} "
            f"{item['mean_abs_pitch']:>13.3f} "
            f"{item['rms_mag']:>10.3f} "
            f"{item['mean_screen_err']:>10.3f} "
            f"{item['rms_screen_err']:>10.3f} "
            f"{item['worst_name']:>9} "
            f"{item['worst_mag']:>10.3f}"
        )


def main() -> None:
    """Run calibration session analysis and print per-point error summaries."""
    parser = argparse.ArgumentParser(
        description="Analyze calibration session error summaries"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="calibration_data",
        help="Directory containing calibration_session_*.json files",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Explicit calibration session JSON file path",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Analyze only the most recent calibration session file",
    )
    args = parser.parse_args()

    files = find_session_files(
        data_dir=Path(args.data_dir),
        explicit_file=args.file,
        latest_only=bool(args.latest),
    )
    results: List[Dict[str, Any]] = []
    for path in files:
        session = load_session_file(path)
        summary = print_session_summary(path, session)
        if summary is not None:
            results.append(summary)

    print_cross_session_summary(results)


if __name__ == "__main__":
    main()
