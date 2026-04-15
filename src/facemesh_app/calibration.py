from dataclasses import dataclass
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .facemesh_dao import FaceMeshEvent, safe_float
from .gaze_primitives import (
    project_head_angles_to_screen_xy,
    screen_xy_to_head_angles,
)


logger = logging.getLogger(__name__)

DEFAULT_CENTER_ZETA = 1200.0
CALIBRATION_MODEL_VERSION = 5


@dataclass(frozen=True)
class CalibrationMatrix:
    center_yaw: float = 0.0
    center_pitch: float = 0.0
    face_center_yaw: float = 0.0
    face_center_pitch: float = 0.0
    center_zeta: float = DEFAULT_CENTER_ZETA
    matrix_yaw_yaw: float = 1.0
    matrix_yaw_pitch: float = 0.0
    matrix_pitch_yaw: float = 0.0
    matrix_pitch_pitch: float = 1.0
    face_center_x: float = 0.0
    face_center_y: float = 0.0
    face_center_z: float = DEFAULT_CENTER_ZETA
    screen_center_cam_x: float = 0.0
    screen_center_cam_y: float = 0.0
    screen_center_cam_z: float = DEFAULT_CENTER_ZETA
    screen_axis_x_x: float = 1.0
    screen_axis_x_y: float = 0.0
    screen_axis_x_z: float = 0.0
    screen_axis_y_x: float = 0.0
    screen_axis_y_y: float = 1.0
    screen_axis_y_z: float = 0.0
    screen_fit_rmse: float = -1.0
    sample_count: int = 0
    timestamp_ms: int = 0


@dataclass(frozen=True)
class CalibrationPoint:
    name: str
    screen_x: float
    screen_y: float
    raw_eye_yaw: float
    raw_eye_pitch: float
    raw_left_eye_yaw: float
    raw_left_eye_pitch: float
    raw_right_eye_yaw: float
    raw_right_eye_pitch: float
    sample_count: int
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    zeta: float = DEFAULT_CENTER_ZETA
    head_x: float = 0.0
    head_y: float = 0.0
    head_z: float = DEFAULT_CENTER_ZETA


def _positive_or(v: Any, fallback: float) -> float:
    x = safe_float(v, fallback)
    return x if x > 1e-9 else fallback


def _build_screen_geometry(
    center_point: CalibrationPoint,
    center_zeta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    center_head_x = safe_float(getattr(center_point, "head_x", 0.0), 0.0)
    center_head_y = safe_float(getattr(center_point, "head_y", 0.0), 0.0)
    center_head_z = _positive_or(getattr(center_point, "head_z", center_zeta), center_zeta)
    depth = _positive_or(center_zeta, DEFAULT_CENTER_ZETA)
    screen_center = np.array(
        [
            center_head_x,
            center_head_y,
            center_head_z - depth,
        ],
        dtype=float,
    )
    screen_axis_x = np.array([1.0, 0.0, 0.0], dtype=float)
    screen_axis_y = np.array([0.0, 1.0, 0.0], dtype=float)
    return screen_center, screen_axis_x, screen_axis_y, 0.0


def _fit_rotation_scale_matrix(
    inputs: np.ndarray,
    targets: np.ndarray,
) -> Tuple[float, float, float, float]:
    if inputs.shape[0] < 2 or targets.shape[0] < 2:
        return 1.0, 0.0, 0.0, 1.0

    x1 = inputs[:, 0]
    x2 = inputs[:, 1]
    y1 = targets[:, 0]
    y2 = targets[:, 1]

    def _evaluate(theta: float) -> Tuple[float, float, float]:
        c = math.cos(theta)
        s = math.sin(theta)
        u = x1 * c + x2 * s
        v = -x1 * s + x2 * c
        uu = float(np.dot(u, u))
        vv = float(np.dot(v, v))
        sx = float(np.dot(u, y1) / uu) if uu > 1e-12 else 1.0
        sy = float(np.dot(v, y2) / vv) if vv > 1e-12 else 1.0
        err = float(np.sum((sx * u - y1) ** 2 + (sy * v - y2) ** 2))
        return err, sx, sy

    best_theta = 0.0
    best_sx = 1.0
    best_sy = 1.0
    best_err = float("inf")

    for theta in np.linspace(-math.pi, math.pi, 1441):
        err, sx, sy = _evaluate(float(theta))
        if err < best_err:
            best_err = err
            best_theta = float(theta)
            best_sx = sx
            best_sy = sy

    for span in (math.radians(2.0), math.radians(0.2), math.radians(0.02)):
        lo = best_theta - span
        hi = best_theta + span
        for theta in np.linspace(lo, hi, 401):
            err, sx, sy = _evaluate(float(theta))
            if err < best_err:
                best_err = err
                best_theta = float(theta)
                best_sx = sx
                best_sy = sy

    c = math.cos(best_theta)
    s = math.sin(best_theta)
    return (
        float(c * best_sx),
        float(-s * best_sy),
        float(s * best_sx),
        float(c * best_sy),
    )


def compute_calibration_matrix(points: List[CalibrationPoint]) -> CalibrationMatrix:
    if len(points) < 9:
        raise ValueError(f"Calibration requires at least 9 points, got {len(points)}")

    center_point = next((p for p in points if p.name == "C"), None)
    if center_point is None:
        raise ValueError("Calibration points must include a center point named 'C'")

    center_zeta = _positive_or(
        getattr(center_point, "zeta", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
    )
    center_head_x = safe_float(getattr(center_point, "head_x", 0.0), 0.0)
    center_head_y = safe_float(getattr(center_point, "head_y", 0.0), 0.0)
    center_head_z = _positive_or(getattr(center_point, "head_z", center_zeta), center_zeta)
    (
        screen_center_cam,
        screen_axis_x,
        screen_axis_y,
        screen_fit_rmse,
    ) = _build_screen_geometry(
        center_point=center_point,
        center_zeta=center_zeta,
    )

    A_rows: List[List[float]] = []
    b_yaw: List[float] = []
    b_pitch: List[float] = []

    for point in points:
        if point.name == "C":
            continue

        eye_dyaw = safe_float(point.raw_eye_yaw) - safe_float(center_point.raw_eye_yaw)
        eye_dpitch = safe_float(point.raw_eye_pitch) - safe_float(center_point.raw_eye_pitch)

        if abs(eye_dyaw) <= 1e-9 and abs(eye_dpitch) <= 1e-9:
            continue

        target_total = screen_xy_to_head_angles(
            screen_x=point.screen_x,
            screen_y=point.screen_y,
            head_x=safe_float(getattr(point, "head_x", center_head_x), center_head_x),
            head_y=safe_float(getattr(point, "head_y", center_head_y), center_head_y),
            head_z=_positive_or(getattr(point, "head_z", center_head_z), center_head_z),
            center_zeta=center_zeta,
            screen_center_cam_x=float(screen_center_cam[0]),
            screen_center_cam_y=float(screen_center_cam[1]),
            screen_center_cam_z=float(screen_center_cam[2]),
            screen_axis_x_x=float(screen_axis_x[0]),
            screen_axis_x_y=float(screen_axis_x[1]),
            screen_axis_x_z=float(screen_axis_x[2]),
            screen_axis_y_x=float(screen_axis_y[0]),
            screen_axis_y_y=float(screen_axis_y[1]),
            screen_axis_y_z=float(screen_axis_y[2]),
            screen_fit_rmse=screen_fit_rmse,
            origin_x=center_point.screen_x,
            origin_y=center_point.screen_y,
        )
        if target_total is None:
            continue
        target_total_yaw, target_total_pitch = target_total

        face_delta_yaw = safe_float(getattr(point, "head_yaw", 0.0)) - safe_float(
            getattr(center_point, "head_yaw", 0.0)
        )
        face_delta_pitch = safe_float(getattr(point, "head_pitch", 0.0)) - safe_float(
            getattr(center_point, "head_pitch", 0.0)
        )

        target_eye_yaw = target_total_yaw - face_delta_yaw
        target_eye_pitch = target_total_pitch - face_delta_pitch

        A_rows.append([eye_dyaw, eye_dpitch])
        b_yaw.append(target_eye_yaw)
        b_pitch.append(target_eye_pitch)

    matrix_yaw_yaw = 1.0
    matrix_yaw_pitch = 0.0
    matrix_pitch_yaw = 0.0
    matrix_pitch_pitch = 1.0

    if len(A_rows) >= 2:
        A = np.array(A_rows, dtype=float)
        B = np.column_stack((np.array(b_yaw, dtype=float), np.array(b_pitch, dtype=float)))
        try:
            (
                matrix_yaw_yaw,
                matrix_yaw_pitch,
                matrix_pitch_yaw,
                matrix_pitch_pitch,
            ) = _fit_rotation_scale_matrix(A, B)
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning(f"Calibration solve failed, using identity matrix: {exc}")

    total_sample_count = sum(int(p.sample_count) for p in points)

    return CalibrationMatrix(
        center_yaw=safe_float(center_point.raw_eye_yaw),
        center_pitch=safe_float(center_point.raw_eye_pitch),
        face_center_yaw=safe_float(getattr(center_point, "head_yaw", 0.0)),
        face_center_pitch=safe_float(getattr(center_point, "head_pitch", 0.0)),
        center_zeta=center_zeta,
        matrix_yaw_yaw=matrix_yaw_yaw,
        matrix_yaw_pitch=matrix_yaw_pitch,
        matrix_pitch_yaw=matrix_pitch_yaw,
        matrix_pitch_pitch=matrix_pitch_pitch,
        face_center_x=center_head_x,
        face_center_y=center_head_y,
        face_center_z=center_head_z,
        screen_center_cam_x=float(screen_center_cam[0]),
        screen_center_cam_y=float(screen_center_cam[1]),
        screen_center_cam_z=float(screen_center_cam[2]),
        screen_axis_x_x=float(screen_axis_x[0]),
        screen_axis_x_y=float(screen_axis_x[1]),
        screen_axis_x_z=float(screen_axis_x[2]),
        screen_axis_y_x=float(screen_axis_y[0]),
        screen_axis_y_y=float(screen_axis_y[1]),
        screen_axis_y_z=float(screen_axis_y[2]),
        screen_fit_rmse=screen_fit_rmse,
        sample_count=total_sample_count,
        timestamp_ms=int(time.time() * 1000),
    )


def _profile_token(raw_profile: str) -> str:
    if not raw_profile:
        return "default"
    out: List[str] = []
    for ch in raw_profile:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("-")
    sanitized = "".join(out).strip("._-")
    return sanitized if sanitized else "default"


def _profile_filename(profile_token: str) -> str:
    return f"calibration-{profile_token}.json"


def _profile_load_candidates(profile_token: str) -> List[str]:
    filename = _profile_filename(profile_token)
    if profile_token != "default":
        return [filename]
    return [filename, "calibration.json"]


def save_calibration(
    calib: CalibrationMatrix, points: List[CalibrationPoint], profile: str = ""
) -> Path:
    profile_token = _profile_token(profile)
    filename = _profile_filename(profile_token)

    payload = {
        "timestamp": int(time.time() * 1000),
        "profile": profile_token,
        "calibration": {
            "modelVersion": int(CALIBRATION_MODEL_VERSION),
            "centerYaw": float(calib.center_yaw),
            "centerPitch": float(calib.center_pitch),
            "faceCenterYaw": float(calib.face_center_yaw),
            "faceCenterPitch": float(calib.face_center_pitch),
            "centerZeta": float(calib.center_zeta),
            "matrixYawYaw": float(calib.matrix_yaw_yaw),
            "matrixYawPitch": float(calib.matrix_yaw_pitch),
            "matrixPitchYaw": float(calib.matrix_pitch_yaw),
            "matrixPitchPitch": float(calib.matrix_pitch_pitch),
            "faceCenterX": float(calib.face_center_x),
            "faceCenterY": float(calib.face_center_y),
            "faceCenterZ": float(calib.face_center_z),
            "screenCenterCamX": float(calib.screen_center_cam_x),
            "screenCenterCamY": float(calib.screen_center_cam_y),
            "screenCenterCamZ": float(calib.screen_center_cam_z),
            "screenAxisXX": float(calib.screen_axis_x_x),
            "screenAxisXY": float(calib.screen_axis_x_y),
            "screenAxisXZ": float(calib.screen_axis_x_z),
            "screenAxisYX": float(calib.screen_axis_y_x),
            "screenAxisYY": float(calib.screen_axis_y_y),
            "screenAxisYZ": float(calib.screen_axis_y_z),
            "screenFitRmse": float(calib.screen_fit_rmse),
            "sampleCount": int(calib.sample_count),
        },
        "points": [
            {
                "name": str(point.name),
                "screenX": float(point.screen_x),
                "screenY": float(point.screen_y),
                "rawEyeYaw": float(point.raw_eye_yaw),
                "rawEyePitch": float(point.raw_eye_pitch),
                "rawLeftEyeYaw": float(point.raw_left_eye_yaw),
                "rawLeftEyePitch": float(point.raw_left_eye_pitch),
                "rawRightEyeYaw": float(point.raw_right_eye_yaw),
                "rawRightEyePitch": float(point.raw_right_eye_pitch),
                "headYaw": float(getattr(point, "head_yaw", 0.0)),
                "headPitch": float(getattr(point, "head_pitch", 0.0)),
                "zeta": float(getattr(point, "zeta", DEFAULT_CENTER_ZETA)),
                "headX": float(getattr(point, "head_x", 0.0)),
                "headY": float(getattr(point, "head_y", 0.0)),
                "headZ": float(getattr(point, "head_z", DEFAULT_CENTER_ZETA)),
                "sampleCount": int(point.sample_count),
            }
            for point in points
        ],
    }

    file_path = Path(filename)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return file_path


def load_calibration(
    profile: str = "",
) -> Tuple[CalibrationMatrix, List[CalibrationPoint]]:
    profile_token = _profile_token(profile)
    filenames = _profile_load_candidates(profile_token)
    file_path: Optional[Path] = None
    for filename in filenames:
        candidate = Path(filename)
        if candidate.exists():
            file_path = candidate
            break
    if file_path is None:
        file_path = Path(filenames[0])

    empty = CalibrationMatrix(
        center_yaw=0.0,
        center_pitch=0.0,
        face_center_yaw=0.0,
        face_center_pitch=0.0,
        center_zeta=DEFAULT_CENTER_ZETA,
        matrix_yaw_yaw=1.0,
        matrix_yaw_pitch=0.0,
        matrix_pitch_yaw=0.0,
        matrix_pitch_pitch=1.0,
        face_center_x=0.0,
        face_center_y=0.0,
        face_center_z=DEFAULT_CENTER_ZETA,
        screen_center_cam_x=0.0,
        screen_center_cam_y=0.0,
        screen_center_cam_z=DEFAULT_CENTER_ZETA,
        screen_axis_x_x=1.0,
        screen_axis_x_y=0.0,
        screen_axis_x_z=0.0,
        screen_axis_y_x=0.0,
        screen_axis_y_y=1.0,
        screen_axis_y_z=0.0,
        screen_fit_rmse=-1.0,
        sample_count=0,
        timestamp_ms=int(time.time() * 1000),
    )

    if not file_path.exists():
        return empty, []

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning(f"Failed to load calibration from {file_path.name}: {exc}")
        return empty, []

    calib_data = data.get("calibration", {})
    model_version = int(safe_float(calib_data.get("modelVersion", 1), 1))
    if model_version != CALIBRATION_MODEL_VERSION:
        logger.info(
            f"Calibration file {file_path.name} has model version {model_version}, expected {CALIBRATION_MODEL_VERSION}. Discarding."
        )
        return empty, []

    calib = CalibrationMatrix(
        center_yaw=safe_float(calib_data.get("centerYaw", 0.0)),
        center_pitch=safe_float(calib_data.get("centerPitch", 0.0)),
        face_center_yaw=safe_float(calib_data.get("faceCenterYaw", 0.0)),
        face_center_pitch=safe_float(calib_data.get("faceCenterPitch", 0.0)),
        center_zeta=_positive_or(
            calib_data.get("centerZeta", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
        ),
        matrix_yaw_yaw=safe_float(calib_data.get("matrixYawYaw", 1.0)),
        matrix_yaw_pitch=safe_float(calib_data.get("matrixYawPitch", 0.0)),
        matrix_pitch_yaw=safe_float(calib_data.get("matrixPitchYaw", 0.0)),
        matrix_pitch_pitch=safe_float(calib_data.get("matrixPitchPitch", 1.0)),
        face_center_x=safe_float(calib_data.get("faceCenterX", 0.0)),
        face_center_y=safe_float(calib_data.get("faceCenterY", 0.0)),
        face_center_z=_positive_or(
            calib_data.get("faceCenterZ", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
        ),
        screen_center_cam_x=safe_float(calib_data.get("screenCenterCamX", 0.0)),
        screen_center_cam_y=safe_float(calib_data.get("screenCenterCamY", 0.0)),
        screen_center_cam_z=safe_float(
            calib_data.get("screenCenterCamZ", DEFAULT_CENTER_ZETA),
            DEFAULT_CENTER_ZETA,
        ),
        screen_axis_x_x=safe_float(calib_data.get("screenAxisXX", 1.0)),
        screen_axis_x_y=safe_float(calib_data.get("screenAxisXY", 0.0)),
        screen_axis_x_z=safe_float(calib_data.get("screenAxisXZ", 0.0)),
        screen_axis_y_x=safe_float(calib_data.get("screenAxisYX", 0.0)),
        screen_axis_y_y=safe_float(calib_data.get("screenAxisYY", 1.0)),
        screen_axis_y_z=safe_float(calib_data.get("screenAxisYZ", 0.0)),
        screen_fit_rmse=safe_float(calib_data.get("screenFitRmse", -1.0)),
        sample_count=int(calib_data.get("sampleCount", 0)),
        timestamp_ms=int(data.get("timestamp", time.time() * 1000)),
    )

    points: List[CalibrationPoint] = []
    for point_data in data.get("points", []):
        points.append(
            CalibrationPoint(
                name=str(point_data.get("name", "")),
                screen_x=safe_float(point_data.get("screenX", 0.0)),
                screen_y=safe_float(point_data.get("screenY", 0.0)),
                raw_eye_yaw=safe_float(point_data.get("rawEyeYaw", 0.0)),
                raw_eye_pitch=safe_float(point_data.get("rawEyePitch", 0.0)),
                raw_left_eye_yaw=safe_float(point_data.get("rawLeftEyeYaw", 0.0)),
                raw_left_eye_pitch=safe_float(point_data.get("rawLeftEyePitch", 0.0)),
                raw_right_eye_yaw=safe_float(point_data.get("rawRightEyeYaw", 0.0)),
                raw_right_eye_pitch=safe_float(point_data.get("rawRightEyePitch", 0.0)),
                sample_count=int(point_data.get("sampleCount", 0)),
                head_yaw=safe_float(point_data.get("headYaw", 0.0)),
                head_pitch=safe_float(point_data.get("headPitch", 0.0)),
                zeta=_positive_or(
                    point_data.get("zeta", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
                ),
                head_x=safe_float(point_data.get("headX", 0.0)),
                head_y=safe_float(point_data.get("headY", 0.0)),
                head_z=_positive_or(
                    point_data.get("headZ", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
                ),
            )
        )

    return calib, points


def apply_calibration_model(
    raw_eye_yaw: Optional[float],
    raw_eye_pitch: Optional[float],
    head_yaw: Optional[float],
    head_pitch: Optional[float],
    *,
    head_x: Optional[float] = None,
    head_y: Optional[float] = None,
    head_z: Optional[float] = None,
    center_eye_yaw: float = 0.0,
    center_eye_pitch: float = 0.0,
    face_center_yaw: float = 0.0,
    face_center_pitch: float = 0.0,
    matrix_yaw_yaw: float = 1.0,
    matrix_yaw_pitch: float = 0.0,
    matrix_pitch_yaw: float = 0.0,
    matrix_pitch_pitch: float = 1.0,
    center_zeta: float = DEFAULT_CENTER_ZETA,
    face_center_x: float = 0.0,
    face_center_y: float = 0.0,
    face_center_z: float = DEFAULT_CENTER_ZETA,
    screen_center_cam_x: float = 0.0,
    screen_center_cam_y: float = 0.0,
    screen_center_cam_z: float = DEFAULT_CENTER_ZETA,
    screen_axis_x_x: float = 1.0,
    screen_axis_x_y: float = 0.0,
    screen_axis_x_z: float = 0.0,
    screen_axis_y_x: float = 0.0,
    screen_axis_y_y: float = 1.0,
    screen_axis_y_z: float = 0.0,
    screen_fit_rmse: float = -1.0,
    origin_x: Optional[float] = None,
    origin_y: Optional[float] = None,
) -> Dict[str, Any]:
    if raw_eye_yaw is None or raw_eye_pitch is None:
        raise ValueError("Raw eye angles are required")
    if head_yaw is None or head_pitch is None:
        raise ValueError("Head angles are required")
    if head_x is None or head_y is None or head_z is None:
        raise ValueError("Head position is required")

    center_eye_yaw_value = safe_float(center_eye_yaw, 0.0)
    center_eye_pitch_value = safe_float(center_eye_pitch, 0.0)
    face_center_yaw_value = safe_float(face_center_yaw, 0.0)
    face_center_pitch_value = safe_float(face_center_pitch, 0.0)

    raw_eye_yaw_value = safe_float(raw_eye_yaw, float("nan"))
    raw_eye_pitch_value = safe_float(raw_eye_pitch, float("nan"))
    head_yaw_value = safe_float(head_yaw, float("nan"))
    head_pitch_value = safe_float(head_pitch, float("nan"))
    head_x_value = safe_float(head_x, float("nan"))
    head_y_value = safe_float(head_y, float("nan"))
    head_z_value = safe_float(head_z, float("nan"))

    if not (
        math.isfinite(raw_eye_yaw_value)
        and math.isfinite(raw_eye_pitch_value)
        and math.isfinite(head_yaw_value)
        and math.isfinite(head_pitch_value)
        and math.isfinite(head_x_value)
        and math.isfinite(head_y_value)
        and math.isfinite(head_z_value)
    ):
        raise ValueError("Calibration input values must be finite")
    if head_z_value <= 1e-9:
        raise ValueError("Head Z position must be positive")

    raw_screen_axis_x = np.array(
        [
            safe_float(screen_axis_x_x, 1.0),
            safe_float(screen_axis_x_y, 0.0),
            safe_float(screen_axis_x_z, 0.0),
        ],
        dtype=float,
    )
    raw_screen_axis_y = np.array(
        [
            safe_float(screen_axis_y_x, 0.0),
            safe_float(screen_axis_y_y, 1.0),
            safe_float(screen_axis_y_z, 0.0),
        ],
        dtype=float,
    )

    axis_x_norm = float(np.linalg.norm(raw_screen_axis_x))
    if axis_x_norm <= 1e-9:
        screen_axis_x = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        screen_axis_x = raw_screen_axis_x / axis_x_norm

    axis_y_ortho = raw_screen_axis_y - float(
        np.dot(raw_screen_axis_y, screen_axis_x)
    ) * screen_axis_x
    axis_y_norm = float(np.linalg.norm(axis_y_ortho))
    if axis_y_norm <= 1e-9:
        screen_axis_y = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        screen_axis_y = axis_y_ortho / axis_y_norm

    screen_normal = np.cross(screen_axis_x, screen_axis_y)
    normal_norm = float(np.linalg.norm(screen_normal))
    if normal_norm <= 1e-9:
        screen_normal = np.array([0.0, 0.0, -1.0], dtype=float)
    else:
        screen_normal = screen_normal / normal_norm

    head_origin = np.array([head_x_value, head_y_value, head_z_value], dtype=float)
    face_origin = np.array(
        [
            safe_float(face_center_x, 0.0),
            safe_float(face_center_y, 0.0),
            _positive_or(face_center_z, DEFAULT_CENTER_ZETA),
        ],
        dtype=float,
    )
    head_delta = head_origin - face_origin
    head_ref_x = float(np.dot(head_delta, screen_axis_x))
    head_ref_y = float(np.dot(head_delta, screen_axis_y))
    head_ref_z = head_z_value

    uncalibrated_mode = safe_float(screen_fit_rmse, -1.0) < 0.0
    if uncalibrated_mode:
        eye_delta_yaw = raw_eye_yaw_value
        eye_delta_pitch = raw_eye_pitch_value
        face_delta_yaw = head_yaw_value
        face_delta_pitch = head_pitch_value
        corrected_eye_yaw = raw_eye_yaw_value
        corrected_eye_pitch = raw_eye_pitch_value
        corrected_yaw_linear = head_yaw_value + raw_eye_yaw_value
        corrected_pitch_linear = head_pitch_value + raw_eye_pitch_value
    else:
        eye_delta_yaw = raw_eye_yaw_value - center_eye_yaw_value
        eye_delta_pitch = raw_eye_pitch_value - center_eye_pitch_value
        face_delta_yaw = head_yaw_value - face_center_yaw_value
        face_delta_pitch = head_pitch_value - face_center_pitch_value

        corrected_eye_yaw = safe_float(matrix_yaw_yaw, 1.0) * eye_delta_yaw + safe_float(
            matrix_yaw_pitch, 0.0
        ) * eye_delta_pitch
        corrected_eye_pitch = safe_float(
            matrix_pitch_yaw, 0.0
        ) * eye_delta_yaw + safe_float(matrix_pitch_pitch, 1.0) * eye_delta_pitch

        corrected_yaw_linear = face_delta_yaw + corrected_eye_yaw
        corrected_pitch_linear = face_delta_pitch + corrected_eye_pitch

    corrected_yaw = corrected_yaw_linear
    corrected_pitch = corrected_pitch_linear
    corrected_screen_x = None
    corrected_screen_y = None
    screen_offset_x = None
    screen_offset_y = None
    screen_projection_t = None
    screen_depth = _positive_or(head_z_value, DEFAULT_CENTER_ZETA)
    projected = project_head_angles_to_screen_xy(
        yaw_deg=corrected_yaw_linear,
        pitch_deg=corrected_pitch_linear,
        head_x=head_x_value,
        head_y=head_y_value,
        head_z=head_z_value,
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
    if projected is not None:
        corrected_screen_x = float(projected["screen_x"])
        corrected_screen_y = float(projected["screen_y"])
        screen_offset_x = float(projected["offset_x"])
        screen_offset_y = float(projected["offset_y"])
        screen_projection_t = float(projected["projection_t"])

    return {
        "raw_eye_yaw": raw_eye_yaw_value,
        "raw_eye_pitch": raw_eye_pitch_value,
        "head_yaw": head_yaw_value,
        "head_pitch": head_pitch_value,
        "head_x": head_x_value,
        "head_y": head_y_value,
        "head_z": head_z_value,
        "eye_delta_yaw": eye_delta_yaw,
        "eye_delta_pitch": eye_delta_pitch,
        "face_delta_yaw": face_delta_yaw,
        "face_delta_pitch": face_delta_pitch,
        "corrected_eye_yaw": corrected_eye_yaw,
        "corrected_eye_pitch": corrected_eye_pitch,
        "corrected_yaw_linear": corrected_yaw_linear,
        "corrected_pitch_linear": corrected_pitch_linear,
        "corrected_yaw": corrected_yaw,
        "corrected_pitch": corrected_pitch,
        "corrected_screen_x": corrected_screen_x,
        "corrected_screen_y": corrected_screen_y,
        "screen_offset_x": screen_offset_x,
        "screen_offset_y": screen_offset_y,
        "screen_projection_t": screen_projection_t,
        "screen_depth": screen_depth,
        "head_ref_x": head_ref_x,
        "head_ref_y": head_ref_y,
        "head_ref_z": head_ref_z,
        "screen_normal_x": float(screen_normal[0]),
        "screen_normal_y": float(screen_normal[1]),
        "screen_normal_z": float(screen_normal[2]),
    }


@dataclass(frozen=True)
class CalibratedFaceAndGazeEvent:
    face_mesh_event: FaceMeshEvent
    pitch_calibration: float
    yaw_calibration: float
    roll_calibration: float
    face_center_yaw: float = 0.0
    face_center_pitch: float = 0.0
    center_zeta: float = DEFAULT_CENTER_ZETA
    matrix_yaw_yaw: float = 1.0
    matrix_yaw_pitch: float = 0.0
    matrix_pitch_yaw: float = 0.0
    matrix_pitch_pitch: float = 1.0
    face_center_x: float = 0.0
    face_center_y: float = 0.0
    face_center_z: float = DEFAULT_CENTER_ZETA
    screen_center_cam_x: float = 0.0
    screen_center_cam_y: float = 0.0
    screen_center_cam_z: float = DEFAULT_CENTER_ZETA
    screen_axis_x_x: float = 1.0
    screen_axis_x_y: float = 0.0
    screen_axis_x_z: float = 0.0
    screen_axis_y_x: float = 0.0
    screen_axis_y_y: float = 1.0
    screen_axis_y_z: float = 0.0
    screen_fit_rmse: float = -1.0
    display_width: int = 1920
    display_height: int = 1080
    origin_x: float = 960.0
    origin_y: float = 540.0

    @property
    def raw_eye_yaw(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.combined_eye_gaze_yaw

    @property
    def raw_eye_pitch(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.combined_eye_gaze_pitch

    @property
    def head_yaw(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.head_yaw

    @property
    def head_pitch(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.head_pitch

    @property
    def head_x(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.camera_x

    @property
    def head_y(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.camera_y

    @property
    def head_z(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.camera_z

    @property
    def calibrated_components(self) -> Dict[str, Any]:
        return apply_calibration_model(
            raw_eye_yaw=self.raw_eye_yaw,
            raw_eye_pitch=self.raw_eye_pitch,
            head_yaw=self.head_yaw,
            head_pitch=self.head_pitch,
            head_x=self.head_x,
            head_y=self.head_y,
            head_z=self.head_z,
            center_eye_yaw=self.yaw_calibration,
            center_eye_pitch=self.pitch_calibration,
            face_center_yaw=self.face_center_yaw,
            face_center_pitch=self.face_center_pitch,
            matrix_yaw_yaw=self.matrix_yaw_yaw,
            matrix_yaw_pitch=self.matrix_yaw_pitch,
            matrix_pitch_yaw=self.matrix_pitch_yaw,
            matrix_pitch_pitch=self.matrix_pitch_pitch,
            center_zeta=self.center_zeta,
            face_center_x=self.face_center_x,
            face_center_y=self.face_center_y,
            face_center_z=self.face_center_z,
            screen_center_cam_x=self.screen_center_cam_x,
            screen_center_cam_y=self.screen_center_cam_y,
            screen_center_cam_z=self.screen_center_cam_z,
            screen_axis_x_x=self.screen_axis_x_x,
            screen_axis_x_y=self.screen_axis_x_y,
            screen_axis_x_z=self.screen_axis_x_z,
            screen_axis_y_x=self.screen_axis_y_x,
            screen_axis_y_y=self.screen_axis_y_y,
            screen_axis_y_z=self.screen_axis_y_z,
            screen_fit_rmse=self.screen_fit_rmse,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
        )

    @property
    def face_delta_yaw(self) -> float:
        return self.calibrated_components["face_delta_yaw"]

    @property
    def face_delta_pitch(self) -> float:
        return self.calibrated_components["face_delta_pitch"]

    @property
    def corrected_eye_yaw(self) -> float:
        return self.calibrated_components["corrected_eye_yaw"]

    @property
    def corrected_eye_pitch(self) -> float:
        return self.calibrated_components["corrected_eye_pitch"]

    @property
    def corrected_yaw(self) -> float:
        return self.calibrated_components["corrected_yaw"]

    @property
    def corrected_pitch(self) -> float:
        return self.calibrated_components["corrected_pitch"]

    @property
    def corrected_screen_x(self) -> Optional[float]:
        return self.calibrated_components.get("corrected_screen_x")

    @property
    def corrected_screen_y(self) -> Optional[float]:
        return self.calibrated_components.get("corrected_screen_y")

    @property
    def corrected_yaw_linear(self) -> float:
        return self.calibrated_components["corrected_yaw_linear"]

    @property
    def corrected_pitch_linear(self) -> float:
        return self.calibrated_components["corrected_pitch_linear"]

    @property
    def head_ref_x(self) -> float:
        return self.calibrated_components["head_ref_x"]

    @property
    def head_ref_y(self) -> float:
        return self.calibrated_components["head_ref_y"]

    @property
    def head_ref_z(self) -> float:
        return self.calibrated_components["head_ref_z"]
