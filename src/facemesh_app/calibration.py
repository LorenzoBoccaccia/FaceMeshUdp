from dataclasses import dataclass
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .facemesh_dao import FaceMeshEvent, safe_float
from .gaze_primitives import project_head_angles_to_screen_xy


logger = logging.getLogger(__name__)

DEFAULT_CENTER_ZETA = 1200.0
CALIBRATION_MODEL_VERSION = 9


@dataclass(frozen=True)
class CalibrationMatrix:
    center_yaw: float = 0.0
    center_pitch: float = 0.0
    face_center_yaw: float = 0.0
    face_center_pitch: float = 0.0
    center_zeta: float = DEFAULT_CENTER_ZETA
    yaw_coefficient_positive: float = 1.0
    yaw_coefficient_negative: float = 1.0
    pitch_coefficient_positive: float = 1.0
    pitch_coefficient_negative: float = 1.0
    yaw_from_pitch_coupling: float = 0.0
    pitch_from_yaw_coupling: float = 0.0
    eye_yaw_min: float = -1.0
    eye_yaw_max: float = 1.0
    eye_pitch_min: float = -1.0
    eye_pitch_max: float = 1.0
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
    screen_scale_x: float = 1.0
    screen_scale_y: float = 1.0
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
    nose_target_x: Optional[float] = None
    nose_target_y: Optional[float] = None
    eye_target_x: Optional[float] = None
    eye_target_y: Optional[float] = None


def _positive_or(v: Any, fallback: float) -> float:
    x = safe_float(v, fallback)
    return x if x > 1e-9 else fallback


def _positive_coefficient(v: Any, fallback: float) -> float:
    x = abs(safe_float(v, fallback))
    return x if math.isfinite(x) and x > 1e-9 else fallback


def _build_screen_geometry(
    center_point: CalibrationPoint,
    center_zeta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return screen_center, screen_axis_x, screen_axis_y


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


def _coefficient_from_samples(samples: List[float]) -> Optional[float]:
    if not samples:
        return None
    arr = np.array(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _resolve_axis_coefficients(
    positive_samples: List[float],
    negative_samples: List[float],
) -> Tuple[float, float]:
    positive = _coefficient_from_samples(positive_samples)
    negative = _coefficient_from_samples(negative_samples)
    if positive is None and negative is None:
        return 1.0, 1.0
    if positive is None:
        positive = negative
    if negative is None:
        negative = positive
    return _positive_coefficient(positive, 1.0), _positive_coefficient(negative, 1.0)


def _resolve_axis_extension(samples: List[float]) -> Tuple[float, float]:
    if not samples:
        return -1.0, 1.0
    arr = np.array(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    axis_min = float(np.min(arr))
    axis_max = float(np.max(arr))
    if abs(axis_max - axis_min) <= 1e-6:
        axis_min -= 1e-3
        axis_max += 1e-3
    return axis_min, axis_max


def _resolve_screen_scale(
    points: List[CalibrationPoint],
    center_point: CalibrationPoint,
    center_zeta: float,
    screen_center_cam: np.ndarray,
    screen_axis_x: np.ndarray,
    screen_axis_y: np.ndarray,
) -> Tuple[float, float]:
    origin_x = safe_float(center_point.screen_x, 0.0)
    origin_y = safe_float(center_point.screen_y, 0.0)
    center_head_yaw = safe_float(getattr(center_point, "head_yaw", 0.0), 0.0)
    center_head_pitch = safe_float(getattr(center_point, "head_pitch", 0.0), 0.0)

    scale_x_samples: List[float] = []
    scale_y_samples: List[float] = []
    for point in points:
        if point.name == "C":
            continue
        if point.nose_target_x is None or point.nose_target_y is None:
            continue
        head_x = safe_float(getattr(point, "head_x", 0.0), 0.0)
        head_y = safe_float(getattr(point, "head_y", 0.0), 0.0)
        head_z = _positive_or(getattr(point, "head_z", center_zeta), center_zeta)
        face_delta_yaw = safe_float(getattr(point, "head_yaw", 0.0), 0.0) - center_head_yaw
        face_delta_pitch = safe_float(getattr(point, "head_pitch", 0.0), 0.0) - center_head_pitch
        projected = project_head_angles_to_screen_xy(
            yaw_deg=face_delta_yaw,
            pitch_deg=face_delta_pitch,
            head_x=head_x,
            head_y=head_y,
            head_z=head_z,
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
            screen_fit_rmse=0.0,
            screen_scale_x=1.0,
            screen_scale_y=1.0,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        if projected is None:
            continue
        offset_x_base = safe_float(projected.get("screen_x"), origin_x) - origin_x
        offset_y_base = safe_float(projected.get("screen_y"), origin_y) - origin_y
        desired_offset_x = safe_float(point.nose_target_x, origin_x) - origin_x
        desired_offset_y = safe_float(point.nose_target_y, origin_y) - origin_y
        if abs(offset_x_base) > 1e-6:
            candidate_x = desired_offset_x / offset_x_base
            if math.isfinite(candidate_x) and candidate_x > 1e-6:
                scale_x_samples.append(candidate_x)
        if abs(offset_y_base) > 1e-6:
            candidate_y = desired_offset_y / offset_y_base
            if math.isfinite(candidate_y) and candidate_y > 1e-6:
                scale_y_samples.append(candidate_y)

    scale_x = float(np.median(np.array(scale_x_samples, dtype=float))) if scale_x_samples else 1.0
    scale_y = float(np.median(np.array(scale_y_samples, dtype=float))) if scale_y_samples else 1.0
    return _positive_or(scale_x, 1.0), _positive_or(scale_y, 1.0)


def _interpolate_coefficient(
    eye_delta: float,
    axis_min: float,
    axis_max: float,
    negative_coefficient: float,
    positive_coefficient: float,
) -> float:
    span = axis_max - axis_min
    if abs(span) <= 1e-9:
        return 0.5 * (
            _positive_coefficient(negative_coefficient, 1.0)
            + _positive_coefficient(positive_coefficient, 1.0)
        )
    t = (eye_delta - axis_min) / span
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    c_neg = _positive_coefficient(negative_coefficient, 1.0)
    c_pos = _positive_coefficient(positive_coefficient, 1.0)
    return c_neg + (c_pos - c_neg) * t


def _fit_linear_scalar(inputs: List[float], targets: List[float]) -> float:
    if not inputs or not targets:
        return 0.0
    if len(inputs) != len(targets):
        return 0.0
    x = np.array(inputs, dtype=float)
    y = np.array(targets, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return 0.0
    denom = float(np.dot(x, x))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(x, y) / denom)


def _signum(value: float, eps: float = 1e-6) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def compute_calibration_matrix(points: List[CalibrationPoint]) -> CalibrationMatrix:
    if len(points) < 9:
        raise ValueError(f"Calibration requires 9 points, got {len(points)}")
    required_names = {"C", "T", "TL", "L", "BL", "B", "BR", "R", "TR"}
    available_names = {str(p.name) for p in points}
    if not required_names.issubset(available_names):
        missing = sorted(required_names - available_names)
        raise ValueError(f"Calibration points missing required targets: {missing}")

    center_point = next((p for p in points if p.name == "C"), None)
    if center_point is None:
        raise ValueError("Calibration points must include a center point named 'C'")

    center_zeta = _positive_or(
        getattr(center_point, "zeta", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
    )
    center_head_x = safe_float(getattr(center_point, "head_x", 0.0), 0.0)
    center_head_y = safe_float(getattr(center_point, "head_y", 0.0), 0.0)
    center_head_z = _positive_or(getattr(center_point, "head_z", center_zeta), center_zeta)
    screen_center_cam, screen_axis_x, screen_axis_y = _build_screen_geometry(
        center_point=center_point,
        center_zeta=center_zeta,
    )

    yaw_positive_samples: List[float] = []
    yaw_negative_samples: List[float] = []
    pitch_positive_samples: List[float] = []
    pitch_negative_samples: List[float] = []
    eye_yaw_samples: List[float] = []
    eye_pitch_samples: List[float] = []
    sign_issues: List[str] = []

    for point in points:
        if point.name == "C":
            continue
        eye_delta_yaw = safe_float(point.raw_eye_yaw) - safe_float(center_point.raw_eye_yaw)
        eye_delta_pitch = safe_float(point.raw_eye_pitch) - safe_float(center_point.raw_eye_pitch)
        face_delta_yaw = safe_float(getattr(point, "head_yaw", 0.0)) - safe_float(
            getattr(center_point, "head_yaw", 0.0)
        )
        face_delta_pitch = safe_float(getattr(point, "head_pitch", 0.0)) - safe_float(
            getattr(center_point, "head_pitch", 0.0)
        )
        expected_nose_x = safe_float(
            point.nose_target_x if point.nose_target_x is not None else point.screen_x,
            center_point.screen_x,
        )
        expected_nose_y = safe_float(
            point.nose_target_y if point.nose_target_y is not None else point.screen_y,
            center_point.screen_y,
        )
        expected_eye_x = safe_float(
            point.eye_target_x if point.eye_target_x is not None else point.screen_x,
            center_point.screen_x,
        )
        expected_eye_y = safe_float(
            point.eye_target_y if point.eye_target_y is not None else point.screen_y,
            center_point.screen_y,
        )
        expected_face_yaw_sign = _signum(expected_nose_x - center_point.screen_x)
        expected_face_pitch_sign = _signum(center_point.screen_y - expected_nose_y)
        expected_eye_yaw_sign = _signum(expected_eye_x - center_point.screen_x)
        expected_eye_pitch_sign = _signum(center_point.screen_y - expected_eye_y)

        eye_yaw_samples.append(eye_delta_yaw)
        eye_pitch_samples.append(eye_delta_pitch)

        if abs(eye_delta_yaw) > 1e-6 and abs(face_delta_yaw) > 1e-6:
            ratio_yaw = abs(face_delta_yaw / eye_delta_yaw)
            bucket_yaw_sign = (
                expected_eye_yaw_sign
                if expected_eye_yaw_sign != 0
                else _signum(eye_delta_yaw)
            )
            if bucket_yaw_sign >= 0:
                yaw_positive_samples.append(ratio_yaw)
            else:
                yaw_negative_samples.append(ratio_yaw)

        if abs(eye_delta_pitch) > 1e-6 and abs(face_delta_pitch) > 1e-6:
            ratio_pitch = abs(face_delta_pitch / eye_delta_pitch)
            bucket_pitch_sign = (
                expected_eye_pitch_sign
                if expected_eye_pitch_sign != 0
                else _signum(eye_delta_pitch)
            )
            if bucket_pitch_sign >= 0:
                pitch_positive_samples.append(ratio_pitch)
            else:
                pitch_negative_samples.append(ratio_pitch)

        if (
            expected_face_yaw_sign != 0
            and abs(face_delta_yaw) > 0.5
            and _signum(face_delta_yaw) != expected_face_yaw_sign
        ):
            sign_issues.append(f"{point.name}: face yaw sign mismatch")
        if (
            expected_face_pitch_sign != 0
            and abs(face_delta_pitch) > 0.5
            and _signum(face_delta_pitch) != expected_face_pitch_sign
        ):
            sign_issues.append(f"{point.name}: face pitch sign mismatch")
        if (
            expected_eye_yaw_sign != 0
            and abs(eye_delta_yaw) > 0.5
            and _signum(eye_delta_yaw) != expected_eye_yaw_sign
        ):
            sign_issues.append(f"{point.name}: eye yaw sign mismatch")
        if (
            expected_eye_pitch_sign != 0
            and abs(eye_delta_pitch) > 0.5
            and _signum(eye_delta_pitch) != expected_eye_pitch_sign
        ):
            sign_issues.append(f"{point.name}: eye pitch sign mismatch")

    if sign_issues:
        deduped = sorted(set(sign_issues))
        raise ValueError(
            "Calibration sign validation failed: " + "; ".join(deduped)
        )

    yaw_positive, yaw_negative = _resolve_axis_coefficients(
        yaw_positive_samples, yaw_negative_samples
    )
    pitch_positive, pitch_negative = _resolve_axis_coefficients(
        pitch_positive_samples, pitch_negative_samples
    )
    eye_yaw_min, eye_yaw_max = _resolve_axis_extension(eye_yaw_samples)
    eye_pitch_min, eye_pitch_max = _resolve_axis_extension(eye_pitch_samples)
    yaw_cross_inputs: List[float] = []
    yaw_cross_targets: List[float] = []
    pitch_cross_inputs: List[float] = []
    pitch_cross_targets: List[float] = []
    center_head_yaw = safe_float(getattr(center_point, "head_yaw", 0.0), 0.0)
    center_head_pitch = safe_float(getattr(center_point, "head_pitch", 0.0), 0.0)
    center_eye_yaw = safe_float(center_point.raw_eye_yaw, 0.0)
    center_eye_pitch = safe_float(center_point.raw_eye_pitch, 0.0)
    for point in points:
        if point.name == "C":
            continue
        eye_delta_yaw = safe_float(point.raw_eye_yaw, 0.0) - center_eye_yaw
        eye_delta_pitch = safe_float(point.raw_eye_pitch, 0.0) - center_eye_pitch
        face_delta_yaw = safe_float(getattr(point, "head_yaw", 0.0), 0.0) - center_head_yaw
        face_delta_pitch = safe_float(getattr(point, "head_pitch", 0.0), 0.0) - center_head_pitch
        yaw_coeff_point = _interpolate_coefficient(
            eye_delta=eye_delta_yaw,
            axis_min=eye_yaw_min,
            axis_max=eye_yaw_max,
            negative_coefficient=yaw_negative,
            positive_coefficient=yaw_positive,
        )
        pitch_coeff_point = _interpolate_coefficient(
            eye_delta=eye_delta_pitch,
            axis_min=eye_pitch_min,
            axis_max=eye_pitch_max,
            negative_coefficient=pitch_negative,
            positive_coefficient=pitch_positive,
        )
        yaw_cross_inputs.append(eye_delta_pitch)
        yaw_cross_targets.append((-face_delta_yaw) - (yaw_coeff_point * eye_delta_yaw))
        pitch_cross_inputs.append(eye_delta_yaw)
        pitch_cross_targets.append((-face_delta_pitch) - (pitch_coeff_point * eye_delta_pitch))
    yaw_from_pitch_coupling = _fit_linear_scalar(yaw_cross_inputs, yaw_cross_targets)
    pitch_from_yaw_coupling = _fit_linear_scalar(pitch_cross_inputs, pitch_cross_targets)
    screen_scale_x, screen_scale_y = _resolve_screen_scale(
        points=points,
        center_point=center_point,
        center_zeta=center_zeta,
        screen_center_cam=screen_center_cam,
        screen_axis_x=screen_axis_x,
        screen_axis_y=screen_axis_y,
    )
    total_sample_count = sum(int(p.sample_count) for p in points)

    return CalibrationMatrix(
        center_yaw=safe_float(center_point.raw_eye_yaw),
        center_pitch=safe_float(center_point.raw_eye_pitch),
        face_center_yaw=safe_float(getattr(center_point, "head_yaw", 0.0)),
        face_center_pitch=safe_float(getattr(center_point, "head_pitch", 0.0)),
        center_zeta=center_zeta,
        yaw_coefficient_positive=yaw_positive,
        yaw_coefficient_negative=yaw_negative,
        pitch_coefficient_positive=pitch_positive,
        pitch_coefficient_negative=pitch_negative,
        yaw_from_pitch_coupling=yaw_from_pitch_coupling,
        pitch_from_yaw_coupling=pitch_from_yaw_coupling,
        eye_yaw_min=eye_yaw_min,
        eye_yaw_max=eye_yaw_max,
        eye_pitch_min=eye_pitch_min,
        eye_pitch_max=eye_pitch_max,
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
        screen_scale_x=screen_scale_x,
        screen_scale_y=screen_scale_y,
        screen_fit_rmse=0.0,
        sample_count=total_sample_count,
        timestamp_ms=int(time.time() * 1000),
    )


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
            "yawCoefficientPositive": float(calib.yaw_coefficient_positive),
            "yawCoefficientNegative": float(calib.yaw_coefficient_negative),
            "pitchCoefficientPositive": float(calib.pitch_coefficient_positive),
            "pitchCoefficientNegative": float(calib.pitch_coefficient_negative),
            "yawFromPitchCoupling": float(calib.yaw_from_pitch_coupling),
            "pitchFromYawCoupling": float(calib.pitch_from_yaw_coupling),
            "eyeYawMin": float(calib.eye_yaw_min),
            "eyeYawMax": float(calib.eye_yaw_max),
            "eyePitchMin": float(calib.eye_pitch_min),
            "eyePitchMax": float(calib.eye_pitch_max),
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
            "screenScaleX": float(calib.screen_scale_x),
            "screenScaleY": float(calib.screen_scale_y),
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
                "noseTargetX": (
                    float(point.nose_target_x) if point.nose_target_x is not None else None
                ),
                "noseTargetY": (
                    float(point.nose_target_y) if point.nose_target_y is not None else None
                ),
                "eyeTargetX": (
                    float(point.eye_target_x) if point.eye_target_x is not None else None
                ),
                "eyeTargetY": (
                    float(point.eye_target_y) if point.eye_target_y is not None else None
                ),
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
        yaw_coefficient_positive=1.0,
        yaw_coefficient_negative=1.0,
        pitch_coefficient_positive=1.0,
        pitch_coefficient_negative=1.0,
        yaw_from_pitch_coupling=0.0,
        pitch_from_yaw_coupling=0.0,
        eye_yaw_min=-1.0,
        eye_yaw_max=1.0,
        eye_pitch_min=-1.0,
        eye_pitch_max=1.0,
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
        screen_scale_x=1.0,
        screen_scale_y=1.0,
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
    if model_version not in (7, 8, CALIBRATION_MODEL_VERSION):
        logger.info(
            f"Calibration file {file_path.name} has model version {model_version}, expected 7, 8 or {CALIBRATION_MODEL_VERSION}. Discarding."
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
        yaw_coefficient_positive=_positive_coefficient(
            calib_data.get("yawCoefficientPositive", 1.0), 1.0
        ),
        yaw_coefficient_negative=_positive_coefficient(
            calib_data.get("yawCoefficientNegative", 1.0), 1.0
        ),
        pitch_coefficient_positive=_positive_coefficient(
            calib_data.get("pitchCoefficientPositive", 1.0), 1.0
        ),
        pitch_coefficient_negative=_positive_coefficient(
            calib_data.get("pitchCoefficientNegative", 1.0), 1.0
        ),
        yaw_from_pitch_coupling=safe_float(calib_data.get("yawFromPitchCoupling", 0.0), 0.0),
        pitch_from_yaw_coupling=safe_float(calib_data.get("pitchFromYawCoupling", 0.0), 0.0),
        eye_yaw_min=safe_float(calib_data.get("eyeYawMin", -1.0)),
        eye_yaw_max=safe_float(calib_data.get("eyeYawMax", 1.0)),
        eye_pitch_min=safe_float(calib_data.get("eyePitchMin", -1.0)),
        eye_pitch_max=safe_float(calib_data.get("eyePitchMax", 1.0)),
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
        screen_scale_x=_positive_or(calib_data.get("screenScaleX", 1.0), 1.0),
        screen_scale_y=_positive_or(calib_data.get("screenScaleY", 1.0), 1.0),
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
                nose_target_x=safe_float(point_data.get("noseTargetX"), float("nan"))
                if point_data.get("noseTargetX") is not None
                else None,
                nose_target_y=safe_float(point_data.get("noseTargetY"), float("nan"))
                if point_data.get("noseTargetY") is not None
                else None,
                eye_target_x=safe_float(point_data.get("eyeTargetX"), float("nan"))
                if point_data.get("eyeTargetX") is not None
                else None,
                eye_target_y=safe_float(point_data.get("eyeTargetY"), float("nan"))
                if point_data.get("eyeTargetY") is not None
                else None,
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
    yaw_coefficient_positive: float = 1.0,
    yaw_coefficient_negative: float = 1.0,
    pitch_coefficient_positive: float = 1.0,
    pitch_coefficient_negative: float = 1.0,
    yaw_from_pitch_coupling: float = 0.0,
    pitch_from_yaw_coupling: float = 0.0,
    eye_yaw_min: float = -1.0,
    eye_yaw_max: float = 1.0,
    eye_pitch_min: float = -1.0,
    eye_pitch_max: float = 1.0,
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
    screen_scale_x: float = 1.0,
    screen_scale_y: float = 1.0,
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
        applied_yaw_coefficient = 1.0
        applied_pitch_coefficient = 1.0
        applied_yaw_from_pitch_coupling = 0.0
        applied_pitch_from_yaw_coupling = 0.0
        corrected_yaw_linear = head_yaw_value + raw_eye_yaw_value
        corrected_pitch_linear = head_pitch_value + raw_eye_pitch_value
    else:
        eye_delta_yaw = raw_eye_yaw_value - center_eye_yaw_value
        eye_delta_pitch = raw_eye_pitch_value - center_eye_pitch_value
        face_delta_yaw = head_yaw_value - face_center_yaw_value
        face_delta_pitch = head_pitch_value - face_center_pitch_value
        applied_yaw_coefficient = _interpolate_coefficient(
            eye_delta=eye_delta_yaw,
            axis_min=safe_float(eye_yaw_min, -1.0),
            axis_max=safe_float(eye_yaw_max, 1.0),
            negative_coefficient=yaw_coefficient_negative,
            positive_coefficient=yaw_coefficient_positive,
        )
        applied_pitch_coefficient = _interpolate_coefficient(
            eye_delta=eye_delta_pitch,
            axis_min=safe_float(eye_pitch_min, -1.0),
            axis_max=safe_float(eye_pitch_max, 1.0),
            negative_coefficient=pitch_coefficient_negative,
            positive_coefficient=pitch_coefficient_positive,
        )
        applied_yaw_from_pitch_coupling = safe_float(yaw_from_pitch_coupling, 0.0)
        applied_pitch_from_yaw_coupling = safe_float(pitch_from_yaw_coupling, 0.0)
        corrected_eye_yaw = (
            eye_delta_yaw * applied_yaw_coefficient
            + eye_delta_pitch * applied_yaw_from_pitch_coupling
        )
        corrected_eye_pitch = (
            eye_delta_pitch * applied_pitch_coefficient
            + eye_delta_yaw * applied_pitch_from_yaw_coupling
        )
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
        screen_scale_x=screen_scale_x,
        screen_scale_y=screen_scale_y,
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
        "applied_yaw_coefficient": applied_yaw_coefficient,
        "applied_pitch_coefficient": applied_pitch_coefficient,
        "applied_yaw_from_pitch_coupling": applied_yaw_from_pitch_coupling,
        "applied_pitch_from_yaw_coupling": applied_pitch_from_yaw_coupling,
        "corrected_screen_x": corrected_screen_x,
        "corrected_screen_y": corrected_screen_y,
        "screen_offset_x": screen_offset_x,
        "screen_offset_y": screen_offset_y,
        "screen_projection_t": screen_projection_t,
        "screen_depth": screen_depth,
        "screen_scale_x": _positive_or(screen_scale_x, 1.0),
        "screen_scale_y": _positive_or(screen_scale_y, 1.0),
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
    yaw_coefficient_positive: float = 1.0
    yaw_coefficient_negative: float = 1.0
    pitch_coefficient_positive: float = 1.0
    pitch_coefficient_negative: float = 1.0
    yaw_from_pitch_coupling: float = 0.0
    pitch_from_yaw_coupling: float = 0.0
    eye_yaw_min: float = -1.0
    eye_yaw_max: float = 1.0
    eye_pitch_min: float = -1.0
    eye_pitch_max: float = 1.0
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
    screen_scale_x: float = 1.0
    screen_scale_y: float = 1.0
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
            yaw_coefficient_positive=self.yaw_coefficient_positive,
            yaw_coefficient_negative=self.yaw_coefficient_negative,
            pitch_coefficient_positive=self.pitch_coefficient_positive,
            pitch_coefficient_negative=self.pitch_coefficient_negative,
            yaw_from_pitch_coupling=self.yaw_from_pitch_coupling,
            pitch_from_yaw_coupling=self.pitch_from_yaw_coupling,
            eye_yaw_min=self.eye_yaw_min,
            eye_yaw_max=self.eye_yaw_max,
            eye_pitch_min=self.eye_pitch_min,
            eye_pitch_max=self.eye_pitch_max,
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
            screen_scale_x=self.screen_scale_x,
            screen_scale_y=self.screen_scale_y,
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
