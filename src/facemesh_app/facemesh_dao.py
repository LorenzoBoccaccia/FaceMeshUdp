# facemesh_dao.py
"""
Data access and interpretation layer for the FaceMesh app.
Calibration is angle-based:
- centerYaw / centerPitch store the eye-in-head zero at screen center
- faceCenterYaw / faceCenterPitch store the head-pose zero at screen center
- the matrix maps eye deltas to monitor-plane eye-angle deltas
- runtime output compounds face delta + calibrated eye delta
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

import numpy as np


AVERAGE_IPD_MM = 60.0

HORIZONTAL_MAX_DEG = 40.0
DOWNWARD_MAX_DEG = 45.0
UPWARD_MAX_DEG = 30.0
GAZE_COORD_MAX_VALID = 1.25

DEFAULT_CENTER_ZETA = 1200.0
CALIBRATION_MODEL_VERSION = 2

LEFT_IRIS_CENTER_IDX = 468
RIGHT_IRIS_CENTER_IDX = 473
LEFT_IRIS_RING_IDXS = (469, 470, 471, 472)
RIGHT_IRIS_RING_IDXS = (474, 475, 476, 477)
LEFT_IRIS_IDXS = (LEFT_IRIS_CENTER_IDX,) + LEFT_IRIS_RING_IDXS
RIGHT_IRIS_IDXS = (RIGHT_IRIS_CENTER_IDX,) + RIGHT_IRIS_RING_IDXS

LEFT_EYE_INNER_IDX = 133
LEFT_EYE_OUTER_IDX = 33
LEFT_EYE_UPPER_IDX = 159
LEFT_EYE_LOWER_IDX = 145

RIGHT_EYE_INNER_IDX = 362
RIGHT_EYE_OUTER_IDX = 263
RIGHT_EYE_UPPER_IDX = 386
RIGHT_EYE_LOWER_IDX = 374

LEFT_EYE_KEY_IDXS = (
    LEFT_EYE_INNER_IDX,
    LEFT_EYE_OUTER_IDX,
    LEFT_EYE_UPPER_IDX,
    LEFT_EYE_LOWER_IDX,
)
RIGHT_EYE_KEY_IDXS = (
    RIGHT_EYE_INNER_IDX,
    RIGHT_EYE_OUTER_IDX,
    RIGHT_EYE_UPPER_IDX,
    RIGHT_EYE_LOWER_IDX,
)


@dataclass
class CalibrationMatrix:
    # Eye-in-head zero at center, in degrees.
    center_yaw: float = 0.0
    center_pitch: float = 0.0

    # Head-pose zero at center, in degrees.
    face_center_yaw: float = 0.0
    face_center_pitch: float = 0.0

    # Monitor-plane depth proxy at center. Same unit family as screen coords
    # once main.py scales the normalized zeta into display space.
    center_zeta: float = DEFAULT_CENTER_ZETA

    # Eye delta -> monitor-plane eye-angle delta.
    matrix_yaw_yaw: float = 1.0
    matrix_yaw_pitch: float = 0.0
    matrix_pitch_yaw: float = 0.0
    matrix_pitch_pitch: float = 1.0

    sample_count: int = 0
    timestamp_ms: int = 0


@dataclass
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

    # Filled in by main.py during calibration capture.
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    zeta: float = DEFAULT_CENTER_ZETA


def safe_float(v, fallback=0.0):
    try:
        f = float(v)
    except Exception:
        return fallback
    return f if math.isfinite(f) else fallback


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _positive_or(v: float, fallback: float) -> float:
    x = safe_float(v, fallback)
    return x if x > 1e-9 else fallback


def _screen_target_angles_deg(
    screen_x: float,
    screen_y: float,
    center_x: float,
    center_y: float,
    zeta: float,
) -> Tuple[float, float]:
    dx = safe_float(screen_x) - safe_float(center_x)
    dy = safe_float(screen_y) - safe_float(center_y)
    z = _positive_or(zeta, DEFAULT_CENTER_ZETA)
    yaw = math.degrees(math.atan2(dx, z))
    pitch = -math.degrees(math.atan2(dy, z))
    return yaw, pitch


def compute_calibration_matrix(points: List[CalibrationPoint]) -> CalibrationMatrix:
    if len(points) < 9:
        raise ValueError(f"Calibration requires at least 9 points, got {len(points)}")

    center_point = next((p for p in points if p.name == "C"), None)
    if center_point is None:
        raise ValueError("Calibration points must include a center point named 'C'")

    center_zeta = _positive_or(getattr(center_point, "zeta", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA)

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

        point_zeta = _positive_or(getattr(point, "zeta", center_zeta), center_zeta)
        zeta = (center_zeta + point_zeta) * 0.5

        target_total_yaw, target_total_pitch = _screen_target_angles_deg(
            point.screen_x,
            point.screen_y,
            center_point.screen_x,
            center_point.screen_y,
            zeta,
        )

        face_delta_yaw = safe_float(getattr(point, "head_yaw", 0.0)) - safe_float(getattr(center_point, "head_yaw", 0.0))
        face_delta_pitch = safe_float(getattr(point, "head_pitch", 0.0)) - safe_float(getattr(center_point, "head_pitch", 0.0))

        # Eye component only: total monitor-plane angle minus face contribution.
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
        b_yaw_array = np.array(b_yaw, dtype=float)
        b_pitch_array = np.array(b_pitch, dtype=float)

        try:
            coeffs_yaw, _, _, _ = np.linalg.lstsq(A, b_yaw_array, rcond=None)
            coeffs_pitch, _, _, _ = np.linalg.lstsq(A, b_pitch_array, rcond=None)

            matrix_yaw_yaw = float(coeffs_yaw[0])
            matrix_yaw_pitch = float(coeffs_yaw[1])
            matrix_pitch_yaw = float(coeffs_pitch[0])
            matrix_pitch_pitch = float(coeffs_pitch[1])
        except Exception:
            pass

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


def save_calibration(calib: CalibrationMatrix, points: List[CalibrationPoint], profile: str = "") -> Path:
    profile_token = _profile_token(profile)
    filename = "calibration.json" if profile_token == "default" else f"calibration-{profile_token}.json"

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
                "sampleCount": int(point.sample_count),
            }
            for point in points
        ],
    }

    file_path = Path(filename)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return file_path


def load_calibration(profile: str = "") -> Tuple[CalibrationMatrix, List[CalibrationPoint]]:
    profile_token = _profile_token(profile)
    filename = "calibration.json" if profile_token == "default" else f"calibration-{profile_token}.json"
    file_path = Path(filename)

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
        sample_count=0,
        timestamp_ms=int(time.time() * 1000),
    )

    if not file_path.exists():
        return empty, []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return empty, []

    calib_data = data.get("calibration", {})
    model_version = int(safe_float(calib_data.get("modelVersion", 1), 1))
    if model_version != CALIBRATION_MODEL_VERSION:
        return empty, []

    calib = CalibrationMatrix(
        center_yaw=safe_float(calib_data.get("centerYaw", 0.0)),
        center_pitch=safe_float(calib_data.get("centerPitch", 0.0)),
        face_center_yaw=safe_float(calib_data.get("faceCenterYaw", 0.0)),
        face_center_pitch=safe_float(calib_data.get("faceCenterPitch", 0.0)),
        center_zeta=_positive_or(calib_data.get("centerZeta", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA),
        matrix_yaw_yaw=safe_float(calib_data.get("matrixYawYaw", 1.0)),
        matrix_yaw_pitch=safe_float(calib_data.get("matrixYawPitch", 0.0)),
        matrix_pitch_yaw=safe_float(calib_data.get("matrixPitchYaw", 0.0)),
        matrix_pitch_pitch=safe_float(calib_data.get("matrixPitchPitch", 1.0)),
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
                zeta=_positive_or(point_data.get("zeta", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA),
            )
        )

    return calib, points


class FaceMeshEvent:
    def __init__(
        self,
        result: Any = None,
        *,
        face_index: int = 0,
        ts: Optional[int] = None,
        event_type: str = "mesh",
        calibration: Optional["CalibrationMatrix"] = None,
    ):
        self.result = result
        self.face_index = int(face_index)
        self.type = str(event_type)
        self.ts = int(ts if ts is not None else time.time() * 1000)
        self.calibration = calibration

    @classmethod
    def from_landmarker_result(
        cls,
        result: Any,
        *,
        face_index: int = 0,
        ts: Optional[int] = None,
        calibration: Optional["CalibrationMatrix"] = None,
    ):
        return cls(result, face_index=face_index, ts=ts, event_type="mesh", calibration=calibration)

    def _face_item(self, attr_name: str):
        if self.result is None:
            return None
        values = getattr(self.result, attr_name, None)
        if values is None:
            return None
        try:
            if len(values) <= self.face_index:
                return None
            return values[self.face_index]
        except TypeError:
            return values

    @staticmethod
    def _float_or_none(v) -> Optional[float]:
        f = safe_float(v, float("nan"))
        return f if math.isfinite(f) else None

    @staticmethod
    def _normalize_vec3(vec) -> Optional[List[float]]:
        if vec is None:
            return None
        try:
            x = float(vec[0])
            y = float(vec[1])
            z = float(vec[2])
        except Exception:
            return None
        mag = math.sqrt(x * x + y * y + z * z)
        if mag <= 1e-9:
            return None
        return [x / mag, y / mag, z / mag]

    @staticmethod
    def _dist2(a: List[float], b: List[float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _transform_flat_no_fallback(self_or_maybe_self=None) -> Optional[List[float]]:
        self = self_or_maybe_self
        m = self.transform_matrix
        if m is None:
            return None

        values: List[float] = []
        try:
            if hasattr(m, "flatten"):
                raw = m.flatten()
                for v in raw:
                    fv = self._float_or_none(v)
                    if fv is None:
                        return None
                    values.append(fv)
            else:
                for row in m:
                    if hasattr(row, "__iter__") and not isinstance(row, (str, bytes)):
                        for v in row:
                            fv = self._float_or_none(v)
                            if fv is None:
                                return None
                            values.append(fv)
                    else:
                        fv = self._float_or_none(row)
                        if fv is None:
                            return None
                        values.append(fv)
        except Exception:
            return None
        return values or None

    def _transform_m44(self) -> Optional[List[List[float]]]:
        flat = self._transform_flat_no_fallback(self)
        if flat is None or len(flat) < 16:
            return None
        return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]

    @property
    def has_face(self) -> bool:
        lms = self.landmarks
        if lms is None:
            return False
        try:
            return len(lms) > 0
        except Exception:
            return True

    @property
    def head_yaw(self) -> Optional[float]:
        m44 = self._transform_m44()
        if m44 is None:
            return None
        face_forward = self._normalize_vec3((-m44[0][2], -m44[1][2], -m44[2][2]))
        if face_forward is None:
            return None
        return -math.degrees(math.atan2(face_forward[0], -face_forward[2]))

    @property
    def head_pitch(self) -> Optional[float]:
        m44 = self._transform_m44()
        if m44 is None:
            return None
        face_forward = self._normalize_vec3((-m44[0][2], -m44[1][2], -m44[2][2]))
        if face_forward is None:
            return None
        return math.degrees(math.atan2(-face_forward[1], -face_forward[2]))

    @property
    def x(self) -> Optional[float]:
        m44 = self._transform_m44()
        if m44 is not None:
            return m44[0][3]
        flat = self._transform_flat_no_fallback(self)
        if flat is not None and len(flat) > 3:
            return flat[3]
        return None

    @property
    def y(self) -> Optional[float]:
        m44 = self._transform_m44()
        if m44 is not None:
            return m44[1][3]
        flat = self._transform_flat_no_fallback(self)
        if flat is not None and len(flat) > 7:
            return flat[7]
        return None

    @property
    def raw_transform_z(self) -> Optional[float]:
        m44 = self._transform_m44()
        if m44 is not None:
            return m44[2][3]
        flat = self._transform_flat_no_fallback(self)
        if flat is not None and len(flat) > 11:
            return flat[11]
        return None

    @property
    def zeta(self) -> Optional[float]:
        left = self.left_iris_center
        right = self.right_iris_center
        if left is None or right is None:
            return None

        projected_ipd = self._dist2(left, right)
        if projected_ipd <= 1e-9:
            return None

        yaw = abs(safe_float(self.head_yaw, 0.0))
        pitch = abs(safe_float(self.head_pitch, 0.0))
        foreshortening = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
        foreshortening = clamp(foreshortening, 0.25, 1.0)

        frontal_projected_ipd = projected_ipd / foreshortening
        if frontal_projected_ipd <= 1e-9:
            return None

        # Pseudo-depth in "mm per normalized image width".
        return AVERAGE_IPD_MM / frontal_projected_ipd

    @property
    def z(self) -> Optional[float]:
        # Intentionally swapped from raw transform Z to IPD-based depth proxy.
        return self.zeta

    @property
    def roll(self) -> Optional[float]:
        m44 = self._transform_m44()
        if m44 is not None:
            return math.degrees(math.atan2(m44[0][1], m44[0][0]))
        flat = self._transform_flat_no_fallback(self)
        if flat is not None and len(flat) > 1:
            return math.degrees(math.atan2(flat[1], flat[0]))
        return None

    @property
    def landmarks(self) -> Optional[List]:
        return self._face_item("face_landmarks")

    def landmark(self, idx: int):
        lms = self.landmarks
        if lms is None:
            return None
        try:
            if idx < 0 or idx >= len(lms):
                return None
            return lms[idx]
        except Exception:
            return None

    def _landmark_xyz(self, lm) -> Optional[List[float]]:
        if lm is None:
            return None
        if hasattr(lm, "x") and hasattr(lm, "y"):
            return [safe_float(lm.x), safe_float(lm.y), safe_float(getattr(lm, "z", 0.0))]
        if isinstance(lm, (list, tuple)) and len(lm) >= 3:
            return [safe_float(lm[0]), safe_float(lm[1]), safe_float(lm[2])]
        return None

    def landmark_xyz(self, idx: int) -> Optional[List[float]]:
        return self._landmark_xyz(self.landmark(idx))

    def _landmarks_xyz_by_indices(self, indices: tuple[int, ...]) -> List[List[float]]:
        points: List[List[float]] = []
        for idx in indices:
            p = self.landmark_xyz(int(idx))
            if p is not None:
                points.append(p)
        return points

    @staticmethod
    def _center_xyz(points: List[List[float]]) -> Optional[List[float]]:
        if not points:
            return None
        n = float(len(points))
        sx = sum(p[0] for p in points)
        sy = sum(p[1] for p in points)
        sz = sum(p[2] for p in points)
        return [sx / n, sy / n, sz / n]

    @property
    def left_iris_points(self) -> List[List[float]]:
        return self._landmarks_xyz_by_indices(LEFT_IRIS_IDXS)

    @property
    def right_iris_points(self) -> List[List[float]]:
        return self._landmarks_xyz_by_indices(RIGHT_IRIS_IDXS)

    @property
    def left_iris_ring_points(self) -> List[List[float]]:
        return self._landmarks_xyz_by_indices(LEFT_IRIS_RING_IDXS)

    @property
    def right_iris_ring_points(self) -> List[List[float]]:
        return self._landmarks_xyz_by_indices(RIGHT_IRIS_RING_IDXS)

    @property
    def left_iris_center(self) -> Optional[List[float]]:
        return self.landmark_xyz(LEFT_IRIS_CENTER_IDX)

    @property
    def right_iris_center(self) -> Optional[List[float]]:
        return self.landmark_xyz(RIGHT_IRIS_CENTER_IDX)

    @property
    def left_eye_key_points(self) -> List[List[float]]:
        return self._landmarks_xyz_by_indices(LEFT_EYE_KEY_IDXS)

    @property
    def right_eye_key_points(self) -> List[List[float]]:
        return self._landmarks_xyz_by_indices(RIGHT_EYE_KEY_IDXS)

    @staticmethod
    def _normalize_vec2(x: float, y: float) -> Optional[tuple[float, float]]:
        mag = math.hypot(x, y)
        if mag <= 1e-9:
            return None
        return x / mag, y / mag

    def _eye_gaze_raw_yaw_pitch(
        self,
        iris_center: Optional[List[float]],
        inner: Optional[List[float]],
        outer: Optional[List[float]],
        upper: Optional[List[float]],
        lower: Optional[List[float]],
    ) -> Optional[tuple[float, float]]:
        if iris_center is None or inner is None or outer is None or upper is None or lower is None:
            return None

        hx = outer[0] - inner[0]
        hy = outer[1] - inner[1]
        h_hat = self._normalize_vec2(hx, hy)
        if h_hat is None:
            return None
        hux, huy = h_hat

        vx = lower[0] - upper[0]
        vy = lower[1] - upper[1]
        v_dot_h = vx * hux + vy * huy
        vx_ortho = vx - v_dot_h * hux
        vy_ortho = vy - v_dot_h * huy

        v_hat = self._normalize_vec2(vx_ortho, vy_ortho)
        if v_hat is None:
            return None
        vux, vuy = v_hat

        lid_aperture = (lower[0] - upper[0]) * vux + (lower[1] - upper[1]) * vuy
        eye_width = math.hypot(hx, hy)
        if eye_width <= 1e-9 or lid_aperture <= (eye_width * 0.05):
            return None

        mid_x = (inner[0] + outer[0]) * 0.5
        mid_y = (inner[1] + outer[1]) * 0.5
        dx = iris_center[0] - mid_x
        dy = iris_center[1] - mid_y

        horizontal_coord = (dx * hux + dy * huy) / (eye_width * 0.5)
        vertical_coord = -(dx * vux + dy * vuy) / (lid_aperture * 0.5)
        if abs(horizontal_coord) > GAZE_COORD_MAX_VALID or abs(vertical_coord) > GAZE_COORD_MAX_VALID:
            return None

        horizontal_coord = clamp(horizontal_coord, -1.0, 1.0)
        vertical_coord = clamp(vertical_coord, -1.0, 1.0)

        yaw = horizontal_coord * HORIZONTAL_MAX_DEG
        pitch = vertical_coord * (UPWARD_MAX_DEG if vertical_coord >= 0.0 else DOWNWARD_MAX_DEG)

        return yaw, pitch

    @property
    def left_eye_gaze_yaw(self) -> Optional[float]:
        yp = self._eye_gaze_raw_yaw_pitch(
            self.left_iris_center,
            self.landmark_xyz(LEFT_EYE_INNER_IDX),
            self.landmark_xyz(LEFT_EYE_OUTER_IDX),
            self.landmark_xyz(LEFT_EYE_UPPER_IDX),
            self.landmark_xyz(LEFT_EYE_LOWER_IDX),
        )
        return -yp[0] if yp is not None else None

    @property
    def right_eye_gaze_yaw(self) -> Optional[float]:
        yp = self._eye_gaze_raw_yaw_pitch(
            self.right_iris_center,
            self.landmark_xyz(RIGHT_EYE_INNER_IDX),
            self.landmark_xyz(RIGHT_EYE_OUTER_IDX),
            self.landmark_xyz(RIGHT_EYE_UPPER_IDX),
            self.landmark_xyz(RIGHT_EYE_LOWER_IDX),
        )
        return yp[0] if yp is not None else None

    @property
    def left_eye_gaze_pitch(self) -> Optional[float]:
        yp = self._eye_gaze_raw_yaw_pitch(
            self.left_iris_center,
            self.landmark_xyz(LEFT_EYE_INNER_IDX),
            self.landmark_xyz(LEFT_EYE_OUTER_IDX),
            self.landmark_xyz(LEFT_EYE_UPPER_IDX),
            self.landmark_xyz(LEFT_EYE_LOWER_IDX),
        )
        return yp[1] if yp is not None else None

    @property
    def right_eye_gaze_pitch(self) -> Optional[float]:
        yp = self._eye_gaze_raw_yaw_pitch(
            self.right_iris_center,
            self.landmark_xyz(RIGHT_EYE_INNER_IDX),
            self.landmark_xyz(RIGHT_EYE_OUTER_IDX),
            self.landmark_xyz(RIGHT_EYE_UPPER_IDX),
            self.landmark_xyz(RIGHT_EYE_LOWER_IDX),
        )
        return yp[1] if yp is not None else None

    @property
    def combined_eye_gaze_yaw(self) -> Optional[float]:
        left_yaw = self.left_eye_gaze_yaw
        right_yaw = self.right_eye_gaze_yaw
        if left_yaw is None or right_yaw is None:
            return None
        return (left_yaw + right_yaw) / 2.0

    @property
    def combined_eye_gaze_pitch(self) -> Optional[float]:
        left_pitch = self.left_eye_gaze_pitch
        right_pitch = self.right_eye_gaze_pitch
        if left_pitch is None or right_pitch is None:
            return None
        return (left_pitch + right_pitch) / 2.0

    def _apply_calibration(self, raw_yaw: float, raw_pitch: float, calib: CalibrationMatrix) -> tuple[float, float]:
        if calib.sample_count <= 0:
            return raw_yaw, raw_pitch

        eye_dyaw = raw_yaw - calib.center_yaw
        eye_dpitch = raw_pitch - calib.center_pitch

        eye_plane_yaw = calib.matrix_yaw_yaw * eye_dyaw + calib.matrix_yaw_pitch * eye_dpitch
        eye_plane_pitch = calib.matrix_pitch_yaw * eye_dyaw + calib.matrix_pitch_pitch * eye_dpitch

        face_delta_yaw = safe_float(self.head_yaw, calib.face_center_yaw) - calib.face_center_yaw
        face_delta_pitch = safe_float(self.head_pitch, calib.face_center_pitch) - calib.face_center_pitch

        return face_delta_yaw + eye_plane_yaw, face_delta_pitch + eye_plane_pitch

    @property
    def calibrated_left_eye_gaze_yaw(self) -> Optional[float]:
        raw_yaw = self.left_eye_gaze_yaw
        if raw_yaw is None:
            return None
        raw_pitch = self.left_eye_gaze_pitch or 0.0
        calib = self.calibration if self.calibration is not None else CalibrationMatrix()
        calibrated_yaw, _ = self._apply_calibration(raw_yaw, raw_pitch, calib)
        return calibrated_yaw

    @property
    def calibrated_right_eye_gaze_yaw(self) -> Optional[float]:
        raw_yaw = self.right_eye_gaze_yaw
        if raw_yaw is None:
            return None
        raw_pitch = self.right_eye_gaze_pitch or 0.0
        calib = self.calibration if self.calibration is not None else CalibrationMatrix()
        calibrated_yaw, _ = self._apply_calibration(raw_yaw, raw_pitch, calib)
        return calibrated_yaw

    @property
    def calibrated_left_eye_gaze_pitch(self) -> Optional[float]:
        raw_yaw = self.left_eye_gaze_yaw or 0.0
        raw_pitch = self.left_eye_gaze_pitch
        if raw_pitch is None:
            return None
        calib = self.calibration if self.calibration is not None else CalibrationMatrix()
        _, calibrated_pitch = self._apply_calibration(raw_yaw, raw_pitch, calib)
        return calibrated_pitch

    @property
    def calibrated_right_eye_gaze_pitch(self) -> Optional[float]:
        raw_yaw = self.right_eye_gaze_yaw or 0.0
        raw_pitch = self.right_eye_gaze_pitch
        if raw_pitch is None:
            return None
        calib = self.calibration if self.calibration is not None else CalibrationMatrix()
        _, calibrated_pitch = self._apply_calibration(raw_yaw, raw_pitch, calib)
        return calibrated_pitch

    @property
    def calibrated_combined_eye_gaze_yaw(self) -> Optional[float]:
        left_yaw = self.calibrated_left_eye_gaze_yaw
        right_yaw = self.calibrated_right_eye_gaze_yaw
        if left_yaw is None or right_yaw is None:
            return None
        return (left_yaw + right_yaw) / 2.0

    @property
    def calibrated_combined_eye_gaze_pitch(self) -> Optional[float]:
        left_pitch = self.calibrated_left_eye_gaze_pitch
        right_pitch = self.calibrated_right_eye_gaze_pitch
        if left_pitch is None or right_pitch is None:
            return None
        return (left_pitch + right_pitch) / 2.0

    @property
    def blendshapes(self) -> Optional[Dict]:
        return self._face_item("face_blendshapes")

    @property
    def transform_matrix(self) -> Optional[List]:
        return self._face_item("facial_transformation_matrixes")

    @property
    def face_mask_segment(self):
        for key in (
            "face_mask_segments",
            "face_mask_segment",
            "face_masks",
            "face_mask",
            "segmentation_masks",
            "segmentation_mask",
        ):
            value = self._face_item(key)
            if value is not None:
                return value
        return None

    @property
    def landmark_count(self) -> int:
        lms = self.landmarks
        if lms is None:
            return 0
        try:
            return len(lms)
        except Exception:
            return 0

    def landmarks_as_list(self) -> Optional[List[List[float]]]:
        lms = self.landmarks
        if not lms:
            return None
        out: List[List[float]] = []
        for lm in lms:
            xyz = self._landmark_xyz(lm)
            if xyz is not None:
                out.append(xyz)
        return out or None

    def blendshapes_as_dict(self) -> Optional[Dict[str, float]]:
        cats = self.blendshapes
        if not cats:
            return None
        if isinstance(cats, dict):
            return {str(k): safe_float(v) for k, v in cats.items()}
        out: Dict[str, float] = {}
        for cat in cats:
            name = getattr(cat, "category_name", None)
            score = getattr(cat, "score", None)
            if name is not None and score is not None:
                out[str(name)] = safe_float(score)
        return out or None

    def transform_matrix_as_flat(self) -> Optional[List[float]]:
        m = self.transform_matrix
        if m is None:
            return None
        if hasattr(m, "flatten"):
            try:
                return [safe_float(v) for v in m.flatten()]
            except Exception:
                pass
        flat: List[float] = []
        try:
            for row in m:
                if hasattr(row, "__iter__") and not isinstance(row, (str, bytes)):
                    for val in row:
                        flat.append(safe_float(val))
                else:
                    flat.append(safe_float(row))
        except Exception:
            return None
        return flat or None

    def face_mask_segment_meta(self) -> Optional[Dict[str, Any]]:
        seg = self.face_mask_segment
        if seg is None:
            return None
        meta: Dict[str, Any] = {"type": type(seg).__name__}
        shape = getattr(seg, "shape", None)
        if shape is not None:
            try:
                meta["shape"] = [int(v) for v in shape]
            except Exception:
                meta["shape"] = str(shape)
        dtype = getattr(seg, "dtype", None)
        if dtype is not None:
            meta["dtype"] = str(dtype)
        return meta

    def eyes_dict(self) -> Dict[str, Any]:
        return {
            "leftIrisCenterIndex": LEFT_IRIS_CENTER_IDX,
            "rightIrisCenterIndex": RIGHT_IRIS_CENTER_IDX,
            "leftIrisRingIndices": list(LEFT_IRIS_RING_IDXS),
            "rightIrisRingIndices": list(RIGHT_IRIS_RING_IDXS),
            "leftIrisIndices": list(LEFT_IRIS_IDXS),
            "rightIrisIndices": list(RIGHT_IRIS_IDXS),
            "leftEyeKeyIndices": list(LEFT_EYE_KEY_IDXS),
            "rightEyeKeyIndices": list(RIGHT_EYE_KEY_IDXS),
            "leftIrisPoints": self.left_iris_points,
            "rightIrisPoints": self.right_iris_points,
            "leftIrisRingPoints": self.left_iris_ring_points,
            "rightIrisRingPoints": self.right_iris_ring_points,
            "leftIrisCenter": self.left_iris_center,
            "rightIrisCenter": self.right_iris_center,
            "leftEyeGazeYaw": self.left_eye_gaze_yaw,
            "rightEyeGazeYaw": self.right_eye_gaze_yaw,
            "leftEyeGazePitch": self.left_eye_gaze_pitch,
            "rightEyeGazePitch": self.right_eye_gaze_pitch,
            "combinedEyeGazeYaw": self.combined_eye_gaze_yaw,
            "combinedEyeGazePitch": self.combined_eye_gaze_pitch,
            "calibratedCombinedEyeGazeYaw": self.calibrated_combined_eye_gaze_yaw,
            "calibratedCombinedEyeGazePitch": self.calibrated_combined_eye_gaze_pitch,
            "leftEyeKeyPoints": self.left_eye_key_points,
            "rightEyeKeyPoints": self.right_eye_key_points,
            "zeta": self.zeta,
        }

    def to_overlay_dict(self) -> Dict:
        return {
            "type": self.type,
            "hasFace": self.has_face,
            "landmarkCount": self.landmark_count,
            "ts": self.ts,
            "zeta": self.zeta,
        }

    def to_capture_dict(self) -> Dict:
        return {
            "landmarks": self.landmarks_as_list(),
            "blendshapes": self.blendshapes_as_dict(),
            "transformMatrix": self.transform_matrix_as_flat(),
            "faceMaskSegment": self.face_mask_segment_meta(),
            "eyes": self.eyes_dict(),
        }

    def to_capture_dump(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "ts": self.ts,
            "hasFace": self.has_face,
            "landmarkCount": self.landmark_count,
            "pose": {
                "yaw": self.head_yaw,
                "pitch": self.head_pitch,
                "roll": self.roll,
            },
            "translation": {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "rawTransformZ": self.raw_transform_z,
            },
            "meshData": self.to_capture_dict(),
        }

    def to_dict(self) -> Dict:
        return self.to_overlay_dict()


@dataclass
class CalibratedFaceAndGazeEvent:
    """Combined event that merges face mesh data with calibration and display geometry.
    
    This event combines:
    - Face mesh data (from FaceMeshEvent)
    - Calibration data (pitch/yaw/roll calibration values)
    - Display geometry (screen dimensions, origin offset)
    
    This is the primary data structure for the refactored gaze pipeline,
    providing all necessary data in a single, well-typed object.
    """
    # Face mesh data (from FaceMeshEvent)
    face_mesh_event: FaceMeshEvent
    
    # Calibration data
    pitch_calibration: float  # Pitch calibration value in degrees
    yaw_calibration: float    # Yaw calibration value in degrees
    roll_calibration: float   # Roll calibration value in degrees
    
    # Display geometry
    display_width: int        # Display width in pixels
    display_height: int       # Display height in pixels
    origin_x: float           # X coordinate of origin (e.g., screen center)
    origin_y: float           # Y coordinate of origin
