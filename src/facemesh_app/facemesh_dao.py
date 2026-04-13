"""
Data Access Object for FaceMesh application.
Provides basic data structures and utility functions for face mesh data capture.
"""

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
import numpy as np


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
    """Stores calibration matrix coefficients for gaze tracking.
    
    The calibration matrix maps raw eye gaze angles to calibrated screen coordinates
    using a 2x2 transformation matrix with center offset compensation.
    
    Default values create an identity transformation (no change to raw values).
    
    Attributes:
        center_yaw: Eye yaw angle when looking at screen center (radians)
        center_pitch: Eye pitch angle when looking at screen center (radians)
        matrix_yaw_yaw: Scaling coefficient for yaw-to-yaw transformation
        matrix_yaw_pitch: Cross-talk coefficient for pitch-to-yaw transformation
        matrix_pitch_yaw: Cross-talk coefficient for yaw-to-pitch transformation
        matrix_pitch_pitch: Scaling coefficient for pitch-to-pitch transformation
        sample_count: Total number of samples used in calibration
        timestamp_ms: Timestamp when calibration was created (milliseconds since epoch)
    """
    center_yaw: float = 0.0
    center_pitch: float = 0.0
    matrix_yaw_yaw: float = 1.0
    matrix_yaw_pitch: float = 0.0
    matrix_pitch_yaw: float = 0.0
    matrix_pitch_pitch: float = 1.0
    sample_count: int = 0
    timestamp_ms: int = 0


@dataclass
class CalibrationPoint:
    """Stores calibration data for a single screen position.
    
    Each calibration point represents the average eye measurements when the user
    was looking at a specific target position on the screen.
    
    Attributes:
        name: Point identifier ("C", "TL", "TC", "TR", "R", "BR", "BC", "BL", "L")
        screen_x: Target screen X position (normalized 0-1 or pixels)
        screen_y: Target screen Y position (normalized 0-1 or pixels)
        raw_eye_yaw: Average combined eye yaw during sampling (radians)
        raw_eye_pitch: Average combined eye pitch during sampling (radians)
        raw_left_eye_yaw: Left eye specific yaw measurement (radians)
        raw_left_eye_pitch: Left eye specific pitch measurement (radians)
        raw_right_eye_yaw: Right eye specific yaw measurement (radians)
        raw_right_eye_pitch: Right eye specific pitch measurement (radians)
        sample_count: Number of samples averaged at this point
    """
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


def compute_calibration_matrix(points: List[CalibrationPoint]) -> CalibrationMatrix:
    """Compute calibration matrix from a set of calibration points.
    
    This function computes a 2x2 transformation matrix that maps raw eye gaze angles
    to calibrated screen coordinates. It uses least squares regression to find the
    optimal coefficients that minimize the error between predicted and actual screen
    positions across all calibration points.
    
    The calibration requires exactly 9 points: one center point ("C") and 8 edge points
    arranged around it. The center point provides the baseline (center_yaw, center_pitch),
    while the edge points are used to compute the transformation matrix coefficients.
    
    The transformation model is:
        screen_x_dev = matrix_yaw_yaw * raw_yaw_dev + matrix_yaw_pitch * raw_pitch_dev
        screen_y_dev = matrix_pitch_yaw * raw_yaw_dev + matrix_pitch_pitch * raw_pitch_dev
    
    where "dev" denotes deviation from the center point values.
    
    Args:
        points: List of CalibrationPoint objects containing calibration data.
                Must contain exactly 9 points including a center point named "C".
                
    Returns:
        CalibrationMatrix containing the computed transformation coefficients.
        
    Raises:
        ValueError: If fewer than 9 points are provided or no center point exists.
        
    Notes:
        - If least squares computation fails, returns identity matrix values
          (1.0 for diagonals, 0.0 for off-diagonals).
        - The function is robust to numerical issues and falls back gracefully
          when computation cannot be completed successfully.
    """
    # Validate input
    if len(points) < 9:
        raise ValueError(f"Calibration requires at least 9 points, got {len(points)}")
    
    # Find center point
    center_point = None
    for point in points:
        if point.name == "C":
            center_point = point
            break
    
    if center_point is None:
        raise ValueError("Calibration points must include a center point named 'C'")
    
    # Extract edge points (all points except center)
    edge_points = [p for p in points if p.name != "C"]
    
    # Initialize with identity matrix values in case of failure
    matrix_yaw_yaw = 1.0
    matrix_yaw_pitch = 0.0
    matrix_pitch_yaw = 0.0
    matrix_pitch_pitch = 1.0
    
    try:
        # Build system of equations for least squares
        # We have 8 equations (one per edge point) with 4 unknown coefficients
        # Each edge point gives us 2 equations (one for screen_x, one for screen_y)
        
        A_rows = []  # Matrix of raw gaze deviations
        b_yaw = []   # Screen X deviations (target for yaw equation)
        b_pitch = [] # Screen Y deviations (target for pitch equation)
        
        for point in edge_points:
            # Compute deviations from center in raw gaze space
            raw_yaw_dev = point.raw_eye_yaw - center_point.raw_eye_yaw
            raw_pitch_dev = point.raw_eye_pitch - center_point.raw_eye_pitch
            
            # Compute deviations from center in screen space
            screen_x_dev = point.screen_x - center_point.screen_x
            screen_y_dev = point.screen_y - center_point.screen_y
            
            # Add to system: A * coeffs = b
            # For yaw equation: screen_x_dev = c00 * raw_yaw_dev + c01 * raw_pitch_dev
            # For pitch equation: screen_y_dev = c10 * raw_yaw_dev + c11 * raw_pitch_dev
            
            A_rows.append([raw_yaw_dev, raw_pitch_dev])
            b_yaw.append(screen_x_dev)
            b_pitch.append(screen_y_dev)
        
        # Convert to numpy arrays
        A = np.array(A_rows)
        b_yaw_array = np.array(b_yaw)
        b_pitch_array = np.array(b_pitch)
        
        # Solve for coefficients using least squares
        # A * [c00, c01]^T = b_yaw  -> gives first row of matrix
        # A * [c10, c11]^T = b_pitch -> gives second row of matrix
        
        try:
            coeffs_yaw, _, _, _ = np.linalg.lstsq(A, b_yaw_array, rcond=None)
            coeffs_pitch, _, _, _ = np.linalg.lstsq(A, b_pitch_array, rcond=None)
            
            # Extract coefficients
            matrix_yaw_yaw = float(coeffs_yaw[0])
            matrix_yaw_pitch = float(coeffs_yaw[1])
            matrix_pitch_yaw = float(coeffs_pitch[0])
            matrix_pitch_pitch = float(coeffs_pitch[1])
            
        except Exception:
            # If least squares fails, keep identity matrix values
            pass
            
    except Exception:
        # If any computation fails, keep identity matrix values
        pass
    
    # Compute total sample count
    total_sample_count = sum(p.sample_count for p in points)
    
    # Create and return calibration matrix
    return CalibrationMatrix(
        center_yaw=center_point.raw_eye_yaw,
        center_pitch=center_point.raw_eye_pitch,
        matrix_yaw_yaw=matrix_yaw_yaw,
        matrix_yaw_pitch=matrix_yaw_pitch,
        matrix_pitch_yaw=matrix_pitch_yaw,
        matrix_pitch_pitch=matrix_pitch_pitch,
        sample_count=total_sample_count,
        timestamp_ms=int(time.time() * 1000)
    )


def _profile_token(raw_profile: str) -> str:
    """Sanitize profile name for safe filename usage.
    
    Converts a profile name into a safe token that can be used in filenames
    by replacing non-alphanumeric characters with hyphens and stripping
    leading/trailing punctuation characters.
    
    The sanitization rules are:
    - Replace non-alphanumeric characters (except ., _, -) with hyphens
    - Strip leading/trailing ., _, - characters
    - Return "default" if the result is empty
    
    Args:
        raw_profile: The raw profile name to sanitize.
        
    Returns:
        A sanitized profile name safe for use in filenames, or "default"
        if sanitization results in an empty string.
        
    Examples:
        >>> _profile_token("")
        "default"
        >>> _profile_token("my-profile")
        "my-profile"
        >>> _profile_token("my profile!")
        "my-profile-"
        >>> _profile_token("  test  ")
        "--test--"
        >>> _profile_token("...test...")
        "test"
    """
    if not raw_profile:
        return "default"
    
    # Replace non-alphanumeric characters (except ., _, -) with hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '-', raw_profile)
    
    # Strip leading/trailing ., _, - characters
    sanitized = sanitized.strip('._-')
    
    # Return "default" if empty after sanitization
    return sanitized if sanitized else "default"


def save_calibration(calib: CalibrationMatrix, points: List[CalibrationPoint], profile: str = "") -> Path:
    """Save calibration data to a JSON file.
    
    Serializes calibration matrix and calibration points to a JSON file
    for persistent storage. The filename is generated based on the profile
    name, allowing multiple calibration profiles to be stored separately.
    
    The file is saved with UTF-8 encoding and 2-space indentation for
    human readability. The JSON structure includes:
    - timestamp: Unix timestamp in milliseconds when calibration was saved
    - profile: The profile name (sanitized)
    - calibration: The calibration matrix coefficients
    - points: List of calibration points with all measurements
    
    Args:
        calib: The CalibrationMatrix object containing calibration coefficients.
        points: List of CalibrationPoint objects with calibration measurements.
        profile: Optional profile name for identifying the calibration.
                If empty, filename will be "calibration.json".
                Otherwise, filename will be "calibration-{profile}.json".
        
    Returns:
        Path object representing the saved file.
        
    Raises:
        IOError: If the file cannot be written.
        TypeError: If serialization fails (e.g., invalid types).
        
    Examples:
        >>> calib = CalibrationMatrix(...)
        >>> points = [CalibrationPoint(...), ...]
        >>> path = save_calibration(calib, points, "test-profile")
        >>> print(path)
        calibration-test-profile.json
    """
    # Sanitize profile name
    profile_token = _profile_token(profile)
    
    # Generate filename
    if profile_token == "default":
        filename = "calibration.json"
    else:
        filename = f"calibration-{profile_token}.json"
    
    # Build JSON payload
    payload = {
        "timestamp": int(time.time() * 1000),
        "profile": profile_token,
        "calibration": {
            "centerYaw": float(calib.center_yaw),
            "centerPitch": float(calib.center_pitch),
            "matrixYawYaw": float(calib.matrix_yaw_yaw),
            "matrixYawPitch": float(calib.matrix_yaw_pitch),
            "matrixPitchYaw": float(calib.matrix_pitch_yaw),
            "matrixPitchPitch": float(calib.matrix_pitch_pitch),
            "sampleCount": int(calib.sample_count)
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
                "sampleCount": int(point.sample_count)
            }
            for point in points
        ]
    }
    
    # Write to file
    file_path = Path(filename)
    with file_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    
    return file_path


def load_calibration(profile: str = "") -> Tuple[CalibrationMatrix, List[CalibrationPoint]]:
    """Load calibration data from a JSON file.
    
    Loads and deserializes calibration matrix and calibration points from
    a JSON file previously saved by save_calibration. The function handles
    missing files and missing fields gracefully with sensible defaults.
    
    If the calibration file doesn't exist, returns a default empty calibration
    matrix and empty points list. This allows the application to start without
    requiring pre-existing calibration data.
    
    The function is tolerant to missing or malformed fields, using default
    values (0.0 for floats, 0 for integers, empty strings for strings) when
    needed. This ensures robustness when loading calibration files from
    different versions or with partial data.
    
    Args:
        profile: Optional profile name for identifying the calibration.
                Must match the profile name used when saving.
                If empty, looks for "calibration.json".
                Otherwise, looks for "calibration-{profile}.json".
        
    Returns:
        Tuple of (CalibrationMatrix, List[CalibrationPoint]).
        If file doesn't exist, returns (CalibrationMatrix(), []).
        
    Examples:
        >>> calib, points = load_calibration("test-profile")
        >>> if not calib.sample_count:
        ...     print("No calibration data found")
        >>> else:
        ...     print(f"Loaded {len(points)} calibration points")
    """
    # Sanitize profile name
    profile_token = _profile_token(profile)
    
    # Generate filename
    if profile_token == "default":
        filename = "calibration.json"
    else:
        filename = f"calibration-{profile_token}.json"
    
    file_path = Path(filename)
    
    # Return empty calibration if file doesn't exist
    if not file_path.exists():
        return CalibrationMatrix(
            center_yaw=0.0,
            center_pitch=0.0,
            matrix_yaw_yaw=1.0,
            matrix_yaw_pitch=0.0,
            matrix_pitch_yaw=0.0,
            matrix_pitch_pitch=1.0,
            sample_count=0,
            timestamp_ms=int(time.time() * 1000)
        ), []
    
    # Load and parse JSON
    try:
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        # Return empty calibration if file is invalid
        return CalibrationMatrix(
            center_yaw=0.0,
            center_pitch=0.0,
            matrix_yaw_yaw=1.0,
            matrix_yaw_pitch=0.0,
            matrix_pitch_yaw=0.0,
            matrix_pitch_pitch=1.0,
            sample_count=0,
            timestamp_ms=int(time.time() * 1000)
        ), []
    
    # Extract calibration data with defaults for missing fields
    calib_data = data.get("calibration", {})
    calib = CalibrationMatrix(
        center_yaw=safe_float(calib_data.get("centerYaw", 0.0)),
        center_pitch=safe_float(calib_data.get("centerPitch", 0.0)),
        matrix_yaw_yaw=safe_float(calib_data.get("matrixYawYaw", 1.0)),
        matrix_yaw_pitch=safe_float(calib_data.get("matrixYawPitch", 0.0)),
        matrix_pitch_yaw=safe_float(calib_data.get("matrixPitchYaw", 0.0)),
        matrix_pitch_pitch=safe_float(calib_data.get("matrixPitchPitch", 1.0)),
        sample_count=int(calib_data.get("sampleCount", 0)),
        timestamp_ms=int(data.get("timestamp", time.time() * 1000))
    )
    
    # Extract calibration points with defaults for missing fields
    points_data = data.get("points", [])
    points = []
    for point_data in points_data:
        point = CalibrationPoint(
            name=str(point_data.get("name", "")),
            screen_x=safe_float(point_data.get("screenX", 0.0)),
            screen_y=safe_float(point_data.get("screenY", 0.0)),
            raw_eye_yaw=safe_float(point_data.get("rawEyeYaw", 0.0)),
            raw_eye_pitch=safe_float(point_data.get("rawEyePitch", 0.0)),
            raw_left_eye_yaw=safe_float(point_data.get("rawLeftEyeYaw", 0.0)),
            raw_left_eye_pitch=safe_float(point_data.get("rawLeftEyePitch", 0.0)),
            raw_right_eye_yaw=safe_float(point_data.get("rawRightEyeYaw", 0.0)),
            raw_right_eye_pitch=safe_float(point_data.get("rawRightEyePitch", 0.0)),
            sample_count=int(point_data.get("sampleCount", 0))
        )
        points.append(point)
    
    return calib, points


def safe_float(v, fallback=0.0):
    """Safely convert value to float with fallback."""
    try:
        f = float(v)
    except Exception:
        return fallback
    return f if math.isfinite(f) else fallback


def clamp(v, lo, hi):
    """Clamp value between lo and hi."""
    return max(lo, min(hi, v))


class FaceMeshEvent:
    """Wraps a raw MediaPipe FaceLandmarker result for one face."""

    def __init__(self, result: Any = None, *, face_index: int = 0, ts: Optional[int] = None, event_type: str = "mesh", calibration: Optional['CalibrationMatrix'] = None):
        self.result = result
        self.face_index = int(face_index)
        self.type = str(event_type)
        self.ts = int(ts if ts is not None else time.time() * 1000)
        self.calibration = calibration

    @classmethod
    def from_landmarker_result(cls, result: Any, *, face_index: int = 0, ts: Optional[int] = None, calibration: Optional['CalibrationMatrix'] = None):
        """Build event directly from MediaPipe FaceLandmarker result."""
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
        # kept signature-compatible with internal callers
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
        """Check if face is detected."""
        lms = self.landmarks
        if lms is None:
            return False
        try:
            return len(lms) > 0
        except Exception:
            return True

    @property
    def head_yaw(self) -> Optional[float]:
        """Get head yaw angle."""
        m44 = self._transform_m44()
        if m44 is None:
            return None
        face_forward = self._normalize_vec3((-m44[0][2], -m44[1][2], -m44[2][2]))
        if face_forward is None:
            return None
        return math.degrees(math.atan2(face_forward[0], -face_forward[2]))

    @property
    def head_pitch(self) -> Optional[float]:
        """Get head pitch angle."""
        m44 = self._transform_m44()
        if m44 is None:
            return None
        face_forward = self._normalize_vec3((-m44[0][2], -m44[1][2], -m44[2][2]))
        if face_forward is None:
            return None
        return math.degrees(math.atan2(-face_forward[1], -face_forward[2]))

    @property
    def x(self) -> Optional[float]:
        """Get x translation."""
        m44 = self._transform_m44()
        if m44 is not None:
            return m44[0][3]
        flat = self._transform_flat_no_fallback(self)
        if flat is not None and len(flat) > 3:
            return flat[3]
        return None

    @property
    def y(self) -> Optional[float]:
        """Get y translation."""
        m44 = self._transform_m44()
        if m44 is not None:
            return m44[1][3]
        flat = self._transform_flat_no_fallback(self)
        if flat is not None and len(flat) > 7:
            return flat[7]
        return None

    @property
    def z(self) -> Optional[float]:
        """Get z translation."""
        m44 = self._transform_m44()
        if m44 is not None:
            return m44[2][3]
        flat = self._transform_flat_no_fallback(self)
        if flat is not None and len(flat) > 11:
            return flat[11]
        return None

    @property
    def roll(self) -> Optional[float]:
        """Get roll angle."""
        m44 = self._transform_m44()
        if m44 is not None:
            return math.degrees(math.atan2(m44[0][1], m44[0][0]))
        flat = self._transform_flat_no_fallback(self)
        if flat is not None and len(flat) > 1:
            return math.degrees(math.atan2(flat[1], flat[0]))
        return None

    @property
    def landmarks(self) -> Optional[List]:
        """Get raw face landmarks object from MediaPipe result."""
        return self._face_item("face_landmarks")

    def landmark(self, idx: int):
        """Get raw landmark object at index."""
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
        """Get landmark coordinates [x,y,z] at index."""
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
        """Get raw left iris points [x,y,z] (center + ring)."""
        return self._landmarks_xyz_by_indices(LEFT_IRIS_IDXS)

    @property
    def right_iris_points(self) -> List[List[float]]:
        """Get raw right iris points [x,y,z] (center + ring)."""
        return self._landmarks_xyz_by_indices(RIGHT_IRIS_IDXS)

    @property
    def left_iris_ring_points(self) -> List[List[float]]:
        """Get raw left iris ring points [x,y,z]."""
        return self._landmarks_xyz_by_indices(LEFT_IRIS_RING_IDXS)

    @property
    def right_iris_ring_points(self) -> List[List[float]]:
        """Get raw right iris ring points [x,y,z]."""
        return self._landmarks_xyz_by_indices(RIGHT_IRIS_RING_IDXS)

    @property
    def left_iris_center(self) -> Optional[List[float]]:
        """Get left iris center [x,y,z] from center index."""
        return self.landmark_xyz(LEFT_IRIS_CENTER_IDX)

    @property
    def right_iris_center(self) -> Optional[List[float]]:
        """Get right iris center [x,y,z] from center index."""
        return self.landmark_xyz(RIGHT_IRIS_CENTER_IDX)

    @property
    def left_eye_key_points(self) -> List[List[float]]:
        """Get left eye key landmarks [x,y,z]."""
        return self._landmarks_xyz_by_indices(LEFT_EYE_KEY_IDXS)

    @property
    def right_eye_key_points(self) -> List[List[float]]:
        """Get right eye key landmarks [x,y,z]."""
        return self._landmarks_xyz_by_indices(RIGHT_EYE_KEY_IDXS)

    @staticmethod
    def _safe_div(num: float, den: float) -> Optional[float]:
        if not math.isfinite(num) or not math.isfinite(den):
            return None
        if abs(den) <= 1e-9:
            return None
        return num / den

    @staticmethod
    def _image_ordered_eye_refs(
        inner: List[float],
        outer: List[float],
        upper: List[float],
        lower: List[float],
    ) -> Dict[str, List[float]]:
        left_corner = inner if inner[0] <= outer[0] else outer
        right_corner = outer if inner[0] <= outer[0] else inner
        top_lid = upper if upper[1] <= lower[1] else lower
        bottom_lid = lower if upper[1] <= lower[1] else upper
        return {
            "leftCorner": left_corner,
            "rightCorner": right_corner,
            "topLid": top_lid,
            "bottomLid": bottom_lid,
        }

    @staticmethod
    def _proxy_to_angle_deg(v: Optional[float], *, invert: bool = False) -> Optional[float]:
        if v is None:
            return None
        x = -v if invert else v
        if not math.isfinite(x):
            return None
        return math.degrees(math.atan(x))

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

        width2d = self._dist2(inner, outer)
        height2d = self._dist2(upper, lower)
        if width2d <= 1e-9 or height2d <= 1e-9:
            return None

        refs = self._image_ordered_eye_refs(inner, outer, upper, lower)

        # --- YAW ---
        # Use only the canthi midpoint horizontally.
        # This avoids contaminating yaw with upper/lower lid motion.
        x_mid = 0.5 * (refs["leftCorner"][0] + refs["rightCorner"][0])
        center_x = self._safe_div(iris_center[0] - x_mid, width2d)
        if center_x is None:
            return None

        # Optional weak sclera cue. Keep small so calibration can do the rest.
        d_left = iris_center[0] - refs["leftCorner"][0]
        d_right = refs["rightCorner"][0] - iris_center[0]
        h_asym = None
        h_sum = d_left + d_right
        if h_sum > 1e-9:
            h_asym = (d_left - d_right) / h_sum

        # Main signal is canthus-centered iris offset.
        # Sclera asymmetry is only a light stabilizer.
        raw_yaw = center_x
        if h_asym is not None:
            raw_yaw = 0.75 * center_x + 0.25 * h_asym

        # --- PITCH ---
        # Use lid midpoint vertically.
        y_mid = 0.5 * (refs["topLid"][1] + refs["bottomLid"][1])
        center_y = self._safe_div(y_mid - iris_center[1], height2d)  # up is positive
        if center_y is None:
            return None

        d_top = iris_center[1] - refs["topLid"][1]
        d_bottom = refs["bottomLid"][1] - iris_center[1]
        v_asym = None
        v_sum = d_top + d_bottom
        if v_sum > 1e-9:
            v_asym = (d_bottom - d_top) / v_sum  # up is positive

        raw_pitch = center_y
        if v_asym is not None:
            raw_pitch = 0.5 * ((2.0 * center_y) + v_asym)

        return raw_yaw, raw_pitch

    @property
    def left_eye_gaze_yaw(self) -> Optional[float]:
        yp = self._eye_gaze_raw_yaw_pitch(
            self.left_iris_center,
            self.landmark_xyz(LEFT_EYE_INNER_IDX),
            self.landmark_xyz(LEFT_EYE_OUTER_IDX),
            self.landmark_xyz(LEFT_EYE_UPPER_IDX),
            self.landmark_xyz(LEFT_EYE_LOWER_IDX),
        )
        return self._proxy_to_angle_deg(yp[0], invert=True) if yp is not None else None

    @property
    def right_eye_gaze_yaw(self) -> Optional[float]:
        yp = self._eye_gaze_raw_yaw_pitch(
            self.right_iris_center,
            self.landmark_xyz(RIGHT_EYE_INNER_IDX),
            self.landmark_xyz(RIGHT_EYE_OUTER_IDX),
            self.landmark_xyz(RIGHT_EYE_UPPER_IDX),
            self.landmark_xyz(RIGHT_EYE_LOWER_IDX),
        )
        return self._proxy_to_angle_deg(yp[0], invert=True) if yp is not None else None

    @property
    def left_eye_gaze_pitch(self) -> Optional[float]:
        yp = self._eye_gaze_raw_yaw_pitch(
            self.left_iris_center,
            self.landmark_xyz(LEFT_EYE_INNER_IDX),
            self.landmark_xyz(LEFT_EYE_OUTER_IDX),
            self.landmark_xyz(LEFT_EYE_UPPER_IDX),
            self.landmark_xyz(LEFT_EYE_LOWER_IDX),
        )
        return self._proxy_to_angle_deg(yp[1]) if yp is not None else None

    @property
    def right_eye_gaze_pitch(self) -> Optional[float]:
        yp = self._eye_gaze_raw_yaw_pitch(
            self.right_iris_center,
            self.landmark_xyz(RIGHT_EYE_INNER_IDX),
            self.landmark_xyz(RIGHT_EYE_OUTER_IDX),
            self.landmark_xyz(RIGHT_EYE_UPPER_IDX),
            self.landmark_xyz(RIGHT_EYE_LOWER_IDX),
        )
        return self._proxy_to_angle_deg(yp[1]) if yp is not None else None

    def _apply_calibration(self, raw_yaw: float, raw_pitch: float, calib: CalibrationMatrix) -> tuple[float, float]:
        """Apply calibration matrix to raw gaze angles.
        
        Args:
            raw_yaw: Raw yaw angle in degrees
            raw_pitch: Raw pitch angle in degrees
            calib: Calibration matrix containing transformation coefficients
            
        Returns:
            Tuple of (calibrated_yaw, calibrated_pitch) in degrees
        """
        dy = raw_yaw - calib.center_yaw
        dp = raw_pitch - calib.center_pitch
        calibrated_yaw = calib.matrix_yaw_yaw * dy + calib.matrix_yaw_pitch * dp + calib.center_yaw
        calibrated_pitch = calib.matrix_pitch_yaw * dy + calib.matrix_pitch_pitch * dp + calib.center_pitch
        return calibrated_yaw, calibrated_pitch

    @property
    def calibrated_left_eye_gaze_yaw(self) -> Optional[float]:
        """Get calibrated left eye gaze yaw angle.
        
        Returns calibrated yaw when raw value exists, using identity transform
        if no calibration is provided (i.e., returns raw values unchanged).
        """
        raw_yaw = self.left_eye_gaze_yaw
        if raw_yaw is None:
            return None
        raw_pitch = self.left_eye_gaze_pitch or 0.0
        calib = self.calibration if self.calibration is not None else CalibrationMatrix()
        calibrated_yaw, _ = self._apply_calibration(raw_yaw, raw_pitch, calib)
        return calibrated_yaw

    @property
    def calibrated_right_eye_gaze_yaw(self) -> Optional[float]:
        """Get calibrated right eye gaze yaw angle.
        
        Returns calibrated yaw when raw value exists, using identity transform
        if no calibration is provided (i.e., returns raw values unchanged).
        """
        raw_yaw = self.right_eye_gaze_yaw
        if raw_yaw is None:
            return None
        raw_pitch = self.right_eye_gaze_pitch or 0.0
        calib = self.calibration if self.calibration is not None else CalibrationMatrix()
        calibrated_yaw, _ = self._apply_calibration(raw_yaw, raw_pitch, calib)
        return calibrated_yaw

    @property
    def calibrated_left_eye_gaze_pitch(self) -> Optional[float]:
        """Get calibrated left eye gaze pitch angle.
        
        Returns calibrated pitch when raw value exists, using identity transform
        if no calibration is provided (i.e., returns raw values unchanged).
        """
        raw_yaw = self.left_eye_gaze_yaw or 0.0
        raw_pitch = self.left_eye_gaze_pitch
        if raw_pitch is None:
            return None
        calib = self.calibration if self.calibration is not None else CalibrationMatrix()
        _, calibrated_pitch = self._apply_calibration(raw_yaw, raw_pitch, calib)
        return calibrated_pitch

    @property
    def calibrated_right_eye_gaze_pitch(self) -> Optional[float]:
        """Get calibrated right eye gaze pitch angle.
        
        Returns calibrated pitch when raw value exists, using identity transform
        if no calibration is provided (i.e., returns raw values unchanged).
        """
        raw_yaw = self.right_eye_gaze_yaw or 0.0
        raw_pitch = self.right_eye_gaze_pitch
        if raw_pitch is None:
            return None
        calib = self.calibration if self.calibration is not None else CalibrationMatrix()
        _, calibrated_pitch = self._apply_calibration(raw_yaw, raw_pitch, calib)
        return calibrated_pitch

    @property
    def calibrated_combined_eye_gaze_yaw(self) -> Optional[float]:
        """Get combined calibrated eye gaze yaw angle.
        
        Returns the average of calibrated left and right eye yaw values when both exist,
        otherwise returns None.
        """
        left_yaw = self.calibrated_left_eye_gaze_yaw
        right_yaw = self.calibrated_right_eye_gaze_yaw
        if left_yaw is None or right_yaw is None:
            return None
        return (left_yaw + right_yaw) / 2.0

    @property
    def calibrated_combined_eye_gaze_pitch(self) -> Optional[float]:
        """Get combined calibrated eye gaze pitch angle.
        
        Returns the average of calibrated left and right eye pitch values when both exist,
        otherwise returns None.
        """
        left_pitch = self.calibrated_left_eye_gaze_pitch
        right_pitch = self.calibrated_right_eye_gaze_pitch
        if left_pitch is None or right_pitch is None:
            return None
        return (left_pitch + right_pitch) / 2.0

    @property
    def blendshapes(self) -> Optional[Dict]:
        """Get raw blendshape categories object from MediaPipe result."""
        return self._face_item("face_blendshapes")

    @property
    def transform_matrix(self) -> Optional[List]:
        """Get raw face transformation matrix object from MediaPipe result."""
        return self._face_item("facial_transformation_matrixes")

    @property
    def face_mask_segment(self):
        """Get raw face mask/segment object when provided by result."""
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
        """Get count of detected landmarks for this face."""
        lms = self.landmarks
        if lms is None:
            return 0
        try:
            return len(lms)
        except Exception:
            return 0

    def landmarks_as_list(self) -> Optional[List[List[float]]]:
        """Serialize landmarks to numeric triples for JSON output."""
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
        """Serialize blendshapes to name->score for JSON output."""
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
        """Serialize transform matrix to flat row-major list for JSON output."""
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
        """Serialize face mask segment metadata for JSON output."""
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
        """Serialize explicit raw eye landmarks, centers, and gaze angles."""
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
            "leftEyeKeyPoints": self.left_eye_key_points,
            "rightEyeKeyPoints": self.right_eye_key_points,
        }

    def to_overlay_dict(self) -> Dict:
        """Overlay-facing lightweight view of event data."""
        return {
            "type": self.type,
            "hasFace": self.has_face,
            "landmarkCount": self.landmark_count,
            "ts": self.ts,
        }

    def to_capture_dict(self) -> Dict:
        """Capture-facing serializable payload."""
        return {
            "landmarks": self.landmarks_as_list(),
            "blendshapes": self.blendshapes_as_dict(),
            "transformMatrix": self.transform_matrix_as_flat(),
            "faceMaskSegment": self.face_mask_segment_meta(),
            "eyes": self.eyes_dict(),
        }

    def to_capture_dump(self) -> Dict[str, Any]:
        """Canonical capture dump with interpreted event fields and mesh payload."""
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
            },
            "meshData": self.to_capture_dict(),
        }

    def to_dict(self) -> Dict:
        """Convert event to dictionary."""
        return self.to_overlay_dict()