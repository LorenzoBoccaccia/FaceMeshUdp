"""
Calibration model: where on screen the user looks, from head pose and eye measurements.
"""

import json
import logging
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .facemesh_dao import FaceMeshEvent, clamp

logger = logging.getLogger(__name__)

CALIBRATION_MODEL_VERSION = 11
DEFAULT_VIEWING_DISTANCE_MM = 1000.0
REFERENCE_POINT = "C"
POINT_NAMES = ("C", "T", "TL", "L", "BL", "B", "BR", "R", "TR")
MAX_SCREEN_ANGLE_DEG = 30.0

PERSON_AXES = np.diag([-1.0, 1.0, -1.0])


def direction(angles_deg: Sequence[float]) -> np.ndarray:
    """Unit vector for a yaw to the right and a pitch up, in right/up/back axes."""
    yaw, pitch = np.radians(angles_deg)
    return np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            -math.cos(yaw) * math.cos(pitch),
        ]
    )


def angles(vector: np.ndarray) -> np.ndarray:
    """Yaw to the right and pitch up of a direction in right/up/back axes, in degrees."""
    unit = vector / np.linalg.norm(vector)
    return np.degrees(
        [math.atan2(unit[0], -unit[2]), math.asin(clamp(unit[1], -1.0, 1.0))]
    )


def has_gaze_inputs(event: Optional[FaceMeshEvent]) -> bool:
    """Whether a face measurement carries everything the gaze model needs."""
    return (
        event is not None
        and event.has_face
        and event.head_rotation is not None
        and event.eye_position is not None
        and event.combined_eye_gaze_yaw is not None
        and event.combined_eye_gaze_pitch is not None
    )


@dataclass(frozen=True)
class Screen:
    """Display surface: its pixel size and physical pixel density."""

    width_px: int
    height_px: int
    px_per_mm_x: float
    px_per_mm_y: float

    @classmethod
    def from_display(cls, display: Dict[str, Any]) -> Optional["Screen"]:
        width_mm = float(display.get("width_mm") or 0)
        height_mm = float(display.get("height_mm") or 0)
        if width_mm <= 0 or height_mm <= 0:
            return None
        width_px = int(display["width"])
        height_px = int(display["height"])
        return cls(width_px, height_px, width_px / width_mm, height_px / height_mm)

    def offset_mm(self, px: Sequence[float]) -> np.ndarray:
        """Distance right of and above the screen centre, in millimetres."""
        return np.array(
            [
                (px[0] - self.width_px / 2.0) / self.px_per_mm_x,
                -(px[1] - self.height_px / 2.0) / self.px_per_mm_y,
            ]
        )

    def pixel(self, offset_mm: Sequence[float]) -> Tuple[float, float]:
        return (
            float(self.width_px / 2.0 + offset_mm[0] * self.px_per_mm_x),
            float(self.height_px / 2.0 - offset_mm[1] * self.px_per_mm_y),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "widthPx": self.width_px,
            "heightPx": self.height_px,
            "pxPerMmX": self.px_per_mm_x,
            "pxPerMmY": self.px_per_mm_y,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Screen":
        return cls(
            int(data["widthPx"]),
            int(data["heightPx"]),
            float(data["pxPerMmX"]),
            float(data["pxPerMmY"]),
        )


@dataclass(frozen=True)
class CalibrationPoint:
    """Head pose and eye measurement while the nose aims at one target and the eyes fixate another."""

    name: str
    nose_target_px: Tuple[float, float]
    eye_target_px: Tuple[float, float]
    raw_eye: Tuple[float, float]
    head_rotation: Tuple[Tuple[float, float, float], ...]
    eye_position_mm: Tuple[float, float, float]
    sample_count: int

    @classmethod
    def from_samples(
        cls,
        name: str,
        nose_target_px: Tuple[float, float],
        eye_target_px: Tuple[float, float],
        events: Sequence[FaceMeshEvent],
    ) -> Optional["CalibrationPoint"]:
        """Robust summary of the frames captured for one target; None when no frame was usable."""
        usable = [event for event in events if has_gaze_inputs(event)]
        if not usable:
            return None
        raw_eye = np.median(
            [[e.combined_eye_gaze_yaw, e.combined_eye_gaze_pitch] for e in usable],
            axis=0,
        )
        eye_position = np.median([e.eye_position for e in usable], axis=0)
        u, _, vt = np.linalg.svd(np.mean([e.head_rotation for e in usable], axis=0))
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        return cls(
            name=name,
            nose_target_px=(float(nose_target_px[0]), float(nose_target_px[1])),
            eye_target_px=(float(eye_target_px[0]), float(eye_target_px[1])),
            raw_eye=(float(raw_eye[0]), float(raw_eye[1])),
            head_rotation=tuple(tuple(float(v) for v in row) for row in rotation),
            eye_position_mm=tuple(float(v) for v in eye_position),
            sample_count=len(usable),
        )

    @property
    def rotation(self) -> np.ndarray:
        return np.array(self.head_rotation)

    @property
    def position(self) -> np.ndarray:
        return np.array(self.eye_position_mm)

    @property
    def raw(self) -> np.ndarray:
        return np.array(self.raw_eye)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "noseTargetPx": list(self.nose_target_px),
            "eyeTargetPx": list(self.eye_target_px),
            "rawEye": list(self.raw_eye),
            "headRotation": [list(row) for row in self.head_rotation],
            "eyePositionMm": list(self.eye_position_mm),
            "sampleCount": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationPoint":
        return cls(
            name=str(data["name"]),
            nose_target_px=tuple(data["noseTargetPx"]),
            eye_target_px=tuple(data["eyeTargetPx"]),
            raw_eye=tuple(data["rawEye"]),
            head_rotation=tuple(tuple(row) for row in data["headRotation"]),
            eye_position_mm=tuple(data["eyePositionMm"]),
            sample_count=int(data["sampleCount"]),
        )


@dataclass(frozen=True)
class Gaze:
    """Where the user looks: angle on screen at the viewing distance, screen pixel, and its head and eye parts."""

    yaw: float
    pitch: float
    screen_px: Tuple[float, float]
    head_screen_px: Optional[Tuple[float, float]]
    eye_screen_px: Optional[Tuple[float, float]]
    eye_yaw: float
    eye_pitch: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "screenPx": list(self.screen_px),
            "headScreenPx": list(self.head_screen_px) if self.head_screen_px else None,
            "eyeScreenPx": list(self.eye_screen_px) if self.eye_screen_px else None,
            "eyeYaw": self.eye_yaw,
            "eyePitch": self.eye_pitch,
        }


def _screen_axes(roll_deg: float, tilt_deg: float) -> np.ndarray:
    """Screen right/up/toward-viewer axes relative to the line of sight to the screen centre."""
    return Rotation.from_euler("zx", [roll_deg, tilt_deg], degrees=True).as_matrix()


@dataclass(frozen=True)
class GazeModel:
    """Maps a face measurement to the screen point the user looks at.

    The screen is anchored to the reference pose, where the user faces the screen centre
    with head and eyes aligned: its centre lies straight ahead at the viewing distance.
    Head rotations are taken relative to that pose, so a constant bias in the measured
    head orientation cancels out.
    """

    screen: Screen
    viewing_distance_mm: float
    reference: CalibrationPoint
    screen_roll_deg: float
    screen_tilt_deg: float
    eye_matrix: Tuple[Tuple[float, float], Tuple[float, float]]
    head_aim_gain: Tuple[float, float]
    eye_residual_deg: float
    head_residual_deg: float

    @cached_property
    def _reference_axes(self) -> np.ndarray:
        return self.reference.rotation @ PERSON_AXES

    @cached_property
    def _screen_axes_camera(self) -> np.ndarray:
        return self._reference_axes @ _screen_axes(
            self.screen_roll_deg, self.screen_tilt_deg
        )

    @cached_property
    def _screen_center_camera(self) -> np.ndarray:
        return self.reference.position + self._reference_axes @ np.array(
            [0.0, 0.0, -self.viewing_distance_mm]
        )

    def eye_angles(self, raw_eye: np.ndarray) -> np.ndarray:
        """Eye rotation within the head, in degrees right and up."""
        return np.array(self.eye_matrix) @ (raw_eye - self.reference.raw)

    def _screen_offset(
        self, origin: np.ndarray, ray: np.ndarray
    ) -> Optional[np.ndarray]:
        axes = self._screen_axes_camera
        local_origin = axes.T @ (origin - self._screen_center_camera)
        local_ray = axes.T @ ray
        if local_ray[2] >= -1e-9:
            return None
        travel = -local_origin[2] / local_ray[2]
        if travel <= 0:
            return None
        return (local_origin + travel * local_ray)[:2]

    def _pixel(self, origin: np.ndarray, ray: np.ndarray) -> Optional[Tuple[float, float]]:
        offset = self._screen_offset(origin, ray)
        return None if offset is None else self.screen.pixel(offset)

    def project(
        self, head_rotation: np.ndarray, eye_position: np.ndarray, raw_eye: np.ndarray
    ) -> Optional[Gaze]:
        """Gaze for one head pose and eye measurement; None when it does not reach the screen plane."""
        head_axes = head_rotation @ PERSON_AXES
        eye = self.eye_angles(raw_eye)
        offset = self._screen_offset(eye_position, head_axes @ direction(eye))
        if offset is None:
            return None
        return Gaze(
            yaw=math.degrees(math.atan2(offset[0], self.viewing_distance_mm)),
            pitch=math.degrees(math.atan2(offset[1], self.viewing_distance_mm)),
            screen_px=self.screen.pixel(offset),
            head_screen_px=self._pixel(eye_position, head_axes @ direction((0.0, 0.0))),
            eye_screen_px=self._pixel(eye_position, self._reference_axes @ direction(eye)),
            eye_yaw=float(eye[0]),
            eye_pitch=float(eye[1]),
        )

    def gaze(self, event: Optional[FaceMeshEvent]) -> Optional[Gaze]:
        if not has_gaze_inputs(event):
            return None
        return self.project(
            event.head_rotation,
            event.eye_position,
            np.array([event.combined_eye_gaze_yaw, event.combined_eye_gaze_pitch]),
        )

    def point_errors_px(self, point: CalibrationPoint) -> Optional[Tuple[float, float]]:
        """Screen distance from a calibration point's eye target to where the model places the gaze."""
        gaze = self.project(point.rotation, point.position, point.raw)
        if gaze is None:
            return None
        return (
            gaze.screen_px[0] - point.eye_target_px[0],
            gaze.screen_px[1] - point.eye_target_px[1],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewingDistanceMm": self.viewing_distance_mm,
            "screenRollDeg": self.screen_roll_deg,
            "screenTiltDeg": self.screen_tilt_deg,
            "eyeMatrix": [list(row) for row in self.eye_matrix],
            "headAimGain": list(self.head_aim_gain),
            "eyeResidualDeg": self.eye_residual_deg,
            "headResidualDeg": self.head_residual_deg,
            "screen": self.screen.to_dict(),
            "reference": self.reference.to_dict(),
        }

    def describe(self) -> str:
        (a, b), (c, d) = self.eye_matrix
        return (
            f"viewing distance {self.viewing_distance_mm / 10.0:.0f} cm, "
            f"screen roll {self.screen_roll_deg:+.1f} deg tilt {self.screen_tilt_deg:+.1f} deg, "
            f"eye matrix [[{a:.3f} {b:+.3f}] [{c:+.3f} {d:.3f}]], "
            f"head aim gain ({self.head_aim_gain[0]:.2f}, {self.head_aim_gain[1]:.2f}), "
            f"fit residual eye {self.eye_residual_deg:.2f} deg head {self.head_residual_deg:.2f} deg"
        )


def fit_gaze_model(
    points: Sequence[CalibrationPoint], screen: Screen, viewing_distance_mm: float
) -> GazeModel:
    """Fit screen orientation, eye gains and head aim gains to one calibration session.

    At every point the eyes fixate one target while the nose aims at the opposite one, so
    head and eye rotations vary independently. Head rotation is measured in true degrees,
    which ties the eye gains to the head; the measured viewing distance fixes the scale
    that head aim and eye gain could otherwise trade against.
    """
    by_name = {point.name: point for point in points}
    missing = [name for name in POINT_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"Calibration is missing targets: {', '.join(missing)}")
    if viewing_distance_mm <= 0:
        raise ValueError("Viewing distance must be positive")

    reference = by_name[REFERENCE_POINT]
    reference_axes = reference.rotation @ PERSON_AXES
    observations = [
        (
            reference_axes.T @ point.rotation @ PERSON_AXES,
            reference_axes.T @ (point.position - reference.position),
            point.raw - reference.raw,
            screen.offset_mm(point.nose_target_px),
            screen.offset_mm(point.eye_target_px),
        )
        for point in (by_name[name] for name in POINT_NAMES if name != REFERENCE_POINT)
    ]
    center = np.array([0.0, 0.0, -viewing_distance_mm])
    head_forward = direction((0.0, 0.0))

    def residuals(params: np.ndarray) -> np.ndarray:
        axes = _screen_axes(params[0], params[1])[:, :2]
        eye_matrix = params[2:6].reshape(2, 2)
        head_gain = params[6:8]
        out = []
        for rotation, position, raw, nose, eye in observations:
            eye_target = center + axes @ eye - position
            out.extend(eye_matrix @ raw - angles(rotation.T @ eye_target))
            nose_target = center + axes @ nose - position
            out.extend(angles(rotation @ head_forward) - head_gain * angles(nose_target))
        return np.array(out)

    limit = MAX_SCREEN_ANGLE_DEG
    result = least_squares(
        residuals,
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.5]),
        bounds=(
            [-limit, -limit] + [-np.inf] * 6,
            [limit, limit] + [np.inf] * 6,
        ),
        loss="soft_l1",
        f_scale=2.0,
    )
    final = residuals(result.x).reshape(-1, 2, 2)
    eye_matrix = result.x[2:6].reshape(2, 2)
    return GazeModel(
        screen=screen,
        viewing_distance_mm=float(viewing_distance_mm),
        reference=reference,
        screen_roll_deg=float(result.x[0]),
        screen_tilt_deg=float(result.x[1]),
        eye_matrix=tuple(tuple(float(v) for v in row) for row in eye_matrix),
        head_aim_gain=(float(result.x[6]), float(result.x[7])),
        eye_residual_deg=float(np.sqrt(np.mean(final[:, 0] ** 2))),
        head_residual_deg=float(np.sqrt(np.mean(final[:, 1] ** 2))),
    )


@dataclass(frozen=True)
class CalibrationProfile:
    """One recorded calibration: the measured points and the screen and distance they were taken at."""

    screen: Screen
    viewing_distance_mm: float
    points: Tuple[CalibrationPoint, ...]

    def fit(self, viewing_distance_mm: Optional[float] = None) -> GazeModel:
        return fit_gaze_model(
            self.points,
            self.screen,
            self.viewing_distance_mm if viewing_distance_mm is None else viewing_distance_mm,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modelVersion": CALIBRATION_MODEL_VERSION,
            "screen": self.screen.to_dict(),
            "viewingDistanceMm": self.viewing_distance_mm,
            "points": [point.to_dict() for point in self.points],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationProfile":
        return cls(
            screen=Screen.from_dict(data["screen"]),
            viewing_distance_mm=float(data["viewingDistanceMm"]),
            points=tuple(CalibrationPoint.from_dict(p) for p in data["points"]),
        )


def _profile_token(raw_profile: str) -> str:
    if not raw_profile:
        return "default"
    sanitized = "".join(
        ch if ch.isalnum() or ch in "._-" else "-" for ch in raw_profile
    ).strip("._-")
    return sanitized or "default"


def profile_path(profile: str = "") -> Path:
    return Path(f"calibration-{_profile_token(profile)}.json")


def save_calibration(calibration: CalibrationProfile, profile: str = "") -> Path:
    path = profile_path(profile)
    path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    return path


def load_calibration(profile: str = "") -> Optional[CalibrationProfile]:
    """The stored calibration for a profile, or None when there is none usable."""
    path = profile_path(profile)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to read calibration {path.name}: {exc}")
        return None
    version = data.get("modelVersion")
    if version != CALIBRATION_MODEL_VERSION:
        logger.warning(
            f"Calibration {path.name} was recorded with model version {version}; "
            f"version {CALIBRATION_MODEL_VERSION} needs a new calibration."
        )
        return None
    return CalibrationProfile.from_dict(data)


@dataclass(frozen=True)
class CalibratedFaceAndGazeEvent:
    """A face measurement together with the gaze the calibration model derives from it."""

    face_mesh_event: FaceMeshEvent
    model: GazeModel
    gaze: Gaze

    def to_dict(self) -> Dict[str, Any]:
        return {"gaze": self.gaze.to_dict(), "model": self.model.to_dict()}
