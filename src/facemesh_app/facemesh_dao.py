"""
Data access and interpretation layer for the FaceMesh app.
Provides FaceMesh-derived pose, eye, and landmark values.
"""

import logging
import math
import time
from typing import Any, Optional, Dict, List, Tuple

import numpy as np

from .face_geometry import VERTICAL_FOV_DEG
from .face_landmarker import FaceLandmarks

logger = logging.getLogger(__name__)


MM_PER_CM = 10.0

HORIZONTAL_MAX_DEG = 60.0
VERTICAL_MAX_DEG = 15


_CACHE_MISS = object()

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

NOSE_BRIDGE_IDX = 168
NOSE_BASE_IDX = 2

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

def safe_float(v, fallback=0.0):
    try:
        f = float(v)
    except (ValueError, TypeError):
        return fallback
    return f if math.isfinite(f) else fallback


def clamp(v, lo, hi):
    return max(lo, min(hi, v))




class FaceMeshEvent:
    def __init__(
        self,
        face: Optional[FaceLandmarks],
        *,
        image_size: Tuple[int, int],
        ts: Optional[int] = None,
        event_type: str = "mesh",
    ):
        self.face = face
        self.image_width = float(image_size[0])
        self.image_height = float(image_size[1])
        self.type = str(event_type)
        self.ts = int(ts if ts is not None else time.time() * 1000)
        self._cache: Dict[str, Any] = {}

    def _cache_get(self, key: str):
        return self._cache.get(key, _CACHE_MISS)

    def _cache_set(self, key: str, value):
        self._cache[key] = value
        return value

    @property
    def transform_matrix(self) -> Optional[np.ndarray]:
        return self.face.transform if self.face is not None else None

    @property
    def head_rotation(self) -> Optional[np.ndarray]:
        """Rotation from the face's own axes (x to its left, y up, z out of the face) to the camera frame."""
        transform = self.transform_matrix
        return transform[:3, :3] if transform is not None else None

    def _head_frame_xy(self, point: Optional[List[float]]) -> Optional[List[float]]:
        """Landmark position on the face's own left-right and up-down axes, unaffected by head rotation."""
        rotation = self.head_rotation
        if point is None or rotation is None:
            return None
        camera = np.array(
            [
                point[0] * self.image_width,
                -point[1] * self.image_height,
                -point[2] * self.image_width,
            ]
        )
        return list(rotation.T[0:2] @ camera)

    @property
    def has_face(self) -> bool:
        return self.face is not None

    @property
    def head_yaw(self) -> Optional[float]:
        """Head turn toward the user's right, in degrees."""
        rotation = self.head_rotation
        if rotation is None:
            return None
        return math.degrees(math.atan2(-rotation[0][2], rotation[2][2]))

    @property
    def head_pitch(self) -> Optional[float]:
        """Head elevation above the camera's horizontal plane, in degrees."""
        rotation = self.head_rotation
        if rotation is None:
            return None
        return math.degrees(math.asin(clamp(rotation[1][2], -1.0, 1.0)))

    def _translation(self, axis: int) -> Optional[float]:
        transform = self.transform_matrix
        return float(transform[axis, 3]) if transform is not None else None

    @property
    def x(self) -> Optional[float]:
        return self._translation(0)

    @property
    def y(self) -> Optional[float]:
        return self._translation(1)

    @property
    def raw_transform_z(self) -> Optional[float]:
        return self._translation(2)

    @property
    def roll(self) -> Optional[float]:
        """Head tilt toward the user's right shoulder about the face's forward axis, in degrees."""
        rotation = self.head_rotation
        if rotation is None:
            return None
        forward = rotation[:, 2]
        level_right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        norm = np.linalg.norm(level_right)
        if norm <= 1e-9:
            return None
        level_right /= norm
        level_up = np.cross(level_right, forward)
        up = rotation[:, 1]
        return math.degrees(math.atan2(float(up @ level_right), float(up @ level_up)))

    @property
    def landmarks(self) -> Optional[np.ndarray]:
        return self.face.landmarks if self.face is not None else None

    @property
    def mesh_edges(self) -> Optional[np.ndarray]:
        """Landmark index pairs joined in the face surface."""
        return self.face.mesh_edges if self.face is not None else None

    def landmark_xyz(self, idx: int) -> Optional[List[float]]:
        landmarks = self.landmarks
        if landmarks is None or not 0 <= idx < len(landmarks):
            return None
        return [float(v) for v in landmarks[idx]]

    def _landmarks_xyz_by_indices(self, indices: tuple[int, ...]) -> List[List[float]]:
        points: List[List[float]] = []
        for idx in indices:
            p = self.landmark_xyz(int(idx))
            if p is not None:
                points.append(p)
        return points

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
    def eye_position(self) -> Optional[np.ndarray]:
        """Midpoint between the eye corners in millimetres, in the camera frame the head pose is reported in."""
        cached = self._cache_get("eye_position")
        if cached is not _CACHE_MISS:
            return cached
        corners = [
            self.landmark_xyz(idx)
            for idx in (
                LEFT_EYE_INNER_IDX,
                LEFT_EYE_OUTER_IDX,
                RIGHT_EYE_INNER_IDX,
                RIGHT_EYE_OUTER_IDX,
            )
        ]
        head_z_cm = self.raw_transform_z
        if any(c is None for c in corners) or head_z_cm is None or head_z_cm >= 0:
            return self._cache_set("eye_position", None)
        u = sum(c[0] for c in corners) / len(corners) * self.image_width
        v = sum(c[1] for c in corners) / len(corners) * self.image_height
        focal_px = (self.image_height / 2.0) / math.tan(
            math.radians(VERTICAL_FOV_DEG / 2.0)
        )
        depth_mm = -head_z_cm * MM_PER_CM
        return self._cache_set(
            "eye_position",
            np.array(
                [
                    (u - self.image_width / 2.0) * depth_mm / focal_px,
                    -(v - self.image_height / 2.0) * depth_mm / focal_px,
                    -depth_mm,
                ]
            ),
        )

    @property
    def left_eye_key_points(self) -> List[List[float]]:
        return self._landmarks_xyz_by_indices(LEFT_EYE_KEY_IDXS)

    @property
    def right_eye_key_points(self) -> List[List[float]]:
        return self._landmarks_xyz_by_indices(RIGHT_EYE_KEY_IDXS)

    @property
    def eye_opening(self) -> Optional[float]:
        """Gap between the eyelids relative to the eye's width, averaged over both eyes; small during blinks."""
        openings = []
        for upper, lower, outer, inner in (
            (LEFT_EYE_UPPER_IDX, LEFT_EYE_LOWER_IDX, LEFT_EYE_OUTER_IDX, LEFT_EYE_INNER_IDX),
            (RIGHT_EYE_UPPER_IDX, RIGHT_EYE_LOWER_IDX, RIGHT_EYE_OUTER_IDX, RIGHT_EYE_INNER_IDX),
        ):
            points = [self._head_frame_xy(self.landmark_xyz(idx)) for idx in (upper, lower, outer, inner)]
            if any(p is None for p in points):
                return None
            width = math.dist(points[2], points[3])
            if width <= 1e-9:
                return None
            openings.append(math.dist(points[0], points[1]) / width)
        return sum(openings) / len(openings)

    def geometry_inputs(self) -> Dict[str, Any]:
        """Everything the head pose, eye position, eye angles and eye opening are derived from, for offline analysis."""
        return {
            "imageSize": [self.image_width, self.image_height],
            "transformMatrix": self.transform_matrix_as_flat(),
            "landmarks": {
                str(idx): self.landmark_xyz(idx)
                for idx in (
                    LEFT_IRIS_CENTER_IDX,
                    RIGHT_IRIS_CENTER_IDX,
                    LEFT_EYE_INNER_IDX,
                    LEFT_EYE_OUTER_IDX,
                    LEFT_EYE_UPPER_IDX,
                    LEFT_EYE_LOWER_IDX,
                    RIGHT_EYE_INNER_IDX,
                    RIGHT_EYE_OUTER_IDX,
                    RIGHT_EYE_UPPER_IDX,
                    RIGHT_EYE_LOWER_IDX,
                    NOSE_BRIDGE_IDX,
                    NOSE_BASE_IDX,
                )
            },
        }

    @staticmethod
    def _normalize_vec2(x: float, y: float) -> Optional[tuple[float, float]]:
        mag = math.hypot(x, y)
        if mag <= 1e-9:
            return None
        return x / mag, y / mag

    def _eye_gaze_raw_yaw_pitch(
        self,
        iris_center_point: Optional[List[float]],
        inner_canthus_point: Optional[List[float]],
        outer_canthus_point: Optional[List[float]],
    ) -> Optional[tuple[float, float]]:
        iris_center_xy = self._head_frame_xy(iris_center_point)
        inner_canthus_xy = self._head_frame_xy(inner_canthus_point)
        outer_canthus_xy = self._head_frame_xy(outer_canthus_point)
        if iris_center_xy is None or inner_canthus_xy is None or outer_canthus_xy is None:
            return None

        eye_axis_dx = outer_canthus_xy[0] - inner_canthus_xy[0]
        eye_axis_dy = outer_canthus_xy[1] - inner_canthus_xy[1]
        eye_width = math.hypot(eye_axis_dx, eye_axis_dy)
        if eye_width <= 1e-9:
            return None

        inner_to_iris_dx = iris_center_xy[0] - inner_canthus_xy[0]
        inner_to_iris_dy = iris_center_xy[1] - inner_canthus_xy[1]
        horizontal_position_from_inner = (
            inner_to_iris_dx * eye_axis_dx + inner_to_iris_dy * eye_axis_dy
        )
        normalized_horizontal_offset = (
            horizontal_position_from_inner / (eye_width * eye_width)
        ) - 0.5

        nose_bridge_xy = self._head_frame_xy(self.landmark_xyz(NOSE_BRIDGE_IDX))
        nose_base_xy = self._head_frame_xy(self.landmark_xyz(NOSE_BASE_IDX))

        if (
            nose_bridge_xy is None
            or nose_base_xy is None
        ):
            return None

        nose_axis_dx = nose_base_xy[0] - nose_bridge_xy[0]
        nose_axis_dy = nose_base_xy[1] - nose_bridge_xy[1]
        nose_axis_unit_2d = self._normalize_vec2(nose_axis_dx, nose_axis_dy)
        if nose_axis_unit_2d is None:
            return None
        nose_axis_ux, nose_axis_uy = nose_axis_unit_2d

        iris_from_bridge_x = iris_center_xy[0] - nose_bridge_xy[0]
        iris_from_bridge_y = iris_center_xy[1] - nose_bridge_xy[1]
        iris_to_perpendicular_signed = (
            iris_from_bridge_x * nose_axis_ux + iris_from_bridge_y * nose_axis_uy
        )
        normalized_vertical_offset = -(iris_to_perpendicular_signed / eye_width)

        yaw = 4 * normalized_horizontal_offset * HORIZONTAL_MAX_DEG
        pitch = 2 * normalized_vertical_offset * VERTICAL_MAX_DEG
        return yaw, pitch

    def _left_eye_raw_yaw_pitch(self) -> Optional[tuple[float, float]]:
        cached = self._cache_get("left_eye_raw_yaw_pitch")
        if cached is not _CACHE_MISS:
            return cached
        value = self._eye_gaze_raw_yaw_pitch(
            self.left_iris_center,
            self.landmark_xyz(LEFT_EYE_INNER_IDX),
            self.landmark_xyz(LEFT_EYE_OUTER_IDX),
        )
        return self._cache_set("left_eye_raw_yaw_pitch", value)

    def _right_eye_raw_yaw_pitch(self) -> Optional[tuple[float, float]]:
        cached = self._cache_get("right_eye_raw_yaw_pitch")
        if cached is not _CACHE_MISS:
            return cached
        value = self._eye_gaze_raw_yaw_pitch(
            self.right_iris_center,
            self.landmark_xyz(RIGHT_EYE_INNER_IDX),
            self.landmark_xyz(RIGHT_EYE_OUTER_IDX),
        )
        return self._cache_set("right_eye_raw_yaw_pitch", value)

    @property
    def left_eye_gaze_yaw(self) -> Optional[float]:
        yp = self._left_eye_raw_yaw_pitch()
        return yp[0] if yp is not None else None

    @property
    def right_eye_gaze_yaw(self) -> Optional[float]:
        yp = self._right_eye_raw_yaw_pitch()
        return -yp[0] if yp is not None else None

    @property
    def left_eye_gaze_pitch(self) -> Optional[float]:
        yp = self._left_eye_raw_yaw_pitch()
        return yp[1] if yp is not None else None

    @property
    def right_eye_gaze_pitch(self) -> Optional[float]:
        yp = self._right_eye_raw_yaw_pitch()
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

    @property
    def landmark_count(self) -> int:
        landmarks = self.landmarks
        return 0 if landmarks is None else len(landmarks)

    def landmarks_as_list(self) -> Optional[List[List[float]]]:
        landmarks = self.landmarks
        return landmarks.tolist() if landmarks is not None else None

    def blendshapes_as_dict(self) -> Optional[Dict[str, float]]:
        if self.face is None or not self.face.blendshapes:
            return None
        return dict(self.face.blendshapes)

    def transform_matrix_as_flat(self) -> Optional[List[float]]:
        transform = self.transform_matrix
        return transform.flatten().tolist() if transform is not None else None

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
            "leftEyeKeyPoints": self.left_eye_key_points,
            "rightEyeKeyPoints": self.right_eye_key_points,
        }

    def to_capture_dict(self) -> Dict:
        return {
            "landmarks": self.landmarks_as_list(),
            "blendshapes": self.blendshapes_as_dict(),
            "transformMatrix": self.transform_matrix_as_flat(),
            "eyes": self.eyes_dict(),
            "geometryInputs": self.geometry_inputs(),
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
                "z": self.raw_transform_z,
            },
            "eyePositionMm": (
                self.eye_position.tolist() if self.eye_position is not None else None
            ),
            "meshData": self.to_capture_dict(),
        }

