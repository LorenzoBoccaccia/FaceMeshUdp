"""
Data access and interpretation layer for the FaceMesh app.
Provides FaceMesh-derived pose, eye, and landmark values.
"""

import logging
import math
import time
from typing import Any, Optional, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


MEDIAPIPE_VERTICAL_FOV_DEG = 63.0
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
        result: Any = None,
        *,
        image_size: Tuple[int, int],
        face_index: int = 0,
        ts: Optional[int] = None,
        event_type: str = "mesh",
    ):
        self.result = result
        self.image_width = float(image_size[0])
        self.image_height = float(image_size[1])
        self.face_index = int(face_index)
        self.type = str(event_type)
        self.ts = int(ts if ts is not None else time.time() * 1000)
        self._cache: Dict[str, Any] = {}
        self._landmark_xyz_cache: Dict[int, Optional[List[float]]] = {}

    def _cache_get(self, key: str):
        return self._cache.get(key, _CACHE_MISS)

    def _cache_set(self, key: str, value):
        self._cache[key] = value
        return value

    @classmethod
    def from_landmarker_result(
        cls,
        result: Any,
        *,
        image_size: Tuple[int, int],
        face_index: int = 0,
        ts: Optional[int] = None,
    ):
        return cls(
            result,
            image_size=image_size,
            face_index=face_index,
            ts=ts,
            event_type="mesh",
        )

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

    def _transform_flat_no_fallback(self) -> Optional[List[float]]:
        cached = self._cache_get("transform_flat_no_fallback")
        if cached is not _CACHE_MISS:
            return cached

        m = self.transform_matrix
        if m is None:
            return self._cache_set("transform_flat_no_fallback", None)

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
            return self._cache_set("transform_flat_no_fallback", None)
        return self._cache_set("transform_flat_no_fallback", values or None)

    def _transform_m44(self) -> Optional[List[List[float]]]:
        cached = self._cache_get("transform_m44")
        if cached is not _CACHE_MISS:
            return cached

        flat = self._transform_flat_no_fallback()
        if flat is None or len(flat) < 16:
            return self._cache_set("transform_m44", None)
        return self._cache_set("transform_m44", [flat[0:4], flat[4:8], flat[8:12], flat[12:16]])

    @property
    def head_rotation(self) -> Optional[np.ndarray]:
        """Rotation from the face's own axes (x to its left, y up, z out of the face) to the camera frame."""
        cached = self._cache_get("head_rotation")
        if cached is not _CACHE_MISS:
            return cached
        m44 = self._transform_m44()
        if m44 is None:
            return self._cache_set("head_rotation", None)
        return self._cache_set(
            "head_rotation", np.array([row[0:3] for row in m44[0:3]], dtype=float)
        )

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
        cached = self._cache_get("has_face")
        if cached is not _CACHE_MISS:
            return cached

        lms = self.landmarks
        if lms is None:
            return self._cache_set("has_face", False)
        try:
            return self._cache_set("has_face", len(lms) > 0)
        except Exception:
            return self._cache_set("has_face", False)

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

    @property
    def x(self) -> Optional[float]:
        cached = self._cache_get("x")
        if cached is not _CACHE_MISS:
            return cached

        m44 = self._transform_m44()
        if m44 is not None:
            return self._cache_set("x", m44[0][3])
        flat = self._transform_flat_no_fallback()
        if flat is not None and len(flat) > 3:
            return self._cache_set("x", flat[3])
        return self._cache_set("x", None)

    @property
    def y(self) -> Optional[float]:
        cached = self._cache_get("y")
        if cached is not _CACHE_MISS:
            return cached

        m44 = self._transform_m44()
        if m44 is not None:
            return self._cache_set("y", m44[1][3])
        flat = self._transform_flat_no_fallback()
        if flat is not None and len(flat) > 7:
            return self._cache_set("y", flat[7])
        return self._cache_set("y", None)

    @property
    def raw_transform_z(self) -> Optional[float]:
        cached = self._cache_get("raw_transform_z")
        if cached is not _CACHE_MISS:
            return cached

        m44 = self._transform_m44()
        if m44 is not None:
            return self._cache_set("raw_transform_z", m44[2][3])
        flat = self._transform_flat_no_fallback()
        if flat is not None and len(flat) > 11:
            return self._cache_set("raw_transform_z", flat[11])
        return self._cache_set("raw_transform_z", None)

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
    def landmarks(self) -> Optional[List]:
        cached = self._cache_get("landmarks")
        if cached is not _CACHE_MISS:
            return cached
        return self._cache_set("landmarks", self._face_item("face_landmarks"))

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
            return [
                safe_float(lm.x),
                safe_float(lm.y),
                safe_float(getattr(lm, "z", 0.0)),
            ]
        if isinstance(lm, (list, tuple)) and len(lm) >= 3:
            return [safe_float(lm[0]), safe_float(lm[1]), safe_float(lm[2])]
        return None

    def landmark_xyz(self, idx: int) -> Optional[List[float]]:
        key = int(idx)
        if key in self._landmark_xyz_cache:
            return self._landmark_xyz_cache[key]
        value = self._landmark_xyz(self.landmark(key))
        self._landmark_xyz_cache[key] = value
        return value

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
            math.radians(MEDIAPIPE_VERTICAL_FOV_DEG / 2.0)
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
    def blendshapes(self) -> Optional[Dict]:
        cached = self._cache_get("blendshapes")
        if cached is not _CACHE_MISS:
            return cached
        return self._cache_set("blendshapes", self._face_item("face_blendshapes"))

    @property
    def transform_matrix(self) -> Optional[List]:
        cached = self._cache_get("transform_matrix")
        if cached is not _CACHE_MISS:
            return cached
        return self._cache_set(
            "transform_matrix", self._face_item("facial_transformation_matrixes")
        )

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
        cached = self._cache_get("landmark_count")
        if cached is not _CACHE_MISS:
            return cached

        lms = self.landmarks
        if lms is None:
            return self._cache_set("landmark_count", 0)
        try:
            return self._cache_set("landmark_count", len(lms))
        except Exception:
            return self._cache_set("landmark_count", 0)

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
            "leftEyeKeyPoints": self.left_eye_key_points,
            "rightEyeKeyPoints": self.right_eye_key_points,
        }

    def to_capture_dict(self) -> Dict:
        return {
            "landmarks": self.landmarks_as_list(),
            "blendshapes": self.blendshapes_as_dict(),
            "transformMatrix": self.transform_matrix_as_flat(),
            "faceMaskSegment": self.face_mask_segment_meta(),
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

