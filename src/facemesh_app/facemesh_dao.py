# facemesh_dao.py
"""
Data access and interpretation layer for the FaceMesh app.
Provides FaceMesh-derived pose, eye, and landmark values.
"""

import logging
import math
import time
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)


AVERAGE_IPD_MM = 60.0

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
        face_index: int = 0,
        ts: Optional[int] = None,
        event_type: str = "mesh",
    ):
        self.result = result
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
        face_index: int = 0,
        ts: Optional[int] = None,
    ):
        return cls(
            result,
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
    def _dot3(a: List[float], b: List[float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _sub3(a: List[float], b: List[float]) -> List[float]:
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    @staticmethod
    def _cross3(a: List[float], b: List[float]) -> List[float]:
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    @staticmethod
    def _mag3(v: List[float]) -> float:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    @staticmethod
    def _dist2(a: List[float], b: List[float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

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
        cached = self._cache_get("head_yaw")
        if cached is not _CACHE_MISS:
            return cached

        m44 = self._transform_m44()
        if m44 is None:
            return self._cache_set("head_yaw", None)
        face_forward = self._normalize_vec3((-m44[0][2], -m44[1][2], -m44[2][2]))
        if face_forward is None:
            return self._cache_set("head_yaw", None)
        return self._cache_set("head_yaw", -math.degrees(math.atan2(face_forward[0], -face_forward[2])))

    @property
    def head_pitch(self) -> Optional[float]:
        cached = self._cache_get("head_pitch")
        if cached is not _CACHE_MISS:
            return cached

        m44 = self._transform_m44()
        if m44 is None:
            return self._cache_set("head_pitch", None)
        face_forward = self._normalize_vec3((-m44[0][2], -m44[1][2], -m44[2][2]))
        if face_forward is None:
            return self._cache_set("head_pitch", None)
        return self._cache_set("head_pitch", math.degrees(math.atan2(-face_forward[1], -face_forward[2])))

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
    def zeta(self) -> Optional[float]:
        cached = self._cache_get("zeta")
        if cached is not _CACHE_MISS:
            return cached

        left = self.left_iris_center
        right = self.right_iris_center
        if left is None or right is None:
            return self._cache_set("zeta", None)

        projected_ipd = self._dist2(left, right)
        if projected_ipd <= 1e-9:
            return self._cache_set("zeta", None)

        yaw = abs(safe_float(self.head_yaw, 0.0))
        pitch = abs(safe_float(self.head_pitch, 0.0))
        foreshortening = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
        foreshortening = clamp(foreshortening, 0.25, 1.0)

        frontal_projected_ipd = projected_ipd / foreshortening
        if frontal_projected_ipd <= 1e-9:
            return self._cache_set("zeta", None)

        return self._cache_set("zeta", AVERAGE_IPD_MM / frontal_projected_ipd)

    @property
    def z(self) -> Optional[float]:
        # Intentionally swapped from raw transform Z to IPD-based depth proxy.
        return self.zeta

    @property
    def roll(self) -> Optional[float]:
        cached = self._cache_get("roll")
        if cached is not _CACHE_MISS:
            return cached

        m44 = self._transform_m44()
        if m44 is not None:
            return self._cache_set("roll", math.degrees(math.atan2(m44[0][1], m44[0][0])))
        flat = self._transform_flat_no_fallback()
        if flat is not None and len(flat) > 1:
            return self._cache_set("roll", math.degrees(math.atan2(flat[1], flat[0])))
        return self._cache_set("roll", None)

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

    @staticmethod
    def _xy(point: Optional[List[float]]) -> Optional[List[float]]:
        if point is None or len(point) < 2:
            return None
        return [point[0], point[1]]

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
    def camera_x(self) -> Optional[float]:
        left = self.left_iris_center
        right = self.right_iris_center
        zeta = self.zeta
        if left is None or right is None or zeta is None:
            return None
        cx = (safe_float(left[0], 0.5) + safe_float(right[0], 0.5)) * 0.5
        return (cx - 0.5) * zeta

    @property
    def camera_y(self) -> Optional[float]:
        left = self.left_iris_center
        right = self.right_iris_center
        zeta = self.zeta
        if left is None or right is None or zeta is None:
            return None
        cy = (safe_float(left[1], 0.5) + safe_float(right[1], 0.5)) * 0.5
        return (cy - 0.5) * zeta

    @property
    def camera_z(self) -> Optional[float]:
        return self.zeta

    @property
    def left_eye_key_points(self) -> List[List[float]]:
        return self._landmarks_xyz_by_indices(LEFT_EYE_KEY_IDXS)

    @property
    def right_eye_key_points(self) -> List[List[float]]:
        return self._landmarks_xyz_by_indices(RIGHT_EYE_KEY_IDXS)

    def _eye_raw_input_points(
        self,
        iris_center_idx: int,
        inner_canthus_idx: int,
        outer_canthus_idx: int,
        upper_eyelid_idx: int,
        lower_eyelid_idx: int,
    ) -> Dict[str, Optional[List[float]]]:
        iris_center_xyz = self.landmark_xyz(iris_center_idx)
        inner_canthus_xyz = self.landmark_xyz(inner_canthus_idx)
        outer_canthus_xyz = self.landmark_xyz(outer_canthus_idx)
        upper_eyelid_xyz = self.landmark_xyz(upper_eyelid_idx)
        lower_eyelid_xyz = self.landmark_xyz(lower_eyelid_idx)

        iris_center = self._xy(iris_center_xyz)
        inner_canthus = self._xy(inner_canthus_xyz)
        outer_canthus = self._xy(outer_canthus_xyz)
        upper_eyelid = self._xy(upper_eyelid_xyz)
        lower_eyelid = self._xy(lower_eyelid_xyz)
        return {
            "irisCenter": iris_center,
            "innerCanthus": inner_canthus,
            "outerCanthus": outer_canthus,
            "upperEyelid": upper_eyelid,
            "lowerEyelid": lower_eyelid,
            "irisCenterXY": iris_center,
            "innerCanthusXY": inner_canthus,
            "outerCanthusXY": outer_canthus,
            "upperEyelidXY": upper_eyelid,
            "lowerEyelidXY": lower_eyelid,
        }

    def raw_mesh_inputs_dict(self) -> Dict[str, Any]:
        return {
            "gazeAndDepth": {
                "leftEye": self._eye_raw_input_points(
                    LEFT_IRIS_CENTER_IDX,
                    LEFT_EYE_INNER_IDX,
                    LEFT_EYE_OUTER_IDX,
                    LEFT_EYE_UPPER_IDX,
                    LEFT_EYE_LOWER_IDX,
                ),
                "rightEye": self._eye_raw_input_points(
                    RIGHT_IRIS_CENTER_IDX,
                    RIGHT_EYE_INNER_IDX,
                    RIGHT_EYE_OUTER_IDX,
                    RIGHT_EYE_UPPER_IDX,
                    RIGHT_EYE_LOWER_IDX,
                ),
            }
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
        upper_eyelid_point: Optional[List[float]],
        lower_eyelid_point: Optional[List[float]],
    ) -> Optional[tuple[float, float]]:

        if (
            iris_center_point is None
            or inner_canthus_point is None
            or outer_canthus_point is None
            or upper_eyelid_point is None
            or lower_eyelid_point is None
        ):
            return None

        iris_center_xy = self._xy(iris_center_point)
        inner_canthus_xy = self._xy(inner_canthus_point)
        outer_canthus_xy = self._xy(outer_canthus_point)
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

        nose_bridge_point = self.landmark_xyz(NOSE_BRIDGE_IDX)
        nose_base_point = self.landmark_xyz(NOSE_BASE_IDX)

        if (
            nose_bridge_point is None
            or nose_base_point is None
        ):
            return None

        nose_axis_dx = nose_base_point[0] - nose_bridge_point[0]
        nose_axis_dy = nose_base_point[1] - nose_bridge_point[1]
        nose_axis_unit_2d = self._normalize_vec2(nose_axis_dx, nose_axis_dy)
        if nose_axis_unit_2d is None:
            return None
        nose_axis_ux, nose_axis_uy = nose_axis_unit_2d

        iris_from_bridge_x = iris_center_xy[0] - nose_bridge_point[0]
        iris_from_bridge_y = iris_center_xy[1] - nose_bridge_point[1]
        iris_to_perpendicular_signed = (
            iris_from_bridge_x * nose_axis_ux + iris_from_bridge_y * nose_axis_uy
        )
        normalized_vertical_offset = -(iris_to_perpendicular_signed / eye_width)

        yaw = 4 * normalized_horizontal_offset * HORIZONTAL_MAX_DEG
        if normalized_vertical_offset > 0:
            pitch = 2 * normalized_vertical_offset * VERTICAL_MAX_DEG
        else:
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
            self.landmark_xyz(LEFT_EYE_UPPER_IDX),
            self.landmark_xyz(LEFT_EYE_LOWER_IDX),
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
            self.landmark_xyz(RIGHT_EYE_UPPER_IDX),
            self.landmark_xyz(RIGHT_EYE_LOWER_IDX),
        )
        return self._cache_set("right_eye_raw_yaw_pitch", value)

    @property
    def left_eye_gaze_yaw(self) -> Optional[float]:
        yp = self._left_eye_raw_yaw_pitch()
        return -yp[0] if yp is not None else None

    @property
    def right_eye_gaze_yaw(self) -> Optional[float]:
        yp = self._right_eye_raw_yaw_pitch()
        return yp[0] if yp is not None else None

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
            "rawInputs": self.raw_mesh_inputs_dict(),
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
                "cameraX": self.camera_x,
                "cameraY": self.camera_y,
                "cameraZ": self.camera_z,
            },
            "meshData": self.to_capture_dict(),
        }

    def to_dict(self) -> Dict:
        return self.to_overlay_dict()

