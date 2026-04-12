"""
Data Access Object for FaceMesh application.
Provides basic data structures and utility functions for face mesh data capture.
"""

import math
import time
from typing import Any, Optional, Dict, List


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

    def __init__(self, result: Any = None, *, face_index: int = 0, ts: Optional[int] = None, event_type: str = "mesh"):
        self.result = result
        self.face_index = int(face_index)
        self.type = str(event_type)
        self.ts = int(ts if ts is not None else time.time() * 1000)

    @classmethod
    def from_landmarker_result(cls, result: Any, *, face_index: int = 0, ts: Optional[int] = None):
        """Build event directly from MediaPipe FaceLandmarker result."""
        return cls(result, face_index=face_index, ts=ts, event_type="mesh")

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