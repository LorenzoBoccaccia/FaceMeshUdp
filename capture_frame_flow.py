"""
Provide one shared frame flow for capture scripts, and the stored form of the faces they capture.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from facemesh_app.face_landmarker import FaceLandmarker, FaceLandmarks


def measure_face(landmarker: FaceLandmarker, frame_bgr: Any) -> Optional[FaceLandmarks]:
    return landmarker.measure(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def finalize_ui_frame(frame_bgr: Any, mirror_view: bool = True) -> Any:
    if mirror_view:
        return cv2.flip(frame_bgr, 1)
    return frame_bgr


def face_to_raw_result(face: Optional[FaceLandmarks]) -> Dict[str, Any]:
    """JSON form of a captured face, as the analysis scripts read it back."""
    if face is None:
        return {}
    raw: Dict[str, Any] = {
        "face_landmarks": [{"x": x, "y": y, "z": z} for x, y, z in face.landmarks.tolist()]
    }
    if face.transform is not None:
        raw["facial_transformation_matrix"] = face.transform.flatten().tolist()
    if face.blendshapes is not None:
        raw["face_blendshapes"] = [
            {"category": name, "score": score} for name, score in face.blendshapes.items()
        ]
    return raw


def face_from_raw_result(raw: Dict[str, Any]) -> Optional[FaceLandmarks]:
    """Captured face read back from its JSON form, or None when it was stored without a head pose."""
    matrix = raw.get("facial_transformation_matrix")
    if matrix is None:
        return None
    points = [[p["x"], p["y"], p["z"]] for p in raw.get("face_landmarks") or [] if p is not None]
    return FaceLandmarks(
        landmarks=np.array(points, dtype=np.float64).reshape(-1, 3),
        transform=np.array(matrix, dtype=np.float64).reshape(4, 4),
    )
