#!/usr/bin/env python3
"""
Pitch Correlation Capture Script
Captures raw FaceMesh data for head and eye pitch combinations to analyze eye pitch consistency.
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from capture_frame_flow import detect_face_landmarker_result, finalize_ui_frame


def safe_float(v, fallback=0.0):
    try:
        f = float(v)
    except (ValueError, TypeError):
        return fallback
    return f if math.isfinite(f) else fallback


# Constants
MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
OUTPUT_DIR = Path("pitch_correlation")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
HUD_BG = (20, 20, 20)
HUD_BORDER = (230, 230, 230)
HUD_TEXT = (245, 245, 245)

LEFT_EYE_LANDMARKS = (33, 133, 145, 159, 468, 469, 470, 471, 472)
RIGHT_EYE_LANDMARKS = (263, 362, 374, 386, 473, 474, 475, 476, 477)
NOSE_BRIDGE_IDX = 168
NOSE_BASE_IDX = 2
LEFT_IRIS_CENTER_IDX = 468
RIGHT_IRIS_CENTER_IDX = 473

# 9 prompts to capture: 3 head positions x 3 eye positions
# Head positions: down, center, up
# Eye positions: down, center, up
PROMPTS = [
    # Head DOWN row
    {
        "name": "head-down-eye-down",
        "instruction": "Tilt your HEAD DOWN and look with your EYES DOWN",
        "type": "combined",
    },
    {
        "name": "head-down-eye-center",
        "instruction": "Tilt your HEAD DOWN and look with your EYES STRAIGHT AHEAD",
        "type": "combined",
    },
    {
        "name": "head-down-eye-up",
        "instruction": "Tilt your HEAD DOWN and look with your EYES UP",
        "type": "combined",
    },
    # Head CENTER row
    {
        "name": "head-center-eye-down",
        "instruction": "Keep your HEAD CENTERED and look with your EYES DOWN",
        "type": "combined",
    },
    {
        "name": "head-center-eye-center",
        "instruction": "Keep your HEAD CENTERED and look with your EYES STRAIGHT AHEAD",
        "type": "combined",
    },
    {
        "name": "head-center-eye-up",
        "instruction": "Keep your HEAD CENTERED and look with your EYES UP",
        "type": "combined",
    },
    # Head UP row
    {
        "name": "head-up-eye-down",
        "instruction": "Tilt your HEAD UP and look with your EYES DOWN",
        "type": "combined",
    },
    {
        "name": "head-up-eye-center",
        "instruction": "Tilt your HEAD UP and look with your EYES STRAIGHT AHEAD",
        "type": "combined",
    },
    {
        "name": "head-up-eye-up",
        "instruction": "Tilt your HEAD UP and look with your EYES UP",
        "type": "combined",
    },
]


def ensure_model():
    """Download MediaPipe model if not present."""
    if MODEL_PATH.exists():
        return
    print(f"Downloading model from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    print("Model downloaded.")


def open_camera(camera_index: int = 0) -> Tuple[cv2.VideoCapture, Dict]:
    """Open camera with sensible defaults."""
    backends = [
        (cv2.CAP_MSMF, "msmf"),
        (cv2.CAP_DSHOW, "dshow"),
        (None, "any"),
    ]

    for backend, name in backends:
        cap = (
            cv2.VideoCapture(camera_index, backend)
            if backend is not None
            else cv2.VideoCapture(camera_index)
        )
        if not cap.isOpened():
            cap.release()
            continue

        # Set preferred settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ok, frame = cap.read()
        if not ok:
            cap.release()
            continue

        info = {
            "backend": name,
            "index": camera_index,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }
        print(f"Camera opened: {info}")
        return cap, info

    raise RuntimeError("Failed to open camera")


def draw_text_with_background(
    img,
    text: str,
    position: Tuple[int, int],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    thickness=2,
    padding=10,
    bg_color=HUD_BG,
    text_color=HUD_TEXT,
):
    """Draw text with background rectangle."""
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(
        img,
        (x - padding, y - text_h - padding - baseline),
        (x + text_w + padding, y + padding + baseline),
        bg_color,
        -1,
    )
    cv2.rectangle(
        img,
        (x - padding, y - text_h - padding - baseline),
        (x + text_w + padding, y + padding + baseline),
        HUD_BORDER,
        2,
    )
    cv2.putText(
        img,
        text,
        (x, y + baseline),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def _nose_plane_reference(
    bridge_xy: Tuple[float, float],
    base_xy: Tuple[float, float],
    left_iris_xy: Tuple[float, float],
    right_iris_xy: Tuple[float, float],
) -> Optional[Dict[str, Any]]:
    bx, by = bridge_xy
    nbx, nby = base_xy
    axis_x = nbx - bx
    axis_y = nby - by
    axis_len = math.hypot(axis_x, axis_y)
    if axis_len <= 1e-9:
        return None

    n_hat_x = axis_x / axis_len
    n_hat_y = axis_y / axis_len
    p_hat_x = -n_hat_y
    p_hat_y = n_hat_x

    lx, ly = left_iris_xy
    rx, ry = right_iris_xy
    eye_span = math.hypot(rx - lx, ry - ly)
    if eye_span <= 1e-9:
        return None

    def point_projection(ix: float, iy: float) -> Dict[str, float]:
        dx = ix - bx
        dy = iy - by
        signed = dx * n_hat_x + dy * n_hat_y
        along_perp = dx * p_hat_x + dy * p_hat_y
        proj_x = bx + along_perp * p_hat_x
        proj_y = by + along_perp * p_hat_y
        return {
            "signed": signed,
            "alongPerp": along_perp,
            "projX": proj_x,
            "projY": proj_y,
        }

    left_proj = point_projection(lx, ly)
    right_proj = point_projection(rx, ry)
    avg_signed = 0.5 * (left_proj["signed"] + right_proj["signed"])

    return {
        "axisLength": axis_len,
        "eyeSpan": eye_span,
        "nHatX": n_hat_x,
        "nHatY": n_hat_y,
        "pHatX": p_hat_x,
        "pHatY": p_hat_y,
        "leftSigned": left_proj["signed"],
        "rightSigned": right_proj["signed"],
        "avgSigned": avg_signed,
        "leftProjX": left_proj["projX"],
        "leftProjY": left_proj["projY"],
        "rightProjX": right_proj["projX"],
        "rightProjY": right_proj["projY"],
        "leftAlongPerp": left_proj["alongPerp"],
        "rightAlongPerp": right_proj["alongPerp"],
    }


def serialize_mediapipe_result(result) -> Dict[str, Any]:
    if result is None:
        return {}

    def serialize_landmarks(landmarks):
        if landmarks is None:
            return None
        try:
            if hasattr(landmarks, "__iter__") and not isinstance(
                landmarks, (str, bytes)
            ):
                result = []
                for lm in landmarks:
                    x_val = getattr(lm, "x", None) if hasattr(lm, "x") else None
                    y_val = getattr(lm, "y", None) if hasattr(lm, "y") else None
                    z_val = getattr(lm, "z", None) if hasattr(lm, "z") else None

                    lm_data = {
                        "x": safe_float(x_val) if x_val is not None else None,
                        "y": safe_float(y_val) if y_val is not None else None,
                        "z": safe_float(z_val) if z_val is not None else None,
                    }
                    if hasattr(lm, "visibility"):
                        v_val = getattr(lm, "visibility", None)
                        lm_data["visibility"] = (
                            safe_float(v_val) if v_val is not None else None
                        )
                    if hasattr(lm, "presence"):
                        p_val = getattr(lm, "presence", None)
                        lm_data["presence"] = (
                            safe_float(p_val) if p_val is not None else None
                        )
                    result.append(lm_data)
                return result
        except (AttributeError, TypeError, ValueError) as e:
            print(f"Error serializing landmarks: {e}")
        return None

    def serialize_matrix(matrix):
        if matrix is None:
            return None
        try:
            if hasattr(matrix, "flatten"):
                return [float(x) for x in matrix.flatten()]
            elif hasattr(matrix, "__iter__"):
                flat = []
                for row in matrix:
                    if hasattr(row, "__iter__"):
                        flat.extend([float(x) for x in row])
                    else:
                        flat.append(float(row))
                return flat
        except (AttributeError, TypeError, ValueError) as e:
            print(f"Error serializing matrix: {e}")
        return None

    def serialize_blendshapes(blendshapes):
        if blendshapes is None:
            return None
        try:
            if hasattr(blendshapes, "__iter__") and not isinstance(
                blendshapes, (str, bytes)
            ):
                return [
                    {
                        "category": str(bs.category)
                        if hasattr(bs, "category")
                        else None,
                        "score": float(bs.score) if hasattr(bs, "score") else None,
                    }
                    for bs in blendshapes
                ]
        except (AttributeError, TypeError, ValueError) as e:
            print(f"Error serializing blendshapes: {e}")
        return None
        try:
            # MediaPipe face_landmarks is a list of NormalizedLandmarkList
            if hasattr(landmarks, "__iter__") and not isinstance(
                landmarks, (str, bytes)
            ):
                result = []
                for lm in landmarks:
                    # Each landmark should have x, y, z properties
                    # Use safe_float to handle None values
                    x_val = getattr(lm, "x", None) if hasattr(lm, "x") else None
                    y_val = getattr(lm, "y", None) if hasattr(lm, "y") else None
                    z_val = getattr(lm, "z", None) if hasattr(lm, "z") else None

                    lm_data = {
                        "x": safe_float(x_val) if x_val is not None else None,
                        "y": safe_float(y_val) if y_val is not None else None,
                        "z": safe_float(z_val) if z_val is not None else None,
                    }
                    # Optional fields that may not exist in face landmarks
                    if hasattr(lm, "visibility"):
                        v_val = getattr(lm, "visibility", None)
                        lm_data["visibility"] = (
                            safe_float(v_val) if v_val is not None else None
                        )
                    if hasattr(lm, "presence"):
                        p_val = getattr(lm, "presence", None)
                        lm_data["presence"] = (
                            safe_float(p_val) if p_val is not None else None
                        )
                    result.append(lm_data)
                return result
        except Exception as e:
            print(f"Error serializing landmarks: {e}")
        return None

    def serialize_matrix(matrix):
        if matrix is None:
            return None
        try:
            if hasattr(matrix, "flatten"):
                return [float(x) for x in matrix.flatten()]
            elif hasattr(matrix, "__iter__"):
                flat = []
                for row in matrix:
                    if hasattr(row, "__iter__"):
                        flat.extend([float(x) for x in row])
                    else:
                        flat.append(float(row))
                return flat
        except Exception:
            pass
        return None

    def extract_eye_geometry(face_landmarks):
        def get_point(idx):
            if (
                face_landmarks is None
                or idx < 0
                or idx >= len(face_landmarks)
                or face_landmarks[idx] is None
            ):
                return None
            point = face_landmarks[idx]
            x = safe_float(point.get("x"), float("nan"))
            y = safe_float(point.get("y"), float("nan"))
            z = safe_float(point.get("z"), float("nan"))
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                return None
            return {
                "x": x,
                "y": y,
                "z": z,
            }

        def get_xy(idx):
            point = get_point(idx)
            if point is None:
                return None
            return point["x"], point["y"]

        def nose_reference():
            bridge = get_xy(NOSE_BRIDGE_IDX)
            base = get_xy(NOSE_BASE_IDX)
            left_iris = get_xy(LEFT_IRIS_CENTER_IDX)
            right_iris = get_xy(RIGHT_IRIS_CENTER_IDX)
            if (
                bridge is None
                or base is None
                or left_iris is None
                or right_iris is None
            ):
                return None

            ref = _nose_plane_reference(bridge, base, left_iris, right_iris)
            if ref is None:
                return None

            return {
                "bridge": get_point(NOSE_BRIDGE_IDX),
                "base": get_point(NOSE_BASE_IDX),
                "bridgeToBaseUnit": {"x": ref["nHatX"], "y": ref["nHatY"]},
                "perpendicularAtBridgeUnit": {"x": ref["pHatX"], "y": ref["pHatY"]},
                "bridgeToBaseLengthNorm": ref["axisLength"],
                "eyeSpanNorm": ref["eyeSpan"],
                "leftIrisToPerpendicularSignedNorm": ref["leftSigned"],
                "rightIrisToPerpendicularSignedNorm": ref["rightSigned"],
                "avgIrisToPerpendicularSignedNorm": ref["avgSigned"],
                "leftIrisToPerpendicularByEyeSpan": ref["leftSigned"] / ref["eyeSpan"],
                "rightIrisToPerpendicularByEyeSpan": ref["rightSigned"]
                / ref["eyeSpan"],
                "avgIrisToPerpendicularByEyeSpan": ref["avgSigned"] / ref["eyeSpan"],
                "leftIrisPerpendicularFoot": {
                    "x": ref["leftProjX"],
                    "y": ref["leftProjY"],
                },
                "rightIrisPerpendicularFoot": {
                    "x": ref["rightProjX"],
                    "y": ref["rightProjY"],
                },
            }

        geometry = {
            "leftEye": {
                "irisCenter": get_point(LEFT_IRIS_CENTER_IDX),
                "innerCanthus": get_point(133),
                "outerCanthus": get_point(33),
                "upperEyelid": get_point(159),
                "lowerEyelid": get_point(145),
            },
            "rightEye": {
                "irisCenter": get_point(RIGHT_IRIS_CENTER_IDX),
                "innerCanthus": get_point(362),
                "outerCanthus": get_point(263),
                "upperEyelid": get_point(386),
                "lowerEyelid": get_point(374),
            },
        }
        geometry["noseReference"] = nose_reference()
        return geometry

    def serialize_blendshapes(blendshapes):
        if blendshapes is None:
            return None
        try:
            if hasattr(blendshapes, "__iter__") and not isinstance(
                blendshapes, (str, bytes)
            ):
                return [
                    {
                        "category": str(bs.category)
                        if hasattr(bs, "category")
                        else None,
                        "score": float(bs.score) if hasattr(bs, "score") else None,
                    }
                    for bs in blendshapes
                ]
        except Exception:
            pass
        return None

    data = {}

    # Get facial transformation matrixes
    if hasattr(result, "facial_transformation_matrixes"):
        fts = result.facial_transformation_matrixes
        if fts and len(fts) > 0:
            data["facial_transformation_matrix"] = serialize_matrix(fts[0])

    # Get face landmarks
    if hasattr(result, "face_landmarks"):
        fl = result.face_landmarks
        if fl and len(fl) > 0:
            # MediaPipe returns a list, one per face detected
            data["face_landmarks"] = serialize_landmarks(fl[0])
            data["eye_geometry"] = extract_eye_geometry(data["face_landmarks"])
        else:
            print(f"Warning: No face landmarks found in result")

    # Get face blendshapes
    if hasattr(result, "face_blendshapes"):
        fbs = result.face_blendshapes
        if fbs and len(fbs) > 0:
            data["face_blendshapes"] = serialize_blendshapes(fbs[0])

    return data


def _landmark_to_px(landmark, width: int, height: int) -> Tuple[int, int]:
    x = int(round(safe_float(getattr(landmark, "x", 0.0)) * width))
    y = int(round(safe_float(getattr(landmark, "y", 0.0)) * height))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def _draw_eye_landmarks(frame, result) -> Optional[Dict[str, float]]:
    if result is None or not getattr(result, "face_landmarks", None):
        return None
    face_landmarks = result.face_landmarks[0]
    if face_landmarks is None:
        return None

    h, w = frame.shape[:2]

    def draw_index(idx: int, color: Tuple[int, int, int], radius: int) -> None:
        if idx < 0 or idx >= len(face_landmarks):
            return
        point = _landmark_to_px(face_landmarks[idx], w, h)
        cv2.circle(frame, point, radius + 1, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

    for idx in LEFT_EYE_LANDMARKS:
        draw_index(idx, (0, 165, 255), 3 if idx == 468 else 2)
    for idx in RIGHT_EYE_LANDMARKS:
        draw_index(idx, (255, 220, 40), 3 if idx == 473 else 2)

    def get_px(idx: int):
        if idx < 0 or idx >= len(face_landmarks):
            return None
        return _landmark_to_px(face_landmarks[idx], w, h)

    bridge = get_px(NOSE_BRIDGE_IDX)
    base = get_px(NOSE_BASE_IDX)
    left_iris = get_px(LEFT_IRIS_CENTER_IDX)
    right_iris = get_px(RIGHT_IRIS_CENTER_IDX)
    if bridge is None or base is None:
        return None

    cv2.circle(frame, bridge, 6, WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, bridge, 4, RED, -1, cv2.LINE_AA)
    cv2.circle(frame, base, 6, WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, base, 4, YELLOW, -1, cv2.LINE_AA)

    if left_iris is None or right_iris is None:
        return None

    bridge_f = (float(bridge[0]), float(bridge[1]))
    base_f = (float(base[0]), float(base[1]))
    left_f = (float(left_iris[0]), float(left_iris[1]))
    right_f = (float(right_iris[0]), float(right_iris[1]))
    ref = _nose_plane_reference(bridge_f, base_f, left_f, right_f)
    if ref is None:
        return None

    cv2.line(frame, bridge, base, WHITE, 4, cv2.LINE_AA)
    cv2.line(frame, bridge, base, RED, 2, cv2.LINE_AA)

    half_len = int(max(90.0, ref["eyeSpan"] * 2.5))
    p1 = (
        int(round(bridge_f[0] - ref["pHatX"] * half_len)),
        int(round(bridge_f[1] - ref["pHatY"] * half_len)),
    )
    p2 = (
        int(round(bridge_f[0] + ref["pHatX"] * half_len)),
        int(round(bridge_f[1] + ref["pHatY"] * half_len)),
    )
    cv2.line(frame, p1, p2, WHITE, 4, cv2.LINE_AA)
    cv2.line(frame, p1, p2, GREEN, 2, cv2.LINE_AA)

    left_proj = (int(round(ref["leftProjX"])), int(round(ref["leftProjY"])))
    right_proj = (int(round(ref["rightProjX"])), int(round(ref["rightProjY"])))
    cv2.line(frame, left_iris, left_proj, WHITE, 3, cv2.LINE_AA)
    cv2.line(frame, left_iris, left_proj, (0, 165, 255), 1, cv2.LINE_AA)
    cv2.line(frame, right_iris, right_proj, WHITE, 3, cv2.LINE_AA)
    cv2.line(frame, right_iris, right_proj, (255, 220, 40), 1, cv2.LINE_AA)
    cv2.circle(frame, left_proj, 3, WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, right_proj, 3, WHITE, -1, cv2.LINE_AA)

    axis_angle = math.degrees(math.atan2(ref["nHatY"], ref["nHatX"]))

    return {
        "leftSignedPx": ref["leftSigned"],
        "rightSignedPx": ref["rightSigned"],
        "avgSignedPx": ref["avgSigned"],
        "bridgeToBasePx": ref["axisLength"],
        "eyeSpanPx": ref["eyeSpan"],
        "avgByEyeSpan": ref["avgSigned"] / ref["eyeSpan"],
        "axisAngleDeg": axis_angle,
    }


@dataclass
class PitchCorrelationPoint:
    """Data captured for a single pitch correlation point."""

    name: str
    instruction: str
    head_position: str  # 'down', 'center', 'up'
    eye_position: str  # 'down', 'center', 'up'
    timestamp_ms: int
    raw_result: Dict[str, Any]  # Serialized MediaPipe result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "instruction": self.instruction,
            "headPosition": self.head_position,
            "eyePosition": self.eye_position,
            "timestampMs": self.timestamp_ms,
            "rawResult": self.raw_result,
        }


class PitchCorrelationCapture:
    """Main pitch correlation capture class."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.landmarker = None
        self.captured_data: List[PitchCorrelationPoint] = []
        self.current_prompt_index = 0
        self.running = False
        self.mouse_clicked = False
        self.mouse_pos = (0, 0)
        self.last_result = None
        self.last_nose_reference = None

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def init_camera(self):
        """Initialize camera."""
        self.cap, self.camera_info = open_camera(self.camera_index)

    def init_landmarker(self):
        """Initialize MediaPipe FaceLandmarker."""
        ensure_model()
        base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        print("FaceLandmarker initialized")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked = True
            self.mouse_pos = (x, y)

    def draw_ui(self, frame, prompt: Dict[str, Any], progress: int, total: int):
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]

        # Semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 220), HUD_BG, -1)
        frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

        # Progress indicator
        progress_text = f"Progress: {progress + 1}/{total}"
        draw_text_with_background(frame, progress_text, (30, 40), font_scale=0.8)

        # Parse head and eye positions from name
        name_parts = prompt["name"].split("-")
        head_pos = name_parts[1].upper()
        eye_pos = name_parts[3].upper()

        # Position indicators
        pos_text = f"Head: {head_pos} | Eyes: {eye_pos}"
        cv2.putText(
            frame,
            pos_text,
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            YELLOW,
            2,
            cv2.LINE_AA,
        )

        # Main instruction
        instruction = prompt["instruction"]
        draw_text_with_background(
            frame, instruction, (30, 115), font_scale=1.0, bg_color=(60, 60, 80)
        )

        # Type indicator
        type_text = "PITCH CORRELATION TEST"
        cv2.putText(
            frame,
            type_text,
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            GREEN,
            2,
            cv2.LINE_AA,
        )

        # Click instruction
        click_text = "CLICK anywhere or press SPACE to capture"
        cv2.putText(
            frame,
            click_text,
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            1,
            cv2.LINE_AA,
        )

        # Face detection indicator
        if hasattr(self, "last_result") and self.last_result:
            has_face = bool(
                self.last_result.face_landmarks
                and len(self.last_result.face_landmarks) > 0
            )
            face_text = "Face: DETECTED" if has_face else "Face: NOT DETECTED"
            face_color = GREEN if has_face else RED
            cv2.putText(
                frame,
                face_text,
                (w - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                face_color,
                2,
                cv2.LINE_AA,
            )

        if isinstance(self.last_nose_reference, dict):
            left_h = safe_float(self.last_nose_reference.get("leftSignedPx"), 0.0)
            right_h = safe_float(self.last_nose_reference.get("rightSignedPx"), 0.0)
            avg_h = safe_float(self.last_nose_reference.get("avgSignedPx"), 0.0)
            norm_h = safe_float(self.last_nose_reference.get("avgByEyeSpan"), 0.0)
            axis_deg = safe_float(self.last_nose_reference.get("axisAngleDeg"), 0.0)
            x0 = max(30, w - 620)
            cv2.putText(
                frame,
                f"Iris->perp signed(px) L:{left_h:+.1f} R:{right_h:+.1f} A:{avg_h:+.1f}",
                (x0, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                YELLOW,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Norm(A/eyeSpan): {norm_h:+.3f}   Nose-axis: {axis_deg:+.1f} deg",
                (x0, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                WHITE,
                1,
                cv2.LINE_AA,
            )

        return frame

    def capture_point(self, prompt: Dict[str, Any]) -> Optional[PitchCorrelationPoint]:
        """Capture a single pitch correlation point."""
        self.mouse_clicked = False

        # Parse positions from name
        name_parts = prompt["name"].split("-")
        head_position = name_parts[1]
        eye_position = name_parts[3]

        print(f"\n{'=' * 60}")
        print(f"CAPTURE {self.current_prompt_index + 1}/{len(PROMPTS)}")
        print(f"Instruction: {prompt['instruction']}")
        print(f"Head position: {head_position}")
        print(f"Eye position: {eye_position}")
        print(f"Click or press SPACE to capture...")
        print(f"{'=' * 60}\n")

        cv2.namedWindow("Pitch Correlation Capture")
        cv2.setMouseCallback("Pitch Correlation Capture", self.mouse_callback)

        while self.running:
            # Read frame
            ok, frame_bgr = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            result = detect_face_landmarker_result(self.landmarker, frame_bgr)
            self.last_result = result

            frame_with_ui = frame_bgr.copy()
            self.last_nose_reference = _draw_eye_landmarks(frame_with_ui, result)

            frame_with_ui = self.draw_ui(
                frame_with_ui, prompt, self.current_prompt_index, len(PROMPTS)
            )
            frame_with_ui = finalize_ui_frame(frame_with_ui)

            cv2.imshow("Pitch Correlation Capture", frame_with_ui)

            # Check for capture trigger
            key = cv2.waitKey(1) & 0xFF
            if self.mouse_clicked or key == ord(" "):
                # Capture the point
                print(f"Captured: {prompt['name']}")

                # Serialize the result
                raw_data = serialize_mediapipe_result(result)

                # Create pitch correlation point
                point = PitchCorrelationPoint(
                    name=prompt["name"],
                    instruction=prompt["instruction"],
                    head_position=head_position,
                    eye_position=eye_position,
                    timestamp_ms=int(time.time() * 1000),
                    raw_result=raw_data,
                )

                # Also save individual JSON file
                json_file = OUTPUT_DIR / f"{prompt['name']}.json"
                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(point.to_dict(), f, indent=2)

                print(f"Saved: {json_file}")

                # Save a screenshot too
                screenshot_file = OUTPUT_DIR / f"{prompt['name']}.png"
                cv2.imwrite(str(screenshot_file), frame_with_ui)
                print(f"Screenshot saved: {screenshot_file}")

                cv2.waitKey(500)  # Brief pause to show feedback
                break

            if key == ord("q") or key == 27:  # q or ESC
                print("User cancelled")
                self.running = False
                return None

        return point

    def run(self):
        """Run the pitch correlation capture session."""
        print("\n" + "=" * 60)
        print("PITCH CORRELATION CAPTURE")
        print("=" * 60)
        print("\nThis script will capture FaceMesh data for 9 different")
        print("combinations of head and eye positions to analyze")
        print("eye pitch consistency across head positions.")
        print("\nTest pattern:")
        print("  3 head positions (DOWN, CENTER, UP)")
        print("  x 3 eye positions (DOWN, CENTER, UP)")
        print("  = 9 capture points")
        print("\nInstructions:")
        print("- Follow each prompt carefully")
        print("- Keep the position steady when instructed")
        print("- Click anywhere or press SPACE to capture")
        print("- Press 'q' or ESC to quit early")
        print("\n" + "=" * 60 + "\n")

        # Initialize
        self.init_camera()
        self.init_landmarker()
        self.running = True

        try:
            # Capture each prompt
            for i, prompt in enumerate(PROMPTS):
                self.current_prompt_index = i

                if not self.running:
                    break

                point = self.capture_point(prompt)
                if point:
                    self.captured_data.append(point)

            # Save all data to a combined file
            if self.captured_data:
                combined_file = OUTPUT_DIR / "pitch_correlation_combined.json"
                combined_data = {
                    "timestamp": int(time.time() * 1000),
                    "captureCount": len(self.captured_data),
                    "cameraInfo": self.camera_info,
                    "points": [p.to_dict() for p in self.captured_data],
                }

                with combined_file.open("w", encoding="utf-8") as f:
                    json.dump(combined_data, f, indent=2)

                print(f"\nCombined data saved: {combined_file}")

            print(f"\n{'=' * 60}")
            print(f"Pitch correlation capture complete!")
            print(f"Captured {len(self.captured_data)} points")
            print(f"Data saved to: {OUTPUT_DIR.absolute()}")
            print(f"{'=' * 60}\n")

        finally:
            # Cleanup
            cv2.destroyAllWindows()
            if self.cap is not None:
                self.cap.release()
            if self.landmarker is not None:
                self.landmarker.close()


def main():
    parser = argparse.ArgumentParser(
        description="Pitch correlation capture for FaceMesh eye pitch consistency analysis"
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Camera index (default: 0)"
    )
    args = parser.parse_args()

    capture = PitchCorrelationCapture(camera_index=args.camera_index)
    capture.run()


if __name__ == "__main__":
    main()
