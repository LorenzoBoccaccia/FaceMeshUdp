"""
Capture module for FaceMesh application.
Handles mesh data capture, screenshot generation, and test data saving.
"""

import json
import math
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from .facemesh_dao import (
    FaceMeshEvent,
    safe_float,
    clamp,
    LEFT_IRIS_CENTER_IDX,
    RIGHT_IRIS_CENTER_IDX,
    LEFT_IRIS_RING_IDXS,
    RIGHT_IRIS_RING_IDXS,
    LEFT_EYE_KEY_IDXS,
    RIGHT_EYE_KEY_IDXS,
)



# Constants
CAPTURE_DIR = Path("captures")
FACE_MESH_CONNECTIONS = list(vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)

# Colors
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (80, 230, 120)
ORANGE = (0, 165, 255)
CYAN = (255, 220, 40)
MAGENTA = (255, 80, 255)
YELLOW = (0, 255, 255)
HUD_BG = (20, 20, 20)
HUD_BORDER = (230, 230, 230)
HUD_TEXT = (245, 245, 245)


def reset_capture_dir():
    """Reset capture directory by clearing all contents."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    for p in CAPTURE_DIR.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def ms_now():
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def _lm_to_px(lm, w, h, mirror_x: bool = False):
    """Convert landmark to pixel coordinates."""
    if hasattr(lm, "x") and hasattr(lm, "y"):
        lx = safe_float(getattr(lm, "x", 0.0), 0.0)
        ly = safe_float(getattr(lm, "y", 0.0), 0.0)
    elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
        lx = safe_float(lm[0], 0.0)
        ly = safe_float(lm[1], 0.0)
    else:
        lx, ly = 0.0, 0.0
    if mirror_x:
        lx = 1.0 - lx
    x = int(clamp(round(lx * w), 0, w - 1))
    y = int(clamp(round(ly * h), 0, h - 1))
    return x, y


def _fmt_num(value: Any, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    v = safe_float(value, float("nan"))
    if v != v:  # NaN
        return "n/a"
    return f"{v:.{precision}f}"


def _safe_lm_xy(lm) -> Optional[Tuple[float, float]]:
    if hasattr(lm, "x") and hasattr(lm, "y"):
        return safe_float(getattr(lm, "x", 0.0), 0.0), safe_float(getattr(lm, "y", 0.0), 0.0)
    if isinstance(lm, (list, tuple)) and len(lm) >= 2:
        return safe_float(lm[0], 0.0), safe_float(lm[1], 0.0)
    return None


def _lm_points_px(landmarks: List[Any], indices: tuple[int, ...], w: int, h: int, mirror_x: bool) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for idx in indices:
        if 0 <= int(idx) < len(landmarks):
            points.append(_lm_to_px(landmarks[int(idx)], w, h, mirror_x=mirror_x))
    return points


def _draw_points(img, points: List[Tuple[int, int]], color, radius: int = 2) -> None:
    for p in points:
        cv2.circle(img, p, radius + 1, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)


def _draw_ring(img, points: List[Tuple[int, int]], color) -> None:
    if len(points) < 2:
        return
    arr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [arr], True, WHITE, 3, cv2.LINE_AA)
    cv2.polylines(img, [arr], True, color, 1, cv2.LINE_AA)


def _center_px(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not points:
        return None
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    n = max(1, len(points))
    return int(round(sx / n)), int(round(sy / n))


def _draw_normal_arrow(img, origin: Tuple[int, int], normal: Any, color, length: int) -> None:
    if not isinstance(normal, (list, tuple)) or len(normal) < 3:
        return
    nx = safe_float(normal[0], float("nan"))
    ny = safe_float(normal[1], float("nan"))
    if not (math.isfinite(nx) and math.isfinite(ny)):
        return
    ox, oy = int(origin[0]), int(origin[1])
    dx = nx * float(length)
    dy = (-ny) * float(length)
    raw_len = math.hypot(dx, dy)
    min_vis = max(10.0, float(length) * 0.35)
    # Avoid amplifying near-zero jitter into misleading opposite arrows.
    min_scale_enable = max(1.5, float(length) * 0.06)
    if raw_len >= min_scale_enable and raw_len < min_vis:
        s = min_vis / raw_len
        dx *= s
        dy *= s
    ex = int(round(ox + dx))
    ey = int(round(oy + dy))
    cv2.arrowedLine(img, (ox, oy), (ex, ey), WHITE, 4, cv2.LINE_AA, tipLength=0.24)
    cv2.arrowedLine(img, (ox, oy), (ex, ey), color, 2, cv2.LINE_AA, tipLength=0.24)


def _draw_fitted_ellipse(img, ellipse_payload: Any, color) -> None:
    if not isinstance(ellipse_payload, dict):
        return
    if not bool(ellipse_payload.get("ok")):
        return
    e = ellipse_payload.get("ellipse")
    if not isinstance(e, dict):
        return
    center = e.get("centerPx")
    axes = e.get("axesPx")
    angle = e.get("angleDeg")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return
    if not isinstance(axes, dict):
        return
    if not isinstance(angle, dict):
        return
    major = safe_float(axes.get("major"), float("nan"))
    minor = safe_float(axes.get("minor"), float("nan"))
    major_angle = safe_float(angle.get("major"), float("nan"))
    if not (math.isfinite(major) and math.isfinite(minor) and math.isfinite(major_angle)):
        return
    if major <= 1e-6 or minor <= 1e-6:
        return
    cx = int(round(safe_float(center[0], 0.0)))
    cy = int(round(safe_float(center[1], 0.0)))
    ax = max(1, int(round(major * 0.5)))
    ay = max(1, int(round(minor * 0.5)))
    cv2.ellipse(img, (cx, cy), (ax, ay), float(major_angle), 0, 360, WHITE, 3, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (ax, ay), float(major_angle), 0, 360, color, 1, cv2.LINE_AA)


def _build_event_lines(snap_evt: Any, landmarks: List[Any]) -> List[str]:
    lines: List[str] = []
    if isinstance(snap_evt, FaceMeshEvent):
        lines.append(f"type={snap_evt.type} ts={snap_evt.ts}")
        lines.append(f"face={snap_evt.has_face} landmarks={snap_evt.landmark_count}")
        lines.append(
            "head y/p/r="
            f"{_fmt_num(snap_evt.head_yaw, 2)}/{_fmt_num(snap_evt.head_pitch, 2)}/{_fmt_num(snap_evt.roll, 2)}"
        )
        lines.append(
            "tx/ty/tz="
            f"{_fmt_num(snap_evt.x, 4)}/{_fmt_num(snap_evt.y, 4)}/{_fmt_num(snap_evt.z, 4)}"
        )

        mask_meta = snap_evt.face_mask_segment_meta()
        if mask_meta:
            mask_shape = mask_meta.get("shape", "n/a")
            mask_dtype = mask_meta.get("dtype", "n/a")
            lines.append(f"mask type={mask_meta.get('type', 'n/a')} shape={mask_shape} dtype={mask_dtype}")
        else:
            lines.append("mask: none")

        blendshape_map = snap_evt.blendshapes_as_dict() or {}
        lines.append(f"blendshapes={len(blendshape_map)}")
        if blendshape_map:
            top_blends = sorted(blendshape_map.items(), key=lambda kv: kv[1], reverse=True)[:8]
            for name, score in top_blends:
                lines.append(f"bs {name}={_fmt_num(score, 4)}")

        transform = snap_evt.transform_matrix_as_flat() or []
        lines.append(f"transform values={len(transform)}")
        if len(transform) >= 16:
            for r in range(4):
                row = transform[r * 4:(r + 1) * 4]
                lines.append("m" + str(r) + ": " + " ".join(_fmt_num(v, 4) for v in row))

        eyes = snap_evt.eyes_dict()
        left_center = eyes.get("leftIrisCenter")
        right_center = eyes.get("rightIrisCenter")
        if left_center is not None and right_center is not None:
            lines.append(
                "irisCtr L(x,y,z)="
                f"{_fmt_num(left_center[0], 4)},{_fmt_num(left_center[1], 4)},{_fmt_num(left_center[2], 4)} "
                "R(x,y,z)="
                f"{_fmt_num(right_center[0], 4)},{_fmt_num(right_center[1], 4)},{_fmt_num(right_center[2], 4)}"
            )
            lines.append(
                "gazeYawPitch L="
                f"{_fmt_num(eyes.get('leftEyeGazeYaw'), 2)}/{_fmt_num(eyes.get('leftEyeGazePitch'), 2)} "
                "R="
                f"{_fmt_num(eyes.get('rightEyeGazeYaw'), 2)}/{_fmt_num(eyes.get('rightEyeGazePitch'), 2)}"
            )
    elif isinstance(snap_evt, dict):
        lines.append(f"type={snap_evt.get('type', 'mesh')}")
        lines.append(f"face={bool(snap_evt.get('hasFace'))} landmarks={len(landmarks)}")
        blendshapes = snap_evt.get("blendshapes")
        if isinstance(blendshapes, dict):
            lines.append(f"blendshapes={len(blendshapes)}")
        transform = snap_evt.get("transformMatrix")
        if isinstance(transform, list):
            lines.append(f"transform values={len(transform)}")
        eyes = snap_evt.get("eyes")
        if isinstance(eyes, dict):
            if isinstance(eyes.get("leftIrisCenter"), list) and isinstance(eyes.get("rightIrisCenter"), list):
                lc = eyes.get("leftIrisCenter")
                rc = eyes.get("rightIrisCenter")
                lines.append(
                    "irisCtr L(x,y,z)="
                    f"{_fmt_num(lc[0], 4)},{_fmt_num(lc[1], 4)},{_fmt_num(lc[2], 4)} "
                    "R(x,y,z)="
                    f"{_fmt_num(rc[0], 4)},{_fmt_num(rc[1], 4)},{_fmt_num(rc[2], 4)}"
                )
            l_gaze_yaw = eyes.get("leftEyeGazeYaw")
            l_gaze_pitch = eyes.get("leftEyeGazePitch")
            r_gaze_yaw = eyes.get("rightEyeGazeYaw")
            r_gaze_pitch = eyes.get("rightEyeGazePitch")
            if l_gaze_yaw is not None or l_gaze_pitch is not None or r_gaze_yaw is not None or r_gaze_pitch is not None:
                lines.append(
                    "gazeYawPitch L="
                    f"{_fmt_num(l_gaze_yaw, 2)}/{_fmt_num(l_gaze_pitch, 2)} "
                    "R="
                    f"{_fmt_num(r_gaze_yaw, 2)}/{_fmt_num(r_gaze_pitch, 2)}"
                )
    else:
        lines.append("event: none")
        lines.append(f"landmarks={len(landmarks)}")

    if len(landmarks) > 473:
        lxy = _safe_lm_xy(landmarks[468])
        rxy = _safe_lm_xy(landmarks[473])
        if lxy and rxy:
            lines.append(
                "iris L(x,y)="
                f"{_fmt_num(lxy[0], 4)},{_fmt_num(lxy[1], 4)} "
                "R(x,y)="
                f"{_fmt_num(rxy[0], 4)},{_fmt_num(rxy[1], 4)}"
            )

    return lines


def _draw_info_panel(img, lines: List[str]) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    thickness = 1
    pad = 8
    line_h = 18
    max_lines = max(1, (img.shape[0] - 24) // line_h)
    draw_lines = lines[:max_lines]
    if len(lines) > max_lines:
        draw_lines[-1] = f"... +{len(lines) - max_lines + 1} more"

    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in draw_lines]
    box_w = min(max(widths) + pad * 2, img.shape[1] - 20) if widths else min(260, img.shape[1] - 20)
    box_h = line_h * len(draw_lines) + pad * 2
    x0, y0 = 10, 10
    x1, y1 = x0 + box_w, y0 + box_h

    cv2.rectangle(img, (x0, y0), (x1, y1), HUD_BG, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), HUD_BORDER, 1, cv2.LINE_AA)

    for i, line in enumerate(draw_lines):
        tx = x0 + pad
        ty = y0 + pad + (i + 1) * line_h - 4
        cv2.putText(img, line, (tx + 1, ty + 1), font, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (tx, ty), font, scale, HUD_TEXT, thickness, cv2.LINE_AA)


def _draw_face_direction_from_ypr(
    img,
    origin: Tuple[int, int],
    yaw_deg: Any,
    pitch_deg: Any,
    roll_deg: Any,
) -> None:
    """Draw face forward vector from yaw/pitch/roll at given origin."""
    yaw = safe_float(yaw_deg, float("nan"))
    pitch = safe_float(pitch_deg, float("nan"))
    roll = safe_float(roll_deg, float("nan"))
    if not (math.isfinite(yaw) and math.isfinite(pitch)):
        return

    yaw_r = math.radians(yaw)
    pitch_r = math.radians(pitch)

    # Reconstruct normalized face-forward direction from yaw/pitch conventions.
    fx = math.sin(yaw_r)
    fy = -math.sin(pitch_r)
    fz = -math.cos(yaw_r) * math.cos(pitch_r)
    mag = math.sqrt(fx * fx + fy * fy + fz * fz)
    if mag <= 1e-9:
        return
    fx /= mag
    fy /= mag

    h, w = img.shape[:2]
    length = int(max(36, min(w, h) * 0.16))
    ox, oy = int(origin[0]), int(origin[1])
    dx = fx * float(length)
    dy = fy * float(length)
    raw_len = math.hypot(dx, dy)
    min_vis = max(14.0, float(length) * 0.3)
    if raw_len > 1e-6 and raw_len < min_vis:
        s = min_vis / raw_len
        dx *= s
        dy *= s
    ex = int(round(ox + dx))
    ey = int(round(oy + dy))

    cv2.arrowedLine(img, (ox, oy), (ex, ey), WHITE, 4, cv2.LINE_AA, tipLength=0.22)
    cv2.arrowedLine(img, (ox, oy), (ex, ey), CYAN, 2, cv2.LINE_AA, tipLength=0.22)

    # Roll rotates around the forward axis; show it as a short nose-centered tick.
    if math.isfinite(roll):
        rr = math.radians(roll)
        tx = math.cos(rr)
        ty = -math.sin(rr)
        tick_len = max(10, int(length * 0.25))
        p1 = (int(round(ox - tx * tick_len * 0.5)), int(round(oy - ty * tick_len * 0.5)))
        p2 = (int(round(ox + tx * tick_len * 0.5)), int(round(oy + ty * tick_len * 0.5)))
        cv2.line(img, p1, p2, WHITE, 3, cv2.LINE_AA)
        cv2.line(img, p1, p2, ORANGE, 1, cv2.LINE_AA)


def render_camera_capture_marked(
    png_path: str,
    snap: Dict,
    overlay_w: float,
    overlay_h: float,
    click_pos: Tuple[float, float],
    draw_click: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Render camera frame with face mesh landmarks and save to file."""
    img, err = build_camera_capture_marked_image(
        snap,
        overlay_w=overlay_w,
        overlay_h=overlay_h,
        click_pos=click_pos,
        draw_click=draw_click,
    )
    if img is None:
        return False, err or "No camera frame available yet."
    ok = cv2.imwrite(str(png_path), img)
    return (ok, None) if ok else (False, "cv2.imwrite failed")


def build_camera_capture_marked_image(
    snap: Dict,
    overlay_w: float,
    overlay_h: float,
    click_pos: Tuple[float, float],
    draw_click: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Build marked camera frame with face mesh, ovals, and vectors."""
    frame = snap.get("frame")
    if frame is None:
        return None, "No camera frame available yet."

    mirror_view = True
    img = cv2.flip(frame, 1) if mirror_view else frame.copy()
    fh, fw = img.shape[:2]
    landmarks = snap.get("landmarks") or []
    snap_evt = snap.get("evt")

    # Draw mesh connections first, then points for clarity.
    if landmarks:
        for conn in FACE_MESH_CONNECTIONS:
            a = int(conn.start)
            b = int(conn.end)
            if a >= len(landmarks) or b >= len(landmarks):
                continue
            p1 = _lm_to_px(landmarks[a], fw, fh, mirror_x=mirror_view)
            p2 = _lm_to_px(landmarks[b], fw, fh, mirror_x=mirror_view)
            cv2.line(img, p1, p2, (45, 140, 45), 1, cv2.LINE_AA)

        for lm in landmarks:
            cv2.circle(img, _lm_to_px(lm, fw, fh, mirror_x=mirror_view), 1, GREEN, -1, cv2.LINE_AA)

    # Draw explicit eye geometry from raw landmarks.
    left_iris_ring = _lm_points_px(landmarks, LEFT_IRIS_RING_IDXS, fw, fh, mirror_x=mirror_view)
    right_iris_ring = _lm_points_px(landmarks, RIGHT_IRIS_RING_IDXS, fw, fh, mirror_x=mirror_view)
    left_eye_keys = _lm_points_px(landmarks, LEFT_EYE_KEY_IDXS, fw, fh, mirror_x=mirror_view)
    right_eye_keys = _lm_points_px(landmarks, RIGHT_EYE_KEY_IDXS, fw, fh, mirror_x=mirror_view)

    _draw_ring(img, left_iris_ring, ORANGE)
    _draw_ring(img, right_iris_ring, CYAN)
    _draw_points(img, left_eye_keys, ORANGE, radius=2)
    _draw_points(img, right_eye_keys, CYAN, radius=2)

    if 0 <= LEFT_IRIS_CENTER_IDX < len(landmarks):
        left_center = _lm_to_px(landmarks[LEFT_IRIS_CENTER_IDX], fw, fh, mirror_x=mirror_view)
    else:
        left_center = _center_px(left_iris_ring)
    if 0 <= RIGHT_IRIS_CENTER_IDX < len(landmarks):
        right_center = _lm_to_px(landmarks[RIGHT_IRIS_CENTER_IDX], fw, fh, mirror_x=mirror_view)
    else:
        right_center = _center_px(right_iris_ring)
    if left_center is not None:
        cv2.circle(img, left_center, 5, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, left_center, 3, MAGENTA, -1, cv2.LINE_AA)
    if right_center is not None:
        cv2.circle(img, right_center, 5, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, right_center, 3, MAGENTA, -1, cv2.LINE_AA)

    if isinstance(snap_evt, FaceMeshEvent):
        eyes = snap_evt.eyes_dict()
        l_gaze_yaw = eyes.get("leftEyeGazeYaw")
        l_gaze_pitch = eyes.get("leftEyeGazePitch")
        r_gaze_yaw = eyes.get("rightEyeGazeYaw")
        r_gaze_pitch = eyes.get("rightEyeGazePitch")
        l_n = None
        r_n = None
        l_e = None
        r_e = None
        gaze = None
    elif isinstance(snap_evt, dict):
        eyes = snap_evt.get("eyes") or {}
        if isinstance(eyes, dict):
            l_gaze_yaw = eyes.get("leftEyeGazeYaw")
            l_gaze_pitch = eyes.get("leftEyeGazePitch")
            r_gaze_yaw = eyes.get("rightEyeGazeYaw")
            r_gaze_pitch = eyes.get("rightEyeGazePitch")
            l_n = None
            r_n = None
            l_e = None
            r_e = None
            gaze = None
        else:
            l_gaze_yaw = None
            l_gaze_pitch = None
            r_gaze_yaw = None
            r_gaze_pitch = None
            l_n = None
            r_n = None
            l_e = None
            r_e = None
            gaze = None
    else:
        l_gaze_yaw = None
        l_gaze_pitch = None
        r_gaze_yaw = None
        r_gaze_pitch = None
        l_n = None
        r_n = None
        l_e = None
        r_e = None
        gaze = None



    _draw_fitted_ellipse(img, l_e, ORANGE)
    _draw_fitted_ellipse(img, r_e, CYAN)
    if left_center is not None:
        _draw_normal_arrow(img, left_center, l_n, ORANGE, max(18, int(min(fw, fh) * 0.06)))
    if right_center is not None:
        _draw_normal_arrow(img, right_center, r_n, CYAN, max(18, int(min(fw, fh) * 0.06)))

    # Get nose position for average gaze vector origin.
    nose = _lm_to_px(landmarks[1], fw, fh, mirror_x=mirror_view) if len(landmarks) > 1 else (fw // 2, fh // 2)

    # Draw gaze vectors from yaw/pitch for each eye from pupil (iris center).
    def _vector_from_yaw_pitch(yaw_deg, pitch_deg):
        """Convert yaw/pitch to 3D direction vector."""
        yaw = safe_float(yaw_deg, float("nan"))
        pitch = safe_float(pitch_deg, float("nan"))
        if not (math.isfinite(yaw) and math.isfinite(pitch)):
            return None
        yaw_r = math.radians(yaw)
        pitch_r = math.radians(pitch)
        vx = math.sin(yaw_r) * math.cos(pitch_r)
        vy = -math.sin(pitch_r)
        vz = -math.cos(yaw_r) * math.cos(pitch_r)
        return [vx, vy, vz]

    l_gaze_vec = _vector_from_yaw_pitch(l_gaze_yaw, l_gaze_pitch)
    r_gaze_vec = _vector_from_yaw_pitch(r_gaze_yaw, r_gaze_pitch)


    if left_center is not None and l_gaze_vec is not None:
        _draw_normal_arrow(img, left_center, l_gaze_vec, YELLOW, max(24, int(min(fw, fh) * 0.08)))
    if right_center is not None and r_gaze_vec is not None:
        _draw_normal_arrow(img, right_center, r_gaze_vec, YELLOW, max(24, int(min(fw, fh) * 0.08)))

    # Draw average gaze vector from nose.
    if l_gaze_vec is not None and r_gaze_vec is not None:
        avg_gaze_vec = [
            (l_gaze_vec[0] + r_gaze_vec[0]) * 0.5,
            (l_gaze_vec[1] + r_gaze_vec[1]) * 0.5,
            (l_gaze_vec[2] + r_gaze_vec[2]) * 0.5,
        ]
        _draw_normal_arrow(img, nose, avg_gaze_vec, YELLOW, max(30, int(min(fw, fh) * 0.10)))

    # Draw face direction vector from nose center using yaw/pitch/roll.
    if isinstance(snap_evt, FaceMeshEvent):
        _draw_face_direction_from_ypr(
            img,
            nose,
            snap_evt.head_yaw,
            snap_evt.head_pitch,
            snap_evt.roll,
        )
    elif isinstance(snap_evt, dict):
        _draw_face_direction_from_ypr(
            img,
            nose,
            snap_evt.get("headYaw", snap_evt.get("yaw")),
            snap_evt.get("headPitch", snap_evt.get("pitch")),
            snap_evt.get("roll"),
        )
    if isinstance(gaze, dict):
        g_n = gaze.get("normal")
        if isinstance(g_n, (list, tuple)) and len(g_n) >= 3:
            _draw_normal_arrow(img, nose, g_n, YELLOW, max(26, int(min(fw, fh) * 0.10)))

    # Click marker mapped from overlay-space to frame-space.
    cx = int(clamp(round((float(click_pos[0]) / max(1.0, float(overlay_w))) * fw), 0, fw - 1))
    cy = int(clamp(round((float(click_pos[1]) / max(1.0, float(overlay_h))) * fh), 0, fh - 1))
    if draw_click:
        cv2.circle(img, (cx, cy), 14, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 11, RED, -1, cv2.LINE_AA)

    _draw_info_panel(img, _build_event_lines(snap_evt, landmarks))
    return img, None




def save_test_capture(display: Dict, w: float, h: float, click_pos: Tuple[float, float], worker) -> None:
    """Save test capture with camera frame and mesh data only."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    ts = ms_now()
    click_x = clamp(float(click_pos[0]), 0.0, w - 1.0)
    click_y = clamp(float(click_pos[1]), 0.0, h - 1.0)

    base = f"mesh_capture_{ts}"
    png_path = CAPTURE_DIR / f"{base}.png"
    raw_png_path = CAPTURE_DIR / f"{base}_raw.png"
    eye_debug_dir = CAPTURE_DIR / f"{base}_eye_debug"
    json_path = CAPTURE_DIR / f"{base}.json"

    snap = worker.capture_snapshot()
    snap_evt = snap.get("evt")
    snap_frame = snap.get("frame")
    snap_landmarks = snap.get("landmarks")

    # Ellipse estimator runs in mirrored view-space to match capture rendering.

    if snap_frame is not None and isinstance(snap_landmarks, list) and snap_landmarks:
        mirrored_frame = cv2.flip(snap_frame, 1)
        

    if isinstance(snap_evt, FaceMeshEvent):
        event_dump = snap_evt.to_capture_dump()
        mesh_data = event_dump.get("meshData") or {}
    elif isinstance(snap_evt, dict):
        event_dump = snap_evt
        eyes = snap_evt.get("eyes")
        mesh_data = {
            "landmarks": snap_evt.get("landmarks"),
            "blendshapes": snap_evt.get("blendshapes"),
            "transformMatrix": snap_evt.get("transformMatrix"),
            "faceMaskSegment": snap_evt.get("faceMaskSegment"),
            "eyes": eyes,
        }
    else:
        event_dump = None
        mesh_data = {
            "landmarks": None,
            "blendshapes": None,
            "transformMatrix": None,
            "faceMaskSegment": None,
            "eyes": None,
        }

    shot_ok, shot_err = render_camera_capture_marked(
        str(png_path),
        snap,
        overlay_w=w,
        overlay_h=h,
        click_pos=(click_x, click_y)
    )
    raw_ok = False
    raw_err: Optional[str] = None
    if snap_frame is not None:
        mirrored_raw = cv2.flip(snap_frame, 1)
        raw_ok = bool(cv2.imwrite(str(raw_png_path), mirrored_raw))
        if not raw_ok:
            raw_err = "cv2.imwrite failed"
    else:
        raw_err = "No camera frame available yet."

    payload = {
        "timestamp": ts,
        "click": {
            "xWindow": click_x,
            "yWindow": click_y,
            "xScreen": float(display["x"]) + click_x,
            "yScreen": float(display["y"]) + click_y,
        },
        "faceMeshEvent": event_dump,
        "meshData": mesh_data,
        "screenshot": {
            "path": str(png_path),
            "ok": shot_ok,
            "error": shot_err,
        },
        "rawCameraScreenshot": {
            "path": str(raw_png_path),
            "ok": raw_ok,
            "error": raw_err,
        },
       
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if shot_ok and raw_ok:
        print(f"[test] Saved {png_path.name}, {raw_png_path.name}, and {json_path.name}", flush=True)
    elif shot_ok and not raw_ok:
        print(f"[test] Saved {png_path.name} and {json_path.name} (raw frame failed: {raw_err})", flush=True)
    else:
        print(
            f"[test] Marked screenshot failed, saved JSON: {json_path.name} "
            f"(marked={shot_err}, raw={'ok' if raw_ok else raw_err})",
            flush=True,
        )
