#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
CALIB_PATH = Path("calibration.json")
CAPTURE_DIR = Path("captures")

PITCH_OFFSET = 6.0
RING_RADIUS = 24
RING_THICKNESS = 5
DOT_RADIUS = 14
SMOOTHING = 0.25
CALIB_INSET = 50
CALIB_BLINK_MS = 1000
CALIB_COUNTDOWN_MS = 3000
CALIB_AVG_MS = 500
CALIB_BLINK_PERIOD_MS = 220

KEY_COLOR = (1, 0, 1)
WHITE = (255, 255, 255)
RED = (255, 40, 40)
BLUE = (70, 180, 255)
BLACK = (0, 0, 0)
HUD_BG = (20, 20, 20)
GREEN = (80, 230, 120)
YELLOW = (40, 220, 255)
CYAN = (255, 220, 40)

FACE_MESH_CONNECTIONS = list(vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)

TRIS = [
    (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5),
    (0, 5, 6), (0, 6, 7), (0, 7, 8), (0, 8, 1),
]


def ms_now():
    return int(time.time() * 1000)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def safe_float(v, fallback=0.0):
    try:
        f = float(v)
    except Exception:
        return fallback
    return f if math.isfinite(f) else fallback


def ensure_model():
    if MODEL_PATH.exists():
        return
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))


def _profile_token(raw_profile):
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw_profile or "").strip())
    token = token.strip("._-")
    return token or "default"


def calib_path_for_profile(profile):
    if not profile:
        return CALIB_PATH
    return Path(f"calibration-{_profile_token(profile)}.json")


def load_calib_points(calib_path=CALIB_PATH):
    try:
        raw = json.loads(calib_path.read_text(encoding="utf-8"))
        points = raw if isinstance(raw, list) else raw.get("points", [])
        return points if isinstance(points, list) else []
    except Exception:
        return []


def save_calib_points(points, calib_path=CALIB_PATH):
    payload = {"timestamp": ms_now(), "points": points}
    calib_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def reset_capture_dir():
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    for p in CAPTURE_DIR.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def center_offsets_from_points(pts):
    if len(pts) < 1:
        return None
    c = pts[0]
    yaw0 = safe_float(c.get("yaw"), None)
    pitch0 = safe_float(c.get("pitch"), None)
    if yaw0 is None or pitch0 is None:
        return None
    return {"yaw0": yaw0, "pitch0": pitch0}


def bs_score(categories, name):
    for cat in categories:
        if cat.category_name == name:
            return float(cat.score)
    return 0.0


def estimate_head_pose(landmarks):
    if len(landmarks) <= 454:
        return None
    left = landmarks[234]
    right = landmarks[454]
    forehead = landmarks[10]
    chin = landmarks[152]

    h = np.array([right.x - left.x, right.y - left.y, right.z - left.z], dtype=np.float64)
    v = np.array([forehead.x - chin.x, forehead.y - chin.y, forehead.z - chin.z], dtype=np.float64)

    n = np.cross(h, v)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return None
    n /= norm
    if n[2] > 0:
        n *= -1.0

    yaw = math.degrees(math.atan2(float(n[0]), float(-n[2])))
    pitch = math.degrees(math.atan2(float(-n[1]), float(-n[2])))
    if not math.isfinite(yaw) or not math.isfinite(pitch):
        return None
    return yaw, pitch


def depth_metric(landmarks):
    if len(landmarks) < 474:
        return None
    l_inner, l_outer, l_iris = landmarks[133], landmarks[33], landmarks[468]
    r_inner, r_outer, r_iris = landmarks[362], landmarks[263], landmarks[473]

    l_width = math.sqrt((l_outer.x - l_inner.x) ** 2 + (l_outer.z - l_inner.z) ** 2)
    r_width = math.sqrt((r_outer.x - r_inner.x) ** 2 + (r_outer.z - r_inner.z) ** 2)
    ipd = math.sqrt((r_iris.x - l_iris.x) ** 2 + (r_iris.y - l_iris.y) ** 2 + (r_iris.z - l_iris.z) ** 2)
    return (l_width + r_width) / 2.0 + ipd


def calc_gaze(result):
    if not result.face_landmarks or not result.face_blendshapes or not result.facial_transformation_matrixes:
        return None

    landmarks = result.face_landmarks[0]
    cats = result.face_blendshapes[0]

    bs_yaw = (
        (bs_score(cats, "eyeLookInRight") - bs_score(cats, "eyeLookOutRight"))
        + (bs_score(cats, "eyeLookOutLeft") - bs_score(cats, "eyeLookInLeft"))
    ) / 2.0
    bs_pitch = (
        (bs_score(cats, "eyeLookUpRight") - bs_score(cats, "eyeLookDownRight"))
        + (bs_score(cats, "eyeLookUpLeft") - bs_score(cats, "eyeLookDownLeft"))
    ) / 2.0

    m_raw = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
    m_flat = m_raw.reshape(-1)
    m44 = m_flat[:16].reshape(4, 4) if m_flat.size >= 16 else None
    hp = estimate_head_pose(landmarks)
    if hp is None:
        if m44 is not None:
            head_yaw = math.degrees(math.asin(float(-m44[0, 2])))
            head_pitch = -math.degrees(math.atan2(float(m44[1, 2]), float(m44[2, 2])))
        else:
            head_yaw = math.degrees(math.asin(float(-m_flat[2]))) if m_flat.size > 2 else 0.0
            head_pitch = -math.degrees(math.atan2(float(m_flat[6]), float(m_flat[10]))) if m_flat.size > 10 else 0.0
    else:
        head_yaw, head_pitch = hp

    if m44 is not None:
        head_roll = math.degrees(math.atan2(float(m44[0, 1]), float(m44[0, 0])))
        tx, ty, tz = float(m44[0, 3]), float(m44[1, 3]), float(m44[2, 3])
    else:
        head_roll = math.degrees(math.atan2(float(m_flat[1]), float(m_flat[0]))) if m_flat.size > 1 else 0.0
        tx = float(m_flat[3]) if m_flat.size > 3 else 0.0
        ty = float(m_flat[7]) if m_flat.size > 7 else 0.0
        tz = float(m_flat[11]) if m_flat.size > 11 else 0.0

    head_pitch += PITCH_OFFSET
    eye_yaw = bs_yaw * 35.0
    eye_pitch = bs_pitch * 35.0

    return {
        "yaw": head_yaw + eye_yaw,
        "pitch": head_pitch + eye_pitch,
        "headYaw": head_yaw,
        "headPitch": head_pitch,
        "eyeYaw": eye_yaw,
        "eyePitch": eye_pitch,
        "roll": head_roll,
        "x": tx,
        "y": ty,
        "z": tz,
        "depth": depth_metric(landmarks),
    }


def open_camera(args):
    backends = {
        "auto": [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW), ("any", None)],
        "msmf": [("msmf", cv2.CAP_MSMF), ("any", None)],
        "dshow": [("dshow", cv2.CAP_DSHOW), ("any", None)],
        "any": [("any", None)],
    }
    errors = []
    for name, backend in backends.get(args.camera_backend, backends["auto"]):
        cap = cv2.VideoCapture(args.camera_index, backend) if backend is not None else cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            cap.release()
            errors.append(f"{name}: open failed")
            continue

        fourcc = (args.camera_fourcc or "").strip().upper()
        if len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if args.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.camera_width))
        if args.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.camera_height))
        if args.camera_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(args.camera_fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, _ = cap.read()
        if not ok:
            cap.release()
            errors.append(f"{name}: read failed")
            continue

        info = {
            "backend": name,
            "index": args.camera_index,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }
        return cap, info

    raise RuntimeError(f"Unable to open webcam ({'; '.join(errors)})")


def make_calib_seq(w, h):
    i = CALIB_INSET
    cx, cy = w / 2.0, h / 2.0
    return [
        {"name": "C", "x": cx, "y": cy},
        {"name": "TL", "x": i, "y": i},
        {"name": "TC", "x": cx, "y": i},
        {"name": "TR", "x": w - i, "y": i},
        {"name": "R", "x": w - i, "y": cy},
        {"name": "BR", "x": w - i, "y": h - i},
        {"name": "BC", "x": cx, "y": h - i},
        {"name": "BL", "x": i, "y": h - i},
        {"name": "L", "x": i, "y": cy},
    ]

def interp_screen(yaw, pitch, depth, points):
    if len(points) < 9 or depth is None or not math.isfinite(depth) or abs(depth) < 1e-9:
        return None
    nx = yaw / depth
    ny = pitch / depth

    best_tri = None
    best_w = None
    best_min = -1e9

    for t in TRIS:
        p1, p2, p3 = points[t[0]], points[t[1]], points[t[2]]
        x1, y1 = float(p1["nX"]), float(p1["nY"])
        x2, y2 = float(p2["nX"]), float(p2["nY"])
        x3, y3 = float(p3["nX"]), float(p3["nY"])

        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(det) < 1e-6:
            continue

        w1 = ((y2 - y3) * (nx - x3) + (x3 - x2) * (ny - y3)) / det
        w2 = ((y3 - y1) * (nx - x3) + (x1 - x3) * (ny - y3)) / det
        w3 = 1.0 - w1 - w2

        mn = min(w1, w2, w3)
        if mn > best_min:
            best_min = mn
            best_tri = t
            best_w = (w1, w2, w3)

    if best_tri is None:
        return None

    p1, p2, p3 = points[best_tri[0]], points[best_tri[1]], points[best_tri[2]]
    w1, w2, w3 = best_w
    return {
        "x": w1 * float(p1["screenX"]) + w2 * float(p2["screenX"]) + w3 * float(p3["screenX"]),
        "y": w1 * float(p1["screenY"]) + w2 * float(p2["screenY"]) + w3 * float(p3["screenY"]),
    }


def target_from_gaze(evt, points, w, h):
    if not evt or not evt.get("hasFace"):
        return None
    yaw = evt.get("yaw")
    pitch = evt.get("pitch")
    if yaw is None or pitch is None:
        return None

    if len(points) >= 9:
        pt = interp_screen(float(yaw), float(pitch), evt.get("depth"), points)
        if pt:
            return {"x": clamp(pt["x"], 0.0, w - 1.0), "y": clamp(pt["y"], 0.0, h - 1.0)}

    return {
        "x": clamp(w / 2.0 + float(yaw) * 14.0, 0.0, w - 1.0),
        "y": clamp(h / 2.0 - float(pitch) * 14.0, 0.0, h - 1.0),
    }


def set_window_transparent(hwnd):
    if sys.platform != "win32" or not hwnd:
        return
    user32 = ctypes.windll.user32
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    LWA_COLORKEY = 0x00000001

    colorref = KEY_COLOR[0] | (KEY_COLOR[1] << 8) | (KEY_COLOR[2] << 16)
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED)
    user32.SetLayeredWindowAttributes(hwnd, colorref, 0, LWA_COLORKEY)


def set_window_topmost(hwnd):
    if sys.platform != "win32" or not hwnd:
        return
    user32 = ctypes.windll.user32
    HWND_TOPMOST = -1
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_SHOWWINDOW = 0x0040
    SWP_NOACTIVATE = 0x0010
    user32.SetWindowPos(
        hwnd,
        HWND_TOPMOST,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW | SWP_NOACTIVATE,
    )


def ps_quote(v):
    return "'" + str(v).replace("'", "''") + "'"


def _lm_to_px(lm, w, h):
    x = int(clamp(round(lm[0] * w), 0, w - 1))
    y = int(clamp(round(lm[1] * h), 0, h - 1))
    return x, y


def _draw_normal_arrow(img, origin, yaw, pitch, color, length):
    ox, oy = int(origin[0]), int(origin[1])
    yaw_r = math.radians(safe_float(yaw, 0.0))
    pitch_r = math.radians(safe_float(pitch, 0.0))
    dx = -math.sin(yaw_r) * length
    dy = -math.sin(pitch_r) * length
    end = (int(round(ox + dx)), int(round(oy + dy)))
    cv2.arrowedLine(img, (ox, oy), end, color, 2, cv2.LINE_AA, tipLength=0.2)


def render_camera_capture_marked(png_path, snap, overlay_w, overlay_h, click_pos):
    frame = snap.get("frame")
    if frame is None:
        return False, "No camera frame available yet."

    img = frame.copy()
    fh, fw = img.shape[:2]
    landmarks = snap.get("landmarks") or []
    evt = snap.get("evt") or {}

    # Draw mesh connections first, then points for clarity.
    if landmarks:
        for conn in FACE_MESH_CONNECTIONS:
            a = int(conn.start)
            b = int(conn.end)
            if a >= len(landmarks) or b >= len(landmarks):
                continue
            p1 = _lm_to_px(landmarks[a], fw, fh)
            p2 = _lm_to_px(landmarks[b], fw, fh)
            cv2.line(img, p1, p2, (45, 140, 45), 1, cv2.LINE_AA)

        for lm in landmarks:
            cv2.circle(img, _lm_to_px(lm, fw, fh), 1, GREEN, -1, cv2.LINE_AA)

    # Face normal from head pose and eye gaze normals from final gaze vector.
    nose = _lm_to_px(landmarks[1], fw, fh) if len(landmarks) > 1 else (fw // 2, fh // 2)
    left_iris = _lm_to_px(landmarks[468], fw, fh) if len(landmarks) > 468 else (fw // 2 - 40, fh // 2)
    right_iris = _lm_to_px(landmarks[473], fw, fh) if len(landmarks) > 473 else (fw // 2 + 40, fh // 2)
    norm_len = int(max(40, min(fw, fh) * 0.16))

    _draw_normal_arrow(
        img,
        nose,
        evt.get("headYaw", evt.get("yaw", 0.0)),
        evt.get("headPitch", evt.get("pitch", 0.0)),
        YELLOW,
        norm_len,
    )
    _draw_normal_arrow(img, left_iris, evt.get("yaw", 0.0), evt.get("pitch", 0.0), CYAN, norm_len)
    _draw_normal_arrow(img, right_iris, evt.get("yaw", 0.0), evt.get("pitch", 0.0), CYAN, norm_len)

    # Click marker mapped from overlay-space to frame-space.
    cx = int(clamp(round((float(click_pos[0]) / max(1.0, float(overlay_w))) * fw), 0, fw - 1))
    cy = int(clamp(round((float(click_pos[1]) / max(1.0, float(overlay_h))) * fh), 0, fh - 1))
    cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)

    ok = cv2.imwrite(str(png_path), img)
    return (ok, None) if ok else (False, "cv2.imwrite failed")


def capture_desktop_marked(png_path, display, click_screen, pred_screen, err_px):
    if sys.platform != "win32":
        return False, "Desktop capture implemented only on Windows"

    rx, ry, rw, rh = int(display["x"]), int(display["y"]), int(display["width"]), int(display["height"])
    cx = int(round(click_screen[0] - rx))
    cy = int(round(click_screen[1] - ry))
    px = int(round(pred_screen[0] - rx))
    py = int(round(pred_screen[1] - ry))

    script = "; ".join([
        "Add-Type -AssemblyName System.Drawing",
        f"$rect = New-Object System.Drawing.Rectangle({rx}, {ry}, {rw}, {rh})",
        "$bmp = New-Object System.Drawing.Bitmap($rect.Width, $rect.Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)",
        "$g = [System.Drawing.Graphics]::FromImage($bmp)",
        "$g.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $rect.Size, [System.Drawing.CopyPixelOperation]::SourceCopy)",
        "$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias",
        "$pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 70, 180, 255), 4)",
        "$dot = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(230, 255, 40, 40))",
        f"$px = {px}; $py = {py}; $cx = {cx}; $cy = {cy}",
        "$g.DrawEllipse($pen, $px - 18, $py - 18, 36, 36)",
        "$g.FillEllipse($dot, $cx - 8, $cy - 8, 16, 16)",
        f"$bmp.Save({ps_quote(png_path)}, [System.Drawing.Imaging.ImageFormat]::Png)",
        "$pen.Dispose(); $dot.Dispose(); $g.Dispose(); $bmp.Dispose()",
    ])

    p = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
    )
    if p.returncode == 0:
        return True, None
    return False, (p.stderr or p.stdout or f"PowerShell exit {p.returncode}").strip()


class UdpSender:
    def __init__(self, host, port, center_offsets=None):
        self.host = host
        self.port = port
        self.center_offsets = center_offsets
        self.sock = None

    def set_center_offsets(self, center_offsets):
        self.center_offsets = center_offsets

    def start(self):
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def stop(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def send_evt(self, evt):
        if not self.sock:
            return
        has_face = bool(evt.get("hasFace"))
        yaw = safe_float(evt.get("yaw"), 0.0) if has_face else 0.0
        pitch = safe_float(evt.get("pitch"), 0.0) if has_face else 0.0
        roll = safe_float(evt.get("roll"), 0.0) if has_face else 0.0
        x = safe_float(evt.get("x"), 0.0) if has_face else 0.0
        y = safe_float(evt.get("y"), 0.0) if has_face else 0.0
        z = safe_float(evt.get("z"), 0.0) if has_face else 0.0
        if has_face and self.center_offsets:
            yaw -= safe_float(self.center_offsets.get("yaw0"), 0.0)
            pitch -= safe_float(self.center_offsets.get("pitch0"), 0.0)

        pkt = struct.pack("<6d", -x, y, z, -yaw, pitch, roll)
        self.sock.sendto(pkt, (self.host, self.port))


class GazeWorker(threading.Thread):
    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.stop_evt = threading.Event()
        self.ready_evt = threading.Event()

        self.lock = threading.Lock()
        self.seq = 0
        self.latest = None
        self.latest_frame = None
        self.latest_landmarks = None
        self.error = None
        self.status = deque(maxlen=80)

    def _status(self, msg, force=False):
        with self.lock:
            self.status.append({"ts": ms_now(), "line": str(msg)})
        if not self.args.quiet or force:
            print(msg, flush=True)

    def snapshot(self):
        with self.lock:
            return self.seq, (dict(self.latest) if self.latest else None)

    def capture_snapshot(self):
        with self.lock:
            snap_evt = dict(self.latest) if self.latest else None
            snap_frame = None if self.latest_frame is None else self.latest_frame.copy()
            snap_landmarks = list(self.latest_landmarks) if self.latest_landmarks else None
            snap_seq = self.seq
        return {"seq": snap_seq, "evt": snap_evt, "frame": snap_frame, "landmarks": snap_landmarks}

    def status_tail(self, n=20):
        with self.lock:
            return list(self.status)[-n:]

    def stop(self):
        self.stop_evt.set()

    def run(self):
        cap = None
        landmarker = None
        try:
            ensure_model()
            base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
            opts = vision.FaceLandmarkerOptions(
                base_options=base,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            )
            landmarker = vision.FaceLandmarker.create_from_options(opts)
            cap, info = open_camera(self.args)

            self._status("FaceLandmarker initialized")
            self._status(
                f"Camera backend={info['backend']} index={info['index']} "
                f"{info['width']}x{info['height']} {info['fps']:.1f}fps",
                force=True,
            )
            self._status("Webcam capture started. Running until terminated.", force=True)
            self.ready_evt.set()

            next_log = time.time() + self.args.log_interval
            while not self.stop_evt.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))
                gaze = calc_gaze(result)
                lms = result.face_landmarks[0] if result.face_landmarks else None
                lms_flat = None
                if lms:
                    lms_flat = [(float(p.x), float(p.y), float(p.z)) for p in lms]

                if gaze is None:
                    evt = {
                        "type": "gaze", "hasFace": False,
                        "yaw": None, "pitch": None, "rawYaw": None, "rawPitch": None,
                        "roll": None, "x": None, "y": None, "z": None, "depth": None,
                        "ts": ms_now(),
                    }
                else:
                    evt = {
                        "type": "gaze", "hasFace": True,
                        "yaw": gaze["yaw"], "pitch": gaze["pitch"],
                        "rawYaw": gaze["yaw"], "rawPitch": gaze["pitch"],
                        "headYaw": gaze["headYaw"], "headPitch": gaze["headPitch"],
                        "eyeYaw": gaze["eyeYaw"], "eyePitch": gaze["eyePitch"],
                        "roll": gaze["roll"], "x": gaze["x"], "y": gaze["y"], "z": gaze["z"],
                        "depth": gaze["depth"], "ts": ms_now(),
                    }

                with self.lock:
                    self.seq += 1
                    self.latest = evt
                    self.latest_frame = frame_bgr
                    self.latest_landmarks = lms_flat

                if self.args.log_interval > 0 and time.time() >= next_log:
                    if evt["hasFace"]:
                        self._status(f"yaw={evt['yaw']:.2f} pitch={evt['pitch']:.2f}", force=True)
                    else:
                        self._status("yaw=NaN pitch=NaN (no face)", force=True)
                    next_log += self.args.log_interval
        except Exception as e:
            self.error = f"Gaze worker failed: {e}"
            self._status(self.error, force=True)
            self.ready_evt.set()
        finally:
            if cap is not None:
                cap.release()
            if landmarker is not None:
                landmarker.close()

def save_test_capture(display, w, h, click_pos, pred_raw, pred_disp, evt, mode, points, seq, idx, worker):
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    ts = ms_now()

    click_x = clamp(float(click_pos[0]), 0.0, w - 1.0)
    click_y = clamp(float(click_pos[1]), 0.0, h - 1.0)
    raw_x, raw_y = float(pred_raw["x"]), float(pred_raw["y"])
    disp_x, disp_y = float(pred_disp["x"]), float(pred_disp["y"])

    click_screen = (display["x"] + click_x, display["y"] + click_y)
    pred_screen = (display["x"] + disp_x, display["y"] + disp_y)

    raw_err = math.hypot(click_x - raw_x, click_y - raw_y)
    disp_err = math.hypot(click_x - disp_x, click_y - disp_y)

    base = f"gaze_debug_{ts}"
    png_path = CAPTURE_DIR / f"{base}.png"
    json_path = CAPTURE_DIR / f"{base}.json"

    snap = worker.capture_snapshot()
    shot_ok, shot_err = render_camera_capture_marked(
        str(png_path),
        snap,
        overlay_w=w,
        overlay_h=h,
        click_pos=(click_x, click_y),
    )

    payload = {
        "timestamp": ts,
        "mode": mode,
        "click": {
            "xWindow": click_x,
            "yWindow": click_y,
            "xScreen": click_screen[0],
            "yScreen": click_screen[1],
        },
        "predicted": {
            "raw": {"xWindow": raw_x, "yWindow": raw_y},
            "displayed": {"xWindow": disp_x, "yWindow": disp_y},
        },
        "errorPx": {
            "raw": raw_err,
            "displayed": disp_err,
        },
        "gaze": evt,
        "diagnostics": {
            "display": display,
            "capture": {
                "source": "camera_frame",
                "workerSeq": snap.get("seq"),
                "workerTs": (snap.get("evt") or {}).get("ts"),
            },
            "calibration": {
                "capturedPoints": len(points),
                "sequenceSize": len(seq),
                "pointIndex": idx,
                "points": points,
            },
            "workerStatus": worker.status_tail(20),
        },
        "screenshot": {
            "path": str(png_path),
            "ok": shot_ok,
            "error": shot_err,
        },
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if shot_ok:
        print(f"[test] Saved {png_path.name} and {json_path.name}", flush=True)
    else:
        print(f"[test] Screenshot failed, saved JSON only: {json_path.name} ({shot_err})", flush=True)


def draw_mouse_triangle(surface, x, y, color=WHITE):
    # Downward-pointing marker under the current mouse cursor.
    tip = (int(round(x)), int(round(y + 14)))
    left = (int(round(x - 10)), int(round(y - 2)))
    right = (int(round(x + 10)), int(round(y - 2)))
    pygame.draw.polygon(surface, color, (left, right, tip), 0)


def draw_capture_hud(surface, font_small, dot_x, dot_y, evt, fps):
    yaw = safe_float(evt.get("yaw"), 0.0) if evt else 0.0
    pitch = safe_float(evt.get("pitch"), 0.0) if evt else 0.0
    depth = safe_float(evt.get("depth"), 0.0) if evt else 0.0
    has_face = bool(evt and evt.get("hasFace"))

    lines = [
        f"FACE: {'YES' if has_face else 'NO'}",
        f"YAW : {yaw:.2f}",
        f"PITCH: {pitch:.2f}",
        f"DEPTH: {depth:.4f}",
        f"FPS  : {fps:0.1f}",
    ]

    pad = 8
    line_h = 20
    box_w = 190
    box_h = pad * 2 + line_h * len(lines)
    bx = int(clamp(dot_x + 30, 8, surface.get_width() - box_w - 8))
    by = int(clamp(dot_y - box_h / 2, 8, surface.get_height() - box_h - 8))

    pygame.draw.rect(surface, HUD_BG, (bx, by, box_w, box_h), 0, border_radius=6)
    pygame.draw.rect(surface, BLUE, (bx, by, box_w, box_h), 1, border_radius=6)

    for i, text in enumerate(lines):
        img = font_small.render(text, True, WHITE)
        surface.blit(img, (bx + pad, by + pad + i * line_h))


def start_capture_loop(args, worker, sender, display, run_duration, calib_path, loaded_points):
    overlay_enabled = bool(args.overlay)
    capture_enabled = bool(args.capture and overlay_enabled)
    emit_udp = bool(args.udp)

    w = int(display["width"])
    h = int(display["height"])

    screen = None
    clock = None
    font = None
    font_small = None
    if overlay_enabled:
        pygame.init()
        pygame.font.init()
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", f"{display['x']},{display['y']}")
        screen = pygame.display.set_mode((w, h), pygame.NOFRAME)
        pygame.display.set_caption("FaceMesh Gaze")
        hwnd = pygame.display.get_wm_info().get("window")
        set_window_transparent(hwnd)
        set_window_topmost(hwnd)
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 80)
        font_small = pygame.font.Font(None, 24)
    else:
        if emit_udp:
            print(f"UDP mode enabled. Sending raw data to {sender.host}:{sender.port} (OpenTrack UDP over network).", flush=True)
        else:
            print("Headless mode enabled (UDP disabled).", flush=True)

    mode = "calibrate" if args.calibrate else "gaze"
    points = [] if args.calibrate else list(loaded_points)
    seq = make_calib_seq(w, h) if overlay_enabled else []
    idx = 0
    calib_phase = "blink_pre"
    phase_start = ms_now()
    calib_samples = []

    if args.calibrate:
        print("Calibration mode enabled (--calibrate).", flush=True)
        print("Flow: 1s blink, 3-2-1 red dot, 0.5s green sampling average, 1s blink timeout.", flush=True)
        print("Calibration target 1/9", flush=True)
        if emit_udp:
            print(f"UDP mode will start sending after calibration to {sender.host}:{sender.port}.", flush=True)
    elif overlay_enabled:
        if len(points) >= 9:
            print(f"Loaded {calib_path.name} ({len(points)} points).", flush=True)
        else:
            print(f"No valid {calib_path.name} found; running uncalibrated.", flush=True)

    target_x = w / 2.0
    target_y = h / 2.0
    current_x = target_x
    current_y = target_y
    mouse_x = w / 2.0
    mouse_y = h / 2.0

    latest_evt = None
    last_seq = -1

    started = time.time()
    running = True
    while running:
        if worker.error:
            raise RuntimeError(worker.error)
        now = ms_now()

        if overlay_enabled:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
                elif e.type == pygame.MOUSEMOTION:
                    mouse_x = clamp(float(e.pos[0]), 0.0, w - 1.0)
                    mouse_y = clamp(float(e.pos[1]), 0.0, h - 1.0)
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if capture_enabled and mode == "gaze" and e.button == 1:
                        save_test_capture(
                            display, w, h,
                            e.pos,
                            {"x": target_x, "y": target_y},
                            {"x": current_x, "y": current_y},
                            latest_evt,
                            mode,
                            points,
                            seq,
                            idx,
                            worker,
                        )

        seq_id, evt = worker.snapshot()
        if seq_id != last_seq and evt and evt.get("type") == "gaze":
            last_seq = seq_id
            latest_evt = evt

            if mode == "gaze":
                if overlay_enabled:
                    pt = target_from_gaze(evt, points, w, h)
                    if pt:
                        target_x = pt["x"]
                        target_y = pt["y"]
                if emit_udp:
                    sender.send_evt(evt)

        if mode == "calibrate":
            elapsed = now - phase_start
            if calib_phase == "blink_pre":
                if elapsed >= CALIB_BLINK_MS:
                    calib_phase = "countdown"
                    phase_start = now
            elif calib_phase == "countdown":
                if elapsed >= CALIB_COUNTDOWN_MS:
                    calib_phase = "sampling"
                    phase_start = now
                    calib_samples = []
            elif calib_phase == "sampling":
                if latest_evt and latest_evt.get("hasFace"):
                    yaw = safe_float(latest_evt.get("yaw"), None)
                    pitch = safe_float(latest_evt.get("pitch"), None)
                    depth = safe_float(latest_evt.get("depth"), None)
                    if yaw is not None and pitch is not None and depth is not None and abs(depth) > 1e-9:
                        calib_samples.append(
                            {
                                "yaw": yaw,
                                "pitch": pitch,
                                "depth": depth,
                                "nX": yaw / depth,
                                "nY": pitch / depth,
                            }
                        )

                if elapsed >= CALIB_AVG_MS:
                    if calib_samples:
                        p = seq[idx]
                        n = float(len(calib_samples))
                        yaw_avg = sum(s["yaw"] for s in calib_samples) / n
                        pitch_avg = sum(s["pitch"] for s in calib_samples) / n
                        depth_avg = sum(s["depth"] for s in calib_samples) / n
                        nx_avg = sum(s["nX"] for s in calib_samples) / n
                        ny_avg = sum(s["nY"] for s in calib_samples) / n
                        points.append(
                            {
                                "screenX": p["x"],
                                "screenY": p["y"],
                                "yaw": yaw_avg,
                                "pitch": pitch_avg,
                                "depthMetric": depth_avg,
                                "nX": nx_avg,
                                "nY": ny_avg,
                                "samples": int(n),
                            }
                        )
                        print(f"Calibration {idx + 1}/9 captured ({int(n)} samples).", flush=True)
                        calib_phase = "blink_post"
                        phase_start = now
                    else:
                        print(f"Calibration {idx + 1}/9 no valid face samples; retrying point.", flush=True)
                        calib_phase = "blink_pre"
                        phase_start = now
            elif calib_phase == "blink_post":
                if elapsed >= CALIB_BLINK_MS:
                    idx += 1
                    if idx >= 9:
                        mode = "gaze"
                        save_calib_points(points, calib_path)
                        if len(points) >= 1:
                            sender.set_center_offsets(
                                {"yaw0": safe_float(points[0].get("yaw"), 0.0), "pitch0": safe_float(points[0].get("pitch"), 0.0)}
                            )
                        print("Calibration complete. Entering gaze mode.", flush=True)
                        print(f"Saved {calib_path.name}", flush=True)
                    else:
                        calib_phase = "blink_pre"
                        phase_start = now
                        print(f"Calibration target {idx + 1}/9", flush=True)

        if overlay_enabled:
            if capture_enabled:
                screen.fill(BLACK)
            else:
                screen.fill(KEY_COLOR)
            if mode == "calibrate":
                p = seq[min(idx, len(seq) - 1)]
                px, py = int(round(p["x"])), int(round(p["y"]))
                phase_elapsed = now - phase_start
                blink_on = ((phase_elapsed // CALIB_BLINK_PERIOD_MS) % 2) == 0

                dot_color = RED if calib_phase == "countdown" else GREEN if calib_phase == "sampling" else WHITE
                draw_dot = True
                if calib_phase in ("blink_pre", "blink_post"):
                    draw_dot = blink_on

                if draw_dot:
                    pygame.draw.circle(screen, WHITE, (px, py), DOT_RADIUS + 3)
                    pygame.draw.circle(screen, dot_color, (px, py), DOT_RADIUS)

                if calib_phase == "countdown":
                    left = CALIB_COUNTDOWN_MS - phase_elapsed
                    digit = 3 if left > 2000 else 2 if left > 1000 else 1 if left > 0 else None
                    if digit is not None:
                        txt = font.render(str(digit), True, WHITE)
                        rect = txt.get_rect(center=(px, py + DOT_RADIUS + 35))
                        screen.blit(txt, rect)
            else:
                current_x += (target_x - current_x) * SMOOTHING
                current_y += (target_y - current_y) * SMOOTHING
                pygame.draw.circle(screen, BLUE, (int(round(current_x)), int(round(current_y))), RING_RADIUS, RING_THICKNESS)
                if capture_enabled:
                    draw_capture_hud(screen, font_small, current_x, current_y, latest_evt, clock.get_fps())
                    draw_mouse_triangle(screen, mouse_x, mouse_y)

            pygame.display.update()
            clock.tick(max(1, int(args.overlay_fps)))
        else:
            time.sleep(0.001)

        if run_duration > 0 and (time.time() - started) >= run_duration:
            print(f"{run_duration:.0f}s elapsed, exiting.", flush=True)
            running = False

    if overlay_enabled:
        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Pure Python FaceMesh app (overlay + calibration + UDP)")
    parser.add_argument(
        "--udp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit OpenTrack UDP output",
    )
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show overlay window",
    )
    parser.add_argument(
        "--capture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture/debug mode (black background + HUD + mouse triangle + click-to-save)",
    )
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run 9-point calibration flow",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Run time in seconds (0 = continuous)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-interval", type=float, default=2.0)
    parser.add_argument("--overlay-fps", type=int, default=60)
    parser.add_argument("--profile", type=str, default="", help="Calibration profile name (uses calibration-<profile>.json)")

    parser.add_argument("--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")))
    parser.add_argument("--camera-backend", choices=["auto", "msmf", "dshow", "any"], default=os.getenv("CAMERA_BACKEND", "auto").lower())
    parser.add_argument("--camera-width", type=int, default=int(os.getenv("CAMERA_WIDTH", "0")))
    parser.add_argument("--camera-height", type=int, default=int(os.getenv("CAMERA_HEIGHT", "0")))
    parser.add_argument("--camera-fps", type=float, default=float(os.getenv("CAMERA_FPS", "0")))
    parser.add_argument("--camera-fourcc", type=str, default=os.getenv("CAMERA_FOURCC", "MJPG"))
    return parser.parse_args()


def get_display_geo():
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        return {"name": "Primary Display", "x": 0, "y": 0, "width": int(user32.GetSystemMetrics(0)), "height": int(user32.GetSystemMetrics(1))}
    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    if sizes:
        w, h = sizes[0]
    else:
        info = pygame.display.Info()
        w, h = info.current_w, info.current_h
    return {"name": "Display", "x": 0, "y": 0, "width": int(w), "height": int(h)}


def normalize_runtime_args(args):
    if args.calibrate and not args.overlay:
        raise RuntimeError("Calibration requires overlay. Use --overlay with --calibrate.")
    if args.capture and not args.overlay:
        print("Capture mode requires overlay; ignoring --capture with --no-overlay.", flush=True)
        args.capture = False
    if args.capture:
        reset_capture_dir()
        print(f"Cleared previous capture session at {CAPTURE_DIR.resolve()}", flush=True)


def main():
    args = parse_args()
    run_duration = max(0.0, args.duration)
    normalize_runtime_args(args)

    udp_host = os.getenv("OPENTRACK_HOST", "127.0.0.1")
    udp_port = int(os.getenv("OPENTRACK_PORT", "4242"))
    calib_path = calib_path_for_profile(args.profile)
    loaded_points = load_calib_points(calib_path)

    display = get_display_geo()
    center = center_offsets_from_points(loaded_points)
    if args.profile:
        print(f"Calibration profile '{args.profile}' -> {calib_path.name}", flush=True)
    if args.udp:
        if center:
            print(f"Loaded calibration center offsets from {calib_path.name} (UDP recenter).", flush=True)
        else:
            print(f"No calibration center offsets found in {calib_path.name}; UDP uses raw gaze angles.", flush=True)
    else:
        print("Using raw gaze space for overlay/calibration mapping.", flush=True)

    worker = GazeWorker(args)
    worker.start()
    if not worker.ready_evt.wait(timeout=20):
        raise RuntimeError("Timed out waiting for gaze worker init")
    if worker.error:
        raise RuntimeError(worker.error)

    print(f"App started on '{display['name']}' ({display['width']}x{display['height']}).", flush=True)
    print(f"Camera request: backend={args.camera_backend} index={args.camera_index}", flush=True)
    if args.capture:
        print(f"Capture mode enabled (--capture). Click to save gaze_debug_*.png/json into {CAPTURE_DIR.resolve()}", flush=True)
    if run_duration <= 0:
        print("Runtime: continuous.", flush=True)
    else:
        print(f"Runtime: {int(run_duration)}s test mode.", flush=True)

    sender = UdpSender(udp_host, udp_port, center_offsets=center)
    if args.udp:
        sender.start()

    try:
        start_capture_loop(args, worker, sender, display, run_duration, calib_path, loaded_points)
    finally:
        worker.stop()
        worker.join(timeout=3.0)
        sender.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
