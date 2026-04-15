"""
Capture window manager for capture mode rendering and interaction.
"""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .facemesh_dao import clamp, safe_float
from .gaze_primitives import collect_gaze_primitives, draw_gaze_primitives_cv2


class CaptureWindowManager:
    """Manage capture mode window state, rendering, and mouse interaction."""

    def __init__(self, display: Dict[str, Any], window_name: str = "FaceMesh Capture"):
        self._display = display
        self._window_name = str(window_name)
        self._width = int(display["width"])
        self._height = int(display["height"])
        self._running = False
        self._mouse_x = float(self._width) * 0.5
        self._mouse_y = float(self._height) * 0.5
        self._clicked: Optional[Tuple[float, float]] = None
        self._capture_fps = 0.0
        self._last_tick = time.perf_counter()

    def initialize(self) -> None:
        """Create and configure the capture fullscreen window."""
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(
            self._window_name, int(self._display["x"]), int(self._display["y"])
        )
        cv2.resizeWindow(self._window_name, self._width, self._height)
        cv2.setWindowProperty(
            self._window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.setMouseCallback(self._window_name, self._on_mouse)
        self._running = True

    def shutdown(self) -> None:
        """Destroy capture window resources."""
        cv2.destroyWindow(self._window_name)
        self._running = False

    def is_running(self) -> bool:
        """Return whether capture window should keep running."""
        return self._running

    def get_mouse_position(self) -> Tuple[float, float]:
        """Return current capture mouse position."""
        return self._mouse_x, self._mouse_y

    def consume_click(self) -> Optional[Tuple[float, float]]:
        """Return and clear pending click."""
        clicked = self._clicked
        self._clicked = None
        return clicked

    def render(
        self,
        runtime_evt: Optional[Dict[str, Any]],
        live_img: Optional[np.ndarray],
    ) -> None:
        """Render one capture frame and handle keyboard exit."""
        out = self._build_frame(runtime_evt, live_img)
        cv2.imshow(self._window_name, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self._running = False

    def _on_mouse(self, event, x, y, flags, param) -> None:
        _ = flags
        _ = param
        self._mouse_x = float(max(0, min(self._width - 1, x)))
        self._mouse_y = float(max(0, min(self._height - 1, y)))
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked = (self._mouse_x, self._mouse_y)

    def _build_frame(
        self,
        runtime_evt: Optional[Dict[str, Any]],
        live_img: Optional[np.ndarray],
    ) -> np.ndarray:
        if live_img is not None:
            if live_img.shape[1] != self._width or live_img.shape[0] != self._height:
                out = cv2.resize(
                    live_img, (self._width, self._height), interpolation=cv2.INTER_LINEAR
                )
            else:
                out = live_img.copy()
        else:
            out = np.zeros((self._height, self._width, 3), dtype=np.uint8)

        now_tick = time.perf_counter()
        dt = now_tick - self._last_tick
        if dt > 1e-6:
            inst = 1.0 / dt
            if self._capture_fps <= 0.0:
                self._capture_fps = inst
            else:
                self._capture_fps = 0.9 * self._capture_fps + 0.1 * inst
        self._last_tick = now_tick

        primitives = collect_gaze_primitives(runtime_evt, self._width, self._height)
        draw_gaze_primitives_cv2(out, primitives, radius=14, outline_thickness=2)
        self._draw_hud(out, runtime_evt)
        self._draw_mouse_triangle(out, self._mouse_x, self._mouse_y, (255, 255, 255))
        return out

    def _draw_hud(self, img: np.ndarray, evt: Optional[Dict[str, Any]]) -> None:
        has_face = bool(evt and evt.get("hasFace"))

        raw_head_yaw = evt.get("head_yaw") if evt else None
        raw_head_pitch = evt.get("head_pitch") if evt else None
        raw_eye_yaw = evt.get("raw_combined_eye_gaze_yaw") if evt else None
        raw_eye_pitch = evt.get("raw_combined_eye_gaze_pitch") if evt else None
        corr_head_yaw = evt.get("face_delta_yaw") if evt else None
        corr_head_pitch = evt.get("face_delta_pitch") if evt else None
        corr_eye_yaw = evt.get("corrected_eye_yaw") if evt else None
        corr_eye_pitch = evt.get("corrected_eye_pitch") if evt else None
        corr_sum_yaw = evt.get("corrected_yaw") if evt else None
        corr_sum_pitch = evt.get("corrected_pitch") if evt else None

        raw_sum_yaw = None
        raw_sum_pitch = None
        if (
            raw_head_yaw is not None
            and raw_head_pitch is not None
            and raw_eye_yaw is not None
            and raw_eye_pitch is not None
        ):
            raw_sum_yaw = safe_float(raw_head_yaw, 0.0) + safe_float(raw_eye_yaw, 0.0)
            raw_sum_pitch = safe_float(raw_head_pitch, 0.0) + safe_float(
                raw_eye_pitch, 0.0
            )

        def _fmt(value: Optional[float]) -> str:
            if value is None:
                return "--"
            return f"{safe_float(value, 0.0):0.1f}"

        lines = [
            f"FACE: {'YES' if has_face else 'NO'}",
            f"FPS: {self._capture_fps:0.1f}",
            f"RAW FACE Y/P: {_fmt(raw_head_yaw)} / {_fmt(raw_head_pitch)}",
            f"RAW EYE  Y/P: {_fmt(raw_eye_yaw)} / {_fmt(raw_eye_pitch)}",
            f"RAW SUM  Y/P: {_fmt(raw_sum_yaw)} / {_fmt(raw_sum_pitch)}",
            f"CORR FACE Y/P: {_fmt(corr_head_yaw)} / {_fmt(corr_head_pitch)}",
            f"CORR EYE  Y/P: {_fmt(corr_eye_yaw)} / {_fmt(corr_eye_pitch)}",
            f"CORR SUM  Y/P: {_fmt(corr_sum_yaw)} / {_fmt(corr_sum_pitch)}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        pad = 8
        line_h = 22
        text_sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
        box_w = max(ts[0] for ts in text_sizes) + pad * 2
        box_h = line_h * len(lines) + pad * 2
        bx = int(clamp(self._mouse_x - box_w * 0.5, 8, img.shape[1] - box_w - 8))
        by = int(clamp(self._mouse_y + 24, 8, img.shape[0] - box_h - 8))

        cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), (20, 20, 20), -1)
        cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), (70, 180, 255), 1)
        for i, line in enumerate(lines):
            tx = bx + pad
            ty = by + pad + (i + 1) * line_h - 6
            cv2.putText(
                img,
                line,
                (tx, ty),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    def _draw_mouse_triangle(
        self, img: np.ndarray, x: float, y: float, color: Tuple[int, int, int]
    ) -> None:
        tip = (int(round(x)), int(round(y + 14)))
        left = (int(round(x - 10)), int(round(y - 2)))
        right = (int(round(x + 10)), int(round(y - 2)))
        cv2.fillConvexPoly(
            img,
            np.array([left, right, tip], dtype=np.int32),
            color,
            lineType=cv2.LINE_AA,
        )
