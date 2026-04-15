"""
Runtime overlay window for gaze display and optional capture HUD.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pygame

from .facemesh_dao import clamp, safe_float
from .gaze_primitives import collect_gaze_primitives, draw_gaze_primitives_pygame
from .overlay_common import (
    BLACK,
    BLUE,
    DOT_RADIUS,
    HUD_BG,
    KEY_COLOR,
    WHITE,
    set_window_click_through,
    set_window_topmost,
    set_window_transparent,
)


def draw_mouse_triangle(surface, x: float, y: float, color=WHITE):
    """Draw a cursor marker for capture mode."""
    tip = (int(round(x)), int(round(y + 14)))
    left = (int(round(x - 10)), int(round(y - 2)))
    right = (int(round(x + 10)), int(round(y - 2)))
    pygame.draw.polygon(surface, color, (left, right, tip), 0)


def draw_capture_hud(
    surface, font_small, mouse_x: float, mouse_y: float, evt: Optional[Dict], fps: float
):
    """Render live capture values near cursor position."""
    has_face = bool(evt and evt.get("hasFace"))
    head_yaw = evt.get("head_yaw") if evt else None
    head_pitch = evt.get("head_pitch") if evt else None
    eye_yaw = evt.get("raw_combined_eye_gaze_yaw") if evt else None
    eye_pitch = evt.get("raw_combined_eye_gaze_pitch") if evt else None

    def _fmt(value: Optional[float]) -> str:
        if value is None:
            return "--"
        return f"{safe_float(value, 0.0):0.1f}"

    lines = [
        f"FACE: {'YES' if has_face else 'NO'}",
        f"FPS: {fps:0.1f}",
        f"HEAD YAW: {_fmt(head_yaw)}",
        f"HEAD PITCH: {_fmt(head_pitch)}",
        f"EYE YAW: {_fmt(eye_yaw)}",
        f"EYE PITCH: {_fmt(eye_pitch)}",
    ]

    pad = 8
    line_h = max(18, font_small.get_linesize())
    box_w = max(font_small.size(line)[0] for line in lines) + pad * 2
    box_h = pad * 2 + line_h * len(lines)
    bx = int(clamp(mouse_x - box_w / 2.0, 8, surface.get_width() - box_w - 8))
    by = int(clamp(mouse_y + 24, 8, surface.get_height() - box_h - 8))

    pygame.draw.rect(surface, HUD_BG, (bx, by, box_w, box_h), 0, border_radius=6)
    pygame.draw.rect(surface, BLUE, (bx, by, box_w, box_h), 1, border_radius=6)

    for i, text in enumerate(lines):
        img = font_small.render(text, True, WHITE)
        surface.blit(img, (bx + pad, by + pad + i * line_h))


class RuntimeOverlayManager:
    """Manage a runtime overlay window."""

    def __init__(
        self,
        display: Dict,
        capture_enabled: bool = False,
        overlay_fps: int = 60,
        click_through: bool = False,
    ):
        self._display = display
        self._capture_enabled = capture_enabled
        self._overlay_fps = overlay_fps
        self._click_through = click_through
        self._width = int(display["width"])
        self._height = int(display["height"])

        self._screen = None
        self._clock = None
        self._font_small = None
        self._hwnd = None

        self._mouse_x = self._width / 2.0
        self._mouse_y = self._height / 2.0

        self._running = False
        self._should_exit = False

    def initialize(self):
        """Create and configure runtime overlay window."""
        pygame.init()
        pygame.font.init()
        os.environ.setdefault(
            "SDL_VIDEO_WINDOW_POS", f"{self._display['x']},{self._display['y']}"
        )
        self._screen = pygame.display.set_mode(
            (self._width, self._height), pygame.NOFRAME
        )
        pygame.display.set_caption("FaceMesh Gaze")
        self._hwnd = pygame.display.get_wm_info().get("window")
        if not self._capture_enabled:
            set_window_transparent(self._hwnd)
        set_window_topmost(self._hwnd)
        if self._click_through:
            set_window_click_through(self._hwnd)
        self._clock = pygame.time.Clock()
        self._font_small = pygame.font.Font(None, 24)
        self._running = True

    def shutdown(self):
        """Close runtime overlay resources."""
        if self._screen:
            pygame.quit()
        self._running = False

    def handle_events(self, on_mouse_motion=None, on_click=None):
        """Process window events."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._should_exit = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self._should_exit = True
            elif e.type == pygame.MOUSEMOTION:
                self._mouse_x = clamp(float(e.pos[0]), 0.0, self._width - 1.0)
                self._mouse_y = clamp(float(e.pos[1]), 0.0, self._height - 1.0)
                if on_mouse_motion:
                    on_mouse_motion(self._mouse_x, self._mouse_y)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if on_click and e.button == 1:
                    on_click(e.pos)

    def render_mesh(
        self, evt: Optional[Dict], capture_live_frame: Optional[np.ndarray] = None
    ):
        """Render one runtime frame."""
        if self._capture_enabled:
            if capture_live_frame is not None:
                self._render_capture_live_frame(capture_live_frame)
            else:
                self._screen.fill(BLACK)
            if evt:
                fps = self._clock.get_fps()
                draw_capture_hud(
                    self._screen,
                    self._font_small,
                    self._mouse_x,
                    self._mouse_y,
                    evt,
                    fps,
                )
            draw_mouse_triangle(self._screen, self._mouse_x, self._mouse_y)
        else:
            self._screen.fill(KEY_COLOR)
            self.render_gaze_dot(evt)

        pygame.display.update()
        self._clock.tick(max(1, int(self._overlay_fps)))

    def _render_capture_live_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            self._screen.fill(BLACK)
            return
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
            self._screen.fill(BLACK)
            return
        rgb = np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        if surface.get_width() != self._width or surface.get_height() != self._height:
            surface = pygame.transform.smoothscale(surface, (self._width, self._height))
        self._screen.blit(surface, (0, 0))

    def clear(self):
        """Clear runtime overlay surface."""
        if self._capture_enabled:
            self._screen.fill(BLACK)
        else:
            self._screen.fill(KEY_COLOR)

    def update(self):
        """Flip buffers and enforce target frame rate."""
        pygame.display.update()
        if self._clock:
            self._clock.tick(max(1, int(self._overlay_fps)))

    def get_mouse_position(self) -> Tuple[float, float]:
        """Return current mouse coordinates."""
        return self._mouse_x, self._mouse_y

    def render_gaze_dot(self, evt: Optional[Dict]):
        """Render current raw and correlated gaze primitives."""
        primitives = collect_gaze_primitives(evt, self._width, self._height)
        draw_gaze_primitives_pygame(
            self._screen,
            primitives,
            radius=DOT_RADIUS,
            outline_thickness=2,
        )

    def request_exit(self) -> None:
        """Signal the runtime window to close."""
        self._should_exit = True

    def is_running(self) -> bool:
        """Report whether runtime window should continue."""
        return self._running and not self._should_exit
