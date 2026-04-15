"""
Overlay module for FaceMesh application.
Handles pygame overlay rendering and HUD elements.

Coordinate System Convention:
- All yaw angles use left-positive convention
- All pitch angles use up-positive convention
- Screen mapping: +yaw moves left, +pitch moves up (negative screen Y)
"""

import ctypes
import os
import sys
import time
from typing import Optional, Dict, Tuple, List

import pygame

from .facemesh_dao import clamp, safe_float, CalibrationPoint


# Colors
KEY_COLOR = (1, 0, 1)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 180, 255)
HUD_BG = (20, 20, 20)
RED = (255, 40, 40)
GREEN = (80, 230, 120)

# Calibration constants
CALIB_INSET = 50
CALIB_BLINK_MS = 1000
CALIB_COUNTDOWN_MS = 3000
CALIB_AVG_MS = 500
CALIB_BLINK_PERIOD_MS = 220
DOT_RADIUS = 14


def set_window_transparent(hwnd):
    """Set window to be transparent with color key."""
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
    """Set window to always be on top."""
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


def draw_mouse_triangle(surface, x: float, y: float, color=WHITE):
    """Draw downward-pointing marker under the current mouse cursor."""
    tip = (int(round(x)), int(round(y + 14)))
    left = (int(round(x - 10)), int(round(y - 2)))
    right = (int(round(x + 10)), int(round(y - 2)))
    pygame.draw.polygon(surface, color, (left, right, tip), 0)


def draw_capture_hud(
    surface, font_small, mouse_x: float, mouse_y: float, evt: Optional[Dict], fps: float
):
    """Draw capture HUD near the cursor with live face and gaze values."""
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


class OverlayManager:
    """Manages pygame overlay rendering and interaction."""

    def __init__(
        self,
        display: Dict,
        capture_enabled: bool = False,
        overlay_fps: int = 60,
        calibration_mode: bool = False,
    ):
        self._display = display
        self._capture_enabled = capture_enabled
        self._overlay_fps = overlay_fps
        self._calibration_mode = calibration_mode
        self._width = int(display["width"])
        self._height = int(display["height"])

        self._screen = None
        self._clock = None
        self._font = None
        self._font_small = None
        self._hwnd = None

        self._mouse_x = self._width / 2.0
        self._mouse_y = self._height / 2.0

        self._running = False
        self._should_exit = False

        self._calibration_sequence: List[Dict] = []
        self._current_calib_idx: int = 0
        self._calib_phase: str = "idle"
        self._calib_phase_start: int = 0
        self._calib_samples: List[Dict] = []

    def initialize(self):
        """Initialize pygame and create overlay window."""
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
        set_window_transparent(self._hwnd)
        set_window_topmost(self._hwnd)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 80)
        self._font_small = pygame.font.Font(None, 24)
        self._running = True

    def shutdown(self):
        """Shutdown pygame overlay."""
        if self._screen:
            pygame.quit()
        self._running = False

    def handle_events(self, on_mouse_motion=None, on_click=None):
        """Handle pygame events."""
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

    def render_mesh(self, evt: Optional[Dict]):
        """Render face mesh data overlay."""
        if self._capture_enabled:
            self._screen.fill(BLACK)
        else:
            self._screen.fill(KEY_COLOR)

        if self._calibration_mode and self._calib_phase != "idle":
            current_point = self.get_current_calib_point()
            if current_point:
                current_time = int(time.time() * 1000)
                elapsed_ms = current_time - self._calib_phase_start

                countdown_digit = None
                if self._calib_phase == "countdown":
                    remaining = (CALIB_COUNTDOWN_MS - elapsed_ms) // 1000
                    countdown_digit = max(1, int(remaining) + 1)
                    if countdown_digit > 3:
                        countdown_digit = 3

                self.render_calibration(
                    current_point, self._calib_phase, elapsed_ms, countdown_digit
                )
        elif not self._calibration_mode:
            self.render_gaze_dot(evt)
            if self._capture_enabled and evt:
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

        pygame.display.update()
        self._clock.tick(max(1, int(self._overlay_fps)))

    def clear(self):
        """Clear screen."""
        if self._capture_enabled:
            self._screen.fill(BLACK)
        else:
            self._screen.fill(KEY_COLOR)

    def update(self):
        """Update display."""
        pygame.display.update()
        if self._clock:
            self._clock.tick(max(1, int(self._overlay_fps)))

    def get_mouse_position(self) -> Tuple[float, float]:
        """Get current mouse position."""
        return self._mouse_x, self._mouse_y

    def start_calibration_sequence(self, width: float, height: float):
        """Start a 9-point calibration sequence."""
        self._calibration_sequence = self._make_calib_seq(width, height)
        self._current_calib_idx = 0
        self._calib_phase = "blink_pre"
        self._calib_phase_start = int(time.time() * 1000)
        self._calib_samples = []

    def get_current_calib_point(self) -> Optional[Dict]:
        if 0 <= self._current_calib_idx < len(self._calibration_sequence):
            return self._calibration_sequence[self._current_calib_idx]
        return None

    def update_calibration_state(
        self, evt: Dict
    ) -> Tuple[bool, Optional[CalibrationPoint]]:
        """Update calibration state based on elapsed time.

        Manages phase transitions through: blink_pre -> countdown -> sampling -> blink_post.
        Collects eye gaze samples during sampling phase and returns a complete
        CalibrationPoint with head pose and zeta data included.
        """
        current_time = int(time.time() * 1000)
        elapsed_ms = current_time - self._calib_phase_start
        current_point = self.get_current_calib_point()
        fallback_zeta = max(self._width, self._height) * 0.75

        if current_point is None:
            return True, None

        if self._calib_phase == "blink_pre":
            if elapsed_ms >= CALIB_BLINK_MS:
                self._calib_phase = "countdown"
                self._calib_phase_start = current_time

        elif self._calib_phase == "countdown":
            if elapsed_ms >= CALIB_COUNTDOWN_MS:
                self._calib_phase = "sampling"
                self._calib_phase_start = current_time
                self._calib_samples = []

        elif self._calib_phase == "sampling":
            if evt:
                combined_yaw = evt.get("raw_combined_eye_gaze_yaw")
                combined_pitch = evt.get("raw_combined_eye_gaze_pitch")
                if combined_yaw is not None and combined_pitch is not None:
                    self._calib_samples.append(
                        {
                            "eye_yaw": combined_yaw,
                            "eye_pitch": combined_pitch,
                            "left_eye_yaw": evt.get("raw_left_eye_gaze_yaw") or 0.0,
                            "left_eye_pitch": evt.get("raw_left_eye_gaze_pitch")
                            or 0.0,
                            "right_eye_yaw": evt.get("raw_right_eye_gaze_yaw") or 0.0,
                            "right_eye_pitch": evt.get("raw_right_eye_gaze_pitch")
                            or 0.0,
                        }
                    )

            if elapsed_ms >= CALIB_AVG_MS:
                head_yaw = safe_float(evt.get("head_yaw") if evt else None, 0.0)
                head_pitch = safe_float(evt.get("head_pitch") if evt else None, 0.0)
                zeta = safe_float(evt.get("zeta") if evt else None, fallback_zeta)

                if self._calib_samples:
                    avg_yaw = sum(s["eye_yaw"] for s in self._calib_samples) / len(
                        self._calib_samples
                    )
                    avg_pitch = sum(s["eye_pitch"] for s in self._calib_samples) / len(
                        self._calib_samples
                    )
                    avg_left_yaw = sum(
                        s["left_eye_yaw"] for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_left_pitch = sum(
                        s["left_eye_pitch"] for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_right_yaw = sum(
                        s["right_eye_yaw"] for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_right_pitch = sum(
                        s["right_eye_pitch"] for s in self._calib_samples
                    ) / len(self._calib_samples)

                    calib_point = CalibrationPoint(
                        name=current_point["name"],
                        screen_x=current_point["x"],
                        screen_y=current_point["y"],
                        raw_eye_yaw=avg_yaw,
                        raw_eye_pitch=avg_pitch,
                        raw_left_eye_yaw=avg_left_yaw,
                        raw_left_eye_pitch=avg_left_pitch,
                        raw_right_eye_yaw=avg_right_yaw,
                        raw_right_eye_pitch=avg_right_pitch,
                        sample_count=len(self._calib_samples),
                        head_yaw=head_yaw,
                        head_pitch=head_pitch,
                        zeta=zeta,
                    )
                else:
                    calib_point = CalibrationPoint(
                        name=current_point["name"],
                        screen_x=current_point["x"],
                        screen_y=current_point["y"],
                        raw_eye_yaw=0.0,
                        raw_eye_pitch=0.0,
                        raw_left_eye_yaw=0.0,
                        raw_left_eye_pitch=0.0,
                        raw_right_eye_yaw=0.0,
                        raw_right_eye_pitch=0.0,
                        sample_count=0,
                        head_yaw=head_yaw,
                        head_pitch=head_pitch,
                        zeta=zeta,
                    )

                self._calib_phase = "blink_post"
                self._calib_phase_start = current_time
                return False, calib_point

        elif self._calib_phase == "blink_post":
            if elapsed_ms >= CALIB_BLINK_MS:
                self._current_calib_idx += 1
                current_point = self.get_current_calib_point()

                if current_point is None:
                    return True, None

                self._calib_phase = "blink_pre"
                self._calib_phase_start = current_time
                self._calib_samples = []

        return False, None

    def render_calibration(
        self,
        current_point: Dict,
        phase: str,
        elapsed_ms: int,
        countdown_digit: Optional[int] = None,
    ):
        """Render calibration UI elements."""
        x = int(current_point["x"])
        y = int(current_point["y"])

        if phase == "countdown":
            dot_color = RED
        elif phase == "sampling":
            dot_color = GREEN
        else:
            if (elapsed_ms // CALIB_BLINK_PERIOD_MS) % 2 == 0:
                dot_color = WHITE
            else:
                dot_color = None

        if dot_color:
            pygame.draw.circle(self._screen, dot_color, (x, y), DOT_RADIUS)
            pygame.draw.circle(self._screen, WHITE, (x, y), DOT_RADIUS + 2, 2)

        if phase == "countdown" and countdown_digit is not None:
            digit_text = str(countdown_digit)
            text_img = self._font.render(digit_text, True, WHITE)
            text_rect = text_img.get_rect(center=(x, y))
            self._screen.blit(text_img, text_rect)

    def render_gaze_dot(self, evt: Optional[Dict]):
        """Render three gaze dots showing different angle combinations."""
        if not evt:
            return

        head_yaw = evt.get("head_yaw")
        head_pitch = evt.get("head_pitch")

        eye_yaw = evt.get("raw_combined_eye_gaze_yaw")
        eye_pitch = evt.get("raw_combined_eye_gaze_pitch")

        if head_yaw is None or head_pitch is None:
            return
        if eye_yaw is None or eye_pitch is None:
            return

        SCALE = 14.0

        red_x = self._width / 2 - head_yaw * SCALE
        red_y = self._height / 2 - head_pitch * SCALE
        red_x = clamp(red_x, 0, self._width)
        red_y = clamp(red_y, 0, self._height)

        green_x = self._width / 2 - eye_yaw * SCALE
        green_y = self._height / 2 - eye_pitch * SCALE
        green_x = clamp(green_x, 0, self._width)
        green_y = clamp(green_y, 0, self._height)

        blue_x = self._width / 2 - (head_yaw + eye_yaw) * SCALE
        blue_y = self._height / 2 - (head_pitch + eye_pitch) * SCALE
        blue_x = clamp(blue_x, 0, self._width)
        blue_y = clamp(blue_y, 0, self._height)

        pygame.draw.circle(self._screen, RED, (int(red_x), int(red_y)), DOT_RADIUS)
        pygame.draw.circle(
            self._screen, WHITE, (int(red_x), int(red_y)), DOT_RADIUS + 2, 2
        )

        pygame.draw.circle(
            self._screen, GREEN, (int(green_x), int(green_y)), DOT_RADIUS
        )
        pygame.draw.circle(
            self._screen, WHITE, (int(green_x), int(green_y)), DOT_RADIUS + 2, 2
        )

        pygame.draw.circle(self._screen, BLUE, (int(blue_x), int(blue_y)), DOT_RADIUS)
        pygame.draw.circle(
            self._screen, WHITE, (int(blue_x), int(blue_y)), DOT_RADIUS + 2, 2
        )

    def _make_calib_seq(self, width: float, height: float) -> List[Dict]:
        """Generate 9-point calibration sequence positions."""
        inset = CALIB_INSET
        w = width
        h = height

        return [
            {"name": "C", "x": w / 2, "y": h / 2},
            {"name": "TL", "x": inset, "y": inset},
            {"name": "TC", "x": w / 2, "y": inset},
            {"name": "TR", "x": w - inset, "y": inset},
            {"name": "R", "x": w - inset, "y": h / 2},
            {"name": "BR", "x": w - inset, "y": h - inset},
            {"name": "BC", "x": w / 2, "y": h - inset},
            {"name": "BL", "x": inset, "y": h - inset},
            {"name": "L", "x": inset, "y": h / 2},
        ]

    def request_exit(self) -> None:
        """Signal the overlay to exit on the next event loop iteration."""
        self._should_exit = True

    def is_running(self) -> bool:
        """Check if overlay is still running."""
        return self._running and not self._should_exit


def get_display_geo() -> Dict:
    """Get display geometry for current platform."""
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        return {
            "name": "Primary Display",
            "x": 0,
            "y": 0,
            "width": int(user32.GetSystemMetrics(0)),
            "height": int(user32.GetSystemMetrics(1)),
        }
    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    if sizes:
        w, h = sizes[0]
    else:
        info = pygame.display.Info()
        w, h = info.current_w, info.current_h
    return {"name": "Display", "x": 0, "y": 0, "width": int(w), "height": int(h)}
