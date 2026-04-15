"""
Calibration overlay window for 9-point calibration flow.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import pygame

from .calibration import CalibrationPoint
from .facemesh_dao import safe_float
from .overlay_common import (
    BLACK,
    DOT_RADIUS,
    GREEN,
    RED,
    WHITE,
    set_window_topmost,
)


CALIB_INSET = 50
CALIB_BLINK_MS = 1000
CALIB_COUNTDOWN_MS = 1000
CALIB_AVG_MS = 500
CALIB_BLINK_PERIOD_MS = 220


class CalibrationOverlayManager:
    """Manage calibration overlay window and calibration state machine."""

    def __init__(
        self,
        display: Dict,
        overlay_fps: int = 60,
    ):
        self._display = display
        self._overlay_fps = overlay_fps
        self._width = int(display["width"])
        self._height = int(display["height"])

        self._screen = None
        self._clock = None
        self._hwnd = None

        self._running = False
        self._should_exit = False

        self._calibration_sequence: List[Dict] = []
        self._current_calib_idx: int = 0
        self._calib_phase: str = "idle"
        self._calib_phase_start: int = 0
        self._calib_samples: List[Dict] = []

    def initialize(self):
        """Create and configure calibration overlay window."""
        pygame.init()
        pygame.font.init()
        os.environ.setdefault(
            "SDL_VIDEO_WINDOW_POS", f"{self._display['x']},{self._display['y']}"
        )
        self._screen = pygame.display.set_mode(
            (self._width, self._height), pygame.NOFRAME
        )
        pygame.display.set_caption("FaceMesh Calibration")
        self._hwnd = pygame.display.get_wm_info().get("window")
        set_window_topmost(self._hwnd)
        self._clock = pygame.time.Clock()
        self._running = True

    def shutdown(self):
        """Close calibration overlay resources."""
        if self._screen:
            pygame.quit()
        self._running = False

    def handle_events(self):
        """Process calibration window events."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._should_exit = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self._should_exit = True

    def render_mesh(self, evt: Optional[Dict]):
        """Render one calibration frame."""
        _ = evt
        self._screen.fill(BLACK)
        if self._calib_phase != "idle":
            current_point = self.get_current_calib_point()
            if current_point:
                current_time = int(time.time() * 1000)
                elapsed_ms = current_time - self._calib_phase_start
                self.render_calibration(current_point, self._calib_phase, elapsed_ms)
        pygame.display.update()
        self._clock.tick(max(1, int(self._overlay_fps)))

    def clear(self):
        """Clear calibration overlay surface."""
        self._screen.fill(BLACK)

    def update(self):
        """Flip buffers and enforce target frame rate."""
        pygame.display.update()
        if self._clock:
            self._clock.tick(max(1, int(self._overlay_fps)))

    def start_calibration_sequence(self, width: float, height: float):
        """Initialize calibration sequence targets and reset state."""
        self._calibration_sequence = self._make_calib_seq(width, height)
        self._current_calib_idx = 0
        self._calib_phase = "blink_pre"
        self._calib_phase_start = int(time.time() * 1000)
        self._calib_samples = []

    def get_current_calib_point(self) -> Optional[Dict]:
        """Return current calibration target."""
        if 0 <= self._current_calib_idx < len(self._calibration_sequence):
            return self._calibration_sequence[self._current_calib_idx]
        return None

    def get_calibration_phase(self) -> str:
        """Return current calibration phase."""
        return self._calib_phase

    def update_calibration_state(
        self, evt: Dict
    ) -> Tuple[bool, Optional[CalibrationPoint]]:
        """Advance calibration state and emit points when sampling completes."""
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
                    left_eye_yaw = evt.get("raw_left_eye_gaze_yaw")
                    left_eye_pitch = evt.get("raw_left_eye_gaze_pitch")
                    right_eye_yaw = evt.get("raw_right_eye_gaze_yaw")
                    right_eye_pitch = evt.get("raw_right_eye_gaze_pitch")
                    head_yaw = evt.get("head_yaw")
                    head_pitch = evt.get("head_pitch")
                    zeta = evt.get("zeta")
                    head_x = evt.get("head_x")
                    head_y = evt.get("head_y")
                    head_z = evt.get("head_z")
                    self._calib_samples.append(
                        {
                            "eye_yaw": combined_yaw,
                            "eye_pitch": combined_pitch,
                            "left_eye_yaw": (
                                left_eye_yaw if left_eye_yaw is not None else combined_yaw
                            ),
                            "left_eye_pitch": (
                                left_eye_pitch
                                if left_eye_pitch is not None
                                else combined_pitch
                            ),
                            "right_eye_yaw": (
                                right_eye_yaw
                                if right_eye_yaw is not None
                                else combined_yaw
                            ),
                            "right_eye_pitch": (
                                right_eye_pitch
                                if right_eye_pitch is not None
                                else combined_pitch
                            ),
                            "head_yaw": head_yaw,
                            "head_pitch": head_pitch,
                            "zeta": zeta,
                            "head_x": head_x,
                            "head_y": head_y,
                            "head_z": head_z,
                        }
                    )

            if elapsed_ms >= CALIB_AVG_MS:
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
                    avg_head_yaw = sum(
                        safe_float(s.get("head_yaw"), 0.0) for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_head_pitch = sum(
                        safe_float(s.get("head_pitch"), 0.0)
                        for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_zeta = sum(
                        safe_float(s.get("zeta"), fallback_zeta)
                        for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_head_x = sum(
                        safe_float(s.get("head_x"), 0.0) for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_head_y = sum(
                        safe_float(s.get("head_y"), 0.0) for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_head_z = sum(
                        safe_float(s.get("head_z"), avg_zeta)
                        for s in self._calib_samples
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
                        head_yaw=avg_head_yaw,
                        head_pitch=avg_head_pitch,
                        zeta=avg_zeta,
                        head_x=avg_head_x,
                        head_y=avg_head_y,
                        head_z=avg_head_z,
                    )
                else:
                    head_yaw = safe_float(evt.get("head_yaw") if evt else None, 0.0)
                    head_pitch = safe_float(evt.get("head_pitch") if evt else None, 0.0)
                    zeta = safe_float(evt.get("zeta") if evt else None, fallback_zeta)
                    head_x = safe_float(evt.get("head_x") if evt else None, 0.0)
                    head_y = safe_float(evt.get("head_y") if evt else None, 0.0)
                    head_z = safe_float(evt.get("head_z") if evt else None, zeta)
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
                        head_x=head_x,
                        head_y=head_y,
                        head_z=head_z,
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
    ):
        """Render calibration target for current phase."""
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

    def _make_calib_seq(self, width: float, height: float) -> List[Dict]:
        """Generate the calibration target list."""
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
        """Signal the calibration window to close."""
        self._should_exit = True

    def is_running(self) -> bool:
        """Report whether calibration window should continue."""
        return self._running and not self._should_exit
