"""
Calibration overlay window for 9-point calibration flow.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import pygame

from .calibration import CalibrationPoint
from .facemesh_dao import FaceMeshEvent
from .overlay_common import (
    BLACK,
    BLUE,
    DOT_RADIUS,
    GREEN,
    RED,
    WHITE,
    set_window_topmost,
)


CALIB_INSET = 50
CALIB_BLINK_MS = 500
CALIB_CAPTURE_MS = 500
CALIB_BLINK_PERIOD_MS = 120


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
        self._font = None
        self._hwnd = None

        self._running = False
        self._should_exit = False

        self._calibration_sequence: List[Dict] = []
        self._current_calib_idx: int = 0
        self._calib_phase: str = "idle"
        self._calib_phase_start: int = 0
        self._calib_samples: List[FaceMeshEvent] = []
        self._retry_point = False
        self._center_x: float = 0.0
        self._center_y: float = 0.0
        self._click_pending: bool = False

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
        self._font = pygame.font.Font(None, 34)
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
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self._click_pending = True

    def render(self):
        """Render one calibration frame."""
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
        self._calib_phase = "wait_click"
        self._calib_phase_start = int(time.time() * 1000)
        self._calib_samples = []
        self._center_x = float(width) / 2.0
        self._center_y = float(height) / 2.0
        self._click_pending = False

    def get_current_calib_point(self) -> Optional[Dict]:
        """Return current calibration target."""
        if 0 <= self._current_calib_idx < len(self._calibration_sequence):
            return self._calibration_sequence[self._current_calib_idx]
        return None

    def get_calibration_phase(self) -> str:
        """Return current calibration phase."""
        return self._calib_phase

    def is_aligned(self) -> bool:
        """Report whether the user has committed to the current target."""
        return self._calib_phase in ("blink", "capture")

    def update_calibration_state(
        self, evt: Optional[FaceMeshEvent]
    ) -> Tuple[bool, Optional[CalibrationPoint]]:
        """Advance calibration and emit a point once its capture window held usable face frames."""
        current_time = int(time.time() * 1000)
        elapsed_ms = current_time - self._calib_phase_start
        current_point = self.get_current_calib_point()
        if current_point is None:
            return True, None

        if self._calib_phase == "wait_click":
            if self._click_pending:
                self._enter_phase("blink", current_time)
        elif self._calib_phase == "blink":
            self._click_pending = False
            if elapsed_ms >= CALIB_BLINK_MS:
                self._enter_phase("capture", current_time)
        elif self._calib_phase == "capture":
            if evt is not None:
                self._calib_samples.append(evt)
            if elapsed_ms >= CALIB_CAPTURE_MS:
                calib_point = CalibrationPoint.from_samples(
                    current_point["name"],
                    (current_point["nose_x"], current_point["nose_y"]),
                    (current_point["eye_x"], current_point["eye_y"]),
                    self._calib_samples,
                )
                self._retry_point = calib_point is None
                if calib_point is not None:
                    self._current_calib_idx += 1
                self._enter_phase("wait_click", current_time)
                return self.get_current_calib_point() is None, calib_point

        return False, None

    def _enter_phase(self, phase: str, start_ms: int) -> None:
        self._calib_phase = phase
        self._calib_phase_start = start_ms
        self._calib_samples = []
        self._click_pending = False

    def render_calibration(
        self,
        current_point: Dict,
        phase: str,
        elapsed_ms: int,
    ):
        """Render calibration target for current phase."""
        nose_x = int(round(current_point["nose_x"]))
        nose_y = int(round(current_point["nose_y"]))
        eye_x = int(round(current_point["eye_x"]))
        eye_y = int(round(current_point["eye_y"]))

        if phase == "blink":
            show = (elapsed_ms // CALIB_BLINK_PERIOD_MS) % 2 == 0
            nose_color = WHITE if show else None
            eye_color = WHITE if show else None
        elif phase == "capture":
            nose_color = BLUE
            eye_color = BLUE
        else:
            nose_color = RED
            eye_color = GREEN

        if nose_color is not None:
            pygame.draw.circle(self._screen, nose_color, (nose_x, nose_y), DOT_RADIUS)
            pygame.draw.circle(self._screen, WHITE, (nose_x, nose_y), DOT_RADIUS + 2, 2)
        if eye_color is not None and (eye_x, eye_y) != (nose_x, nose_y):
            pygame.draw.circle(self._screen, eye_color, (eye_x, eye_y), DOT_RADIUS)
            pygame.draw.circle(self._screen, WHITE, (eye_x, eye_y), DOT_RADIUS + 2, 2)

        instruction = current_point.get("instruction", "")
        label = f"Step {self._current_calib_idx + 1}/{len(self._calibration_sequence)}"
        phase_hint = {
            "wait_click": (
                "No face seen — align and CLICK again."
                if self._retry_point
                else "Align, hold steady, then CLICK."
            ),
            "blink": "Hold still — capturing…",
            "capture": "Keep holding — sampling…",
        }.get(phase, "")

        lines = [label]
        lines.extend(line for line in instruction.split("\n") if line)
        if phase_hint:
            lines.append(phase_hint)

        base_y = int(self._height * 3 / 4)
        line_height = self._font.get_linesize()
        for i, line in enumerate(lines):
            text_surface = self._font.render(line, True, WHITE)
            text_rect = text_surface.get_rect(
                center=(self._width // 2, base_y + i * line_height)
            )
            self._screen.blit(text_surface, text_rect)

    def _make_calib_seq(self, width: float, height: float) -> List[Dict]:
        """Generate the calibration target list."""
        inset = CALIB_INSET
        w = width
        h = height
        center_x = w / 2
        center_y = h / 2
        t = (center_x, inset)
        b = (center_x, h - inset)
        l = (inset, center_y)
        r = (w - inset, center_y)
        tl = (inset, inset)
        tr = (w - inset, inset)
        br = (w - inset, h - inset)
        bl = (inset, h - inset)

        click_instruction = (
            "Turn your HEAD so your nose points at the RED dot.\n"
            "Keep your EYES fixed on the GREEN dot (do not move your head to follow it)."
        )

        return [
            {
                "name": "C",
                "nose_x": center_x,
                "nose_y": center_y,
                "eye_x": center_x,
                "eye_y": center_y,
                "instruction": (
                    "Face the screen squarely and look at the CENTRE dot.\n"
                    "Keep your head and eyes aligned straight ahead."
                ),
            },
            {
                "name": "T",
                "nose_x": t[0],
                "nose_y": t[1],
                "eye_x": b[0],
                "eye_y": b[1],
                "instruction": click_instruction,
            },
            {
                "name": "TL",
                "nose_x": tl[0],
                "nose_y": tl[1],
                "eye_x": br[0],
                "eye_y": br[1],
                "instruction": click_instruction,
            },
            {
                "name": "L",
                "nose_x": l[0],
                "nose_y": l[1],
                "eye_x": r[0],
                "eye_y": r[1],
                "instruction": click_instruction,
            },
            {
                "name": "BL",
                "nose_x": bl[0],
                "nose_y": bl[1],
                "eye_x": tr[0],
                "eye_y": tr[1],
                "instruction": click_instruction,
            },
            {
                "name": "B",
                "nose_x": b[0],
                "nose_y": b[1],
                "eye_x": t[0],
                "eye_y": t[1],
                "instruction": click_instruction,
            },
            {
                "name": "BR",
                "nose_x": br[0],
                "nose_y": br[1],
                "eye_x": tl[0],
                "eye_y": tl[1],
                "instruction": click_instruction,
            },
            {
                "name": "R",
                "nose_x": r[0],
                "nose_y": r[1],
                "eye_x": l[0],
                "eye_y": l[1],
                "instruction": click_instruction,
            },
            {
                "name": "TR",
                "nose_x": tr[0],
                "nose_y": tr[1],
                "eye_x": bl[0],
                "eye_y": bl[1],
                "instruction": click_instruction,
            },
        ]

    def request_exit(self) -> None:
        """Signal the calibration window to close."""
        self._should_exit = True

    def is_running(self) -> bool:
        """Report whether calibration window should continue."""
        return self._running and not self._should_exit
