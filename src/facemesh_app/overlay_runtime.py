"""
Runtime overlay window showing where the user looks.
"""

import os
from typing import Dict, Optional

import pygame

from .calibration import CalibratedFaceAndGazeEvent
from .gaze_primitives import collect_gaze_primitives, draw_gaze_primitives_pygame
from .overlay_common import (
    DOT_RADIUS,
    KEY_COLOR,
    set_window_click_through,
    set_window_topmost,
    set_window_transparent,
)


class RuntimeOverlayManager:
    """Manage a transparent, click-through window drawing the gaze markers over the desktop."""

    def __init__(self, display: Dict, overlay_fps: int = 60):
        self._display = display
        self._overlay_fps = overlay_fps
        self._width = int(display["width"])
        self._height = int(display["height"])

        self._screen = None
        self._clock = None
        self._hwnd = None

        self._running = False
        self._should_exit = False

    def initialize(self):
        """Create and configure runtime overlay window."""
        pygame.init()
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
        set_window_click_through(self._hwnd)
        self._clock = pygame.time.Clock()
        self._running = True

    def shutdown(self):
        """Close runtime overlay resources."""
        if self._screen:
            pygame.quit()
        self._running = False

    def handle_events(self):
        """Process window events."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._should_exit = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self._should_exit = True

    def render(self, calibrated_event: Optional[CalibratedFaceAndGazeEvent]):
        """Draw the current gaze markers."""
        self._screen.fill(KEY_COLOR)
        draw_gaze_primitives_pygame(
            self._screen, collect_gaze_primitives(calibrated_event), radius=DOT_RADIUS
        )
        pygame.display.update()
        self._clock.tick(max(1, int(self._overlay_fps)))

    def is_running(self) -> bool:
        """Report whether runtime window should continue."""
        return self._running and not self._should_exit
