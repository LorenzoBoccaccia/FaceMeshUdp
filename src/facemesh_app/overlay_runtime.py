"""
Runtime overlay window showing where the user looks.
"""

from typing import Dict, Optional

import pyglet

from .calibration import CalibratedFaceAndGazeEvent
from .gaze_primitives import RGB_BY_COMPONENT, collect_gaze_primitives
from .overlay_common import DOT_RADIUS, OverlayWindow


class RuntimeOverlayManager:
    """Manage a transparent, click-through window drawing the gaze markers over the desktop."""

    def __init__(self, display: Dict, overlay_fps: int = 60):
        self._display = display
        self._overlay_fps = overlay_fps
        self._window: Optional[OverlayWindow] = None
        self._markers: Dict[str, pyglet.shapes.Circle] = {}

    def initialize(self):
        """Create and configure runtime overlay window."""
        self._window = OverlayWindow(
            self._display, "FaceMesh Gaze", self._overlay_fps, click_through=True
        )
        self._markers = {
            component: pyglet.shapes.Circle(
                0, 0, DOT_RADIUS, color=rgb, batch=self._window.batch
            )
            for component, rgb in RGB_BY_COMPONENT.items()
        }

    def shutdown(self):
        """Close runtime overlay resources."""
        if self._window is not None:
            self._window.close()
            self._window = None

    def handle_events(self):
        """Process window events."""
        self._window.poll()

    def render(self, calibrated_event: Optional[CalibratedFaceAndGazeEvent]):
        """Draw the current gaze markers."""
        points = {
            primitive.component: primitive.point
            for primitive in collect_gaze_primitives(calibrated_event)
        }
        for component, marker in self._markers.items():
            point = points.get(component)
            marker.visible = point is not None
            if point is not None:
                marker.position = self._window.window_point(*point)
        self._window.present()

    def is_running(self) -> bool:
        """Report whether runtime window should continue."""
        return self._window is not None and not self._window.exit_requested
