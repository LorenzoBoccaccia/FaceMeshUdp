"""
Overlay window shared by the calibration and runtime overlays, and the display geometry they cover.
"""

import ctypes
import math
import sys
import time
from typing import Dict, Optional, Tuple

import pyglet

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 180, 255)
RED = (255, 40, 40)
GREEN = (80, 230, 120)
DOT_RADIUS = 14


class OverlayWindow:
    """A borderless window covering the display above all other windows, redrawn at most at a set rate.

    The window is see-through wherever nothing is drawn unless it is given a background. A
    click-through window lets the mouse reach whatever lies underneath; otherwise it collects left
    clicks. Escape or closing the window asks the overlay to end.
    """

    def __init__(
        self,
        display: Dict,
        caption: str,
        fps: int,
        click_through: bool,
        background: Optional[Tuple[int, int, int]] = None,
    ):
        self.width = int(display["width"])
        self.height = int(display["height"])
        self.batch = pyglet.graphics.Batch()
        self._frame_interval = 1.0 / max(1, int(fps))
        self._last_frame = -math.inf
        self._clicked = False
        self._window = pyglet.window.Window(
            self.width,
            self.height,
            caption=caption,
            style=pyglet.window.Window.WINDOW_STYLE_OVERLAY,
            vsync=False,
            visible=False,
        )
        self._window.set_location(int(display["x"]), int(display["y"]))
        self._window.set_mouse_passthrough(click_through)
        if background is not None:
            self._window.switch_to()
            pyglet.gl.glClearColor(*(channel / 255.0 for channel in background), 1.0)
        self._window.push_handlers(on_mouse_press=self._on_mouse_press)
        self._window.set_visible(True)

    @property
    def exit_requested(self) -> bool:
        """Whether the user pressed Escape or closed the window."""
        return self._window.has_exit

    def window_point(self, x: float, y: float) -> Tuple[float, float]:
        """Drawing position of a display pixel counted from the top-left corner."""
        return x, self.height - y

    def poll(self) -> None:
        """Take in the window's pending input."""
        self._window.dispatch_events()

    def take_click(self) -> bool:
        """Whether a left click arrived since the last call."""
        clicked, self._clicked = self._clicked, False
        return clicked

    def present(self) -> None:
        """Show the batch's current content, unless the last frame is more recent than the frame rate allows."""
        now = time.perf_counter()
        if now - self._last_frame < self._frame_interval:
            return
        self._last_frame = now
        self._window.switch_to()
        self._window.clear()
        self.batch.draw()
        self._window.flip()

    def close(self) -> None:
        """Remove the window from the screen."""
        self._window.close()

    def _on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self._clicked = True


def _physical_size_mm() -> Tuple[int, int]:
    if sys.platform != "win32":
        return 0, 0
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    HORZSIZE = 4
    VERTSIZE = 6
    hdc = user32.GetDC(0)
    if not hdc:
        return 0, 0
    width_mm = int(gdi32.GetDeviceCaps(hdc, HORZSIZE))
    height_mm = int(gdi32.GetDeviceCaps(hdc, VERTSIZE))
    user32.ReleaseDC(0, hdc)
    return max(width_mm, 0), max(height_mm, 0)


def get_display_geo() -> Dict:
    """Return primary display geometry, in the pixels overlay windows use, with its physical size in mm when known."""
    screen = pyglet.display.get_display().get_default_screen()
    width_mm, height_mm = _physical_size_mm()
    return {
        "name": "Primary Display",
        "x": int(screen.x),
        "y": int(screen.y),
        "width": int(screen.width),
        "height": int(screen.height),
        "width_mm": width_mm,
        "height_mm": height_mm,
    }
