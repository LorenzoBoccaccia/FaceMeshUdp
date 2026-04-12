"""
Overlay module for FaceMesh application.
Handles pygame overlay rendering and HUD elements.
"""

import ctypes
import os
import sys
from typing import Optional, Dict, Tuple

import pygame

from .facemesh_dao import clamp


# Colors
KEY_COLOR = (1, 0, 1)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 180, 255)
HUD_BG = (20, 20, 20)


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


def draw_capture_hud(surface, font_small, dot_x: float, dot_y: float,
                   evt: Optional[Dict], fps: float):
    """Draw capture HUD with basic face tracking info."""
    has_face = bool(evt and evt.get("hasFace"))

    lines = [
        f"FACE: {'YES' if has_face else 'NO'}",
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


class OverlayManager:
    """Manages pygame overlay rendering and interaction."""
    
    def __init__(self, display: Dict, capture_enabled: bool = False, overlay_fps: int = 60):
        """Initialize overlay manager."""
        self.display = display
        self.capture_enabled = capture_enabled
        self.overlay_fps = overlay_fps
        self.width = int(display["width"])
        self.height = int(display["height"])
        
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self.hwnd = None
        
        self.mouse_x = self.width / 2.0
        self.mouse_y = self.height / 2.0
        
        self.running = False
        self.should_exit = False
    
    def initialize(self):
        """Initialize pygame and create overlay window."""
        pygame.init()
        pygame.font.init()
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", f"{self.display['x']},{self.display['y']}")
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.NOFRAME)
        pygame.display.set_caption("FaceMesh Gaze")
        self.hwnd = pygame.display.get_wm_info().get("window")
        set_window_transparent(self.hwnd)
        set_window_topmost(self.hwnd)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 80)
        self.font_small = pygame.font.Font(None, 24)
        self.running = True
    
    def shutdown(self):
        """Shutdown pygame overlay."""
        if self.screen:
            pygame.quit()
        self.running = False
    
    def handle_events(self, on_mouse_motion=None, on_click=None):
        """Handle pygame events."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.should_exit = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self.should_exit = True
            elif e.type == pygame.MOUSEMOTION:
                self.mouse_x = clamp(float(e.pos[0]), 0.0, self.width - 1.0)
                self.mouse_y = clamp(float(e.pos[1]), 0.0, self.height - 1.0)
                if on_mouse_motion:
                    on_mouse_motion(self.mouse_x, self.mouse_y)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if on_click and e.button == 1:
                    on_click(e.pos)
    
    def render_mesh(self, evt: Optional[Dict]):
        """Render face mesh data overlay."""
        if self.capture_enabled:
            self.screen.fill(BLACK)
        else:
            self.screen.fill(KEY_COLOR)
        
        # No gaze dot to draw - just render mesh data visualization if available
        if self.capture_enabled and evt:
            fps = self.clock.get_fps()
            # Use center of screen as default position for HUD
            draw_capture_hud(self.screen, self.font_small, self.width / 2, self.height / 2, evt, fps)
            draw_mouse_triangle(self.screen, self.mouse_x, self.mouse_y)
        
        pygame.display.update()
        self.clock.tick(max(1, int(self.overlay_fps)))
    
    def clear(self):
        """Clear screen."""
        if self.capture_enabled:
            self.screen.fill(BLACK)
        else:
            self.screen.fill(KEY_COLOR)
    
    def update(self):
        """Update display."""
        pygame.display.update()
        if self.clock:
            self.clock.tick(max(1, int(self.overlay_fps)))
    
    def get_mouse_position(self) -> Tuple[float, float]:
        """Get current mouse position."""
        return self.mouse_x, self.mouse_y
    
    def is_running(self) -> bool:
        """Check if overlay is still running."""
        return self.running and not self.should_exit


def get_display_geo() -> Dict:
    """Get display geometry for current platform."""
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        return {
            "name": "Primary Display",
            "x": 0,
            "y": 0,
            "width": int(user32.GetSystemMetrics(0)),
            "height": int(user32.GetSystemMetrics(1))
        }
    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    if sizes:
        w, h = sizes[0]
    else:
        info = pygame.display.Info()
        w, h = info.current_w, info.current_h
    return {"name": "Display", "x": 0, "y": 0, "width": int(w), "height": int(h)}
