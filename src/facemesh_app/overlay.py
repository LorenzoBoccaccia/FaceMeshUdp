"""
Overlay module for FaceMesh application.
Handles pygame overlay rendering and HUD elements.
"""

import ctypes
import os
import sys
import time
from typing import Optional, Dict, Tuple, List

import pygame

from .facemesh_dao import clamp, CalibrationPoint


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
    
    def __init__(self, display: Dict, capture_enabled: bool = False, overlay_fps: int = 60,
                 calibration_mode: bool = False):
        """Initialize overlay manager.
        
        Args:
            display: Display configuration dict with 'width', 'height', 'x', 'y'
            capture_enabled: Whether capture mode is enabled
            overlay_fps: Target framerate for overlay rendering
            calibration_mode: Whether to enable calibration UI mode
        """
        self.display = display
        self.capture_enabled = capture_enabled
        self.overlay_fps = overlay_fps
        self.calibration_mode = calibration_mode
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
        
        # Calibration state attributes
        self.calibration_sequence: List[Dict] = []
        self.current_calib_idx: int = 0
        self.calib_phase: str = "idle"
        self.calib_phase_start: int = 0
        self.calib_samples: List[Dict] = []
    
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
        """Render face mesh data overlay.
        
        If calibration_mode is enabled, renders calibration UI instead of standard overlay.
        In normal mode, renders gaze dot showing calibrated eye gaze position.
        """
        if self.capture_enabled:
            self.screen.fill(BLACK)
        else:
            self.screen.fill(KEY_COLOR)
        
        # Render calibration UI if in calibration mode
        if self.calibration_mode and self.calib_phase != "idle":
            current_point = self.get_current_calib_point()
            if current_point:
                current_time = int(time.time() * 1000)
                elapsed_ms = current_time - self.calib_phase_start
                
                # Calculate countdown digit
                countdown_digit = None
                if self.calib_phase == "countdown":
                    remaining = (CALIB_COUNTDOWN_MS - elapsed_ms) // 1000
                    countdown_digit = max(1, int(remaining) + 1)
                    if countdown_digit > 3:
                        countdown_digit = 3
                
                self.render_calibration(current_point, self.calib_phase, elapsed_ms, countdown_digit)
        # Normal mode: render gaze dot showing calibrated eye gaze position
        elif not self.calibration_mode:
            self.render_gaze_dot(evt)
            # Render capture HUD if capture mode is enabled
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
    
    def start_calibration_sequence(self, width: float, height: float):
        """Start a 9-point calibration sequence.
        
        Creates calibration points: center, corners, and midpoints.
        Resets calibration state and starts with blink_pre phase.
        
        Args:
            width: Screen width in pixels
            height: Screen height in pixels
        """
        self.calibration_sequence = self._make_calib_seq(width, height)
        self.current_calib_idx = 0
        self.calib_phase = "blink_pre"
        self.calib_phase_start = int(time.time() * 1000)
        self.calib_samples = []
    
    def get_current_calib_point(self) -> Optional[Dict]:
        """Get the current calibration point.
        
        Returns:
            Current calibration point dict with keys 'name', 'x', 'y',
            or None if index is out of range
        """
        if 0 <= self.current_calib_idx < len(self.calibration_sequence):
            return self.calibration_sequence[self.current_calib_idx]
        return None
    
    def update_calibration_state(self, evt: Dict) -> Tuple[bool, Optional[CalibrationPoint]]:
        """Update calibration state based on elapsed time.
        
        Manages phase transitions through: blink_pre -> countdown -> sampling -> blink_post.
        Collects eye gaze samples during sampling phase.
        
        Args:
            evt: Face tracking event dict containing eye gaze data
            
        Returns:
            Tuple of (completed: bool, calibration_point: Optional[CalibrationPoint]).
            completed is True when all 9 points are finished.
            calibration_point is only returned when a single point is fully completed.
        """
        current_time = int(time.time() * 1000)
        elapsed_ms = current_time - self.calib_phase_start
        current_point = self.get_current_calib_point()
        
        if current_point is None:
            return True, None
        
        # Phase transitions
        if self.calib_phase == "blink_pre":
            if elapsed_ms >= CALIB_BLINK_MS:
                self.calib_phase = "countdown"
                self.calib_phase_start = current_time
                
        elif self.calib_phase == "countdown":
            if elapsed_ms >= CALIB_COUNTDOWN_MS:
                self.calib_phase = "sampling"
                self.calib_phase_start = current_time
                self.calib_samples = []
                
        elif self.calib_phase == "sampling":
            # Collect samples
            if evt:
                combined_yaw = evt.get("calibrated_combined_eye_gaze_yaw")
                combined_pitch = evt.get("calibrated_combined_eye_gaze_pitch")
                if combined_yaw is not None and combined_pitch is not None:
                    self.calib_samples.append({
                        "eye_yaw": combined_yaw,
                        "eye_pitch": combined_pitch,
                        "left_eye_yaw": evt.get("calibrated_left_eye_gaze_yaw") or 0.0,
                        "left_eye_pitch": evt.get("calibrated_left_eye_gaze_pitch") or 0.0,
                        "right_eye_yaw": evt.get("calibrated_right_eye_gaze_yaw") or 0.0,
                        "right_eye_pitch": evt.get("calibrated_right_eye_gaze_pitch") or 0.0,
                    })
            
            if elapsed_ms >= CALIB_AVG_MS:
                # Calculate averages and create CalibrationPoint
                if self.calib_samples:
                    avg_yaw = sum(s["eye_yaw"] for s in self.calib_samples) / len(self.calib_samples)
                    avg_pitch = sum(s["eye_pitch"] for s in self.calib_samples) / len(self.calib_samples)
                    avg_left_yaw = sum(s["left_eye_yaw"] for s in self.calib_samples) / len(self.calib_samples)
                    avg_left_pitch = sum(s["left_eye_pitch"] for s in self.calib_samples) / len(self.calib_samples)
                    avg_right_yaw = sum(s["right_eye_yaw"] for s in self.calib_samples) / len(self.calib_samples)
                    avg_right_pitch = sum(s["right_eye_pitch"] for s in self.calib_samples) / len(self.calib_samples)
                    
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
                        sample_count=len(self.calib_samples)
                    )
                else:
                    # No samples collected, create CalibrationPoint with zeros
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
                        sample_count=0
                    )
                
                self.calib_phase = "blink_post"
                self.calib_phase_start = current_time
                return False, calib_point
                
        elif self.calib_phase == "blink_post":
            if elapsed_ms >= CALIB_BLINK_MS:
                # Move to next point
                self.current_calib_idx += 1
                current_point = self.get_current_calib_point()
                
                if current_point is None:
                    # All points completed
                    return True, None
                
                # Reset for next point
                self.calib_phase = "blink_pre"
                self.calib_phase_start = current_time
                self.calib_samples = []
        
        return False, None
    
    def render_calibration(self, current_point: Dict, phase: str, elapsed_ms: int,
                           countdown_digit: Optional[int] = None):
        """Render calibration UI elements.
        
        Renders the calibration dot with appropriate color based on phase:
        - RED during countdown phase
        - GREEN during sampling phase
        - WHITE during blink phases
        
        Handles blinking during blink phases (220ms period).
        Shows countdown digit during countdown phase.
        
        Args:
            current_point: Dict with 'name', 'x', 'y' for current point position
            phase: Current calibration phase ('blink_pre', 'countdown', 'sampling', 'blink_post')
            elapsed_ms: Time elapsed in current phase (milliseconds)
            countdown_digit: Optional digit to display during countdown (3, 2, 1)
        """
        x = int(current_point["x"])
        y = int(current_point["y"])
        
        # Determine dot color based on phase
        if phase == "countdown":
            dot_color = RED
        elif phase == "sampling":
            dot_color = GREEN
        else:
            # Blink phases
            if (elapsed_ms // CALIB_BLINK_PERIOD_MS) % 2 == 0:
                dot_color = WHITE
            else:
                dot_color = None  # Don't draw during off cycle
        
        if dot_color:
            # Draw main dot
            pygame.draw.circle(self.screen, dot_color, (x, y), DOT_RADIUS)
            # Draw white ring around dot
            pygame.draw.circle(self.screen, WHITE, (x, y), DOT_RADIUS + 2, 2)
        
        # Show countdown digit during countdown phase
        if phase == "countdown" and countdown_digit is not None:
            digit_text = str(countdown_digit)
            text_img = self.font.render(digit_text, True, WHITE)
            text_rect = text_img.get_rect(center=(x, y))
            self.screen.blit(text_img, text_rect)
    
    def render_gaze_dot(self, evt: Optional[Dict]):
        """Render blue gaze dot showing calibrated eye gaze position in normal mode.
        
        Converts calibrated eye gaze angles to screen coordinates and draws a blue circle
        with white ring at the computed gaze position.
        
        Args:
            evt: Face tracking event dict containing calibrated gaze data
        """
        if not evt:
            return
        
        # Extract calibrated gaze angles
        gaze_yaw = evt.get("calibrated_combined_eye_gaze_yaw")
        gaze_pitch = evt.get("calibrated_combined_eye_gaze_pitch")
        
        # Only render if calibrated gaze data is available
        if gaze_yaw is None or gaze_pitch is None:
            return
        
        # Convert calibrated gaze angles to screen coordinates
        # Using 14.0 pixels per degree scaling factor
        screen_x = self.width / 2 + gaze_yaw * 14.0
        screen_y = self.height / 2 - gaze_pitch * 14.0  # Negative because screen Y increases downward
        
        # Clamp to screen bounds
        screen_x = clamp(screen_x, 0, self.width)
        screen_y = clamp(screen_y, 0, self.height)
        
        # Draw blue gaze dot with white ring
        pygame.draw.circle(self.screen, BLUE, (int(screen_x), int(screen_y)), DOT_RADIUS)
        pygame.draw.circle(self.screen, WHITE, (int(screen_x), int(screen_y)), DOT_RADIUS + 2, 2)
    
    def _make_calib_seq(self, width: float, height: float) -> List[Dict]:
        """Generate 9-point calibration sequence positions.
        
        Creates calibration points in order: C, TL, TC, TR, R, BR, BC, BL, L
        (Center, Top-Left, Top-Center, Top-Right, Right, Bottom-Right,
         Bottom-Center, Bottom-Left, Left)
        
        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            
        Returns:
            List of dicts with 'name', 'x', 'y' keys for each point
        """
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
